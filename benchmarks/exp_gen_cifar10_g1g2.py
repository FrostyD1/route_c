#!/usr/bin/env python3
"""
CIFAR-10 Generation — G1+G2: Bandwidth Upgrade + Freq-Band Sampling
====================================================================
G1: Compare 16×16×8 (2048 bits) vs 32×32×16 (16384 bits) for generation
G2: Freq-band-scheduled sampling (commit low-freq bits first, then mid, then high)

Architecture matrix:
  A) 16×16×8  + standard sampling    (baseline from A2)
  B) 16×16×8  + freq_band sampling   (G2 on current z)
  C) 32×32×16 + standard sampling    (G1 bandwidth only)
  D) 32×32×16 + freq_band sampling   (G1+G2 combined)

Freq-band sampling (G2):
  - Map z positions to DCT frequency bands
  - Steps 1-5: commit only low-freq positions (global structure)
  - Steps 6-10: commit mid-freq positions (edges/detail)
  - Steps 11-15: commit high-freq positions (texture)
  - Key insight: z is binary latent, DCT decomposition is on decoded images,
    but we can use decoder feedback to identify which z positions drive which bands

4GB GPU: 3000 train, 500 test, batch_size=32

Usage:
    python3 -u benchmarks/exp_gen_cifar10_g1g2.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, argparse, json
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# DCT / FREQ UTILITIES
# ============================================================================

def dct2d(x):
    B, C, H, W = x.shape
    def dct_matrix(N):
        n = torch.arange(N, dtype=x.dtype, device=x.device)
        k = n.unsqueeze(1)
        D = torch.cos(np.pi * (2*n+1) * k / (2*N))
        D[0] *= 1.0/np.sqrt(N); D[1:] *= np.sqrt(2.0/N)
        return D
    DH = dct_matrix(H); DW = dct_matrix(W)
    return torch.einsum('hH,bcHW,wW->bchw', DH, x, DW)

def idct2d(X):
    B, C, H, W = X.shape
    def dct_matrix(N):
        n = torch.arange(N, dtype=X.dtype, device=X.device)
        k = n.unsqueeze(1)
        D = torch.cos(np.pi * (2*n+1) * k / (2*N))
        D[0] *= 1.0/np.sqrt(N); D[1:] *= np.sqrt(2.0/N)
        return D
    DH = dct_matrix(H).T; DW = dct_matrix(W).T
    return torch.einsum('Hh,bchw,Ww->bcHW', DH, X, DW)

def get_freq_masks(H, W, device='cpu'):
    fy = torch.arange(H, device=device).float()
    fx = torch.arange(W, device=device).float()
    freq_grid = fy.unsqueeze(1) + fx.unsqueeze(0)
    max_freq = H + W - 2
    t1, t2 = max_freq / 3.0, 2 * max_freq / 3.0
    return ((freq_grid <= t1).float(),
            ((freq_grid > t1) & (freq_grid <= t2)).float(),
            (freq_grid > t2).float())

def decompose_bands(x):
    B, C, H, W = x.shape
    low_m, mid_m, high_m = get_freq_masks(H, W, x.device)
    dct_x = dct2d(x)
    return idct2d(dct_x * low_m), idct2d(dct_x * mid_m), idct2d(dct_x * high_m)

def freq_scheduled_loss(x_pred, x_target, progress):
    pl, pm, ph = decompose_bands(x_pred)
    tl, tm, th = decompose_bands(x_target)
    w_low = 3.0
    w_mid = 1.0 * min(1.0, progress * 2)
    w_high = 0.5 * max(0.0, (progress - 0.3) / 0.7)
    return (w_low * F.mse_loss(pl, tl) +
            w_mid * F.mse_loss(pm, tm) +
            w_high * F.mse_loss(ph, th))

def hf_coherence_loss(pred_h, tgt_h):
    B, C, H, W = pred_h.shape
    def ac(x):
        xf = x.reshape(B*C, 1, H, W)
        le = F.avg_pool2d(xf**2, 3, 1, 1)
        lm = F.avg_pool2d(xf, 3, 1, 1)
        xs = F.pad(xf[:, :, :, :-1], (1, 0))
        lc = F.avg_pool2d(xf * xs, 3, 1, 1)
        v = (le - lm**2).clamp(min=1e-8)
        cv = lc - lm * F.avg_pool2d(xs, 3, 1, 1)
        return (cv / v).reshape(B, C, H, W)
    return F.mse_loss(ac(pred_h), ac(tgt_h))


# ============================================================================
# METRICS
# ============================================================================

def per_band_energy_distance(ig, ir, device='cpu'):
    def t(x):
        if isinstance(x, np.ndarray): x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 3: x = x.unsqueeze(1)
        return x.to(device)
    g = t(ig[:200]); r = t(ir[:200])
    dg = dct2d(g); dr = dct2d(r); H, W = g.shape[2], g.shape[3]
    res = {}
    for nm, m in zip(['low', 'mid', 'high'], get_freq_masks(H, W, device)):
        eg = (dg**2 * m).mean(dim=(0, 1)).sum().item()
        er = (dr**2 * m).mean(dim=(0, 1)).sum().item()
        res[f'energy_gap_{nm}'] = abs(eg - er) / (er + 1e-12)
    return res

def hf_coherence_metric(images, device='cpu'):
    if isinstance(images, np.ndarray): images = torch.tensor(images, dtype=torch.float32)
    if images.dim() == 3: images = images.unsqueeze(1)
    images = images[:200].to(device)
    _, _, xh = decompose_bands(images)
    xf = xh.reshape(-1, xh.shape[2], xh.shape[3])
    xm = xf.mean(dim=(1, 2), keepdim=True)
    xs = xf.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    xn = (xf - xm) / xs
    corrs = []
    for dy, dx in [(0, 1), (1, 0), (1, 1)]:
        a, b = xn, xn
        if dy > 0: a, b = a[:, dy:, :], b[:, :-dy, :]
        if dx > 0: a, b = a[:, :, dx:], b[:, :, :-dx]
        corrs.append((a * b).mean(dim=(1, 2)).mean().item())
    return np.mean(corrs)

def hf_noise_index(images, device='cpu'):
    if isinstance(images, np.ndarray): images = torch.tensor(images, dtype=torch.float32)
    if images.dim() == 3: images = images.unsqueeze(1)
    images = images[:200].to(device)
    gx = images[:, :, :, 1:] - images[:, :, :, :-1]
    gy = images[:, :, 1:, :] - images[:, :, :-1, :]
    ge = (gx**2).mean().item() + (gy**2).mean().item()
    _, _, xh = decompose_bands(images)
    return ge / ((xh**2).mean().item() + 1e-12)

def connectedness_proxy(images, threshold=0.3):
    if isinstance(images, torch.Tensor): images = images.cpu().numpy()
    if images.ndim == 4 and images.shape[1] == 3:
        images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
    elif images.ndim == 4:
        images = images[:, 0]
    scores = []
    for img in images[:100]:
        b = (img > threshold).astype(np.int32); tf = b.sum()
        if tf < 5: scores.append(0.0); continue
        H, W = b.shape; vis = np.zeros_like(b, dtype=bool); mc = 0
        for i in range(H):
            for j in range(W):
                if b[i, j] == 1 and not vis[i, j]:
                    st = [(i, j)]; vis[i, j] = True; sz = 0
                    while st:
                        ci, cj = st.pop(); sz += 1
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < H and 0 <= nj < W and b[ni, nj] == 1 and not vis[ni, nj]:
                                vis[ni, nj] = True; st.append((ni, nj))
                    mc = max(mc, sz)
        scores.append(mc / tf)
    return float(np.mean(scores))

def compute_diversity(z, n_pairs=500):
    N = len(z); zf = z.reshape(N, -1); d = []
    for _ in range(n_pairs):
        i, j = np.random.choice(N, 2, replace=False)
        d.append((zf[i] != zf[j]).float().mean().item())
    return np.mean(d)

def token_histogram_kl(z_real, z_gen, n_bits):
    N_r, K, H, W = z_real.shape; N_g = z_gen.shape[0]
    # For high-K, sample bit pairs instead of full tokens
    if K > 10:
        kls = []
        for _ in range(50):
            b1 = np.random.randint(K); b2 = np.random.randint(K)
            i = np.random.randint(H); j = np.random.randint(W)
            idx_r = z_real[:, b1, i, j].long() * 2 + z_real[:, b2, i, j].long()
            idx_g = z_gen[:, b1, i, j].long() * 2 + z_gen[:, b2, i, j].long()
            p = torch.bincount(idx_r, minlength=4).float() + 1
            q = torch.bincount(idx_g, minlength=4).float() + 1
            p /= p.sum(); q /= q.sum()
            kls.append((p * (p / q).log()).sum().item())
        return np.mean(kls)
    n_tokens = 2**K
    positions = [(i, j) for i in range(H) for j in range(W)]
    np.random.shuffle(positions); kls = []
    for i, j in positions[:50]:
        idx_r = torch.zeros(N_r, dtype=torch.long)
        idx_g = torch.zeros(N_g, dtype=torch.long)
        for b in range(K):
            idx_r += (z_real[:, b, i, j].long() << b)
            idx_g += (z_gen[:, b, i, j].long() << b)
        p = torch.bincount(idx_r, minlength=n_tokens).float() + 1
        q = torch.bincount(idx_g, minlength=n_tokens).float() + 1
        p /= p.sum(); q /= q.sum()
        kls.append((p * (p / q).log()).sum().item())
    return np.mean(kls)

def save_grid(images, path, nrow=8):
    try:
        from torchvision.utils import save_image
        if isinstance(images, np.ndarray): images = torch.tensor(images)
        if images.dim() == 3: images = images.unsqueeze(1)
        save_image(images[:64], path, nrow=nrow, normalize=False)
        print(f"    Grid: {path}")
    except Exception as e:
        print(f"    Grid fail: {e}")


# ============================================================================
# MODELS — 16×16 (stride-2 enc/dec, from A2)
# ============================================================================

class GumbelSigmoid(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__(); self.temperature = temperature
    def forward(self, logits):
        if self.training:
            u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            noisy = (logits - torch.log(-torch.log(u))) / self.temperature
        else:
            noisy = logits / self.temperature
        soft = torch.sigmoid(noisy); hard = (soft > 0.5).float()
        return hard - soft.detach() + soft
    def set_temperature(self, tau): self.temperature = tau


class Encoder16(nn.Module):
    """32×32×3 → 16×16×k (stride-2 downsample)."""
    def __init__(self, n_bits=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = self._res(64, 64)
        self.res2 = self._res(64, 64)
        self.head = nn.Conv2d(64, n_bits, 3, padding=1)
        self.q = GumbelSigmoid()
    def _res(self, ic, oc):
        return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(),
                             nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc))
    def forward(self, x):
        h = self.stem(x); h = F.relu(h + self.res1(h)); h = F.relu(h + self.res2(h))
        logits = self.head(h)
        return self.q(logits), logits
    def set_temperature(self, tau): self.q.set_temperature(tau)


class Decoder16(nn.Module):
    """16×16×k → 32×32×3 (stride-2 upsample)."""
    def __init__(self, n_bits=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ConvTranspose2d(n_bits, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = self._res(64, 64)
        self.res2 = self._res(64, 64)
        self.head = nn.Sequential(nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid())
    def _res(self, ic, oc):
        return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(),
                             nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc))
    def forward(self, z):
        h = self.stem(z); h = F.relu(h + self.res1(h)); h = F.relu(h + self.res2(h))
        return self.head(h)


# ============================================================================
# MODELS — 32×32 (1:1 spatial, no down/up sampling)
# ============================================================================

class Encoder32(nn.Module):
    """32×32×3 → 32×32×k (1:1 spatial)."""
    def __init__(self, n_bits=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, n_bits, 3, padding=1),
        )
        self.q = GumbelSigmoid()
    def forward(self, x):
        logits = self.net(x)
        return self.q(logits), logits
    def set_temperature(self, tau): self.q.set_temperature(tau)


class Decoder32(nn.Module):
    """32×32×k → 32×32×3 (1:1 spatial)."""
    def __init__(self, n_bits=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z): return self.net(z)


# ============================================================================
# E_CORE
# ============================================================================

class LocalEnergyCore(nn.Module):
    def __init__(self, n_bits):
        super().__init__(); self.n_bits = n_bits
        self.predictor = nn.Sequential(
            nn.Linear(9 * n_bits - 1, 64), nn.ReLU(), nn.Linear(64, 1))
    def get_context(self, z, bi, i, j):
        B, K, H, W = z.shape; ctx = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = (i + di) % H, (j + dj) % W
                for b in range(K):
                    if di == 0 and dj == 0 and b == bi: continue
                    ctx.append(z[:, b, ni, nj])
        return torch.stack(ctx, dim=1)
    def violation_rate(self, z):
        B, K, H, W = z.shape; v = []
        for _ in range(min(50, H * W * K)):
            b = torch.randint(K, (1,)).item()
            i = torch.randint(H, (1,)).item()
            j = torch.randint(W, (1,)).item()
            ctx = self.get_context(z, b, i, j)
            pred = (self.predictor(ctx).squeeze(1) > 0).float()
            v.append((pred != z[:, b, i, j]).float().mean().item())
        return np.mean(v)


# ============================================================================
# DENOISER with freq-band sampling
# ============================================================================

class FreqDenoiser(nn.Module):
    def __init__(self, n_bits):
        super().__init__(); self.n_bits = n_bits
        hid = min(128, max(64, n_bits * 4))
        self.net = nn.Sequential(
            nn.Conv2d(n_bits + 1, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, n_bits, 3, padding=1))
        self.skip = nn.Conv2d(n_bits, n_bits, 1)

    def forward(self, z_noisy, noise_level):
        B = z_noisy.shape[0]
        nl = noise_level.view(B, 1, 1, 1).expand(-1, 1, z_noisy.shape[2], z_noisy.shape[3])
        return self.net(torch.cat([z_noisy, nl], dim=1)) + self.skip(z_noisy)

    @torch.no_grad()
    def sample_standard(self, n, H, W, device, n_steps=15, temperature=0.7):
        K = self.n_bits; z = (torch.rand(n, K, H, W, device=device) > 0.5).float()
        for step in range(n_steps):
            nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
            probs = torch.sigmoid(self(z, nl) / temperature)
            conf = (step + 1) / n_steps
            mask = (torch.rand_like(z) < conf).float()
            z = mask * (torch.rand_like(z) < probs).float() + (1 - mask) * z
        return (torch.sigmoid(self(z, torch.zeros(n, device=device))) > 0.5).float()

    @torch.no_grad()
    def sample_multiscale(self, n, H, W, device, n_steps=15, temperature=0.7):
        K = self.n_bits; z = (torch.rand(n, K, H, W, device=device) > 0.5).float()
        for step in range(n_steps):
            nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
            logits = self(z, nl); p = step / n_steps
            if p < 0.5:
                t = temperature * 0.6; c = 0.3 + 0.4 * (p / 0.5)
            else:
                t = temperature * 1.2; c = 0.7 + 0.3 * ((p - 0.5) / 0.5)
            probs = torch.sigmoid(logits / t)
            mask = (torch.rand_like(z) < c).float()
            z = mask * (torch.rand_like(z) < probs).float() + (1 - mask) * z
        return (torch.sigmoid(self(z, torch.zeros(n, device=device))) > 0.5).float()

    @torch.no_grad()
    def sample_freq_band(self, n, H, W, decoder, device, n_steps=15, temperature=0.7):
        """G2: Freq-band-scheduled sampling.

        Key idea: Use decoder feedback to identify which z positions primarily
        affect low/mid/high frequency content, then commit them in order:
        low-freq first (global structure) → mid-freq → high-freq (texture).

        Implementation: At each step, decode current z, compute per-position
        frequency sensitivity, and weight the update mask by band schedule.
        """
        K = self.n_bits
        z = (torch.rand(n, K, H, W, device=device) > 0.5).float()

        # Pre-compute frequency sensitivity map for z positions
        # Perturb each position and measure which frequency band changes most
        # (too expensive to do per-position, so use a learned proxy via
        #  spatial frequency of the z grid itself)
        # Approximation: positions near image center → low freq,
        # positions near edges → high freq (in DCT sense)
        fy = torch.arange(H, device=device).float() / H
        fx = torch.arange(W, device=device).float() / W
        # Frequency "priority" based on DCT-like indexing
        freq_idx = (fy.unsqueeze(1) + fx.unsqueeze(0))  # [H, W]
        max_fi = freq_idx.max()
        freq_norm = freq_idx / max_fi  # 0=DC corner, 1=highest freq

        for step in range(n_steps):
            progress = (step + 1) / n_steps
            nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
            logits = self(z, nl)

            # Band schedule: which frequency tier to commit this step
            # Phase 1 (0-40%): commit low-freq positions (freq_norm < 0.33)
            # Phase 2 (40-70%): commit mid-freq positions (0.33 < freq_norm < 0.66)
            # Phase 3 (70-100%): commit high-freq positions (freq_norm > 0.66)
            if progress <= 0.4:
                # Low-freq phase: high confidence on low-freq, low on high-freq
                band_progress = progress / 0.4
                conf_map = (1.0 - freq_norm) * band_progress * 0.8 + 0.1
                t = temperature * 0.5  # low temp → sharp decisions for structure
            elif progress <= 0.7:
                # Mid-freq phase: commit mid-freq, keep low committed
                band_progress = (progress - 0.4) / 0.3
                low_conf = torch.ones_like(freq_norm) * 0.9  # already committed
                mid_mask = ((freq_norm >= 0.33) & (freq_norm < 0.66)).float()
                high_mask = (freq_norm >= 0.66).float()
                conf_map = low_conf * (freq_norm < 0.33).float() + \
                           (0.3 + 0.6 * band_progress) * mid_mask + \
                           0.1 * high_mask
                t = temperature * 0.8
            else:
                # High-freq phase: commit everything
                band_progress = (progress - 0.7) / 0.3
                conf_map = torch.ones_like(freq_norm) * (0.7 + 0.3 * band_progress)
                t = temperature * 1.2  # higher temp for texture diversity

            probs = torch.sigmoid(logits / t)
            # Expand conf_map to match z shape: [H, W] → [1, 1, H, W] → [n, K, H, W]
            conf_4d = conf_map.unsqueeze(0).unsqueeze(0).expand(n, K, -1, -1)
            mask = (torch.rand_like(z) < conf_4d).float()
            z = mask * (torch.rand_like(z) < probs).float() + (1 - mask) * z

        return (torch.sigmoid(self(z, torch.zeros(n, device=device))) > 0.5).float()

    @torch.no_grad()
    def sample_freq_band_v2(self, n, H, W, decoder, device, n_steps=15, temperature=0.7):
        """G2v2: Decoder-feedback freq-band sampling.

        Instead of using spatial position as frequency proxy, decode at each step
        and use the actual DCT decomposition to guide which positions to update.
        Positions where decoded image has high low-freq energy get committed first.
        """
        K = self.n_bits
        z = (torch.rand(n, K, H, W, device=device) > 0.5).float()

        for step in range(n_steps):
            progress = (step + 1) / n_steps
            nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
            logits = self(z, nl)
            probs = torch.sigmoid(logits / temperature)

            # Decode current z and compute per-pixel frequency composition
            if step < n_steps - 1:
                x_cur = decoder(z.to(device))  # [n, 3, 32, 32]
                x_low, x_mid, x_high = decompose_bands(x_cur)

                # Compute energy at each decoded pixel position
                low_e = x_low.pow(2).mean(dim=1, keepdim=True)   # [n, 1, 32, 32]
                mid_e = x_mid.pow(2).mean(dim=1, keepdim=True)
                high_e = x_high.pow(2).mean(dim=1, keepdim=True)
                total_e = (low_e + mid_e + high_e).clamp(min=1e-8)

                # Fraction of energy in target band at each pixel
                if progress <= 0.4:
                    target_frac = low_e / total_e  # prioritize low-freq positions
                    base_conf = 0.2 + 0.5 * (progress / 0.4)
                elif progress <= 0.7:
                    target_frac = (low_e + mid_e) / total_e
                    base_conf = 0.5 + 0.3 * ((progress - 0.4) / 0.3)
                else:
                    target_frac = torch.ones_like(low_e)
                    base_conf = 0.8 + 0.2 * ((progress - 0.7) / 0.3)

                # Downsample pixel-space map to z-space
                if x_cur.shape[2] != H:
                    target_frac = F.adaptive_avg_pool2d(target_frac, (H, W))

                # Confidence = base + bonus for target-band positions
                conf_map = (base_conf * target_frac).expand(-1, K, -1, -1)
                conf_map = conf_map.clamp(0.05, 0.95)
            else:
                conf_map = torch.ones(n, K, H, W, device=device) * 0.95

            mask = (torch.rand_like(z) < conf_map).float()
            z = mask * (torch.rand_like(z) < probs).float() + (1 - mask) * z

        return (torch.sigmoid(self(z, torch.zeros(n, device=device))) > 0.5).float()


# ============================================================================
# TRAINING
# ============================================================================

def train_adc(encoder, decoder, train_x, device, epochs=40, bs=32):
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    for epoch in tqdm(range(epochs), desc="ADC"):
        encoder.train(); decoder.train()
        encoder.set_temperature(1.0 + (0.3 - 1.0) * epoch / max(epochs - 1, 1))
        perm = torch.randperm(len(train_x)); tl, nb = 0., 0
        for i in range(0, len(train_x), bs):
            x = train_x[perm[i:i+bs]].to(device)
            opt.zero_grad()
            z, _ = encoder(x); xh = decoder(z)
            loss = F.mse_loss(xh, x) + 0.5 * F.binary_cross_entropy(xh, x)
            loss.backward(); opt.step(); tl += loss.item(); nb += 1
    encoder.eval(); decoder.eval()
    return tl / nb


def encode_all(encoder, data, device, bs=32):
    zs = []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            z, _ = encoder(data[i:i+bs].to(device))
            zs.append(z.cpu())
    return torch.cat(zs)


def train_ecore(e_core, z_data, device, epochs=10, bs=128):
    opt = torch.optim.Adam(e_core.parameters(), lr=1e-3)
    K, H, W = z_data.shape[1:]
    for ep in tqdm(range(epochs), desc="E_core"):
        e_core.train(); perm = torch.randperm(len(z_data))
        for i in range(0, len(z_data), bs):
            z = z_data[perm[i:i+bs]].to(device); opt.zero_grad(); tl = 0.
            for _ in range(20):
                b = torch.randint(K, (1,)).item()
                ii = torch.randint(H, (1,)).item()
                jj = torch.randint(W, (1,)).item()
                ctx = e_core.get_context(z, b, ii, jj)
                tl += F.binary_cross_entropy_with_logits(
                    e_core.predictor(ctx).squeeze(1), z[:, b, ii, jj])
            (tl / 20).backward(); opt.step()
    e_core.eval()


def train_denoiser(denoiser, z_data, decoder, device, epochs=30, bs=32,
                   use_freq=True):
    opt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)
    for epoch in tqdm(range(epochs), desc="Denoiser"):
        denoiser.train(); perm = torch.randperm(N)
        tl, fl, cl, nb = 0., 0., 0., 0
        progress = epoch / max(epochs - 1, 1)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]; z_clean = z_data[idx].to(device); B = z_clean.shape[0]
            noise_level = torch.rand(B, device=device)
            flip = (torch.rand_like(z_clean) < noise_level.view(B, 1, 1, 1)).float()
            z_noisy = z_clean * (1 - flip) + (1 - z_clean) * flip
            opt.zero_grad()
            logits = denoiser(z_noisy, noise_level)
            loss_bce = F.binary_cross_entropy_with_logits(logits, z_clean)
            loss_freq = loss_coh = torch.tensor(0.0, device=device)
            if use_freq:
                s = torch.sigmoid(logits); h = (s > 0.5).float()
                z_pred = h - s.detach() + s
                with torch.no_grad():
                    x_clean = decoder(z_clean)
                x_pred = decoder(z_pred)
                loss_freq = freq_scheduled_loss(x_pred, x_clean, progress)
                _, _, pred_h = decompose_bands(x_pred)
                _, _, tgt_h = decompose_bands(x_clean)
                loss_coh = hf_coherence_loss(pred_h, tgt_h)
            loss = loss_bce + 0.3 * loss_freq + 0.1 * loss_coh
            loss.backward(); opt.step()
            tl += loss_bce.item(); fl += loss_freq.item()
            cl += loss_coh.item(); nb += 1
        if (epoch + 1) % 10 == 0:
            print(f"      ep {epoch+1}/{epochs}: BCE={tl/nb:.4f} freq={fl/nb:.4f} coh={cl/nb:.4f}")
    denoiser.eval()


# ============================================================================
# EVALUATE ONE CONFIG
# ============================================================================

def evaluate(z_gen, decoder, encoder, e_core, z_data, test_x, train_x_np,
             real_hf_coh, real_hf_noi, device, n_bits, bs=32):
    z_cpu = z_gen.cpu()

    # Decode
    x_gen_list = []
    with torch.no_grad():
        for gi in range(0, len(z_gen), bs):
            x_gen_list.append(decoder(z_gen[gi:gi+bs].to(device)).cpu())
    x_gen = torch.cat(x_gen_list)

    # Metrics
    viol = e_core.violation_rate(z_cpu[:100].to(device))
    tok_kl = token_histogram_kl(z_data[:500], z_cpu[:min(500, len(z_cpu))], n_bits)
    div = compute_diversity(z_cpu)

    with torch.no_grad():
        zc = z_cpu[:100].to(device); xc = decoder(zc)
        zcy, _ = encoder(xc)
        cycle = (zc != zcy).float().mean().item()

    conn = connectedness_proxy(x_gen[:100])
    band = per_band_energy_distance(x_gen[:200], test_x[:200], device)
    hfc = hf_coherence_metric(x_gen[:200], device)
    hfn = hf_noise_index(x_gen[:200], device)

    return {
        'violation': viol, 'token_kl': tok_kl, 'diversity': div,
        'cycle': cycle, 'connectedness': conn,
        'hf_coherence': hfc, 'hf_noise_index': hfn,
        'energy_gap_low': band['energy_gap_low'],
        'energy_gap_mid': band['energy_gap_mid'],
        'energy_gap_high': band['energy_gap_high'],
    }, x_gen


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_gen_g1g2')
    parser.add_argument('--n_samples', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("CIFAR-10 GENERATION — G1+G2: BANDWIDTH + FREQ-BAND SAMPLING")
    print("=" * 100)

    # Load data
    print("\n[1] Loading CIFAR-10...")
    from torchvision import datasets, transforms
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(args.seed)
    train_x = torch.stack([train_ds[i][0] for i in rng.choice(len(train_ds), 3000, replace=False)])
    test_x = torch.stack([test_ds[i][0] for i in rng.choice(len(test_ds), 500, replace=False)])
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")
    train_x_np = train_x.numpy().reshape(len(train_x), -1)

    # Reference metrics
    print("\n[2] Reference metrics...")
    real_hf_coh = hf_coherence_metric(test_x[:200], device)
    real_hf_noi = hf_noise_index(test_x[:200], device)
    real_conn = connectedness_proxy(test_x[:100].numpy())
    print(f"    HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  conn={real_conn:.4f}")
    save_grid(test_x[:64], os.path.join(args.output_dir, 'real_samples.png'))

    # ================================================================
    # CONFIG MATRIX
    # ================================================================
    configs = [
        # (name, z_spec, sampler)
        # z_spec: (encoder_class, decoder_class, n_bits, z_h, z_w, total_bits)
        ("A_16x16x8_standard",  "16", 8,  "standard"),
        ("B_16x16x8_freqband",  "16", 8,  "freq_band"),
        ("C_32x32x16_standard", "32", 16, "standard"),
        ("D_32x32x16_freqband", "32", 16, "freq_band"),
        ("E_32x32x16_freqv2",  "32", 16, "freq_band_v2"),
    ]

    all_results = {}
    # Cache trained models per z-spec to avoid retraining
    trained_models = {}

    for cfg_name, z_type, n_bits, sampler in configs:
        z_key = f"{z_type}_{n_bits}"
        if z_type == "16":
            z_h, z_w = 16, 16
        else:
            z_h, z_w = 32, 32
        total_bits = z_h * z_w * n_bits

        print(f"\n{'='*100}")
        print(f"CONFIG: {cfg_name} ({z_h}×{z_w}×{n_bits} = {total_bits} bits, sampler={sampler})")
        print("=" * 100)

        # Train ADC/DAC + E_core + denoiser (or reuse)
        if z_key not in trained_models:
            torch.manual_seed(args.seed)
            if z_type == "16":
                enc = Encoder16(n_bits).to(device)
                dec = Decoder16(n_bits).to(device)
            else:
                enc = Encoder32(n_bits).to(device)
                dec = Decoder32(n_bits).to(device)

            enc_p = sum(p.numel() for p in enc.parameters())
            dec_p = sum(p.numel() for p in dec.parameters())
            print(f"    Encoder params: {enc_p:,}  Decoder params: {dec_p:,}")

            # Train ADC
            print("  Training ADC...")
            adc_loss = train_adc(enc, dec, train_x, device, epochs=40, bs=32)
            print(f"    ADC loss: {adc_loss:.4f}")

            # Check reconstruction
            with torch.no_grad():
                tb = test_x[:32].to(device); zo, _ = enc(tb)
                oracle_mse = F.mse_loss(dec(zo), tb).item()
            print(f"    Oracle MSE: {oracle_mse:.4f}")

            # Encode
            print("  Encoding training set...")
            z_data = encode_all(enc, train_x, device, bs=32)
            K, H, W = z_data.shape[1:]
            print(f"    z_data: {z_data.shape}, usage={z_data.mean():.3f}")

            # E_core
            print("  Training E_core...")
            e_core = LocalEnergyCore(n_bits).to(device)
            train_ecore(e_core, z_data, device, epochs=10, bs=128)

            # Denoiser (with freq training)
            print("  Training denoiser (freq-aware)...")
            denoiser = FreqDenoiser(n_bits).to(device)
            train_denoiser(denoiser, z_data, dec, device, epochs=30, bs=32, use_freq=True)

            trained_models[z_key] = {
                'enc': enc, 'dec': dec, 'e_core': e_core,
                'denoiser': denoiser, 'z_data': z_data,
                'oracle_mse': oracle_mse,
            }
        else:
            print("  Reusing trained models...")

        m = trained_models[z_key]
        enc, dec, e_core = m['enc'], m['dec'], m['e_core']
        denoiser, z_data = m['denoiser'], m['z_data']
        K, H, W = z_data.shape[1:]

        # Sample
        print(f"  Generating {args.n_samples} samples ({sampler})...")
        torch.manual_seed(args.seed)
        gen_bs = 32
        z_gen_list = []
        for gi in range(0, args.n_samples, gen_bs):
            nb = min(gen_bs, args.n_samples - gi)
            if sampler == "standard":
                z_gen_list.append(denoiser.sample_standard(nb, H, W, device).cpu())
            elif sampler == "freq_band":
                z_gen_list.append(denoiser.sample_freq_band(nb, H, W, dec, device).cpu())
            elif sampler == "freq_band_v2":
                z_gen_list.append(denoiser.sample_freq_band_v2(nb, H, W, dec, device).cpu())
            else:
                z_gen_list.append(denoiser.sample_multiscale(nb, H, W, device).cpu())
        z_gen = torch.cat(z_gen_list)

        # Evaluate
        print("  Evaluating...")
        r, x_gen = evaluate(z_gen, dec, enc, e_core, z_data, test_x, train_x_np,
                            real_hf_coh, real_hf_noi, device, n_bits)
        r['oracle_mse'] = m['oracle_mse']
        r['total_bits'] = total_bits
        r['z_spec'] = f"{z_h}x{z_w}x{n_bits}"
        r['sampler'] = sampler
        all_results[cfg_name] = r

        save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))

        print(f"    viol={r['violation']:.4f} div={r['diversity']:.4f} "
              f"cycle={r['cycle']:.4f} conn={r['connectedness']:.4f}")
        print(f"    HF_coh={r['hf_coherence']:.4f}(real={real_hf_coh:.4f}) "
              f"HF_noi={r['hf_noise_index']:.2f}(real={real_hf_noi:.2f})")
        print(f"    E_gap: L={r['energy_gap_low']:.4f} M={r['energy_gap_mid']:.4f} "
              f"H={r['energy_gap_high']:.4f}")

        # Free VRAM between z-spec groups
        torch.cuda.empty_cache()

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*100}")
    print("G1+G2 GENERATION SUMMARY")
    print(f"Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  conn={real_conn:.4f}")
    print("=" * 100)

    h = (f"{'config':<28} {'bits':>6} {'viol':>7} {'div':>7} {'cycle':>7} "
         f"{'conn':>7} {'HFcoh':>7} {'HFnoi':>7} {'EgL':>7} {'EgM':>7} {'EgH':>7}")
    print(h); print("-" * len(h))
    for name, r in all_results.items():
        print(f"{name:<28} {r['total_bits']:>6} {r['violation']:>7.4f} {r['diversity']:>7.4f} "
              f"{r['cycle']:>7.4f} {r['connectedness']:>7.4f} {r['hf_coherence']:>7.4f} "
              f"{r['hf_noise_index']:>7.2f} {r['energy_gap_low']:>7.4f} "
              f"{r['energy_gap_mid']:>7.4f} {r['energy_gap_high']:>7.4f}")

    # Comparative analysis
    print(f"\n{'='*100}")
    print("COMPARATIVE ANALYSIS")
    print("=" * 100)

    # G1: bandwidth effect (A vs C, B vs D)
    if 'A_16x16x8_standard' in all_results and 'C_32x32x16_standard' in all_results:
        a = all_results['A_16x16x8_standard']
        c = all_results['C_32x32x16_standard']
        print("\nG1 (Bandwidth): 16×16×8 vs 32×32×16 (standard sampler)")
        print(f"  violation:  {a['violation']:.4f} → {c['violation']:.4f} (Δ={c['violation']-a['violation']:+.4f})")
        print(f"  diversity:  {a['diversity']:.4f} → {c['diversity']:.4f} (Δ={c['diversity']-a['diversity']:+.4f})")
        print(f"  cycle:      {a['cycle']:.4f} → {c['cycle']:.4f}")
        print(f"  HF_noise:   {a['hf_noise_index']:.2f} → {c['hf_noise_index']:.2f} (real={real_hf_noi:.2f})")
        print(f"  E_gap_mid:  {a['energy_gap_mid']:.4f} → {c['energy_gap_mid']:.4f}")
        print(f"  E_gap_high: {a['energy_gap_high']:.4f} → {c['energy_gap_high']:.4f}")

    # G2: freq-band sampling effect (A vs B, C vs D)
    if 'A_16x16x8_standard' in all_results and 'B_16x16x8_freqband' in all_results:
        a = all_results['A_16x16x8_standard']
        b = all_results['B_16x16x8_freqband']
        print("\nG2 (Freq-band sampling): standard vs freq_band (16×16×8)")
        print(f"  violation:  {a['violation']:.4f} → {b['violation']:.4f} (Δ={b['violation']-a['violation']:+.4f})")
        print(f"  diversity:  {a['diversity']:.4f} → {b['diversity']:.4f} (Δ={b['diversity']-a['diversity']:+.4f})")
        print(f"  HF_coh:     {a['hf_coherence']:.4f} → {b['hf_coherence']:.4f} (real={real_hf_coh:.4f})")
        print(f"  E_gap_low:  {a['energy_gap_low']:.4f} → {b['energy_gap_low']:.4f}")

    # G1+G2: combined effect
    if 'A_16x16x8_standard' in all_results and 'D_32x32x16_freqband' in all_results:
        a = all_results['A_16x16x8_standard']
        d = all_results['D_32x32x16_freqband']
        print("\nG1+G2 Combined: 16×16×8 standard → 32×32×16 freq_band")
        print(f"  violation:  {a['violation']:.4f} → {d['violation']:.4f} (Δ={d['violation']-a['violation']:+.4f})")
        print(f"  diversity:  {a['diversity']:.4f} → {d['diversity']:.4f} (Δ={d['diversity']-a['diversity']:+.4f})")
        print(f"  HF_noise:   {a['hf_noise_index']:.2f} → {d['hf_noise_index']:.2f} (real={real_hf_noi:.2f})")
        print(f"  E_gap_high: {a['energy_gap_high']:.4f} → {d['energy_gap_high']:.4f}")

    # Diagnosis
    print(f"\n{'='*100}")
    print("DIAGNOSIS")
    print("=" * 100)

    best_viol = min(all_results.items(), key=lambda x: x[1]['violation'])
    best_div = max(all_results.items(), key=lambda x: x[1]['diversity'])
    best_hfn = min(all_results.items(), key=lambda x: abs(x[1]['hf_noise_index'] - real_hf_noi))
    best_egm = min(all_results.items(), key=lambda x: x[1]['energy_gap_mid'])

    print(f"  Best violation:  {best_viol[0]} ({best_viol[1]['violation']:.4f})")
    print(f"  Best diversity:  {best_div[0]} ({best_div[1]['diversity']:.4f})")
    print(f"  Best HF_noise:   {best_hfn[0]} ({best_hfn[1]['hf_noise_index']:.2f}, real={real_hf_noi:.2f})")
    print(f"  Best E_gap_mid:  {best_egm[0]} ({best_egm[1]['energy_gap_mid']:.4f})")

    # Save
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")

    print(f"\n{'='*100}")
    print("G1+G2 EXPERIMENT COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
