#!/usr/bin/env python3
"""
CIFAR-10 Generation — G3: Stride-2 High-Channel + Regression Denoiser
======================================================================
Findings from G1+G2:
  - 32×32×16 (1:1 spatial) has great diversity/violation but HF_noise=1838 (real=264)
  - 16×16×8 has controlled HF_noise=136 but limited diversity
  - Root cause: 1:1 spatial mapping → every z bit → pixel, no spatial abstraction

G3 hypothesis: 16×16×16 (stride-2, 4096 bits) gives spatial abstraction + more bits.
Plus: regression denoiser that predicts continuous logits rather than hard classification.

Architecture matrix:
  A) 16×16×8  standard  (baseline)
  B) 16×16×16 standard  (G3: more channels with spatial abstraction)
  C) 16×16×16 freqband  (G3 + G2)
  D) 16×16×16 regression denoiser + freqband  (G3 full)

Regression denoiser: predicts continuous residual δ = z_clean - z_noisy,
then z_pred = z_noisy + σ(denoiser(z_noisy)), allowing smooth interpolation.

4GB GPU: 3000 train, 500 test, batch_size=32

Usage:
    python3 -u benchmarks/exp_gen_cifar10_g3.py --device cuda
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
# METRICS (compact)
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
    elif images.ndim == 4: images = images[:, 0]
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
# MODELS — 16×16 stride-2 encoder/decoder (parameterized n_bits)
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
# DENOISERS
# ============================================================================

class FreqDenoiser(nn.Module):
    """Standard binary denoiser (from A2)."""
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
    def sample_freq_band(self, n, H, W, device, n_steps=15, temperature=0.7):
        """Freq-band scheduled sampling (from G2)."""
        K = self.n_bits
        z = (torch.rand(n, K, H, W, device=device) > 0.5).float()
        fy = torch.arange(H, device=device).float() / H
        fx = torch.arange(W, device=device).float() / W
        freq_norm = (fy.unsqueeze(1) + fx.unsqueeze(0))
        freq_norm = freq_norm / freq_norm.max()

        for step in range(n_steps):
            progress = (step + 1) / n_steps
            nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
            logits = self(z, nl)

            if progress <= 0.4:
                band_progress = progress / 0.4
                conf_map = (1.0 - freq_norm) * band_progress * 0.8 + 0.1
                t = temperature * 0.5
            elif progress <= 0.7:
                band_progress = (progress - 0.4) / 0.3
                mid_mask = ((freq_norm >= 0.33) & (freq_norm < 0.66)).float()
                high_mask = (freq_norm >= 0.66).float()
                conf_map = torch.ones_like(freq_norm) * 0.9 * (freq_norm < 0.33).float() + \
                           (0.3 + 0.6 * band_progress) * mid_mask + \
                           0.1 * high_mask
                t = temperature * 0.8
            else:
                band_progress = (progress - 0.7) / 0.3
                conf_map = torch.ones_like(freq_norm) * (0.7 + 0.3 * band_progress)
                t = temperature * 1.2

            probs = torch.sigmoid(logits / t)
            conf_4d = conf_map.unsqueeze(0).unsqueeze(0).expand(n, K, -1, -1)
            mask = (torch.rand_like(z) < conf_4d).float()
            z = mask * (torch.rand_like(z) < probs).float() + (1 - mask) * z

        return (torch.sigmoid(self(z, torch.zeros(n, device=device))) > 0.5).float()


class RegressionDenoiser(nn.Module):
    """G3: Regression denoiser — predicts continuous residual.

    Instead of binary classification (predict p(z_i=1)),
    predicts continuous δ and uses smooth annealing to binary.
    Key difference: final sampling uses temperature-scaled sigmoid
    with residual correction, not hard thresholding.
    """
    def __init__(self, n_bits):
        super().__init__(); self.n_bits = n_bits
        hid = min(128, max(64, n_bits * 4))
        self.net = nn.Sequential(
            nn.Conv2d(n_bits + 1, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, n_bits, 3, padding=1))
        self.skip = nn.Conv2d(n_bits, n_bits, 1)
        # Learnable temperature per channel
        self.log_temp = nn.Parameter(torch.zeros(1, n_bits, 1, 1))

    def forward(self, z_noisy, noise_level):
        B = z_noisy.shape[0]
        nl = noise_level.view(B, 1, 1, 1).expand(-1, 1, z_noisy.shape[2], z_noisy.shape[3])
        logits = self.net(torch.cat([z_noisy, nl], dim=1)) + self.skip(z_noisy)
        return logits

    @torch.no_grad()
    def sample_freq_band(self, n, H, W, device, n_steps=20, temperature=0.7):
        """Regression sampling with freq-band schedule.

        Uses continuous predictions with gradual annealing to binary.
        Key: maintain soft z during early steps, only snap to binary at end.
        """
        K = self.n_bits
        # Start with soft initialization (continuous [0,1])
        z = torch.rand(n, K, H, W, device=device) * 0.5 + 0.25  # [0.25, 0.75]

        fy = torch.arange(H, device=device).float() / H
        fx = torch.arange(W, device=device).float() / W
        freq_norm = (fy.unsqueeze(1) + fx.unsqueeze(0))
        freq_norm = freq_norm / freq_norm.max()

        for step in range(n_steps):
            progress = (step + 1) / n_steps
            nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)

            # Snap to binary for denoiser input
            z_binary = (z > 0.5).float()
            logits = self(z_binary, nl)
            temp = self.log_temp.exp() * temperature

            # Continuous prediction
            soft_pred = torch.sigmoid(logits / temp)

            # Freq-band update weights
            if progress <= 0.4:
                bp = progress / 0.4
                weight = (1.0 - freq_norm) * bp * 0.8 + 0.1
            elif progress <= 0.7:
                bp = (progress - 0.4) / 0.3
                mid = ((freq_norm >= 0.33) & (freq_norm < 0.66)).float()
                high = (freq_norm >= 0.66).float()
                weight = torch.ones_like(freq_norm) * 0.9 * (freq_norm < 0.33).float() + \
                         (0.3 + 0.6 * bp) * mid + 0.1 * high
            else:
                bp = (progress - 0.7) / 0.3
                weight = torch.ones_like(freq_norm) * (0.7 + 0.3 * bp)

            weight_4d = weight.unsqueeze(0).unsqueeze(0).expand(n, K, -1, -1)

            # Smooth update: blend current z with prediction
            # Early steps: small weight → keep exploring
            # Late steps: large weight → commit
            anneal = min(1.0, progress * 2)  # 0→1 over first half
            z = z * (1 - weight_4d * anneal) + soft_pred * weight_4d * anneal

            # Add small noise in early steps for exploration
            if progress < 0.5:
                noise_scale = 0.1 * (1 - progress * 2)
                z = z + torch.randn_like(z) * noise_scale
                z = z.clamp(0, 1)

        # Final: snap to binary
        return (z > 0.5).float()


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


def train_denoiser(denoiser, z_data, decoder, device, epochs=30, bs=32):
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
            # Freq loss
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


def train_regression_denoiser(denoiser, z_data, decoder, device, epochs=30, bs=32):
    """Train regression denoiser with smooth MSE loss instead of BCE."""
    opt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)
    for epoch in tqdm(range(epochs), desc="RegDen"):
        denoiser.train(); perm = torch.randperm(N)
        tl, fl, nb = 0., 0., 0
        progress = epoch / max(epochs - 1, 1)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]; z_clean = z_data[idx].to(device); B = z_clean.shape[0]
            noise_level = torch.rand(B, device=device)
            flip = (torch.rand_like(z_clean) < noise_level.view(B, 1, 1, 1)).float()
            z_noisy = z_clean * (1 - flip) + (1 - z_clean) * flip
            opt.zero_grad()
            logits = denoiser(z_noisy, noise_level)
            # Primary: BCE (still works well for binary targets)
            loss_bce = F.binary_cross_entropy_with_logits(logits, z_clean)
            # Secondary: continuous MSE on soft predictions
            soft = torch.sigmoid(logits)
            loss_mse = F.mse_loss(soft, z_clean) * 2.0
            # Freq loss
            h = (soft > 0.5).float()
            z_pred = h - soft.detach() + soft
            with torch.no_grad():
                x_clean = decoder(z_clean)
            x_pred = decoder(z_pred)
            loss_freq = freq_scheduled_loss(x_pred, x_clean, progress)
            loss = loss_bce + 0.5 * loss_mse + 0.3 * loss_freq
            loss.backward(); opt.step()
            tl += (loss_bce.item() + 0.5 * loss_mse.item()); fl += loss_freq.item(); nb += 1
        if (epoch + 1) % 10 == 0:
            print(f"      ep {epoch+1}/{epochs}: loss={tl/nb:.4f} freq={fl/nb:.4f}")
    denoiser.eval()


# ============================================================================
# EVALUATE
# ============================================================================

def evaluate(z_gen, decoder, encoder, e_core, z_data, test_x,
             real_hf_coh, real_hf_noi, device, n_bits, bs=32):
    z_cpu = z_gen.cpu()
    x_gen_list = []
    with torch.no_grad():
        for gi in range(0, len(z_gen), bs):
            x_gen_list.append(decoder(z_gen[gi:gi+bs].to(device)).cpu())
    x_gen = torch.cat(x_gen_list)

    viol = e_core.violation_rate(z_cpu[:100].to(device))
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
        'violation': viol, 'diversity': div, 'cycle': cycle,
        'connectedness': conn, 'hf_coherence': hfc, 'hf_noise_index': hfn,
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
    parser.add_argument('--output_dir', default='outputs/exp_gen_g3')
    parser.add_argument('--n_samples', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("CIFAR-10 GENERATION — G3: STRIDE-2 HIGH-CHANNEL + REGRESSION DENOISER")
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

    # Reference
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
        # (name, n_bits, sampler_type, denoiser_type)
        ("A_16x16x8_standard",   8,  "standard",  "freq"),
        ("B_16x16x16_standard", 16, "standard",  "freq"),
        ("C_16x16x16_freqband", 16, "freq_band", "freq"),
        ("D_16x16x16_regression", 16, "freq_band", "regression"),
    ]

    all_results = {}
    trained_models = {}

    for cfg_name, n_bits, sampler, den_type in configs:
        model_key = f"{n_bits}_{den_type}"
        total_bits = 16 * 16 * n_bits

        print(f"\n{'='*100}")
        print(f"CONFIG: {cfg_name} (16×16×{n_bits} = {total_bits} bits, "
              f"sampler={sampler}, denoiser={den_type})")
        print("=" * 100)

        if model_key not in trained_models:
            torch.manual_seed(args.seed)
            enc = Encoder16(n_bits).to(device)
            dec = Decoder16(n_bits).to(device)

            enc_p = sum(p.numel() for p in enc.parameters())
            dec_p = sum(p.numel() for p in dec.parameters())
            print(f"    Encoder params: {enc_p:,}  Decoder params: {dec_p:,}")

            print("  Training ADC...")
            adc_loss = train_adc(enc, dec, train_x, device, epochs=40, bs=32)
            print(f"    ADC loss: {adc_loss:.4f}")

            with torch.no_grad():
                tb = test_x[:32].to(device); zo, _ = enc(tb)
                oracle_mse = F.mse_loss(dec(zo), tb).item()
            print(f"    Oracle MSE: {oracle_mse:.4f}")
            save_grid(dec(zo).cpu(), os.path.join(args.output_dir, f'recon_{cfg_name}.png'))

            print("  Encoding training set...")
            z_data = encode_all(enc, train_x, device, bs=32)
            K, H, W = z_data.shape[1:]
            print(f"    z_data: {z_data.shape}, usage={z_data.mean():.3f}")

            print("  Training E_core...")
            e_core = LocalEnergyCore(n_bits).to(device)
            train_ecore(e_core, z_data, device, epochs=10, bs=128)

            if den_type == "freq":
                print("  Training freq denoiser...")
                denoiser = FreqDenoiser(n_bits).to(device)
                train_denoiser(denoiser, z_data, dec, device, epochs=30, bs=32)
            else:
                print("  Training regression denoiser...")
                denoiser = RegressionDenoiser(n_bits).to(device)
                train_regression_denoiser(denoiser, z_data, dec, device, epochs=30, bs=32)

            trained_models[model_key] = {
                'enc': enc, 'dec': dec, 'e_core': e_core,
                'denoiser': denoiser, 'z_data': z_data,
                'oracle_mse': oracle_mse,
            }
        else:
            print("  Reusing trained models...")

        m = trained_models[model_key]
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
                z_gen_list.append(denoiser.sample_freq_band(nb, H, W, device).cpu())
        z_gen = torch.cat(z_gen_list)

        # Evaluate
        print("  Evaluating...")
        r, x_gen = evaluate(z_gen, dec, enc, e_core, z_data, test_x,
                            real_hf_coh, real_hf_noi, device, n_bits)
        r['oracle_mse'] = m['oracle_mse']
        r['total_bits'] = total_bits
        r['z_spec'] = f"16x16x{n_bits}"
        r['sampler'] = sampler
        r['denoiser'] = den_type
        all_results[cfg_name] = r

        save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))

        print(f"    viol={r['violation']:.4f} div={r['diversity']:.4f} "
              f"cycle={r['cycle']:.4f} conn={r['connectedness']:.4f}")
        print(f"    HF_coh={r['hf_coherence']:.4f}(real={real_hf_coh:.4f}) "
              f"HF_noi={r['hf_noise_index']:.2f}(real={real_hf_noi:.2f})")
        print(f"    E_gap: L={r['energy_gap_low']:.4f} M={r['energy_gap_mid']:.4f} "
              f"H={r['energy_gap_high']:.4f}")

        torch.cuda.empty_cache()

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*100}")
    print("G3 GENERATION SUMMARY")
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

    # Compare
    print(f"\n{'='*100}")
    print("COMPARATIVE ANALYSIS")
    print("=" * 100)

    if 'A_16x16x8_standard' in all_results and 'B_16x16x16_standard' in all_results:
        a = all_results['A_16x16x8_standard']
        b = all_results['B_16x16x16_standard']
        print("\nG3 Core: 16×16×8 → 16×16×16 (both stride-2, standard sampler)")
        print(f"  violation:  {a['violation']:.4f} → {b['violation']:.4f}")
        print(f"  diversity:  {a['diversity']:.4f} → {b['diversity']:.4f}")
        print(f"  HF_noise:   {a['hf_noise_index']:.2f} → {b['hf_noise_index']:.2f} (real={real_hf_noi:.2f})")
        print(f"  oracle_mse: {a['oracle_mse']:.4f} → {b['oracle_mse']:.4f}")

    if 'C_16x16x16_freqband' in all_results and 'D_16x16x16_regression' in all_results:
        c = all_results['C_16x16x16_freqband']
        d = all_results['D_16x16x16_regression']
        print("\nRegression vs Freq denoiser (both 16×16×16, freqband sampling)")
        print(f"  violation:  {c['violation']:.4f} → {d['violation']:.4f}")
        print(f"  diversity:  {c['diversity']:.4f} → {d['diversity']:.4f}")
        print(f"  HF_noise:   {c['hf_noise_index']:.2f} → {d['hf_noise_index']:.2f}")
        print(f"  E_gap_low:  {c['energy_gap_low']:.4f} → {d['energy_gap_low']:.4f}")

    # Diagnosis
    best_viol = min(all_results.items(), key=lambda x: x[1]['violation'])
    best_div = max(all_results.items(), key=lambda x: x[1]['diversity'])
    best_hfn = min(all_results.items(), key=lambda x: abs(x[1]['hf_noise_index'] - real_hf_noi))
    print(f"\n  Best violation: {best_viol[0]} ({best_viol[1]['violation']:.4f})")
    print(f"  Best diversity: {best_div[0]} ({best_div[1]['diversity']:.4f})")
    print(f"  Best HF_noise:  {best_hfn[0]} ({best_hfn[1]['hf_noise_index']:.2f}, real={real_hf_noi:.2f})")

    # Save
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*100}")
    print("G3 EXPERIMENT COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
