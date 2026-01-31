#!/usr/bin/env python3
"""
CIFAR-10 Generation — G4: Energy-Guided U-Net Denoiser
=======================================================
Core diagnosis from G1-G3:
  - Shallow flat CNN denoiser cannot capture multi-scale structure
  - E_core is disconnected from generation (only used for evaluation)
  - No fine texture emerges because denoiser has no global context

G4 addresses all three:
  1. U-Net denoiser in z-space (16→8→4→8→16 with skip connections)
  2. Self-attention at 4×4 bottleneck (global structure)
  3. E_core energy term in training: denoiser learns to produce low-energy z
  4. MaskGIT-style confidence unmasking instead of linear annealing
  5. E_core-guided rejection during sampling

Architecture:
  - z: 16×16×16 (4096 bits, stride-2 encoder from G3)
  - U-Net denoiser: ~200K params (vs 50K flat CNN)
  - Self-attention at 4×4 resolution (affordable)
  - Training: BCE + E_core_energy + freq_loss
  - Sampling: confidence-based unmasking + E_core rejection

Configs:
  A) flat_denoiser  (baseline, same as G3-B)
  B) unet_denoiser  (U-Net only, no energy)
  C) unet_energy    (U-Net + E_core in training)
  D) unet_energy_maskgit (U-Net + E_core + MaskGIT sampling)

4GB GPU: 3000 train, 500 test, batch_size=32

Usage:
    python3 -u benchmarks/exp_gen_cifar10_g4_energy_unet.py --device cuda
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
    def dm(N):
        n = torch.arange(N, dtype=x.dtype, device=x.device)
        k = n.unsqueeze(1)
        D = torch.cos(np.pi * (2*n+1) * k / (2*N))
        D[0] *= 1.0/np.sqrt(N); D[1:] *= np.sqrt(2.0/N)
        return D
    return torch.einsum('hH,bcHW,wW->bchw', dm(H), x, dm(W))

def idct2d(X):
    B, C, H, W = X.shape
    def dm(N):
        n = torch.arange(N, dtype=X.dtype, device=X.device)
        k = n.unsqueeze(1)
        D = torch.cos(np.pi * (2*n+1) * k / (2*N))
        D[0] *= 1.0/np.sqrt(N); D[1:] *= np.sqrt(2.0/N)
        return D
    return torch.einsum('Hh,bchw,Ww->bcHW', dm(H).T, X, dm(W).T)

def get_freq_masks(H, W, device='cpu'):
    fy = torch.arange(H, device=device).float()
    fx = torch.arange(W, device=device).float()
    fg = fy.unsqueeze(1) + fx.unsqueeze(0)
    mf = H + W - 2; t1, t2 = mf/3.0, 2*mf/3.0
    return (fg <= t1).float(), ((fg > t1) & (fg <= t2)).float(), (fg > t2).float()

def decompose_bands(x):
    lm, mm, hm = get_freq_masks(x.shape[2], x.shape[3], x.device)
    d = dct2d(x)
    return idct2d(d * lm), idct2d(d * mm), idct2d(d * hm)

def freq_scheduled_loss(x_pred, x_target, progress):
    pl, pm, ph = decompose_bands(x_pred)
    tl, tm, th = decompose_bands(x_target)
    wl = 3.0; wm = min(1.0, progress * 2); wh = 0.5 * max(0, (progress - 0.3) / 0.7)
    return wl * F.mse_loss(pl, tl) + wm * F.mse_loss(pm, tm) + wh * F.mse_loss(ph, th)


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
# ADC/DAC (16×16 stride-2)
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
    def __init__(self, n_bits=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = self._res(64, 64)
        self.res2 = self._res(64, 64)
        self.head = nn.Conv2d(64, n_bits, 3, padding=1)
        self.q = GumbelSigmoid()
    def _res(self, ic, oc):
        return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(),
                             nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc))
    def forward(self, x):
        h = self.stem(x); h = F.relu(h + self.res1(h)); h = F.relu(h + self.res2(h))
        return self.q(self.head(h)), self.head(h)
    def set_temperature(self, tau): self.q.set_temperature(tau)

class Decoder16(nn.Module):
    def __init__(self, n_bits=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ConvTranspose2d(n_bits, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
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
# E_CORE (differentiable version for training)
# ============================================================================

class DiffEnergyCore(nn.Module):
    """E_core that can be used in training loop (differentiable).

    Computes pseudo-likelihood energy: E(z) = -Σ log p(z_i | neighbors).
    Approximated by sampling random positions.
    """
    def __init__(self, n_bits):
        super().__init__(); self.n_bits = n_bits
        # Use conv-based predictor for efficiency (no manual context extraction)
        self.predictor = nn.Sequential(
            nn.Conv2d(n_bits, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, n_bits, 3, padding=1),
        )

    def energy(self, z):
        """Compute per-position energy: -log p(z_i | context).
        Returns scalar mean energy."""
        # Predict each bit from its 3×3 context
        logits = self.predictor(z)  # [B, K, H, W]
        # Energy = BCE (negative log-likelihood)
        return F.binary_cross_entropy_with_logits(logits, z, reduction='mean')

    def violation_rate(self, z):
        """Hard violation: predicted bit != actual bit."""
        with torch.no_grad():
            logits = self.predictor(z)
            pred = (logits > 0).float()
            return (pred != z).float().mean().item()


# ============================================================================
# FLAT DENOISER (baseline)
# ============================================================================

class FlatDenoiser(nn.Module):
    def __init__(self, n_bits):
        super().__init__(); self.n_bits = n_bits
        hid = 64
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


# ============================================================================
# U-NET DENOISER (G4 core)
# ============================================================================

class SelfAttention(nn.Module):
    """Lightweight self-attention for small spatial sizes (4×4)."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = C ** -0.5
        attn = torch.bmm(q.transpose(1, 2), k) * scale  # [B, HW, HW]
        attn = attn.softmax(dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(out)


class UNetDenoiser(nn.Module):
    """U-Net denoiser operating in z-space (16×16).

    Architecture:
      16×16 → 8×8 → 4×4 (+ attention) → 8×8 → 16×16
    Skip connections at each resolution level.
    """
    def __init__(self, n_bits, base_ch=64):
        super().__init__(); self.n_bits = n_bits

        # Encoder path
        self.enc1 = nn.Sequential(  # 16×16
            nn.Conv2d(n_bits + 1, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch), nn.SiLU(),
        )
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)  # → 8×8

        self.enc2 = nn.Sequential(  # 8×8
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2), nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2), nn.SiLU(),
        )
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)  # → 4×4

        # Bottleneck with attention (4×4)
        self.mid = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.GroupNorm(8, base_ch * 4), nn.SiLU(),
        )
        self.mid_attn = SelfAttention(base_ch * 4)
        self.mid2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.GroupNorm(8, base_ch * 4), nn.SiLU(),
        )

        # Decoder path
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)  # → 8×8
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),  # cat with skip
            nn.GroupNorm(8, base_ch * 2), nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2), nn.SiLU(),
        )

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)  # → 16×16
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),  # cat with skip
            nn.GroupNorm(8, base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch), nn.SiLU(),
        )

        self.out = nn.Conv2d(base_ch, n_bits, 1)
        self.skip_proj = nn.Conv2d(n_bits, n_bits, 1)

        # Noise level embedding
        self.noise_emb = nn.Sequential(
            nn.Linear(1, base_ch), nn.SiLU(), nn.Linear(base_ch, base_ch),
        )

    def forward(self, z_noisy, noise_level):
        B, K, H, W = z_noisy.shape
        nl = noise_level.view(B, 1, 1, 1).expand(-1, 1, H, W)
        x = torch.cat([z_noisy, nl], dim=1)  # [B, K+1, H, W]

        # Noise conditioning (additive to first layer)
        n_emb = self.noise_emb(noise_level.view(B, 1))  # [B, base_ch]

        # Encoder
        h1 = self.enc1(x)  # [B, 64, 16, 16]
        h1 = h1 + n_emb.view(B, -1, 1, 1)  # condition on noise level
        d1 = self.down1(h1)  # [B, 128, 8, 8]

        h2 = self.enc2(d1)  # [B, 128, 8, 8]
        d2 = self.down2(h2)  # [B, 256, 4, 4]

        # Bottleneck
        m = self.mid(d2)
        m = self.mid_attn(m)
        m = self.mid2(m)

        # Decoder
        u2 = self.up2(m)  # [B, 128, 8, 8]
        u2 = self.dec2(torch.cat([u2, h2], dim=1))

        u1 = self.up1(u2)  # [B, 64, 16, 16]
        u1 = self.dec1(torch.cat([u1, h1], dim=1))

        return self.out(u1) + self.skip_proj(z_noisy)


# ============================================================================
# SAMPLING STRATEGIES
# ============================================================================

@torch.no_grad()
def sample_standard(denoiser, n, K, H, W, device, n_steps=15, temperature=0.7):
    """Standard linear annealing (baseline)."""
    z = (torch.rand(n, K, H, W, device=device) > 0.5).float()
    for step in range(n_steps):
        nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
        probs = torch.sigmoid(denoiser(z, nl) / temperature)
        conf = (step + 1) / n_steps
        mask = (torch.rand_like(z) < conf).float()
        z = mask * (torch.rand_like(z) < probs).float() + (1 - mask) * z
    return (torch.sigmoid(denoiser(z, torch.zeros(n, device=device))) > 0.5).float()


@torch.no_grad()
def sample_maskgit(denoiser, n, K, H, W, device, n_steps=15, temperature=0.7):
    """MaskGIT-style confidence-based unmasking.

    Instead of randomly choosing which positions to update,
    update the most confident predictions first.
    """
    z = (torch.rand(n, K, H, W, device=device) > 0.5).float()
    # Track which positions are "committed"
    committed = torch.zeros(n, K, H, W, device=device, dtype=torch.bool)

    total_positions = K * H * W
    positions_per_step = total_positions // n_steps

    for step in range(n_steps):
        nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
        logits = denoiser(z, nl)
        probs = torch.sigmoid(logits / temperature)

        # Confidence = |prob - 0.5| (how sure we are about 0 or 1)
        confidence = (probs - 0.5).abs()
        # Mask out already committed positions
        confidence[committed] = -1.0

        # Select top-k most confident positions to commit
        n_to_commit = min(positions_per_step * 2, total_positions - committed.sum(dim=(1,2,3)).min().item())
        if n_to_commit <= 0:
            break

        # Flatten and select top-k per sample
        conf_flat = confidence.reshape(n, -1)  # [n, K*H*W]
        _, topk_idx = conf_flat.topk(int(n_to_commit), dim=1)

        # Create commit mask
        new_commit = torch.zeros_like(conf_flat, dtype=torch.bool)
        new_commit.scatter_(1, topk_idx, True)
        new_commit = new_commit.reshape(n, K, H, W)

        # Apply predictions at newly committed positions
        z_proposed = (torch.rand_like(z) < probs).float()
        z = torch.where(new_commit & ~committed, z_proposed, z)
        committed = committed | new_commit

    # Final pass
    logits = denoiser(z, torch.zeros(n, device=device))
    z = (torch.sigmoid(logits) > 0.5).float()
    return z


@torch.no_grad()
def sample_energy_guided(denoiser, e_core, n, K, H, W, device,
                          n_steps=15, temperature=0.7, energy_weight=0.3):
    """E_core-guided sampling: reject updates that increase energy."""
    z = (torch.rand(n, K, H, W, device=device) > 0.5).float()
    committed = torch.zeros(n, K, H, W, device=device, dtype=torch.bool)
    total_positions = K * H * W
    positions_per_step = total_positions // n_steps

    for step in range(n_steps):
        nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
        logits = denoiser(z, nl)
        probs = torch.sigmoid(logits / temperature)

        confidence = (probs - 0.5).abs()
        confidence[committed] = -1.0

        n_to_commit = min(positions_per_step * 2,
                          total_positions - committed.sum(dim=(1, 2, 3)).min().item())
        if n_to_commit <= 0:
            break

        conf_flat = confidence.reshape(n, -1)
        _, topk_idx = conf_flat.topk(int(n_to_commit), dim=1)
        new_commit = torch.zeros_like(conf_flat, dtype=torch.bool)
        new_commit.scatter_(1, topk_idx, True)
        new_commit = new_commit.reshape(n, K, H, W)

        z_proposed = (torch.rand_like(z) < probs).float()
        z_candidate = torch.where(new_commit & ~committed, z_proposed, z)

        # E_core rejection: compare energy before and after
        if step > 0 and step < n_steps - 1:
            e_old = e_core.energy(z)
            e_new = e_core.energy(z_candidate)
            # Accept if energy decreases, or with probability exp(-(e_new - e_old))
            accept_prob = torch.exp(-energy_weight * (e_new - e_old)).clamp(max=1.0)
            if accept_prob.item() > 0.3:  # mostly accept
                z = z_candidate
            else:
                # Partial accept: only commit the positions where confidence > 0.8
                high_conf = confidence > 0.3
                z = torch.where(new_commit & ~committed & high_conf, z_proposed, z)
        else:
            z = z_candidate

        committed = committed | new_commit

    logits = denoiser(z, torch.zeros(n, device=device))
    return (torch.sigmoid(logits) > 0.5).float()


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
            z, _ = encoder(data[i:i+bs].to(device)); zs.append(z.cpu())
    return torch.cat(zs)


def train_ecore(e_core, z_data, device, epochs=15, bs=128):
    """Train differentiable E_core (conv-based)."""
    opt = torch.optim.Adam(e_core.parameters(), lr=1e-3)
    for ep in tqdm(range(epochs), desc="E_core"):
        e_core.train(); perm = torch.randperm(len(z_data))
        tl, nb = 0., 0
        for i in range(0, len(z_data), bs):
            z = z_data[perm[i:i+bs]].to(device)
            opt.zero_grad()
            loss = e_core.energy(z)
            loss.backward(); opt.step()
            tl += loss.item(); nb += 1
    e_core.eval()
    return tl / nb


def train_denoiser_base(denoiser, z_data, decoder, device, epochs=30, bs=32):
    """Standard denoiser training (BCE + freq)."""
    opt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)
    for epoch in tqdm(range(epochs), desc="Denoiser"):
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
            loss_bce = F.binary_cross_entropy_with_logits(logits, z_clean)
            # Freq loss
            s = torch.sigmoid(logits); h = (s > 0.5).float()
            z_pred = h - s.detach() + s
            with torch.no_grad(): x_clean = decoder(z_clean)
            x_pred = decoder(z_pred)
            loss_freq = freq_scheduled_loss(x_pred, x_clean, progress)
            loss = loss_bce + 0.3 * loss_freq
            loss.backward(); opt.step()
            tl += loss_bce.item(); fl += loss_freq.item(); nb += 1
        if (epoch + 1) % 10 == 0:
            print(f"      ep {epoch+1}/{epochs}: BCE={tl/nb:.4f} freq={fl/nb:.4f}")
    denoiser.eval()


def train_denoiser_energy(denoiser, e_core, z_data, decoder, device, epochs=30, bs=32):
    """Energy-guided denoiser training: BCE + freq + E_core energy."""
    opt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)
    for epoch in tqdm(range(epochs), desc="EnDen"):
        denoiser.train(); perm = torch.randperm(N)
        tl, fl, el, nb = 0., 0., 0., 0
        progress = epoch / max(epochs - 1, 1)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]; z_clean = z_data[idx].to(device); B = z_clean.shape[0]
            noise_level = torch.rand(B, device=device)
            flip = (torch.rand_like(z_clean) < noise_level.view(B, 1, 1, 1)).float()
            z_noisy = z_clean * (1 - flip) + (1 - z_clean) * flip
            opt.zero_grad()
            logits = denoiser(z_noisy, noise_level)
            loss_bce = F.binary_cross_entropy_with_logits(logits, z_clean)

            # STE prediction
            s = torch.sigmoid(logits); h = (s > 0.5).float()
            z_pred = h - s.detach() + s

            # Freq loss
            with torch.no_grad(): x_clean = decoder(z_clean)
            x_pred = decoder(z_pred)
            loss_freq = freq_scheduled_loss(x_pred, x_clean, progress)

            # E_core energy: denoiser output should have low energy
            loss_energy = e_core.energy(z_pred)

            # Energy weight ramps up during training
            energy_w = 0.1 * min(1.0, progress * 2)
            loss = loss_bce + 0.3 * loss_freq + energy_w * loss_energy
            loss.backward(); opt.step()
            tl += loss_bce.item(); fl += loss_freq.item()
            el += loss_energy.item(); nb += 1
        if (epoch + 1) % 10 == 0:
            print(f"      ep {epoch+1}/{epochs}: BCE={tl/nb:.4f} freq={fl/nb:.4f} energy={el/nb:.4f}")
    denoiser.eval()


# ============================================================================
# EVALUATE
# ============================================================================

def evaluate(z_gen, decoder, encoder, e_core, z_data, test_x,
             real_hf_coh, real_hf_noi, device, bs=32):
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
    parser.add_argument('--output_dir', default='outputs/exp_gen_g4')
    parser.add_argument('--n_samples', type=int, default=256)
    parser.add_argument('--n_bits', type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    N_BITS = args.n_bits

    print("=" * 100)
    print("CIFAR-10 G4: ENERGY-GUIDED U-NET DENOISER")
    print("=" * 100)

    # Load
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

    # [3] Shared ADC/DAC
    print(f"\n[3] Training shared ADC/DAC (16×16×{N_BITS})...")
    torch.manual_seed(args.seed)
    encoder = Encoder16(N_BITS).to(device)
    decoder = Decoder16(N_BITS).to(device)
    adc_loss = train_adc(encoder, decoder, train_x, device, epochs=40, bs=32)
    print(f"    ADC loss: {adc_loss:.4f}")

    with torch.no_grad():
        tb = test_x[:32].to(device); zo, _ = encoder(tb)
        oracle_mse = F.mse_loss(decoder(zo), tb).item()
    print(f"    Oracle MSE: {oracle_mse:.4f}")
    save_grid(decoder(zo).cpu(), os.path.join(args.output_dir, 'oracle_recon.png'))

    # [4] Encode
    print("\n[4] Encoding training set...")
    z_data = encode_all(encoder, train_x, device, bs=32)
    K, H, W = z_data.shape[1:]
    print(f"    z: {z_data.shape}, usage={z_data.mean():.3f}")

    # [5] Train E_core (differentiable)
    print("\n[5] Training differentiable E_core...")
    e_core = DiffEnergyCore(N_BITS).to(device)
    e_loss = train_ecore(e_core, z_data, device, epochs=15, bs=128)
    print(f"    E_core loss: {e_loss:.4f}")
    print(f"    Violation on train data: {e_core.violation_rate(z_data[:100].to(device)):.4f}")

    # [6] CONFIG MATRIX
    configs = [
        ("A_flat_standard",     "flat",  "base",    "standard"),
        ("B_unet_standard",     "unet",  "base",    "standard"),
        ("C_unet_maskgit",      "unet",  "base",    "maskgit"),
        ("D_unet_energy",       "unet",  "energy",  "standard"),
        ("E_unet_energy_maskgit", "unet", "energy", "maskgit"),
        ("F_unet_energy_guided", "unet", "energy",  "energy_guided"),
    ]

    all_results = {}
    trained_denoisers = {}

    for cfg_name, arch, train_mode, sampler in configs:
        den_key = f"{arch}_{train_mode}"

        print(f"\n{'='*100}")
        print(f"CONFIG: {cfg_name} (arch={arch}, train={train_mode}, sample={sampler})")
        print("=" * 100)

        if den_key not in trained_denoisers:
            torch.manual_seed(args.seed)
            if arch == "flat":
                denoiser = FlatDenoiser(N_BITS).to(device)
            else:
                denoiser = UNetDenoiser(N_BITS, base_ch=64).to(device)

            den_p = sum(p.numel() for p in denoiser.parameters())
            print(f"    Denoiser params: {den_p:,}")

            if train_mode == "base":
                print("  Training denoiser (BCE + freq)...")
                train_denoiser_base(denoiser, z_data, decoder, device, epochs=30, bs=32)
            else:
                print("  Training denoiser (BCE + freq + E_core)...")
                train_denoiser_energy(denoiser, e_core, z_data, decoder, device, epochs=30, bs=32)

            trained_denoisers[den_key] = denoiser
        else:
            print("  Reusing trained denoiser...")

        denoiser = trained_denoisers[den_key]

        # Sample
        print(f"  Generating {args.n_samples} samples ({sampler})...")
        torch.manual_seed(args.seed)
        gen_bs = 32
        z_gen_list = []
        for gi in range(0, args.n_samples, gen_bs):
            nb = min(gen_bs, args.n_samples - gi)
            if sampler == "standard":
                z_gen_list.append(sample_standard(denoiser, nb, K, H, W, device).cpu())
            elif sampler == "maskgit":
                z_gen_list.append(sample_maskgit(denoiser, nb, K, H, W, device).cpu())
            elif sampler == "energy_guided":
                z_gen_list.append(sample_energy_guided(
                    denoiser, e_core, nb, K, H, W, device).cpu())
        z_gen = torch.cat(z_gen_list)

        # Evaluate
        print("  Evaluating...")
        r, x_gen = evaluate(z_gen, decoder, encoder, e_core, z_data, test_x,
                            real_hf_coh, real_hf_noi, device)
        r['oracle_mse'] = oracle_mse
        r['arch'] = arch
        r['train_mode'] = train_mode
        r['sampler'] = sampler
        all_results[cfg_name] = r

        save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))

        print(f"    viol={r['violation']:.4f} div={r['diversity']:.4f} "
              f"cycle={r['cycle']:.4f} conn={r['connectedness']:.4f}")
        print(f"    HF_coh={r['hf_coherence']:.4f}(real={real_hf_coh:.4f}) "
              f"HF_noi={r['hf_noise_index']:.2f}(real={real_hf_noi:.2f})")
        print(f"    E_gap: L={r['energy_gap_low']:.4f} M={r['energy_gap_mid']:.4f} "
              f"H={r['energy_gap_high']:.4f}")

        torch.cuda.empty_cache()

    # SUMMARY
    print(f"\n{'='*100}")
    print("G4 ENERGY-GUIDED U-NET SUMMARY")
    print(f"Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  conn={real_conn:.4f}")
    print("=" * 100)

    h = (f"{'config':<28} {'viol':>7} {'div':>7} {'cycle':>7} "
         f"{'conn':>7} {'HFcoh':>7} {'HFnoi':>7} {'EgL':>7} {'EgM':>7} {'EgH':>7}")
    print(h); print("-" * len(h))
    for name, r in all_results.items():
        print(f"{name:<28} {r['violation']:>7.4f} {r['diversity']:>7.4f} "
              f"{r['cycle']:>7.4f} {r['connectedness']:>7.4f} {r['hf_coherence']:>7.4f} "
              f"{r['hf_noise_index']:>7.2f} {r['energy_gap_low']:>7.4f} "
              f"{r['energy_gap_mid']:>7.4f} {r['energy_gap_high']:>7.4f}")

    # Comparative
    print(f"\n{'='*100}")
    print("COMPARATIVE ANALYSIS")
    print("=" * 100)

    if 'A_flat_standard' in all_results and 'B_unet_standard' in all_results:
        a, b = all_results['A_flat_standard'], all_results['B_unet_standard']
        print(f"\nFlat → U-Net (both standard sampling, base training):")
        print(f"  violation:  {a['violation']:.4f} → {b['violation']:.4f}")
        print(f"  diversity:  {a['diversity']:.4f} → {b['diversity']:.4f}")
        print(f"  HF_noise:   {a['hf_noise_index']:.2f} → {b['hf_noise_index']:.2f}")
        print(f"  E_gap_mid:  {a['energy_gap_mid']:.4f} → {b['energy_gap_mid']:.4f}")

    if 'B_unet_standard' in all_results and 'D_unet_energy' in all_results:
        b, d = all_results['B_unet_standard'], all_results['D_unet_energy']
        print(f"\nBase → Energy training (U-Net, standard sampling):")
        print(f"  violation:  {b['violation']:.4f} → {d['violation']:.4f}")
        print(f"  diversity:  {b['diversity']:.4f} → {d['diversity']:.4f}")

    if 'C_unet_maskgit' in all_results and 'B_unet_standard' in all_results:
        b, c = all_results['B_unet_standard'], all_results['C_unet_maskgit']
        print(f"\nStandard → MaskGIT sampling (U-Net, base training):")
        print(f"  violation:  {b['violation']:.4f} → {c['violation']:.4f}")
        print(f"  diversity:  {b['diversity']:.4f} → {c['diversity']:.4f}")

    best_viol = min(all_results.items(), key=lambda x: x[1]['violation'])
    best_div = max(all_results.items(), key=lambda x: x[1]['diversity'])
    best_hfn = min(all_results.items(), key=lambda x: abs(x[1]['hf_noise_index'] - real_hf_noi))
    print(f"\n  Best violation: {best_viol[0]} ({best_viol[1]['violation']:.4f})")
    print(f"  Best diversity: {best_div[0]} ({best_div[1]['diversity']:.4f})")
    print(f"  Best HF_noise:  {best_hfn[0]} ({best_hfn[1]['hf_noise_index']:.2f})")

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*100}")
    print("G4 EXPERIMENT COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
