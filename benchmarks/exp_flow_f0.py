#!/usr/bin/env python3
"""
Flow F0: Unified Descent Operator — Minimal Closed Loop
=========================================================
Core hypothesis: Replace one-shot denoiser with iterative descent operator
that minimizes E_tot = λ_obs·E_obs + λ_core·E_core.

Same operator serves generation (from noise) and repair (from corrupted z).
Classification reads out the converged z.

u_{t+1} = u_t + Δt · f_φ(u_t, ∇E_core(u_t), t) + σ(t)·ε

Key differences from G4:
  - Continuous internal state u (logits), z = Q(u) = (u > 0)
  - Iterative: T steps with schedule (not one-shot)
  - Energy descent hinge loss in training
  - Soft energy surrogate for differentiability through Q
  - Langevin noise for generation, deterministic for repair
  - Energy-driven early stopping

Configs:
  A) flat_oneshot    — Flat denoiser, one-shot (baseline from G4)
  B) flat_flow_T10   — Same flat denoiser wrapped in T=10 flow
  C) unet_flow_T10   — U-Net step function, T=10
  D1) unet_energy_train — U-Net flow + energy descent hinge in training
  D2) unet_energy_infer — U-Net flow (base train) + E_core projection at inference

New metrics (operator-level):
  - energy_mono_rate: fraction of steps where E_tot decreases
  - mean_steps_to_converge: avg steps before ΔE < ε
  - e_core_trajectory: E_core at each step (should decrease)

4GB GPU: 3000 train, 500 test, batch_size=32, 16×16×16 z
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
# DCT / FREQ UTILITIES (shared with G4)
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
# E_CORE (differentiable)
# ============================================================================

class DiffEnergyCore(nn.Module):
    """Conv-based E_core: E(z) = -Σ log p(z_i | neighbors)."""
    def __init__(self, n_bits):
        super().__init__(); self.n_bits = n_bits
        self.predictor = nn.Sequential(
            nn.Conv2d(n_bits, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, n_bits, 3, padding=1))

    def energy(self, z):
        logits = self.predictor(z)
        return F.binary_cross_entropy_with_logits(logits, z, reduction='mean')

    def energy_per_position(self, z):
        """Return per-position energy [B, K, H, W]."""
        logits = self.predictor(z)
        return F.binary_cross_entropy_with_logits(logits, z, reduction='none')

    def soft_energy(self, u):
        """Soft energy on continuous u ∈ R (not quantized).
        Uses sigmoid(u) as soft z for smooth gradients."""
        z_soft = torch.sigmoid(u)
        logits = self.predictor(z_soft)
        return F.binary_cross_entropy_with_logits(logits, z_soft, reduction='mean')

    def violation_rate(self, z):
        with torch.no_grad():
            logits = self.predictor(z)
            pred = (logits > 0).float()
            return (pred != z).float().mean().item()


# ============================================================================
# STEP FUNCTIONS (the f_φ in u_{t+1} = u_t + Δt·f_φ(...))
# ============================================================================

class FlatStepFn(nn.Module):
    """Flat 3-layer CNN step function (same capacity as G4 FlatDenoiser)."""
    def __init__(self, n_bits):
        super().__init__(); self.n_bits = n_bits
        hid = 64
        # Input: u (n_bits) + t_emb (1) + ∇E_core (n_bits) = 2*n_bits + 1
        self.net = nn.Sequential(
            nn.Conv2d(n_bits * 2 + 1, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, n_bits, 3, padding=1))
        self.skip = nn.Conv2d(n_bits, n_bits, 1)

    def forward(self, u, e_grad, t_scalar):
        """
        u: [B, K, H, W] continuous logits
        e_grad: [B, K, H, W] gradient of soft E_core w.r.t. u
        t_scalar: [B] time step (1.0 = start, 0.0 = end)
        """
        B = u.shape[0]
        t_map = t_scalar.view(B, 1, 1, 1).expand(-1, 1, u.shape[2], u.shape[3])
        inp = torch.cat([u, e_grad, t_map], dim=1)
        return self.net(inp) + self.skip(u)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.bmm(q.transpose(1, 2), k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(out)


class UNetStepFn(nn.Module):
    """U-Net step function: predicts Δu given (u, ∇E_core, t).

    Architecture: 16×16 → 8×8 → 4×4 (attention) → 8×8 → 16×16
    Input channels: n_bits (u) + n_bits (∇E_core) + 1 (t) = 2K+1
    """
    def __init__(self, n_bits, base_ch=48):
        super().__init__(); self.n_bits = n_bits
        in_ch = n_bits * 2 + 1

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch), nn.SiLU())
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2), nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2), nn.SiLU())
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)

        self.mid = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.GroupNorm(8, base_ch * 4), nn.SiLU())
        self.mid_attn = SelfAttention(base_ch * 4)
        self.mid2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.GroupNorm(8, base_ch * 4), nn.SiLU())

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2), nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2), nn.SiLU())

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch), nn.SiLU())

        self.out = nn.Conv2d(base_ch, n_bits, 1)

        # Time embedding
        self.t_emb = nn.Sequential(
            nn.Linear(1, base_ch), nn.SiLU(), nn.Linear(base_ch, base_ch))

    def forward(self, u, e_grad, t_scalar):
        B, K, H, W = u.shape
        t_map = t_scalar.view(B, 1, 1, 1).expand(-1, 1, H, W)
        x = torch.cat([u, e_grad, t_map], dim=1)

        t_emb = self.t_emb(t_scalar.view(B, 1))

        h1 = self.enc1(x)
        h1 = h1 + t_emb.view(B, -1, 1, 1)
        d1 = self.down1(h1)
        h2 = self.enc2(d1)
        d2 = self.down2(h2)

        m = self.mid(d2)
        m = self.mid_attn(m)
        m = self.mid2(m)

        u2 = self.up2(m)
        u2 = self.dec2(torch.cat([u2, h2], dim=1))
        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, h1], dim=1))

        return self.out(u1)  # Δu (residual update, no skip — this is the flow direction)


# ============================================================================
# FLOW SAMPLING (the core inference loop)
# ============================================================================

def quantize(u):
    """Hard quantize: z = (u > 0)."""
    return (u > 0).float()

def soft_quantize(u, temperature=1.0):
    """Soft quantize for differentiable forward pass."""
    return torch.sigmoid(u / temperature)


@torch.no_grad()
def compute_e_core_grad(e_core, u):
    """Compute ∇_u E_core(sigmoid(u)) via finite difference (no autograd needed).
    Fast approximation: use E_core predictor output as pseudo-gradient."""
    z_soft = torch.sigmoid(u)
    logits = e_core.predictor(z_soft)
    # Pseudo-gradient: direction that E_core thinks z should move
    # If logit > 0, E_core predicts bit=1; if z_soft < 0.5, gradient pushes u up
    return (torch.sigmoid(logits) - z_soft)  # [B, K, H, W]


@torch.no_grad()
def sample_oneshot(denoiser_or_step, e_core, n, K, H, W, device,
                   n_steps=15, temperature=0.7, is_step_fn=False):
    """One-shot baseline: same as G4 standard sampling."""
    z = (torch.rand(n, K, H, W, device=device) > 0.5).float()
    for step in range(n_steps):
        nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
        if is_step_fn:
            e_grad = compute_e_core_grad(e_core, z * 2 - 1)  # z→u space
            logits = denoiser_or_step(z * 2 - 1, e_grad, nl)
        else:
            # Legacy flat denoiser: (z, noise_level) → logits
            logits = denoiser_or_step(z, nl)
        probs = torch.sigmoid(logits / temperature)
        conf = (step + 1) / n_steps
        mask = (torch.rand_like(z) < conf).float()
        z = mask * (torch.rand_like(z) < probs).float() + (1 - mask) * z
    # Final pass
    if is_step_fn:
        e_grad = compute_e_core_grad(e_core, z * 2 - 1)
        logits = denoiser_or_step(z * 2 - 1, e_grad, torch.zeros(n, device=device))
    else:
        logits = denoiser_or_step(z, torch.zeros(n, device=device))
    return (torch.sigmoid(logits) > 0.5).float()


@torch.no_grad()
def sample_flow(step_fn, e_core, n, K, H, W, device,
                T=10, dt=0.5, sigma_schedule='cosine',
                temperature=1.0, use_langevin=True,
                use_projection=False, projection_strength=0.1):
    """Flow-based sampling: iterative descent in u-space.

    u_0 ~ N(0, 1) or uniform
    u_{t+1} = u_t + dt · f_φ(u_t, ∇E_core, t) + σ(t)·ε
    z = Q(u_T)

    Returns z (binary) and trajectory info.
    """
    # Initialize u from random (generation mode)
    u = torch.randn(n, K, H, W, device=device) * 0.5

    trajectory = {'e_core': [], 'e_soft': []}

    for step in range(T):
        t_frac = 1.0 - step / T  # 1.0 → 0.0
        t_tensor = torch.tensor([t_frac], device=device).expand(n)

        # Compute E_core gradient (pseudo-gradient from predictor)
        e_grad = compute_e_core_grad(e_core, u)

        # Step function: predict update direction
        delta_u = step_fn(u, e_grad, t_tensor)

        # Apply update
        u = u + dt * delta_u

        # Langevin noise (decreasing over time)
        if use_langevin:
            if sigma_schedule == 'cosine':
                sigma = 0.3 * np.cos(np.pi * step / (2 * T))  # decreases to 0
            elif sigma_schedule == 'linear':
                sigma = 0.3 * (1 - step / T)
            else:
                sigma = 0.0
            if sigma > 0.01:
                u = u + sigma * torch.randn_like(u)

        # E_core projection (D2): locally correct high-violation positions
        if use_projection and step > 0:
            z_hard = quantize(u)
            violation_map = e_core.energy_per_position(z_hard)
            # Only project positions with high violation
            high_viol = (violation_map > violation_map.mean() + violation_map.std())
            # Nudge u toward E_core prediction
            proj_dir = e_grad  # direction E_core wants
            u = u + projection_strength * proj_dir * high_viol.float()

        # Track trajectory
        z_cur = quantize(u)
        trajectory['e_core'].append(e_core.energy(z_cur).item())
        trajectory['e_soft'].append(e_core.soft_energy(u).item())

    z_final = quantize(u)
    return z_final, trajectory


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
    opt = torch.optim.Adam(e_core.parameters(), lr=1e-3)
    for ep in tqdm(range(epochs), desc="E_core"):
        e_core.train(); perm = torch.randperm(len(z_data))
        tl, nb = 0., 0
        for i in range(0, len(z_data), bs):
            z = z_data[perm[i:i+bs]].to(device)
            opt.zero_grad(); loss = e_core.energy(z)
            loss.backward(); opt.step(); tl += loss.item(); nb += 1
    e_core.eval()
    return tl / nb


def train_step_fn_base(step_fn, e_core, z_data, decoder, device,
                       epochs=30, bs=32, T_unroll=3):
    """Train step function with single-step regression + optional T-step unroll.

    For each training sample:
      1. Sample z_clean from data
      2. Create z_noisy by flipping bits
      3. Convert to u-space: u = z*2-1 + noise
      4. Run step function for 1 step (or T_unroll steps for multi-step)
      5. Loss = BCE(Q(u_final), z_clean) + freq_loss
    """
    opt = torch.optim.Adam(step_fn.parameters(), lr=1e-3)
    N = len(z_data)

    for epoch in tqdm(range(epochs), desc="StepFn"):
        step_fn.train(); perm = torch.randperm(N)
        tl, fl, nb = 0., 0., 0
        progress = epoch / max(epochs - 1, 1)

        for i in range(0, N, bs):
            idx = perm[i:i+bs]; z_clean = z_data[idx].to(device); B = z_clean.shape[0]

            # Create noisy u in continuous space
            noise_level = torch.rand(B, device=device)
            flip_prob = noise_level.view(B, 1, 1, 1)
            # u_clean = z_clean * 2 - 1 (maps {0,1} to {-1,+1})
            u_clean = z_clean * 2.0 - 1.0
            # Add noise proportional to noise_level
            u_noisy = u_clean + torch.randn_like(u_clean) * flip_prob * 2.0

            opt.zero_grad()

            # Run T_unroll steps
            u = u_noisy
            for t_step in range(T_unroll):
                t_frac = 1.0 - t_step / T_unroll
                t_tensor = torch.full((B,), t_frac, device=device)
                e_grad = (torch.sigmoid(e_core.predictor(torch.sigmoid(u))) -
                          torch.sigmoid(u))
                delta_u = step_fn(u, e_grad, t_tensor)
                u = u + 0.5 * delta_u  # dt=0.5

            # Loss on final u
            z_pred_soft = torch.sigmoid(u)
            loss_bce = F.binary_cross_entropy(z_pred_soft, z_clean)

            # Freq loss (STE to get hard z for decoder)
            z_hard = (z_pred_soft > 0.5).float()
            z_ste = z_hard - z_pred_soft.detach() + z_pred_soft
            with torch.no_grad(): x_clean = decoder(z_clean)
            x_pred = decoder(z_ste)
            loss_freq = freq_scheduled_loss(x_pred, x_clean, progress)

            loss = loss_bce + 0.3 * loss_freq
            loss.backward(); opt.step()
            tl += loss_bce.item(); fl += loss_freq.item(); nb += 1

        if (epoch + 1) % 10 == 0:
            print(f"      ep {epoch+1}/{epochs}: BCE={tl/nb:.4f} freq={fl/nb:.4f}")

    step_fn.eval()


def train_step_fn_energy(step_fn, e_core, z_data, decoder, device,
                         epochs=30, bs=32, T_unroll=3):
    """Train with energy descent hinge loss (D1).

    Additional loss: L_↓ = softplus(E_soft(u_{t+1}) - E_soft(u_t) + δ)
    Uses soft energy on sigmoid(u) for smooth gradients.
    """
    opt = torch.optim.Adam(step_fn.parameters(), lr=1e-3)
    N = len(z_data)

    for epoch in tqdm(range(epochs), desc="StepFn+E"):
        step_fn.train(); perm = torch.randperm(N)
        tl, fl, el, nb = 0., 0., 0., 0
        progress = epoch / max(epochs - 1, 1)

        for i in range(0, N, bs):
            idx = perm[i:i+bs]; z_clean = z_data[idx].to(device); B = z_clean.shape[0]

            noise_level = torch.rand(B, device=device)
            flip_prob = noise_level.view(B, 1, 1, 1)
            u_clean = z_clean * 2.0 - 1.0
            u_noisy = u_clean + torch.randn_like(u_clean) * flip_prob * 2.0

            opt.zero_grad()

            # Run T_unroll steps, accumulate energy descent loss
            u = u_noisy
            loss_descent = torch.tensor(0.0, device=device)
            delta_margin = 0.01  # δ in hinge

            for t_step in range(T_unroll):
                t_frac = 1.0 - t_step / T_unroll
                t_tensor = torch.full((B,), t_frac, device=device)

                # Soft energy before step
                e_before = e_core.soft_energy(u)

                e_grad = (torch.sigmoid(e_core.predictor(torch.sigmoid(u))) -
                          torch.sigmoid(u))
                delta_u = step_fn(u, e_grad, t_tensor)
                u = u + 0.5 * delta_u

                # Soft energy after step
                e_after = e_core.soft_energy(u)

                # Hinge: penalize energy increase
                loss_descent = loss_descent + F.softplus(e_after - e_before + delta_margin)

            loss_descent = loss_descent / T_unroll

            # Reconstruction loss on final u
            z_pred_soft = torch.sigmoid(u)
            loss_bce = F.binary_cross_entropy(z_pred_soft, z_clean)

            # Freq loss
            z_hard = (z_pred_soft > 0.5).float()
            z_ste = z_hard - z_pred_soft.detach() + z_pred_soft
            with torch.no_grad(): x_clean = decoder(z_clean)
            x_pred = decoder(z_ste)
            loss_freq = freq_scheduled_loss(x_pred, x_clean, progress)

            # Energy weight ramps up
            energy_w = 0.2 * min(1.0, progress * 2)
            loss = loss_bce + 0.3 * loss_freq + energy_w * loss_descent
            loss.backward(); opt.step()
            tl += loss_bce.item(); fl += loss_freq.item()
            el += loss_descent.item(); nb += 1

        if (epoch + 1) % 10 == 0:
            print(f"      ep {epoch+1}/{epochs}: BCE={tl/nb:.4f} "
                  f"freq={fl/nb:.4f} E_desc={el/nb:.4f}")

    step_fn.eval()


# ============================================================================
# LEGACY FLAT DENOISER (for Config A baseline)
# ============================================================================

class FlatDenoiser(nn.Module):
    """Same as G4 flat denoiser for baseline comparison."""
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


def train_denoiser_base(denoiser, z_data, decoder, device, epochs=30, bs=32):
    """Legacy one-shot denoiser training."""
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


# ============================================================================
# EVALUATE
# ============================================================================

def evaluate(z_gen, decoder, encoder, e_core, z_data, test_x,
             real_hf_coh, real_hf_noi, device, bs=32, trajectory=None):
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

    result = {
        'violation': viol, 'diversity': div, 'cycle': cycle,
        'connectedness': conn, 'hf_coherence': hfc, 'hf_noise_index': hfn,
        'energy_gap_low': band['energy_gap_low'],
        'energy_gap_mid': band['energy_gap_mid'],
        'energy_gap_high': band['energy_gap_high'],
    }

    # Operator-level metrics from trajectory
    if trajectory and len(trajectory['e_core']) > 1:
        e_seq = trajectory['e_core']
        # Energy monotonicity: fraction of steps where E decreases
        mono_count = sum(1 for i in range(1, len(e_seq)) if e_seq[i] < e_seq[i-1])
        result['energy_mono_rate'] = mono_count / (len(e_seq) - 1)
        result['e_core_start'] = e_seq[0]
        result['e_core_end'] = e_seq[-1]
        result['e_core_drop'] = e_seq[0] - e_seq[-1]

        # Steps to convergence (ΔE < 0.001 for 2 consecutive)
        converge_step = len(e_seq)
        for i in range(2, len(e_seq)):
            if (abs(e_seq[i] - e_seq[i-1]) < 0.001 and
                abs(e_seq[i-1] - e_seq[i-2]) < 0.001):
                converge_step = i
                break
        result['converge_step'] = converge_step

    return result, x_gen


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_flow_f0')
    parser.add_argument('--n_samples', type=int, default=256)
    parser.add_argument('--n_bits', type=int, default=16)
    parser.add_argument('--T', type=int, default=10, help='Flow steps')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    N_BITS = args.n_bits; T = args.T

    print("=" * 100)
    print("FLOW F0: UNIFIED DESCENT OPERATOR")
    print("=" * 100)

    # ========== SHARED SETUP ==========
    print("\n[1] Loading CIFAR-10...")
    from torchvision import datasets, transforms
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(args.seed)
    train_x = torch.stack([train_ds[i][0] for i in rng.choice(len(train_ds), 3000, replace=False)])
    test_x = torch.stack([test_ds[i][0] for i in rng.choice(len(test_ds), 500, replace=False)])
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")

    print("\n[2] Reference metrics...")
    real_hf_coh = hf_coherence_metric(test_x[:200], device)
    real_hf_noi = hf_noise_index(test_x[:200], device)
    real_conn = connectedness_proxy(test_x[:100].numpy())
    print(f"    HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  conn={real_conn:.4f}")
    save_grid(test_x[:64], os.path.join(args.output_dir, 'real_samples.png'))

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

    print("\n[4] Encoding training set...")
    z_data = encode_all(encoder, train_x, device, bs=32)
    K, H, W = z_data.shape[1:]
    print(f"    z: {z_data.shape}, usage={z_data.mean():.3f}")

    print("\n[5] Training differentiable E_core...")
    e_core = DiffEnergyCore(N_BITS).to(device)
    e_loss = train_ecore(e_core, z_data, device, epochs=15, bs=128)
    print(f"    E_core loss: {e_loss:.4f}")
    print(f"    Violation on train data: {e_core.violation_rate(z_data[:100].to(device)):.4f}")

    # ========== CONFIG MATRIX ==========
    configs = [
        # (name,           arch,   train_mode, sample_mode,     kwargs)
        ("A_flat_oneshot",  "flat_legacy", "base", "oneshot",   {}),
        ("B_flat_flow",     "flat_step",   "base", "flow",      {'T': T}),
        ("C_unet_flow",     "unet_step",   "base", "flow",      {'T': T}),
        ("D1_unet_energy_train", "unet_step", "energy", "flow", {'T': T}),
        ("D2_unet_energy_infer", "unet_step", "base", "flow",   {'T': T, 'use_projection': True}),
    ]

    all_results = {}
    trained_models = {}

    for cfg_name, arch, train_mode, sample_mode, kwargs in configs:
        model_key = f"{arch}_{train_mode}"

        print(f"\n{'='*100}")
        print(f"CONFIG: {cfg_name} (arch={arch}, train={train_mode}, sample={sample_mode})")
        print("=" * 100)

        if model_key not in trained_models:
            torch.manual_seed(args.seed)

            if arch == "flat_legacy":
                model = FlatDenoiser(N_BITS).to(device)
            elif arch == "flat_step":
                model = FlatStepFn(N_BITS).to(device)
            elif arch == "unet_step":
                model = UNetStepFn(N_BITS, base_ch=48).to(device)
            else:
                raise ValueError(f"Unknown arch: {arch}")

            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Model params: {n_params:,}")

            if arch == "flat_legacy":
                print("  Training legacy denoiser (BCE + freq)...")
                train_denoiser_base(model, z_data, decoder, device, epochs=30, bs=32)
            elif train_mode == "base":
                print("  Training step function (BCE + freq, T_unroll=3)...")
                train_step_fn_base(model, e_core, z_data, decoder, device,
                                   epochs=30, bs=32, T_unroll=3)
            elif train_mode == "energy":
                print("  Training step function (BCE + freq + energy descent, T_unroll=3)...")
                train_step_fn_energy(model, e_core, z_data, decoder, device,
                                     epochs=30, bs=32, T_unroll=3)

            trained_models[model_key] = model
        else:
            print("  Reusing trained model...")

        model = trained_models[model_key]

        # ========== SAMPLE ==========
        print(f"  Generating {args.n_samples} samples...")
        torch.manual_seed(args.seed + 100)  # Different seed for sampling
        gen_bs = 32
        z_gen_list = []; all_trajectories = []

        for gi in range(0, args.n_samples, gen_bs):
            nb = min(gen_bs, args.n_samples - gi)

            if sample_mode == "oneshot":
                z_batch = sample_oneshot(model, e_core, nb, K, H, W, device,
                                        is_step_fn=(arch != "flat_legacy"))
                traj = None
            elif sample_mode == "flow":
                use_proj = kwargs.get('use_projection', False)
                z_batch, traj = sample_flow(
                    model, e_core, nb, K, H, W, device,
                    T=kwargs.get('T', T), dt=0.5,
                    use_langevin=True, use_projection=use_proj)
                all_trajectories.append(traj)
            else:
                raise ValueError(f"Unknown sample_mode: {sample_mode}")

            z_gen_list.append(z_batch.cpu())

        z_gen = torch.cat(z_gen_list)

        # Aggregate trajectory info
        agg_traj = None
        if all_trajectories:
            agg_traj = {
                'e_core': [np.mean([t['e_core'][s] for t in all_trajectories])
                           for s in range(len(all_trajectories[0]['e_core']))],
                'e_soft': [np.mean([t['e_soft'][s] for t in all_trajectories])
                           for s in range(len(all_trajectories[0]['e_soft']))],
            }

        # ========== EVALUATE ==========
        print("  Evaluating...")
        r, x_gen = evaluate(z_gen, decoder, encoder, e_core, z_data, test_x,
                            real_hf_coh, real_hf_noi, device, trajectory=agg_traj)
        r['oracle_mse'] = oracle_mse
        r['arch'] = arch
        r['train_mode'] = train_mode
        r['sample_mode'] = sample_mode
        all_results[cfg_name] = r

        save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))

        print(f"    viol={r['violation']:.4f}  div={r['diversity']:.4f}  "
              f"cycle={r['cycle']:.4f}  conn={r['connectedness']:.4f}")
        print(f"    HF_coh={r['hf_coherence']:.4f}(real={real_hf_coh:.4f})  "
              f"HF_noi={r['hf_noise_index']:.2f}(real={real_hf_noi:.2f})")
        print(f"    E_gap: L={r['energy_gap_low']:.4f}  M={r['energy_gap_mid']:.4f}  "
              f"H={r['energy_gap_high']:.4f}")

        if 'energy_mono_rate' in r:
            print(f"    [OPERATOR] mono_rate={r['energy_mono_rate']:.3f}  "
                  f"E_drop={r.get('e_core_drop', 0):.4f}  "
                  f"converge_step={r.get('converge_step', 'N/A')}")
            if agg_traj:
                print(f"    [TRAJECTORY] E_core: "
                      f"{' → '.join(f'{e:.4f}' for e in agg_traj['e_core'][:5])} ... "
                      f"→ {agg_traj['e_core'][-1]:.4f}")

        torch.cuda.empty_cache()

    # ========== SUMMARY ==========
    print(f"\n{'='*100}")
    print("FLOW F0 SUMMARY: UNIFIED DESCENT OPERATOR")
    print(f"Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  conn={real_conn:.4f}")
    print("=" * 100)

    header = (f"{'config':<25} {'viol':>7} {'div':>7} {'cycle':>7} {'conn':>7} "
              f"{'HFcoh':>7} {'HFnoi':>7} {'EgL':>7} {'mono':>6} {'Edrop':>7}")
    print(header); print("-" * len(header))
    for name, r in all_results.items():
        mono = r.get('energy_mono_rate', -1)
        edrop = r.get('e_core_drop', -1)
        mono_s = f"{mono:>6.3f}" if mono >= 0 else "   N/A"
        edrop_s = f"{edrop:>7.4f}" if edrop >= 0 else "    N/A"
        print(f"{name:<25} {r['violation']:>7.4f} {r['diversity']:>7.4f} "
              f"{r['cycle']:>7.4f} {r['connectedness']:>7.4f} "
              f"{r['hf_coherence']:>7.4f} {r['hf_noise_index']:>7.2f} "
              f"{r['energy_gap_low']:>7.4f} {mono_s} {edrop_s}")

    # Comparative analysis
    print(f"\n{'='*100}")
    print("COMPARATIVE ANALYSIS")
    print("=" * 100)

    if 'A_flat_oneshot' in all_results and 'B_flat_flow' in all_results:
        a, b = all_results['A_flat_oneshot'], all_results['B_flat_flow']
        print(f"\nOne-shot → Flow (same flat capacity):")
        print(f"  violation:  {a['violation']:.4f} → {b['violation']:.4f}")
        print(f"  diversity:  {a['diversity']:.4f} → {b['diversity']:.4f}")
        print(f"  HF_noise:   {a['hf_noise_index']:.2f} → {b['hf_noise_index']:.2f}")

    if 'B_flat_flow' in all_results and 'C_unet_flow' in all_results:
        b, c = all_results['B_flat_flow'], all_results['C_unet_flow']
        print(f"\nFlat → U-Net (both flow T={T}):")
        print(f"  violation:  {b['violation']:.4f} → {c['violation']:.4f}")
        print(f"  diversity:  {b['diversity']:.4f} → {c['diversity']:.4f}")
        print(f"  HF_noise:   {b['hf_noise_index']:.2f} → {c['hf_noise_index']:.2f}")

    if 'C_unet_flow' in all_results and 'D1_unet_energy_train' in all_results:
        c, d1 = all_results['C_unet_flow'], all_results['D1_unet_energy_train']
        print(f"\nBase → Energy training (U-Net flow):")
        print(f"  violation:  {c['violation']:.4f} → {d1['violation']:.4f}")
        print(f"  mono_rate:  {c.get('energy_mono_rate', 0):.3f} → {d1.get('energy_mono_rate', 0):.3f}")
        print(f"  E_drop:     {c.get('e_core_drop', 0):.4f} → {d1.get('e_core_drop', 0):.4f}")

    if 'C_unet_flow' in all_results and 'D2_unet_energy_infer' in all_results:
        c, d2 = all_results['C_unet_flow'], all_results['D2_unet_energy_infer']
        print(f"\nBase → E_core projection at inference (U-Net flow):")
        print(f"  violation:  {c['violation']:.4f} → {d2['violation']:.4f}")
        print(f"  mono_rate:  {c.get('energy_mono_rate', 0):.3f} → {d2.get('energy_mono_rate', 0):.3f}")

    best_viol = min(all_results.items(), key=lambda x: x[1]['violation'])
    best_div = max(all_results.items(), key=lambda x: x[1]['diversity'])
    best_hfn = min(all_results.items(), key=lambda x: abs(x[1]['hf_noise_index'] - real_hf_noi))
    print(f"\n  Best violation: {best_viol[0]} ({best_viol[1]['violation']:.4f})")
    print(f"  Best diversity: {best_div[0]} ({best_div[1]['diversity']:.4f})")
    print(f"  Best HF_noise:  {best_hfn[0]} ({best_hfn[1]['hf_noise_index']:.2f})")

    if any('energy_mono_rate' in r for r in all_results.values()):
        best_mono = max(
            [(k, v) for k, v in all_results.items() if 'energy_mono_rate' in v],
            key=lambda x: x[1]['energy_mono_rate'])
        print(f"  Best mono_rate: {best_mono[0]} ({best_mono[1]['energy_mono_rate']:.3f})")

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*100}")
    print("F0 EXPERIMENT COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
