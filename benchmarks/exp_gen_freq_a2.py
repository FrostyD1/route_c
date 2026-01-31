#!/usr/bin/env python3
"""
Route A2: Structured High-Frequency Generation via Freq-Aware Energy Geometry
==============================================================================
Key insight from A1v2: "有高频 ≠ 有纹理"
  - Denoise suppresses ALL high-freq (including texture) → "轮廓像但糊"
  - AR generates NOISY high-freq (random, not structured) → "花但碎"

Solution: teach the denoiser to generate STRUCTURED high-freq by:
  1. Spatial frequency decomposition (full-image DCT → band-specific images)
  2. Band-specific spatial residual maps (not scalar BEP)
  3. HF coherence loss (encourage spatially correlated high-freq = texture)
  4. Frequency-scheduled training (low-freq first, high-freq later)
  5. Continuous spatial guidance during sampling

New metrics (non-task-specific):
  - Per-band energy distance (low/mid/high separately)
  - HF coherence: local spatial autocorrelation of high-freq components
  - HF noise index: edge_energy / hf_energy (>1 = noisy HF, <1 = structured HF)

Experiment matrix (8 runs on FMNIST):
  1. baseline:           No freq (standard denoiser)
  2. freq_amp:           Band energy matching only (A1v2 freq_train_03 equivalent)
  3. freq_amp_coh:       Band energy + HF coherence loss
  4. freq_sched:         Frequency-scheduled training (low→high)
  5. freq_sched_coh:     Scheduled + coherence
  6. freq_sched_coh_ms:  Scheduled + coherence + multiscale sampling
  7. freq_full:          All: scheduled + coherence + continuous spatial guidance
  8. freq_full_ms:       All + multiscale

Usage:
    python3 -u benchmarks/exp_gen_freq_a2.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, csv, argparse
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# SPATIAL FREQUENCY DECOMPOSITION
# ============================================================================

def dct2d(x):
    """2D DCT (Type-II, orthonormal). x: (B, C, H, W)."""
    B, C, H, W = x.shape
    def dct_matrix(N):
        n = torch.arange(N, dtype=x.dtype, device=x.device)
        k = n.unsqueeze(1)
        D = torch.cos(np.pi * (2 * n + 1) * k / (2 * N))
        D[0] *= 1.0 / np.sqrt(N)
        D[1:] *= np.sqrt(2.0 / N)
        return D
    DH = dct_matrix(H); DW = dct_matrix(W)
    return torch.einsum('hH,bcHW,wW->bchw', DH, x, DW)


def idct2d(X):
    """2D inverse DCT. X: (B, C, H, W)."""
    B, C, H, W = X.shape
    def dct_matrix(N):
        n = torch.arange(N, dtype=X.dtype, device=X.device)
        k = n.unsqueeze(1)
        D = torch.cos(np.pi * (2 * n + 1) * k / (2 * N))
        D[0] *= 1.0 / np.sqrt(N)
        D[1:] *= np.sqrt(2.0 / N)
        return D
    # iDCT = D^T (orthonormal)
    DH = dct_matrix(H).T
    DW = dct_matrix(W).T
    return torch.einsum('Hh,bchw,Ww->bcHW', DH, X, DW)


def get_freq_masks(H, W, device='cpu'):
    """Low/mid/high frequency masks based on Manhattan distance from DC."""
    fy = torch.arange(H, device=device).float()
    fx = torch.arange(W, device=device).float()
    freq_grid = fy.unsqueeze(1) + fx.unsqueeze(0)
    max_freq = H + W - 2
    t1, t2 = max_freq / 3.0, 2 * max_freq / 3.0
    return ((freq_grid <= t1).float(),
            ((freq_grid > t1) & (freq_grid <= t2)).float(),
            (freq_grid > t2).float())


def decompose_bands(x):
    """Decompose image into low/mid/high frequency band images.
    x: (B, C, H, W) → returns (x_low, x_mid, x_high) each same shape.
    """
    B, C, H, W = x.shape
    low_m, mid_m, high_m = get_freq_masks(H, W, x.device)
    dct_x = dct2d(x)
    x_low = idct2d(dct_x * low_m)
    x_mid = idct2d(dct_x * mid_m)
    x_high = idct2d(dct_x * high_m)
    return x_low, x_mid, x_high


# ============================================================================
# HF COHERENCE: Spatial autocorrelation of high-freq components
# ============================================================================

def hf_local_autocorrelation(x_high, shifts=[(0, 1), (1, 0), (1, 1)]):
    """Compute mean local autocorrelation of high-freq band image.
    High value = structured texture; Low value = random noise.
    x_high: (B, C, H, W).
    Returns: scalar (mean autocorrelation across shifts).
    """
    B, C, H, W = x_high.shape
    # Normalize
    x_flat = x_high.reshape(B * C, H, W)
    x_mean = x_flat.mean(dim=(1, 2), keepdim=True)
    x_std = x_flat.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    x_norm = (x_flat - x_mean) / x_std

    correlations = []
    for dy, dx in shifts:
        if dy > 0:
            a, b = x_norm[:, dy:, :], x_norm[:, :-dy, :]
        else:
            a, b = x_norm, x_norm
        if dx > 0:
            a, b = a[:, :, dx:], b[:, :, :-dx]

        corr = (a * b).mean(dim=(1, 2))  # (B*C,)
        correlations.append(corr.mean().item())

    return np.mean(correlations)


def hf_coherence_loss(x_pred_high, x_target_high):
    """Loss that encourages high-freq to be spatially coherent like target.
    Penalizes when predicted HF has lower autocorrelation than target HF.
    """
    B, C, H, W = x_pred_high.shape

    def autocorr_map(x):
        """Per-position local autocorrelation (3×3 neighborhood)."""
        # Use unfold to get local patches
        x_flat = x.reshape(B * C, 1, H, W)
        # Average pool of x^2 gives local energy
        local_energy = F.avg_pool2d(x_flat ** 2, 3, stride=1, padding=1)
        # Average pool of x gives local mean
        local_mean = F.avg_pool2d(x_flat, 3, stride=1, padding=1)
        # Average pool of x * shifted_x gives cross-correlation
        # Use shift-right correlation as proxy
        x_shift = F.pad(x_flat[:, :, :, :-1], (1, 0))
        local_cross = F.avg_pool2d(x_flat * x_shift, 3, stride=1, padding=1)
        # Correlation = (cross - mean^2) / (energy - mean^2 + eps)
        var = (local_energy - local_mean ** 2).clamp(min=1e-8)
        cov = local_cross - local_mean * F.avg_pool2d(x_shift, 3, stride=1, padding=1)
        return (cov / var).reshape(B, C, H, W)

    corr_pred = autocorr_map(x_pred_high)
    corr_target = autocorr_map(x_target_high)

    # Loss: predicted correlation should match target correlation
    return F.mse_loss(corr_pred, corr_target)


# ============================================================================
# BAND-SPECIFIC SPATIAL RESIDUAL
# ============================================================================

def band_spatial_residual(x_pred, x_target):
    """Compute per-band MSE residual maps.
    Returns (res_low, res_mid, res_high) each (B, C, H, W).
    """
    pred_l, pred_m, pred_h = decompose_bands(x_pred)
    tgt_l, tgt_m, tgt_h = decompose_bands(x_target)
    return (pred_l - tgt_l) ** 2, (pred_m - tgt_m) ** 2, (pred_h - tgt_h) ** 2


def freq_scheduled_loss(x_pred, x_target, progress):
    """Frequency-scheduled loss: emphasize low-freq early, add high-freq later.
    progress: 0.0 (start) → 1.0 (end of training).
    """
    pred_l, pred_m, pred_h = decompose_bands(x_pred)
    tgt_l, tgt_m, tgt_h = decompose_bands(x_target)

    loss_low = F.mse_loss(pred_l, tgt_l)
    loss_mid = F.mse_loss(pred_m, tgt_m)
    loss_high = F.mse_loss(pred_h, tgt_h)

    # Schedule: low always on; mid ramps in; high ramps in later
    w_low = 3.0
    w_mid = 1.0 * min(1.0, progress * 2)          # 0→1 over first half
    w_high = 0.5 * max(0.0, (progress - 0.3) / 0.7)  # 0→0.5, starts at 30%

    return w_low * loss_low + w_mid * loss_mid + w_high * loss_high


# ============================================================================
# NEW METRICS
# ============================================================================

def per_band_energy_distance(images_gen, images_real, device='cpu'):
    """Compute energy distance per band (not aggregated BEP)."""
    def batch_to(x):
        if isinstance(x, np.ndarray): x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 3: x = x.unsqueeze(1)
        return x.to(device)

    gen = batch_to(images_gen[:200])
    real = batch_to(images_real[:200])

    dct_gen = dct2d(gen)
    dct_real = dct2d(real)
    H, W = gen.shape[2], gen.shape[3]
    low_m, mid_m, high_m = get_freq_masks(H, W, device)

    results = {}
    for name, mask in [('low', low_m), ('mid', mid_m), ('high', high_m)]:
        e_gen = (dct_gen ** 2 * mask).mean(dim=(0, 1)).sum().item()
        e_real = (dct_real ** 2 * mask).mean(dim=(0, 1)).sum().item()
        results[f'energy_{name}_gen'] = e_gen
        results[f'energy_{name}_real'] = e_real
        results[f'energy_gap_{name}'] = abs(e_gen - e_real) / (e_real + 1e-12)
    return results


def hf_noise_index(images, device='cpu'):
    """HF noise index = gradient_energy / hf_energy.
    >1 means high-freq is dominated by sharp edges/noise.
    <1 means high-freq is smoother/more textured.
    """
    if isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float32)
    if images.dim() == 3:
        images = images.unsqueeze(1)
    images = images[:200].to(device)

    # Gradient energy (Sobel-like)
    gx = images[:, :, :, 1:] - images[:, :, :, :-1]
    gy = images[:, :, 1:, :] - images[:, :, :-1, :]
    grad_energy = (gx ** 2).mean().item() + (gy ** 2).mean().item()

    # High-freq energy from DCT
    _, _, x_high = decompose_bands(images)
    hf_energy = (x_high ** 2).mean().item() + 1e-12

    return grad_energy / hf_energy


def hf_coherence_metric(images, device='cpu'):
    """HF spatial coherence (autocorrelation of high-freq band)."""
    if isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float32)
    if images.dim() == 3:
        images = images.unsqueeze(1)
    images = images[:200].to(device)
    _, _, x_high = decompose_bands(images)
    return hf_local_autocorrelation(x_high)


def connectedness_proxy(images, threshold=0.5):
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if images.ndim == 4:
        images = images[:, 0]
    scores = []
    for img in images[:100]:
        binary = (img > threshold).astype(np.int32)
        total_fg = binary.sum()
        if total_fg < 5:
            scores.append(0.0); continue
        H, W = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        max_comp = 0
        for i in range(H):
            for j in range(W):
                if binary[i, j] == 1 and not visited[i, j]:
                    stack = [(i, j)]; visited[i, j] = True; sz = 0
                    while stack:
                        ci, cj = stack.pop(); sz += 1
                        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ni, nj = ci+di, cj+dj
                            if 0<=ni<H and 0<=nj<W and binary[ni,nj]==1 and not visited[ni,nj]:
                                visited[ni,nj] = True; stack.append((ni,nj))
                    max_comp = max(max_comp, sz)
        scores.append(max_comp / total_fg)
    return float(np.mean(scores))


# ============================================================================
# ADC/DAC
# ============================================================================

class GumbelSigmoid(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    def forward(self, logits):
        if self.training:
            u = torch.rand_like(logits).clamp(1e-8, 1-1e-8)
            noisy = (logits - torch.log(-torch.log(u))) / self.temperature
        else:
            noisy = logits / self.temperature
        soft = torch.sigmoid(noisy)
        hard = (soft > 0.5).float()
        return hard - soft.detach() + soft
    def set_temperature(self, tau): self.temperature = tau


class Encoder14(nn.Module):
    def __init__(self, in_ch=1, n_bits=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, n_bits, 3, padding=1),
        )
        self.quantizer = GumbelSigmoid()
    def forward(self, x):
        logits = self.conv(x)
        return self.quantizer(logits), logits
    def set_temperature(self, tau):
        self.quantizer.set_temperature(tau)


class Decoder14(nn.Module):
    def __init__(self, out_ch=1, n_bits=8):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, out_ch, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z):
        return self.deconv(z)


# ============================================================================
# E_CORE
# ============================================================================

class LocalEnergyCore(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        context_size = 9 * n_bits - 1
        self.predictor = nn.Sequential(
            nn.Linear(context_size, 64), nn.ReLU(),
            nn.Linear(64, 1))
    def get_context(self, z, bit_idx, i, j):
        B, K, H, W = z.shape
        contexts = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = (i+di)%H, (j+dj)%W
                for b in range(K):
                    if di==0 and dj==0 and b==bit_idx: continue
                    contexts.append(z[:, b, ni, nj])
        return torch.stack(contexts, dim=1)
    def violation_rate(self, z):
        B, K, H, W = z.shape
        violations = []
        for _ in range(min(50, H*W*K)):
            b = torch.randint(K, (1,)).item()
            i = torch.randint(H, (1,)).item()
            j = torch.randint(W, (1,)).item()
            ctx = self.get_context(z, b, i, j)
            logit = self.predictor(ctx).squeeze(1)
            pred = (logit > 0).float()
            actual = z[:, b, i, j]
            violations.append((pred != actual).float().mean().item())
        return np.mean(violations)


# ============================================================================
# DENOISER WITH CONFIGURABLE FREQUENCY TRAINING
# ============================================================================

class FreqDenoiser(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.net = nn.Sequential(
            nn.Conv2d(n_bits + 1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, n_bits, 3, padding=1),
        )
        self.skip = nn.Conv2d(n_bits, n_bits, 1)

    def forward(self, z_noisy, noise_level):
        B = z_noisy.shape[0]
        nl = noise_level.view(B, 1, 1, 1).expand(-1, 1, z_noisy.shape[2], z_noisy.shape[3])
        inp = torch.cat([z_noisy, nl], dim=1)
        return self.net(inp) + self.skip(z_noisy)

    @torch.no_grad()
    def sample_standard(self, n, H, W, device, n_steps=15, temperature=0.7):
        """Standard denoising sampling."""
        K = self.n_bits
        z = (torch.rand(n, K, H, W, device=device) > 0.5).float()
        for step in range(n_steps):
            nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
            logits = self(z, nl)
            probs = torch.sigmoid(logits / temperature)
            confidence = (step + 1) / n_steps
            mask = (torch.rand_like(z) < confidence).float()
            z_new = (torch.rand_like(z) < probs).float()
            z = mask * z_new + (1 - mask) * z
        logits = self(z, torch.zeros(n, device=device))
        return (torch.sigmoid(logits) > 0.5).float()

    @torch.no_grad()
    def sample_multiscale(self, n, H, W, device, n_steps=15, temperature=0.7):
        """Multi-scale: low temp early (global structure), higher temp later (details)."""
        K = self.n_bits
        z = (torch.rand(n, K, H, W, device=device) > 0.5).float()
        for step in range(n_steps):
            nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
            logits = self(z, nl)
            progress = step / n_steps
            if progress < 0.5:
                step_temp = temperature * 0.6
                confidence = 0.3 + 0.4 * (progress / 0.5)
            else:
                step_temp = temperature * 1.2
                confidence = 0.7 + 0.3 * ((progress - 0.5) / 0.5)
            probs = torch.sigmoid(logits / step_temp)
            mask = (torch.rand_like(z) < confidence).float()
            z_new = (torch.rand_like(z) < probs).float()
            z = mask * z_new + (1 - mask) * z
        logits = self(z, torch.zeros(n, device=device))
        return (torch.sigmoid(logits) > 0.5).float()

    @torch.no_grad()
    def sample_guided(self, n, H, W, decoder, device, n_steps=15, temperature=0.7):
        """Continuous spatial guidance: use freq residual map as update weight.
        Where high-freq residual is large → allow more updates (need more detail).
        Where low-freq is already good → conservative updates.
        """
        K = self.n_bits
        z = (torch.rand(n, K, H, W, device=device) > 0.5).float()

        for step in range(n_steps):
            nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
            logits = self(z, nl)
            progress = (step + 1) / n_steps

            # Phase-dependent temperature
            if progress < 0.5:
                step_temp = temperature * 0.6
            else:
                step_temp = temperature * 1.0

            probs = torch.sigmoid(logits / step_temp)
            z_proposed = (torch.rand_like(z) < probs).float()

            # Compute spatial guidance from frequency residual
            if step > 0 and step < n_steps - 1:
                x_cur = decoder(z)
                x_prop = decoder(z_proposed)

                # Get high-freq energy difference at each spatial position
                _, _, h_cur = decompose_bands(x_cur)
                _, _, h_prop = decompose_bands(x_prop)

                # Spatial residual: where proposed has MORE structured HF, prefer it
                # Use local variance as proxy for "structured" vs "flat"
                hf_energy_prop = F.avg_pool2d(
                    h_prop.abs().mean(dim=1, keepdim=True), 3, 1, 1)
                hf_energy_cur = F.avg_pool2d(
                    h_cur.abs().mean(dim=1, keepdim=True), 3, 1, 1)

                # Guidance: where proposed has more HF energy, increase update probability
                # But scale by progress (more HF updates in later steps)
                hf_advantage = (hf_energy_prop - hf_energy_cur).clamp(min=0)
                # Normalize to [0, 1]
                hf_max = hf_advantage.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
                guidance = (hf_advantage / hf_max)  # (B, 1, H_img, W_img)

                # Downsample guidance to z resolution
                guidance_z = F.interpolate(guidance, size=(H, W), mode='bilinear',
                                           align_corners=False)
                guidance_z = guidance_z.expand(-1, K, -1, -1)

                # Update probability: base confidence + HF guidance boost
                base_conf = progress
                boost = 0.3 * max(0.0, (progress - 0.3) / 0.7)  # HF boost in later steps
                update_prob = (base_conf + boost * guidance_z).clamp(0, 1)
            else:
                update_prob = torch.full_like(z, progress)

            mask = (torch.rand_like(z) < update_prob).float()
            z = mask * z_proposed + (1 - mask) * z

        logits = self(z, torch.zeros(n, device=device))
        return (torch.sigmoid(logits) > 0.5).float()


def train_denoiser(denoiser, z_data, decoder, device, cfg, epochs=30, batch_size=64):
    """Train denoiser with configurable frequency losses.

    cfg keys:
      use_freq_amp: bool - use band energy matching loss
      use_freq_coh: bool - use HF coherence loss
      use_schedule: bool - use frequency-scheduled loss weights
      lam_freq: float - weight for frequency losses
      lam_coh: float - weight for coherence loss
    """
    opt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)

    for epoch in range(epochs):
        denoiser.train()
        perm = torch.randperm(N)
        tl, fl, cl, nb = 0., 0., 0., 0
        progress = epoch / max(epochs - 1, 1)  # 0→1

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            z_clean = z_data[idx].to(device)
            B = z_clean.shape[0]

            noise_level = torch.rand(B, device=device)
            flip_mask = (torch.rand_like(z_clean) < noise_level.view(B, 1, 1, 1)).float()
            z_noisy = z_clean * (1 - flip_mask) + (1 - z_clean) * flip_mask

            opt.zero_grad()
            logits = denoiser(z_noisy, noise_level)
            loss_bce = F.binary_cross_entropy_with_logits(logits, z_clean)

            loss_freq = torch.tensor(0.0, device=device)
            loss_coh = torch.tensor(0.0, device=device)

            if cfg.get('use_freq_amp') or cfg.get('use_freq_coh'):
                # STE decode
                z_pred_soft = torch.sigmoid(logits)
                z_pred_hard = (z_pred_soft > 0.5).float()
                z_pred = z_pred_hard - z_pred_soft.detach() + z_pred_soft

                with torch.no_grad():
                    x_clean = decoder(z_clean)
                x_pred = decoder(z_pred)

                if cfg.get('use_freq_amp'):
                    if cfg.get('use_schedule'):
                        loss_freq = freq_scheduled_loss(x_pred, x_clean, progress)
                    else:
                        # Fixed weights (like A1v2 freq_train_03)
                        pred_l, pred_m, pred_h = decompose_bands(x_pred)
                        tgt_l, tgt_m, tgt_h = decompose_bands(x_clean)
                        loss_freq = (3.0 * F.mse_loss(pred_l, tgt_l) +
                                     1.0 * F.mse_loss(pred_m, tgt_m) +
                                     0.3 * F.mse_loss(pred_h, tgt_h))

                if cfg.get('use_freq_coh'):
                    _, _, pred_h = decompose_bands(x_pred)
                    _, _, tgt_h = decompose_bands(x_clean)
                    loss_coh = hf_coherence_loss(pred_h, tgt_h)

            lam_freq = cfg.get('lam_freq', 0.3)
            lam_coh = cfg.get('lam_coh', 0.1)
            loss = loss_bce + lam_freq * loss_freq + lam_coh * loss_coh
            loss.backward(); opt.step()

            tl += loss_bce.item()
            fl += loss_freq.item()
            cl += loss_coh.item()
            nb += 1

        if (epoch + 1) % 10 == 0:
            msg = f"    epoch {epoch+1}/{epochs}: BCE={tl/nb:.4f}"
            if cfg.get('use_freq_amp'): msg += f" freq={fl/nb:.4f}"
            if cfg.get('use_freq_coh'): msg += f" coh={cl/nb:.4f}"
            print(msg)


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def compute_diversity(z_samples, n_pairs=500):
    N = len(z_samples)
    z_flat = z_samples.reshape(N, -1)
    dists = []
    for _ in range(n_pairs):
        i, j = np.random.choice(N, 2, replace=False)
        dists.append((z_flat[i] != z_flat[j]).float().mean().item())
    return np.mean(dists), np.std(dists)


def compute_1nn_distance(gen_samples, train_samples, n_check=100):
    gen_flat = gen_samples[:n_check].reshape(n_check, -1)
    train_flat = train_samples.reshape(len(train_samples), -1)
    dists = []
    for i in range(n_check):
        d = ((train_flat - gen_flat[i:i+1])**2).sum(1)
        dists.append(d.min().item())
    return np.mean(dists), np.std(dists)


def token_histogram_kl(z_real, z_gen, n_bits=8):
    N_r, K, H, W = z_real.shape
    N_g = z_gen.shape[0]
    n_tokens = 2 ** K
    kls = []
    for i in range(H):
        for j in range(W):
            idx_r = torch.zeros(N_r, dtype=torch.long)
            idx_g = torch.zeros(N_g, dtype=torch.long)
            for b in range(K):
                idx_r += (z_real[:, b, i, j].long() << b)
                idx_g += (z_gen[:, b, i, j].long() << b)
            p = torch.bincount(idx_r, minlength=n_tokens).float() + 1
            q = torch.bincount(idx_g, minlength=n_tokens).float() + 1
            p /= p.sum(); q /= q.sum()
            kls.append((p * (p/q).log()).sum().item())
    return np.mean(kls)


def save_grid(images, path, nrow=8):
    try:
        from torchvision.utils import save_image
        if isinstance(images, np.ndarray):
            images = torch.tensor(images)
        if images.dim() == 3:
            images = images.unsqueeze(1)
        save_image(images[:64], path, nrow=nrow, normalize=False)
        print(f"    Grid: {path}")
    except Exception as e:
        print(f"    Grid save failed: {e}")


# ============================================================================
# EXPERIMENT CONFIGS
# ============================================================================

CONFIGS = OrderedDict([
    ("baseline", {
        "use_freq_amp": False, "use_freq_coh": False, "use_schedule": False,
        "lam_freq": 0.0, "lam_coh": 0.0,
        "sampler": "standard",
    }),
    ("freq_amp", {
        "use_freq_amp": True, "use_freq_coh": False, "use_schedule": False,
        "lam_freq": 0.3, "lam_coh": 0.0,
        "sampler": "standard",
    }),
    ("freq_amp_coh", {
        "use_freq_amp": True, "use_freq_coh": True, "use_schedule": False,
        "lam_freq": 0.3, "lam_coh": 0.1,
        "sampler": "standard",
    }),
    ("freq_sched", {
        "use_freq_amp": True, "use_freq_coh": False, "use_schedule": True,
        "lam_freq": 0.3, "lam_coh": 0.0,
        "sampler": "standard",
    }),
    ("freq_sched_coh", {
        "use_freq_amp": True, "use_freq_coh": True, "use_schedule": True,
        "lam_freq": 0.3, "lam_coh": 0.1,
        "sampler": "standard",
    }),
    ("freq_sched_coh_ms", {
        "use_freq_amp": True, "use_freq_coh": True, "use_schedule": True,
        "lam_freq": 0.3, "lam_coh": 0.1,
        "sampler": "multiscale",
    }),
    ("freq_full", {
        "use_freq_amp": True, "use_freq_coh": True, "use_schedule": True,
        "lam_freq": 0.3, "lam_coh": 0.1,
        "sampler": "guided",
    }),
    ("freq_full_ms", {
        "use_freq_amp": True, "use_freq_coh": True, "use_schedule": True,
        "lam_freq": 0.3, "lam_coh": 0.1,
        "sampler": "multiscale",  # ms version with full training
    }),
])


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', default='fmnist', choices=['mnist', 'fmnist', 'kmnist'])
    parser.add_argument('--output_dir', default='outputs/exp_gen_freq_a2')
    parser.add_argument('--n_bits', type=int, default=8)
    parser.add_argument('--n_samples', type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print(f"ROUTE A2: STRUCTURED HF GENERATION — {args.dataset.upper()} — 8-run matrix")
    print("=" * 100)

    # [1] Load
    print("\n[1] Loading dataset...")
    from torchvision import datasets, transforms
    ds_class = {'mnist': datasets.MNIST, 'fmnist': datasets.FashionMNIST,
                'kmnist': datasets.KMNIST}[args.dataset]
    train_ds = ds_class('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = ds_class('./data', train=False, download=True, transform=transforms.ToTensor())

    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(len(train_ds), 5000, replace=False)
    test_idx = rng.choice(len(test_ds), 1000, replace=False)
    train_x = torch.stack([train_ds[i][0] for i in train_idx])
    test_x = torch.stack([test_ds[i][0] for i in test_idx])

    # [2] Shared ADC/DAC
    print(f"\n[2] Training ADC/DAC (28→14×14×{args.n_bits})...")
    encoder = Encoder14(in_ch=1, n_bits=args.n_bits).to(device)
    decoder = Decoder14(out_ch=1, n_bits=args.n_bits).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    for epoch in range(25):
        encoder.train(); decoder.train()
        tau = 1.0 + (0.3 - 1.0) * epoch / 24
        encoder.set_temperature(tau)
        perm = torch.randperm(len(train_x))
        tl, nb = 0., 0
        for i in range(0, len(train_x), 64):
            idx = perm[i:i+64]
            x = train_x[idx].to(device)
            opt.zero_grad()
            z, _ = encoder(x)
            x_hat = decoder(z)
            loss = F.binary_cross_entropy(x_hat, x)
            loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        if (epoch+1) % 5 == 0:
            print(f"    epoch {epoch+1}/25: BCE={tl/nb:.4f}")
    encoder.eval(); decoder.eval()

    # [3] Encode
    print("\n[3] Encoding training set...")
    z_data = []
    with torch.no_grad():
        for i in range(0, len(train_x), 64):
            x = train_x[i:i+64].to(device)
            z, _ = encoder(x)
            z_data.append(z.cpu())
    z_data = torch.cat(z_data)
    K, H, W = z_data.shape[1:]
    print(f"    z_data: {z_data.shape}, bit usage: {z_data.mean():.3f}")

    # [4] E_core
    print("\n[4] Training E_core...")
    e_core = LocalEnergyCore(args.n_bits).to(device)
    e_opt = torch.optim.Adam(e_core.parameters(), lr=1e-3)
    for epoch in range(10):
        e_core.train()
        perm = torch.randperm(len(z_data))
        for i in range(0, len(z_data), 64):
            idx = perm[i:i+64]
            z = z_data[idx].to(device)
            e_opt.zero_grad()
            total_loss = 0.
            for _ in range(20):
                b = torch.randint(K,(1,)).item()
                ii = torch.randint(H,(1,)).item()
                jj = torch.randint(W,(1,)).item()
                ctx = e_core.get_context(z, b, ii, jj)
                logit = e_core.predictor(ctx).squeeze(1)
                target = z[:, b, ii, jj]
                total_loss += F.binary_cross_entropy_with_logits(logit, target)
            (total_loss / 20).backward(); e_opt.step()
    e_core.eval()

    # [5] Reference metrics
    print("\n[5] Reference structural metrics on REAL data...")
    real_conn = connectedness_proxy(test_x[:100].numpy())
    real_hf_coh = hf_coherence_metric(test_x[:200], device)
    real_hf_noise = hf_noise_index(test_x[:200], device)
    real_band = per_band_energy_distance(test_x[:200], test_x[:200], device)
    print(f"    Connectedness: {real_conn:.4f}")
    print(f"    HF coherence:  {real_hf_coh:.4f}")
    print(f"    HF noise index: {real_hf_noise:.4f}")

    # [6] Run all configs
    print("\n" + "=" * 100)
    print("RUNNING 8-CONFIG EXPERIMENT MATRIX")
    print("=" * 100)

    all_results = []
    train_x_np = train_x.numpy().reshape(len(train_x), -1)

    for config_name, cfg in CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"CONFIG: {config_name}")
        print(f"  freq_amp={cfg['use_freq_amp']} coh={cfg['use_freq_coh']} "
              f"sched={cfg['use_schedule']} sampler={cfg['sampler']}")
        print("=" * 80)

        torch.manual_seed(args.seed + hash(config_name) % 10000)

        # Train denoiser
        denoiser = FreqDenoiser(args.n_bits).to(device)
        print(f"  Training denoiser...")
        train_denoiser(denoiser, z_data, decoder, device, cfg, epochs=30)
        denoiser.eval()

        # Sample
        print(f"  Sampling {args.n_samples}...")
        torch.manual_seed(args.seed)
        if cfg['sampler'] == 'standard':
            z_gen = denoiser.sample_standard(args.n_samples, H, W, device)
        elif cfg['sampler'] == 'multiscale':
            z_gen = denoiser.sample_multiscale(args.n_samples, H, W, device)
        elif cfg['sampler'] == 'guided':
            z_gen = denoiser.sample_guided(args.n_samples, H, W, decoder, device)

        with torch.no_grad():
            x_gen = decoder(z_gen.to(device)).cpu()

        save_grid(x_gen, os.path.join(out_dir, f'gen_{config_name}.png'))

        # === FULL EVALUATION ===
        z_gen_cpu = z_gen.cpu()

        # Protocol metrics
        viol = e_core.violation_rate(z_gen_cpu[:100].to(device))
        tok_kl = token_histogram_kl(z_data[:500], z_gen_cpu[:500], args.n_bits)
        div_mean, div_std = compute_diversity(z_gen_cpu)

        # Cycle
        with torch.no_grad():
            z_check = z_gen_cpu[:100].to(device)
            x_check = decoder(z_check)
            z_cyc, _ = encoder(x_check)
            cycle_ham = (z_check != z_cyc).float().mean().item()

        # 1-NN
        x_gen_np = x_gen.numpy().reshape(len(x_gen), -1)
        nn_mean, _ = compute_1nn_distance(x_gen_np, train_x_np)

        # Structural
        conn = connectedness_proxy(x_gen[:100])

        # NEW: Three-metric freq evaluation
        band_dist = per_band_energy_distance(x_gen[:200], test_x[:200], device)
        gen_hf_coh = hf_coherence_metric(x_gen[:200], device)
        gen_hf_noise = hf_noise_index(x_gen[:200], device)

        r = {
            'config': config_name,
            'violation': viol,
            'token_kl': tok_kl,
            'diversity': div_mean,
            'cycle_hamming': cycle_ham,
            'nn_dist': nn_mean,
            'connectedness': conn,
            'hf_coherence': gen_hf_coh,
            'hf_noise_index': gen_hf_noise,
            'energy_gap_low': band_dist['energy_gap_low'],
            'energy_gap_mid': band_dist['energy_gap_mid'],
            'energy_gap_high': band_dist['energy_gap_high'],
        }
        all_results.append(r)

        print(f"    viol={viol:.4f} div={div_mean:.4f} cycle={cycle_ham:.4f} conn={conn:.4f}")
        print(f"    HF_coh={gen_hf_coh:.4f}(real={real_hf_coh:.4f}) "
              f"HF_noise={gen_hf_noise:.2f}(real={real_hf_noise:.2f})")
        print(f"    E_gap: low={band_dist['energy_gap_low']:.4f} "
              f"mid={band_dist['energy_gap_mid']:.4f} "
              f"high={band_dist['energy_gap_high']:.4f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 100)
    print(f"A2 STRUCTURED HF SUMMARY ({args.dataset.upper()})")
    print(f"Real: conn={real_conn:.4f}  HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noise:.2f}")
    print("=" * 100)

    header = (f"{'config':<20} {'viol':>7} {'div':>7} {'cycle':>7} "
              f"{'conn':>7} {'HF_coh':>7} {'HF_noi':>7} "
              f"{'Egap_L':>7} {'Egap_M':>7} {'Egap_H':>7}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['config']:<20} {r['violation']:>7.4f} {r['diversity']:>7.4f} "
              f"{r['cycle_hamming']:>7.4f} {r['connectedness']:>7.4f} "
              f"{r['hf_coherence']:>7.4f} {r['hf_noise_index']:>7.2f} "
              f"{r['energy_gap_low']:>7.4f} {r['energy_gap_mid']:>7.4f} "
              f"{r['energy_gap_high']:>7.4f}")

    # Gate check
    print("\n" + "-" * 60)
    print("GATE CHECK + FREQ QUALITY (vs baseline):")
    bl = all_results[0]
    for r in all_results[1:]:
        viol_d = (r['violation'] - bl['violation']) / (bl['violation'] + 1e-8) * 100
        div_d = r['diversity'] - bl['diversity']
        conn_d = r['connectedness'] - bl['connectedness']
        coh_d = r['hf_coherence'] - bl['hf_coherence']
        noi_d = r['hf_noise_index'] - bl['hf_noise_index']
        g1 = "PASS" if viol_d < 20 else "FAIL"
        g2 = "PASS" if r['cycle_hamming'] - bl['cycle_hamming'] <= 0.01 else "FAIL"
        # Freq quality: good if HF_coh closer to real, HF_noise closer to real
        coh_improve = abs(r['hf_coherence'] - real_hf_coh) < abs(bl['hf_coherence'] - real_hf_coh)
        noi_improve = abs(r['hf_noise_index'] - real_hf_noise) < abs(bl['hf_noise_index'] - real_hf_noise)
        freq_verdict = "BETTER" if (coh_improve and noi_improve) else (
            "MIXED" if (coh_improve or noi_improve) else "WORSE")
        print(f"  {r['config']:<20} viol[{g1}] cycle[{g2}] "
              f"div={div_d:+.4f} conn={conn_d:+.4f} "
              f"HFcoh={coh_d:+.4f} HFnoi={noi_d:+.2f} → freq:{freq_verdict}")

    # CSV
    csv_path = os.path.join(out_dir, "freq_a2_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(all_results[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV: {csv_path}")

    save_grid(test_x[:64], os.path.join(out_dir, 'real_samples.png'))

    print("\n" + "=" * 100)
    print("Route A2 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
