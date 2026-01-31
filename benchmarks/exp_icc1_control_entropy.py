#!/usr/bin/env python3
"""
ICC-1: Information Circulation Constraint — Control Port + Conditional Entropy
==============================================================================
First-principles motivation (Information Circulation):

  Route C protocol must satisfy four constraints simultaneously:
    R (Rate):           fixed bit budget per position (hardware constraint)
    O (Observability):  z carries enough info about x for repair/recon
    C (Controllability): explicit control port c can steer global mode
    D (Decodability):   z's info can be read out by probe/DAC

  The "information wheel" x → z → x̂ must circulate without collapse.
  Current failure modes:
    - Same-color generation → C deficit (no control over global mode)
    - Dead bits → O deficit (decoder ignores z-bits)
    - Probe ceiling ~51% → D deficit (info exists but unreadable)

  ICC-1 tests whether adding:
    (1) Conditional entropy regularizer (anti-collapse, fights O+D deficit)
    (2) Explicit control port c ∈ {0,1}^8 (fights C deficit)
    (3) ANOVA-style ctrl loss on global statistics (makes c actually control modes)
  can improve all three modes (repair/generation/classification) simultaneously.

Design principles:
  - ControlPort = concat on z-bus (netlist-friendly: just 8 extra wires)
  - L_ent = conditional entropy using E_core's pseudo-likelihood (not marginal)
  - L_ctrl = -FisherRatio(s(x̂) | c) using 8-dim statistics vector
  - NO discriminator, NO MI estimator, NO extra networks for losses
  - Switch matrix: ent_weight and ctrl_weight default to 0, calibrated by O0/S0

Configs (4, clean switch matrix):
  ICC1-A: Base (16bit, no extras) — control
  ICC1-B: +Entropy (conditional, ent_weight=0.1) — O+D constraint
  ICC1-C: +ControlPort (c_bits=8, ctrl_weight=0) — ablation: port without loss
  ICC1-D: +ControlPort + CtrlLoss (ctrl_weight=0.1) — full ICC

New metrics (ICC-specific):
  - effective_entropy: per-position conditional entropy of z
  - control_rank: Fisher ratio of s(x̂) across different c values
  - dead_ratio: fraction of bits with negligible decoder influence

Pre-registered gates:
  Hard: ham_unmasked == 0.000, cycle ≤ baseline + 0.02
  Success (any 2 of 3):
    - HueVar ≥ 2× baseline  OR  acc_clean +2%
    - control_rank(D) > control_rank(A) (controllability improved)
    - div ≥ baseline - 0.03

4GB GPU: 3000 train, 500 test, batch_size=32
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

from exp_flow_f0 import (
    dct2d, idct2d, get_freq_masks, decompose_bands, freq_scheduled_loss,
    per_band_energy_distance, hf_coherence_metric, hf_noise_index,
    connectedness_proxy, compute_diversity, save_grid,
    GumbelSigmoid, DiffEnergyCore,
    quantize, compute_e_core_grad,
    evaluate
)

from exp_flow_f0c_fixes import FlatStepFn_Norm, get_sigma

from exp_g2_protocol_density import (
    Encoder16, Decoder16,
    encode_all, train_ecore, train_step_fn
)

from exp_e2a_global_prior import compute_hue_var, compute_marginal_kl


# ============================================================================
# CONTROL PORT — Decoder with c-bus (concat, netlist-friendly)
# ============================================================================

class Decoder16_ControlPort(nn.Module):
    """Decoder that accepts z (n_bits channels) + c (c_bits channels, broadcast).

    c is concatenated to z before decoding. This is just extra input wires
    on the z-bus — no AdaIN/FiLM, no multiplicative interaction.

    For synthesis: c_bits additional lines → same ConvTranspose2d but wider input.
    """
    def __init__(self, n_bits=16, c_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.c_bits = c_bits
        total_in = n_bits + c_bits
        self.stem = nn.Sequential(
            nn.ConvTranspose2d(total_in, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = self._res(64, 64)
        self.res2 = self._res(64, 64)
        self.head = nn.Sequential(nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid())

    def _res(self, ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(),
            nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc))

    def forward(self, z, c=None):
        """z: (B, n_bits, H, W), c: (B, c_bits, H, W) or None.
        If c is None, uses zeros (backward-compatible with base decoder)."""
        if c is not None:
            zc = torch.cat([z, c], dim=1)
        else:
            B, _, H, W = z.shape
            zc = torch.cat([z, torch.zeros(B, self.c_bits, H, W, device=z.device)], dim=1)
        h = self.stem(zc)
        h = F.relu(h + self.res1(h))
        h = F.relu(h + self.res2(h))
        return self.head(h)


class ControlEncoder(nn.Module):
    """Small encoder that extracts global control code c from image.

    c ∈ {0,1}^c_bits — Gumbel-Sigmoid quantized.
    Global average pool → linear → c.
    This is the "what global mode does this image belong to" extractor.
    """
    def __init__(self, c_bits=8):
        super().__init__()
        self.c_bits = c_bits
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, c_bits))
        self.q = GumbelSigmoid()

    def forward(self, x):
        """Returns c_hard (B, c_bits), c_logits (B, c_bits)."""
        logits = self.net(x)  # (B, c_bits)
        c_hard = self.q(logits)  # (B, c_bits) via Gumbel-Sigmoid STE
        return c_hard, logits

    def set_temperature(self, tau):
        self.q.set_temperature(tau)

    def broadcast(self, c, H, W):
        """Broadcast c (B, c_bits) → (B, c_bits, H, W) for concat with z."""
        return c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)


# ============================================================================
# CONDITIONAL ENTROPY LOSS — using E_core's pseudo-likelihood
# ============================================================================

def compute_conditional_entropy(logits, e_core_predictor=None):
    """Conditional entropy of z given neighbors, estimated from logits.

    H(z_i | neighbors_i) = -p_i log p_i - (1-p_i) log(1-p_i)

    where p_i = sigmoid(logit_i). This is the binary entropy of each position.
    Under the MRF pseudo-likelihood interpretation, these logits already
    condition on the neighborhood (encoder sees spatial context).

    Maximizing this entropy prevents dead bits and mode collapse,
    while the reconstruction loss prevents it from becoming pure noise.

    Returns:
        ent_loss: negative conditional entropy (minimize this to maximize entropy)
        mean_entropy: diagnostic — average per-bit entropy in nats
    """
    p = torch.sigmoid(logits)  # (B, K, H, W)
    # Clamp for numerical stability
    p = p.clamp(1e-6, 1 - 1e-6)

    # Binary entropy per position
    h = -(p * p.log() + (1 - p) * (1 - p).log())  # (B, K, H, W)
    mean_h = h.mean()

    # Loss: negative entropy (we want to MAXIMIZE entropy → MINIMIZE -H)
    ent_loss = -mean_h

    return ent_loss, mean_h.item()


# ============================================================================
# CONTROLLABILITY LOSS — ANOVA / Fisher Ratio on global statistics
# ============================================================================

def compute_global_stats(x):
    """Compute low-dimensional global statistics s(x̂) for controllability measurement.

    s(x̂) = [mean_R, mean_G, mean_B, low_freq_energy, hue_hist_4bins]
    Total: 8 dimensions. Stable, cheap, no learned components.

    Args:
        x: (B, 3, H, W) — decoded images, in [0,1]

    Returns:
        stats: (B, 8) — per-image statistics vector
    """
    B = x.shape[0]
    device = x.device

    # Mean RGB (3 dims)
    mean_rgb = x.mean(dim=(2, 3))  # (B, 3)

    # Low-frequency energy (1 dim) — average of 2×2 pooled image variance
    x_low = F.avg_pool2d(x, 4)  # (B, 3, H/4, W/4)
    low_freq_e = x_low.var(dim=(1, 2, 3))  # (B,)

    # Hue histogram (4 bins) — simplified from RGB
    # Use approximate hue: atan2(sqrt(3)*(G-B), 2R-G-B) / (2π) → [0,1]
    R, G, Bl = x[:, 0], x[:, 1], x[:, 2]
    hue_angle = torch.atan2(1.732 * (G - Bl), 2 * R - G - Bl + 1e-8)
    hue_norm = (hue_angle / (2 * 3.14159) + 0.5).clamp(0, 1)  # → [0,1]

    # Soft histogram: 4 bins
    hue_flat = hue_norm.reshape(B, -1)  # (B, H*W)
    centers = torch.tensor([0.125, 0.375, 0.625, 0.875], device=device)
    # Gaussian kernel for soft binning
    diffs = hue_flat.unsqueeze(2) - centers.unsqueeze(0).unsqueeze(0)  # (B, N, 4)
    weights = torch.exp(-diffs.pow(2) / (2 * 0.05**2))  # σ=0.05
    hue_hist = weights.mean(dim=1)  # (B, 4)
    hue_hist = hue_hist / (hue_hist.sum(dim=1, keepdim=True) + 1e-8)

    # Concatenate: (B, 3+1+4=8)
    stats = torch.cat([mean_rgb, low_freq_e.unsqueeze(1), hue_hist], dim=1)
    return stats


def compute_ctrl_loss_anova(stats_by_c, min_samples=4):
    """ANOVA-style controllability loss: maximize between-class / within-class variance.

    Given stats grouped by control code c, compute Fisher ratio.
    Loss = -log(Fisher + eps) — drives c to separate global modes.

    Args:
        stats_by_c: dict {c_idx: (N_c, 8) tensor} — stats grouped by c value

    Returns:
        ctrl_loss: scalar
        fisher_ratio: diagnostic
    """
    # Compute overall mean
    all_stats = torch.cat(list(stats_by_c.values()), dim=0)  # (N_total, 8)
    grand_mean = all_stats.mean(dim=0)  # (8,)

    # Between-class variance
    var_between = 0.0
    n_total = 0
    for c_idx, stats in stats_by_c.items():
        if len(stats) < min_samples:
            continue
        class_mean = stats.mean(dim=0)
        var_between = var_between + len(stats) * (class_mean - grand_mean).pow(2).sum()
        n_total += len(stats)

    if n_total == 0:
        return torch.tensor(0.0, device=all_stats.device), 0.0

    var_between = var_between / n_total

    # Within-class variance
    var_within = 0.0
    for c_idx, stats in stats_by_c.items():
        if len(stats) < min_samples:
            continue
        class_mean = stats.mean(dim=0, keepdim=True)
        var_within = var_within + (stats - class_mean).pow(2).sum()

    var_within = var_within / max(n_total - len(stats_by_c), 1)

    fisher = var_between / (var_within + 1e-8)
    ctrl_loss = -torch.log(fisher + 1e-6)

    return ctrl_loss, fisher.item()


def compute_grad_norm(loss, params):
    """Compute gradient norm for auto-normalization."""
    grads = torch.autograd.grad(loss, params, create_graph=False,
                                retain_graph=True, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.data.norm().item() ** 2
    return total ** 0.5


# ============================================================================
# ICC-AWARE ADC TRAINING
# ============================================================================

def train_adc_icc(encoder, decoder, train_x, device,
                  control_encoder=None,  # None = no control port
                  ent_weight=0.0,        # 0 = entropy off (calibrate from O0/S0)
                  ctrl_weight=0.0,       # 0 = ctrl loss off
                  c_bits=8, epochs=40, bs=64):
    """Train ADC/DAC with ICC constraints (switch matrix design).

    Losses:
      L_recon: MSE + 0.5*BCE (always on)
      L_ent:   -H(z_i|context)  (on if ent_weight > 0)
      L_ctrl:  -Fisher(s(x̂)|c)  (on if ctrl_weight > 0 and control_encoder is not None)

    λ auto-normalized on first batch for each extra loss.
    """
    params = list(encoder.parameters()) + list(decoder.parameters())
    if control_encoder is not None:
        params += list(control_encoder.parameters())

    opt = torch.optim.Adam(params, lr=1e-3)
    lambda_ent = None
    lambda_ctrl = None

    for epoch in tqdm(range(epochs), desc="ADC"):
        encoder.train(); decoder.train()
        if control_encoder is not None:
            control_encoder.train()
        if hasattr(encoder, 'set_temperature'):
            encoder.set_temperature(1.0 + (0.3 - 1.0) * epoch / max(epochs - 1, 1))
        if control_encoder is not None and hasattr(control_encoder, 'set_temperature'):
            control_encoder.set_temperature(1.0 + (0.3 - 1.0) * epoch / max(epochs - 1, 1))

        perm = torch.randperm(len(train_x))
        tl_recon, tl_ent, tl_ctrl, nb = 0., 0., 0., 0

        for i in range(0, len(train_x), bs):
            x = train_x[perm[i:i+bs]].to(device)
            opt.zero_grad()

            z, logits = encoder(x)
            B, K, H, W = z.shape

            # Decode (with or without control port)
            if control_encoder is not None:
                c_hard, c_logits = control_encoder(x)
                c_spatial = control_encoder.broadcast(c_hard, H, W)
                xh = decoder(z, c_spatial)
            else:
                xh = decoder(z)

            # L_recon (always on)
            loss_recon = F.mse_loss(xh, x) + 0.5 * F.binary_cross_entropy(xh, x)
            loss = loss_recon

            # L_ent (conditional entropy, switch-controlled)
            if ent_weight > 0:
                ent_loss, mean_ent = compute_conditional_entropy(logits)

                if lambda_ent is None:
                    g_recon = compute_grad_norm(loss_recon, params)
                    g_ent = compute_grad_norm(ent_loss, params)
                    lambda_ent = g_recon / max(g_ent, 1e-8)
                    lambda_ent = min(lambda_ent, 5.0)
                    print(f"    Auto-norm λ_ent = {lambda_ent:.4f} "
                          f"(g_recon={g_recon:.4f}, g_ent={g_ent:.4f})")

                loss = loss + ent_weight * lambda_ent * ent_loss
                tl_ent += ent_loss.item()

            # L_ctrl (ANOVA controllability, switch-controlled)
            if ctrl_weight > 0 and control_encoder is not None:
                # Group decoded images by c value
                # Discretize c to integer index for grouping
                with torch.no_grad():
                    c_idx = c_hard.sum(dim=1).long()  # simple hash: sum of bits

                stats = compute_global_stats(xh)

                # Group stats by c
                stats_by_c = {}
                for ci in range(B):
                    key = c_idx[ci].item()
                    if key not in stats_by_c:
                        stats_by_c[key] = []
                    stats_by_c[key].append(stats[ci:ci+1])

                # Need at least 2 groups with enough samples
                valid_groups = {k: torch.cat(v) for k, v in stats_by_c.items()
                                if len(v) >= 2}

                if len(valid_groups) >= 2:
                    ctrl_loss, fisher = compute_ctrl_loss_anova(valid_groups)

                    if lambda_ctrl is None:
                        g_recon = compute_grad_norm(loss_recon, params)
                        g_ctrl = compute_grad_norm(ctrl_loss, params)
                        lambda_ctrl = g_recon / max(g_ctrl, 1e-8)
                        lambda_ctrl = min(lambda_ctrl, 5.0)
                        print(f"    Auto-norm λ_ctrl = {lambda_ctrl:.4f} "
                              f"(g_recon={g_recon:.4f}, g_ctrl={g_ctrl:.4f})")

                    loss = loss + ctrl_weight * lambda_ctrl * ctrl_loss
                    tl_ctrl += ctrl_loss.item()

            loss.backward()
            opt.step()
            tl_recon += loss_recon.item()
            nb += 1

    encoder.eval(); decoder.eval()
    if control_encoder is not None:
        control_encoder.eval()

    return {
        'recon_loss': tl_recon / max(nb, 1),
        'ent_loss': tl_ent / max(nb, 1),
        'ctrl_loss': tl_ctrl / max(nb, 1),
        'lambda_ent': lambda_ent,
        'lambda_ctrl': lambda_ctrl,
    }


# ============================================================================
# DEAD-BIT DIAGNOSTIC (from O0, simplified)
# ============================================================================

@torch.no_grad()
def measure_dead_bits(encoder, decoder, test_x, device,
                      n_probes=200, threshold=1e-5, n_images=100,
                      control_encoder=None):
    """Measure fraction of z-bits with negligible decoder influence."""
    x_sub = test_x[:n_images]
    z_list, c_list = [], []
    for i in range(0, len(x_sub), 32):
        batch = x_sub[i:i+32].to(device)
        z, _ = encoder(batch)
        z_list.append(z.cpu())
        if control_encoder is not None:
            c, _ = control_encoder(batch)
            c_list.append(c.cpu())
    z_all = torch.cat(z_list)
    c_all = torch.cat(c_list) if c_list else None

    # Decode originals
    x_orig_list = []
    for i in range(0, len(z_all), 32):
        z_batch = z_all[i:i+32].to(device)
        if c_all is not None:
            c_batch = c_all[i:i+32].to(device)
            B, K, H, W = z_batch.shape
            c_sp = c_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            x_orig_list.append(decoder(z_batch, c_sp).cpu())
        else:
            x_orig_list.append(decoder(z_batch).cpu())
    x_orig = torch.cat(x_orig_list)

    B, K, H, W = z_all.shape
    influences = []

    for _ in range(n_probes):
        k = torch.randint(K, (1,)).item()
        h = torch.randint(H, (1,)).item()
        w = torch.randint(W, (1,)).item()

        z_flip = z_all.clone()
        z_flip[:, k, h, w] = 1.0 - z_flip[:, k, h, w]

        x_flip_list = []
        for i in range(0, len(z_flip), 32):
            z_batch = z_flip[i:i+32].to(device)
            if c_all is not None:
                c_batch = c_all[i:i+32].to(device)
                c_sp = c_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
                x_flip_list.append(decoder(z_batch, c_sp).cpu())
            else:
                x_flip_list.append(decoder(z_batch).cpu())
        x_flip = torch.cat(x_flip_list)

        influence = (x_orig - x_flip).pow(2).mean().item()
        influences.append(influence)

    influences = np.array(influences)
    return {
        'dead_ratio': float((influences < threshold).mean()),
        'mean_influence': float(influences.mean()),
        'p10_influence': float(np.percentile(influences, 10)),
    }


# ============================================================================
# EFFECTIVE ENTROPY MEASUREMENT
# ============================================================================

@torch.no_grad()
def measure_effective_entropy(encoder, data, device, bs=32):
    """Measure per-position entropy of z across the dataset."""
    logits_list = []
    for i in range(0, len(data), bs):
        _, logits = encoder(data[i:i+bs].to(device))
        logits_list.append(logits.cpu())
    all_logits = torch.cat(logits_list)  # (N, K, H, W)

    p = torch.sigmoid(all_logits)
    p = p.clamp(1e-6, 1 - 1e-6)
    h = -(p * p.log() + (1 - p) * (1 - p).log())  # binary entropy

    return {
        'mean_entropy': h.mean().item(),
        'min_entropy': h.min().item(),
        'entropy_below_01': float((h < 0.1).float().mean().item()),  # fraction near-deterministic
    }


# ============================================================================
# CONTROL RANK MEASUREMENT (post-hoc, for any config)
# ============================================================================

@torch.no_grad()
def measure_control_rank(decoder, e_core, step_fn, K, H, W, device,
                         control_encoder=None, c_bits=8,
                         n_per_c=32, n_c_values=16, T=10):
    """Measure controllability: how much does c change the generated output statistics?

    For configs without control port: measure variance across random seeds only.
    For configs with control port: measure variance across different c values.

    Returns Fisher-like ratio (between-c variance / within-c variance).
    """
    stats_by_c = {}

    for ci in range(n_c_values):
        torch.manual_seed(ci * 9973)

        if control_encoder is not None:
            # Sample specific c value
            c = torch.zeros(n_per_c, c_bits, device=device)
            # Use ci to set specific bits
            for bit in range(c_bits):
                if (ci >> bit) & 1:
                    c[:, bit] = 1.0
        else:
            c = None

        # Generate via flow
        u = torch.randn(n_per_c, K, H, W, device=device) * 0.5
        for step in range(T):
            t_frac = 1.0 - step / T
            t_tensor = torch.full((n_per_c,), t_frac, device=device)
            e_grad = compute_e_core_grad(e_core, u)
            delta_u = step_fn(u, e_grad, t_tensor)
            u = u + 0.5 * delta_u
            sigma = get_sigma('cosine', step, T)
            if sigma > 0.01:
                u = u + sigma * torch.randn_like(u)

        z_gen = quantize(u)

        # Decode
        if c is not None:
            c_sp = c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            x_gen = decoder(z_gen, c_sp)
        else:
            x_gen = decoder(z_gen)

        stats = compute_global_stats(x_gen)
        stats_by_c[ci] = stats.cpu()

    # Compute Fisher ratio
    all_stats = torch.cat(list(stats_by_c.values()))
    grand_mean = all_stats.mean(dim=0)

    var_between = 0.0
    var_within = 0.0
    n_total = 0

    for ci, stats in stats_by_c.items():
        class_mean = stats.mean(dim=0)
        var_between += len(stats) * (class_mean - grand_mean).pow(2).sum().item()
        var_within += (stats - class_mean.unsqueeze(0)).pow(2).sum().item()
        n_total += len(stats)

    var_between /= max(n_total, 1)
    var_within /= max(n_total - len(stats_by_c), 1)

    fisher = var_between / max(var_within, 1e-8)

    return {
        'fisher_ratio': fisher,
        'var_between': var_between,
        'var_within': var_within,
        'n_c_values': n_c_values,
    }


# ============================================================================
# REPAIR / GEN / CLASSIFY (standard three-mode eval)
# ============================================================================

def make_center_mask(B, K, H, W, device):
    mask = torch.ones(B, 1, H, W, device=device)
    h4, w4 = H // 4, W // 4
    mask[:, :, h4:3*h4, w4:3*w4] = 0.0
    return mask


@torch.no_grad()
def repair_flow(step_fn, e_core, z_clean, mask, device, T=10, dt=0.5):
    B, K, H, W = z_clean.shape
    if mask.shape[1] == 1:
        mask = mask.expand(-1, K, -1, -1)
    u_clean = torch.where(z_clean > 0.5,
                          torch.tensor(2.0, device=device),
                          torch.tensor(-2.0, device=device))
    u_noise = torch.randn(B, K, H, W, device=device) * 0.3
    u = mask * u_clean + (1 - mask) * u_noise

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((B,), t_frac, device=device)
        e_grad = compute_e_core_grad(e_core, u)
        delta_u = step_fn(u, e_grad, t_tensor)
        u = u + dt * delta_u
        sigma = get_sigma('cosine', step, T)
        if sigma > 0.01:
            u = u + sigma * torch.randn_like(u) * (1 - mask)
        u = mask * u_clean + (1 - mask) * u

    return mask * z_clean + (1 - mask) * quantize(u)


@torch.no_grad()
def sample_flow_gen(step_fn, e_core, n, K, H, W, device,
                    T=10, dt=0.5, decoder=None, control_encoder=None, c=None):
    """Generate via flow. Optionally uses control port."""
    u = torch.randn(n, K, H, W, device=device) * 0.5
    trajectory = {'e_core': [], 'delta_u_norm': []}
    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((n,), t_frac, device=device)
        e_grad = compute_e_core_grad(e_core, u)
        delta_u = step_fn(u, e_grad, t_tensor)
        trajectory['delta_u_norm'].append(delta_u.abs().mean().item())
        u = u + dt * delta_u
        sigma = get_sigma('cosine', step, T)
        if sigma > 0.01:
            u = u + sigma * torch.randn_like(u)
        z_cur = quantize(u)
        trajectory['e_core'].append(e_core.energy(z_cur).item())
    return quantize(u), trajectory


class ConvProbe(nn.Module):
    def __init__(self, n_bits, H, W, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(64 * 4 * 4, n_classes))

    def forward(self, z):
        return self.net(z.float())


def train_probe_mixed(probe, z_clean, z_repair, labels, device, epochs=30, bs=128):
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    N = len(z_clean)
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            use_repair = torch.rand(len(idx)) < 0.5
            z_batch = torch.where(use_repair.view(-1, 1, 1, 1),
                                  z_repair[idx], z_clean[idx]).to(device)
            y = labels[idx].to(device)
            loss = F.cross_entropy(probe(z_batch), y)
            opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def eval_probe(probe, z, labels, device, bs=128):
    correct, total = 0, 0
    for i in range(0, len(z), bs):
        pred = probe(z[i:i+bs].to(device)).argmax(dim=1)
        correct += (pred == labels[i:i+bs].to(device)).sum().item()
        total += len(labels[i:i+bs])
    return correct / total


def hue_variance(x):
    return compute_hue_var(x)['hue_var']


def color_kl(x_gen, x_real):
    gen_means = x_gen.mean(dim=(2, 3))
    real_means = x_real.mean(dim=(2, 3))
    kl_total = 0.0
    for c_idx in range(gen_means.shape[1]):
        g_hist = torch.histc(gen_means[:, c_idx], bins=50, min=0, max=1) + 1e-8
        r_hist = torch.histc(real_means[:, c_idx], bins=50, min=0, max=1) + 1e-8
        g_hist = g_hist / g_hist.sum()
        r_hist = r_hist / r_hist.sum()
        kl_total += (g_hist * (g_hist / r_hist).log()).sum().item()
    return kl_total / gen_means.shape[1]


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_icc1_control_entropy')
    parser.add_argument('--n_gen', type=int, default=256)
    parser.add_argument('--T', type=int, default=10)
    # Calibration switches (can be overridden after O0/S0 results)
    parser.add_argument('--ent_weight', type=float, default=0.1)
    parser.add_argument('--ctrl_weight', type=float, default=0.1)
    parser.add_argument('--c_bits', type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    N_BITS = 16  # base: 16bit (C channel value more visible at lower bandwidth)

    print("=" * 100)
    print("ICC-1: INFORMATION CIRCULATION CONSTRAINT — CONTROL PORT + CONDITIONAL ENTROPY")
    print("=" * 100)
    print(f"Device: {device}  |  Seed: {args.seed}  |  T: {args.T}  |  n_bits: {N_BITS}")
    print(f"ent_weight: {args.ent_weight}  |  ctrl_weight: {args.ctrl_weight}  |  c_bits: {args.c_bits}")
    print(f"\nICC constraints:")
    print(f"  R (Rate):           {N_BITS} bits/pos × 16×16 = {N_BITS*256} total")
    print(f"  O (Observability):  cycle + repair + dead-bit ratio")
    print(f"  C (Controllability): control port c∈{{0,1}}^{args.c_bits} + Fisher ratio")
    print(f"  D (Decodability):   probe accuracy + effective entropy")
    print()

    # ========== DATA ==========
    print("[1] Loading CIFAR-10...")
    from torchvision import datasets, transforms
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(args.seed)

    train_idx = rng.choice(len(train_ds), 3000, replace=False)
    test_idx = rng.choice(len(test_ds), 500, replace=False)
    train_x = torch.stack([train_ds[i][0] for i in train_idx])
    test_x = torch.stack([test_ds[i][0] for i in test_idx])
    train_y = torch.tensor([train_ds[i][1] for i in train_idx])
    test_y = torch.tensor([test_ds[i][1] for i in test_idx])
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")

    # ========== REFERENCE ==========
    print("\n[2] Reference metrics...")
    real_hf_coh = hf_coherence_metric(test_x[:200], device)
    real_hf_noi = hf_noise_index(test_x[:200], device)
    real_hue_var = hue_variance(test_x[:200])
    print(f"    Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")

    # ========== CONFIGS (switch matrix) ==========
    configs = [
        {
            'name': 'ICC1A_base',
            'use_ctrl': False,
            'ent_weight': 0.0,
            'ctrl_weight': 0.0,
            'desc': 'Base 16bit — control (no ICC extras)',
        },
        {
            'name': 'ICC1B_entropy',
            'use_ctrl': False,
            'ent_weight': args.ent_weight,
            'ctrl_weight': 0.0,
            'desc': f'+Entropy (conditional, λ_ent=auto×{args.ent_weight})',
        },
        {
            'name': 'ICC1C_ctrl_port',
            'use_ctrl': True,
            'ent_weight': 0.0,
            'ctrl_weight': 0.0,
            'desc': '+ControlPort (c_bits=8, NO ctrl loss) — ablation',
        },
        {
            'name': 'ICC1D_ctrl_full',
            'use_ctrl': True,
            'ent_weight': args.ent_weight,
            'ctrl_weight': args.ctrl_weight,
            'desc': f'+ControlPort + CtrlLoss + Entropy — full ICC',
        },
    ]

    all_results = {}

    for cfg in configs:
        cfg_name = cfg['name']
        print(f"\n{'='*80}")
        print(f"CONFIG: {cfg_name}")
        print(f"  {cfg['desc']}")
        print(f"  use_ctrl={cfg['use_ctrl']}  ent_weight={cfg['ent_weight']}  ctrl_weight={cfg['ctrl_weight']}")
        print("=" * 80)

        result = {k: v for k, v in cfg.items()}

        # --- Build models ---
        torch.manual_seed(args.seed)
        enc = Encoder16(N_BITS).to(device)

        if cfg['use_ctrl']:
            dec = Decoder16_ControlPort(N_BITS, args.c_bits).to(device)
            ctrl_enc = ControlEncoder(args.c_bits).to(device)
        else:
            dec = Decoder16(N_BITS).to(device)
            ctrl_enc = None

        # --- Train ADC/DAC ---
        adc_info = train_adc_icc(
            enc, dec, train_x, device,
            control_encoder=ctrl_enc,
            ent_weight=cfg['ent_weight'],
            ctrl_weight=cfg['ctrl_weight'],
            c_bits=args.c_bits,
            epochs=40, bs=64)

        print(f"    ADC: recon={adc_info['recon_loss']:.4f}  "
              f"ent={adc_info['ent_loss']:.4f}  ctrl={adc_info['ctrl_loss']:.4f}")
        result['adc'] = adc_info

        # --- ICC Diagnostics ---
        print(f"\n  [ICC-O] Dead-bit ratio...")
        dead_info = measure_dead_bits(enc, dec, test_x, device,
                                       n_probes=200, n_images=100,
                                       control_encoder=ctrl_enc)
        print(f"    Dead: {dead_info['dead_ratio']:.3f}  "
              f"mean_inf={dead_info['mean_influence']:.6f}  "
              f"p10={dead_info['p10_influence']:.6f}")
        result['dead_bits'] = dead_info

        print(f"  [ICC-D] Effective entropy...")
        ent_info = measure_effective_entropy(enc, test_x, device)
        print(f"    mean_H={ent_info['mean_entropy']:.4f}  "
              f"near_deterministic={ent_info['entropy_below_01']:.3f}")
        result['entropy'] = ent_info

        # --- Encode ---
        z_train = encode_all(enc, train_x, device, bs=32)
        z_test = encode_all(enc, test_x, device, bs=32)
        K, H, W = z_train.shape[1:]
        usage = z_train.float().mean().item()
        print(f"    z: {z_train.shape}, usage={usage:.3f}")
        result['z_usage'] = usage

        # --- Train E_core ---
        e_core = DiffEnergyCore(N_BITS).to(device)
        train_ecore(e_core, z_train, device, epochs=15, bs=128)

        # --- Train StepFn ---
        step_fn = FlatStepFn_Norm(N_BITS).to(device)
        # StepFn needs a decode function — for control port, use zeros c (base mode)
        if cfg['use_ctrl']:
            dec_for_stepfn = lambda z: dec(z, None)
        else:
            dec_for_stepfn = dec
        train_step_fn(step_fn, e_core, z_train, dec_for_stepfn, device, epochs=30, bs=48)

        # --- ICC-C: Control Rank ---
        print(f"\n  [ICC-C] Control rank (Fisher ratio)...")
        ctrl_rank = measure_control_rank(
            dec, e_core, step_fn, K, H, W, device,
            control_encoder=ctrl_enc, c_bits=args.c_bits,
            n_per_c=32, n_c_values=min(16, 2**args.c_bits), T=args.T)
        print(f"    Fisher: {ctrl_rank['fisher_ratio']:.4f}  "
              f"Var_B={ctrl_rank['var_between']:.6f}  Var_W={ctrl_rank['var_within']:.6f}")
        result['ctrl_rank'] = ctrl_rank

        # ==================================================
        # REPAIR
        # ==================================================
        print(f"\n  --- REPAIR (center mask) ---")
        mask_center_test = make_center_mask(len(z_test), K, H, W, device='cpu')
        z_rep_list = []
        for ri in range(0, len(z_test), 32):
            nb = min(32, len(z_test) - ri)
            z_batch = z_test[ri:ri+nb].to(device)
            m_batch = mask_center_test[0:1].expand(nb, -1, -1, -1).to(device)
            z_rep = repair_flow(step_fn, e_core, z_batch, m_batch, device,
                                T=args.T, dt=0.5)
            z_rep_list.append(z_rep.cpu())
        z_test_repaired = torch.cat(z_rep_list)

        mask_exp = mask_center_test[0:1].expand(len(z_test), K, -1, -1)
        diff = (z_test != z_test_repaired).float()
        ham_unmasked = (diff * mask_exp).sum() / max(mask_exp.sum().item(), 1)
        ham_masked = (diff * (1 - mask_exp)).sum() / max((1 - mask_exp).sum().item(), 1)

        # Cycle
        with torch.no_grad():
            x_rep = []
            for ri in range(0, len(z_test_repaired), 32):
                zb = z_test_repaired[ri:ri+32].to(device)
                if cfg['use_ctrl']:
                    x_rep.append(dec(zb, None).cpu())
                else:
                    x_rep.append(dec(zb).cpu())
            x_recon_rep = torch.cat(x_rep)
            z_cycle = encode_all(enc, x_recon_rep, device, bs=32)
        cycle_repair = (z_test_repaired != z_cycle).float().mean().item()

        result['ham_unmasked'] = ham_unmasked.item()
        result['ham_masked'] = ham_masked.item()
        result['cycle_repair'] = cycle_repair
        print(f"    ham_unmask={ham_unmasked.item():.4f}  ham_mask={ham_masked.item():.4f}  "
              f"cycle={cycle_repair:.4f}")

        # ==================================================
        # GENERATION
        # ==================================================
        print(f"\n  --- GENERATION ---")
        torch.manual_seed(args.seed + 100)
        z_gen_list, all_traj = [], []
        for gi in range(0, args.n_gen, 32):
            nb = min(32, args.n_gen - gi)
            z_batch, traj = sample_flow_gen(step_fn, e_core, nb, K, H, W, device,
                                            T=args.T, dt=0.5)
            z_gen_list.append(z_batch.cpu())
            all_traj.append(traj)
        z_gen = torch.cat(z_gen_list)

        agg_traj = {
            key: [np.mean([t[key][s] for t in all_traj])
                  for s in range(len(all_traj[0][key]))]
            for key in all_traj[0].keys()
        }

        # Decode generated
        with torch.no_grad():
            x_gen_list = []
            for gi in range(0, len(z_gen), 32):
                zb = z_gen[gi:gi+32].to(device)
                if cfg['use_ctrl']:
                    # For generation: use random c
                    c_rand = torch.randint(0, 2, (len(zb), args.c_bits), device=device).float()
                    c_sp = c_rand.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
                    x_gen_list.append(dec(zb, c_sp).cpu())
                else:
                    x_gen_list.append(dec(zb).cpu())
            x_gen = torch.cat(x_gen_list)

        eval_result = evaluate(z_gen, dec if not cfg['use_ctrl'] else (lambda z: dec(z, None)),
                               enc, e_core, z_train, test_x,
                               real_hf_coh, real_hf_noi, device, trajectory=agg_traj)
        # evaluate() returns (dict, x_gen) or just dict depending on version
        if isinstance(eval_result, tuple):
            r_gen, _ = eval_result
        else:
            r_gen = eval_result
        # Override with control-decoded versions for visual metrics
        r_gen['hue_var'] = hue_variance(x_gen)
        r_gen['color_kl'] = color_kl(x_gen, test_x[:len(x_gen)])

        result['gen'] = {k: v for k, v in r_gen.items()
                         if not isinstance(v, (list, np.ndarray))}

        save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))

        print(f"    viol={r_gen['violation']:.4f}  div={r_gen['diversity']:.4f}  "
              f"HF_noise={r_gen['hf_noise_index']:.2f}")
        print(f"    HueVar={r_gen['hue_var']:.4f}  ColorKL={r_gen['color_kl']:.4f}  "
              f"conn={r_gen['connectedness']:.4f}")

        # ==================================================
        # CLASSIFICATION
        # ==================================================
        print(f"\n  --- CLASSIFICATION (conv probe, mixed) ---")
        mask_center_train = make_center_mask(1, K, H, W, device='cpu')
        z_train_rep_list = []
        for ri in range(0, len(z_train), 32):
            nb = min(32, len(z_train) - ri)
            z_batch = z_train[ri:ri+nb].to(device)
            m_batch = mask_center_train.expand(nb, -1, -1, -1).to(device)
            z_rep = repair_flow(step_fn, e_core, z_batch, m_batch, device,
                                T=args.T, dt=0.5)
            z_train_rep_list.append(z_rep.cpu())
        z_train_repaired = torch.cat(z_train_rep_list)

        probe = ConvProbe(N_BITS, H, W).to(device)
        train_probe_mixed(probe, z_train, z_train_repaired, train_y, device,
                          epochs=30, bs=128)

        acc_clean = eval_probe(probe, z_test, test_y, device)
        acc_repair = eval_probe(probe, z_test_repaired, test_y, device)
        gap = acc_clean - acc_repair

        result['acc_clean'] = acc_clean
        result['acc_repair'] = acc_repair
        result['gap'] = gap
        print(f"    acc_clean={acc_clean:.3f}  acc_repair={acc_repair:.3f}  gap={gap:.3f}")

        all_results[cfg_name] = result

        del enc, dec, e_core, step_fn, probe, ctrl_enc
        torch.cuda.empty_cache()

    # ========== SUMMARY ==========
    print(f"\n{'='*100}")
    print("ICC-1 SUMMARY: INFORMATION CIRCULATION CONSTRAINT")
    print(f"Real: HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")
    print("=" * 100)

    # --- ICC 4-constraint dashboard ---
    print(f"\n--- ICC CONSTRAINT DASHBOARD ---")
    icc_h = (f"{'config':<22} {'R(bits)':>7} {'O(cycle)':>8} {'O(dead%)':>8} "
             f"{'C(Fisher)':>9} {'D(clean)':>8} {'D(ent)':>7}")
    print(icc_h); print("-" * len(icc_h))
    for name, r in all_results.items():
        print(f"{name:<22} {N_BITS*256:>7} {r['cycle_repair']:>8.4f} "
              f"{r['dead_bits']['dead_ratio']:>8.3f} "
              f"{r['ctrl_rank']['fisher_ratio']:>9.4f} "
              f"{r['acc_clean']:>8.3f} {r['entropy']['mean_entropy']:>7.4f}")

    # --- Repair ---
    print(f"\n--- REPAIR CONTRACT ---")
    rh = f"{'config':<22} {'ham_un':>7} {'ham_m':>7} {'cycle':>7}"
    print(rh); print("-" * len(rh))
    for name, r in all_results.items():
        print(f"{name:<22} {r['ham_unmasked']:>7.4f} {r['ham_masked']:>7.4f} "
              f"{r['cycle_repair']:>7.4f}")

    # --- Generation ---
    print(f"\n--- GENERATION ---")
    gh = (f"{'config':<22} {'viol':>7} {'div':>7} {'HFnoi':>7} "
          f"{'HueV':>7} {'ColKL':>7} {'conn':>7}")
    print(gh); print("-" * len(gh))
    for name, r in all_results.items():
        g = r['gen']
        print(f"{name:<22} {g['violation']:>7.4f} {g['diversity']:>7.4f} "
              f"{g['hf_noise_index']:>7.2f} {g.get('hue_var', 0):>7.4f} "
              f"{g.get('color_kl', 0):>7.4f} {g['connectedness']:>7.4f}")

    # --- Classification ---
    print(f"\n--- CLASSIFICATION ---")
    ch = f"{'config':<22} {'clean':>7} {'repair':>7} {'gap':>7}"
    print(ch); print("-" * len(ch))
    for name, r in all_results.items():
        print(f"{name:<22} {r['acc_clean']:>7.3f} {r['acc_repair']:>7.3f} {r['gap']:>7.3f}")

    # --- Delta table ---
    a0 = all_results.get('ICC1A_base')
    if a0:
        print(f"\n--- DELTA vs ICC1A_base ---")
        dh = (f"{'config':<22} {'Δclean':>7} {'Δgap':>7} {'Δdiv':>7} "
              f"{'ΔHueV':>8} {'ΔFisher':>8} {'Δdead%':>7}")
        print(dh); print("-" * len(dh))
        for name, r in all_results.items():
            if name == 'ICC1A_base':
                continue
            print(f"{name:<22} "
                  f"{r['acc_clean']-a0['acc_clean']:>+7.3f} "
                  f"{r['gap']-a0['gap']:>+7.3f} "
                  f"{r['gen']['diversity']-a0['gen']['diversity']:>+7.3f} "
                  f"{r['gen'].get('hue_var',0)-a0['gen'].get('hue_var',0):>+8.4f} "
                  f"{r['ctrl_rank']['fisher_ratio']-a0['ctrl_rank']['fisher_ratio']:>+8.4f} "
                  f"{r['dead_bits']['dead_ratio']-a0['dead_bits']['dead_ratio']:>+7.3f}")

    # --- Gate check ---
    print(f"\n--- PRE-REGISTERED GATE CHECK ---")
    if a0:
        a0_cycle = a0['cycle_repair']
        a0_hue = a0['gen'].get('hue_var', 0)
        a0_fisher = a0['ctrl_rank']['fisher_ratio']

        for name, r in all_results.items():
            if name == 'ICC1A_base':
                continue
            hard_pass = (r['ham_unmasked'] < 0.001 and
                         r['cycle_repair'] <= a0_cycle + 0.02)

            criteria = 0
            hue_ratio = r['gen'].get('hue_var', 0) / max(a0_hue, 1e-8)
            d_clean = r['acc_clean'] - a0['acc_clean']
            d_div = r['gen']['diversity'] - a0['gen']['diversity']
            d_fisher = r['ctrl_rank']['fisher_ratio'] - a0_fisher

            if hue_ratio >= 2.0 or d_clean >= 0.02:
                criteria += 1
            if d_fisher > 0:
                criteria += 1
            if d_div >= -0.03:
                criteria += 1

            verdict = "PASS" if hard_pass and criteria >= 2 else "FAIL"
            if not hard_pass:
                verdict = "HARD_FAIL"
            print(f"  {name}: hard={'PASS' if hard_pass else 'FAIL'}  "
                  f"criteria={criteria}/3  verdict={verdict}")

    # Save
    save_results = {}
    for name, r in all_results.items():
        save_r = {k: v for k, v in r.items()}
        save_results[name] = save_r

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\n{'='*100}")
    print("ICC-1 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
