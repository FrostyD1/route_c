#!/usr/bin/env python3
"""
ICC-2: Spectral Equalization + Augmentation Invariance
======================================================

Core hypothesis (from S0 #76): The real bottleneck is NOT rank (94% observable,
99.6% controllable) but SPECTRAL GAP (560,000×). Most z-directions carry
negligible decoder influence. Reconstruction-only training creates a degenerate
code where capacity exists but isn't used.

Three orthogonal, task-agnostic fixes:
  1. Spectral equalization: equalize decoder Jacobian singular values
     → D-optimal design: maximize log det(J^T J + εI)
     → makes ALL z-directions equally important to decoder
  2. Augmentation invariance: encoder maps augmented views to same z
     → makes z invariant to irrelevant variations (pose, lighting)
     → semantic content dominates z's principal components
  3. VICReg channel decorrelation: z-channels are independent
     → no redundancy, maximum information per bit

Theory:
  - Kalman (1960): controllability/observability = rank conditions
  - Numerical reality: condition number κ = σ_max/σ_min determines usability
  - D-optimal design: maximize det(Fisher info) = maximize volume of
    sensitivity ellipsoid = equalize singular values
  - Hutchinson estimator: tr(A) ≈ 1/m Σ v^T A v for random v

All fixes are task-agnostic (no labels, no task-specific loss).
All fixes are training-only (deployment is still discrete LUT + flow).

5 configs:
  A: baseline (16bit, reproduce ICC1A)
  B: +spectral_eq (decoder Jacobian conditioning)
  C: +aug_inv (encoder invariance to augmentations)
  D: +spectral_eq + aug_inv (both fixes)
  E: +spectral_eq + aug_inv + vicreg (all three)

Pre-registered gates:
  Hard: ham_unmasked == 0.000, cycle ≤ baseline + 0.02
  Success (any 2 of 3):
    - classify: acc_clean ≥ baseline + 0.02
    - generation: HueVar ≥ 2× baseline OR div ≥ baseline
    - spectral: condition_number ≤ baseline / 2

4GB GPU: 3000 train, 500 test, ADC bs=64, StepFn bs=48
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
sys.path.insert(0, SCRIPT_DIR)

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
# SPECTRAL EQUALIZATION — D-optimal design on decoder Jacobian
# ============================================================================

def compute_spectral_loss(decoder, z, n_probes=4, eps=0.01):
    """Maximize log det(J^T J + εI) via random Jacobian probing.

    D-optimal design principle: maximize the volume of the decoder's
    output sensitivity ellipsoid. This makes the decoder equally
    sensitive to all z-directions, preventing spectral collapse.

    Uses batched random perturbations (Hutchinson-style) for efficiency.
    Only decoder parameters get gradients (z is used as evaluation point).

    Computational cost: 1 batched decoder forward pass of size (n_probes × B).

    Args:
        decoder: decoder network
        z: (B, C, H, W) current z-codes (will be detached for perturbation base)
        n_probes: number of random directions to probe (4-8 sufficient)
        eps: perturbation magnitude

    Returns:
        loss: -log det(G + εI) / n_probes (scalar, minimize to equalize spectrum)
    """
    B, C, H, W = z.shape

    # Use z as-is (not detached) so decoder params get gradients through both
    # base and perturbed forward passes
    x_base = decoder(z)  # [B, 3, 32, 32]
    x_flat = x_base.reshape(B, -1)  # [B, D_out]

    # Generate random perturbation directions in z-space
    vs = torch.randn(n_probes, B, C, H, W, device=z.device)
    vs = vs / (vs.reshape(n_probes, B, -1).norm(dim=2, keepdim=True).reshape(
        n_probes, B, 1, 1, 1) + 1e-8)

    # Batched forward pass: all probes at once
    z_expanded = z.unsqueeze(0).expand(n_probes, -1, -1, -1, -1)  # [P, B, C, H, W]
    z_pert = (z_expanded + eps * vs).reshape(n_probes * B, C, H, W)

    # Single batched decoder call
    x_pert = decoder(z_pert)  # [P*B, 3, 32, 32]
    x_pert = x_pert.reshape(n_probes, B, -1)  # [P, B, D_out]

    # Jacobian-vector products (finite difference approximation)
    responses = (x_pert - x_flat.unsqueeze(0)) / eps  # [P, B, D_out]

    # Gram matrix of responses: G_ij = <Jv_i, Jv_j>
    R = responses.permute(1, 0, 2)  # [B, P, D_out]
    G = torch.bmm(R, R.transpose(1, 2))  # [B, P, P]
    G = G + 1e-6 * torch.eye(n_probes, device=G.device).unsqueeze(0)

    # -log det(G) = minimize to maximize volume of sensitivity ellipsoid
    _, logdet = torch.linalg.slogdet(G)

    return -logdet.mean() / n_probes


# ============================================================================
# AUGMENTATION INVARIANCE — encoder stability to irrelevant perturbations
# ============================================================================

def augment_batch(x):
    """Task-agnostic augmentations: flip + mild color jitter + small noise.

    These are variations that don't change semantic content.
    Biologically: invariance to viewpoint/lighting changes.
    """
    B = x.shape[0]
    x_aug = x.clone()

    # Random horizontal flip (50%)
    flip_mask = torch.rand(B, device=x.device) > 0.5
    if flip_mask.any():
        x_aug[flip_mask] = x_aug[flip_mask].flip(-1)

    # Mild brightness/contrast jitter (±20%)
    brightness = 1.0 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.4
    x_aug = (x_aug * brightness).clamp(0, 1)

    # Small Gaussian noise (σ=0.03)
    x_aug = (x_aug + torch.randn_like(x_aug) * 0.03).clamp(0, 1)

    return x_aug


def compute_aug_inv_loss(encoder, x, x_aug):
    """Augmentation invariance: encoder(x_aug) should match encoder(x).

    Stop-gradient on original: only push augmented toward original,
    preserving the original encoding quality.
    """
    _, logits_orig = encoder(x)
    _, logits_aug = encoder(x_aug)

    return F.mse_loss(logits_aug, logits_orig.detach())


# ============================================================================
# VICReg CHANNEL DECORRELATION — maximize information per bit
# ============================================================================

def compute_vicreg_loss(z):
    """VICReg-style regularization on z-channels (Bardes et al., 2022).

    Variance: each channel should vary across samples (avoid collapse).
    Covariance: channels should be uncorrelated (avoid redundancy).

    Operates on channel dimension (C=16), not full z-space (4096-dim
    covariance matrix would be too large for 4GB GPU).

    Args:
        z: (B, C, H, W) soft z-codes

    Returns:
        loss: λ_var * var_loss + λ_cov * cov_loss
    """
    B, C, Hz, Wz = z.shape

    # Per-channel activation rate across batch and space
    # z: [B, C, H, W] → per-channel statistics
    z_chan = z.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]

    # Variance loss: push std above threshold (avoid collapse)
    std = z_chan.std(dim=0)  # [C]
    var_loss = F.relu(0.3 - std).mean()  # want std > 0.3

    # Covariance loss: off-diagonal of correlation matrix → 0
    z_centered = z_chan - z_chan.mean(dim=0, keepdim=True)
    N = z_centered.shape[0]
    cov = (z_centered.T @ z_centered) / max(N - 1, 1)  # [C, C]

    # Only penalize off-diagonal
    mask = ~torch.eye(C, dtype=torch.bool, device=z.device)
    cov_loss = (cov[mask] ** 2).mean()

    return 1.0 * var_loss + 0.04 * cov_loss


# ============================================================================
# SPECTRAL DIAGNOSTICS — measure condition number and participation ratio
# ============================================================================

@torch.no_grad()
def spectral_diagnostics(decoder, z_test, device, n_probes=32, n_images=100,
                         eps=0.01):
    """Estimate decoder Jacobian spectral properties.

    Uses random probing to estimate:
    - condition_number: σ_max / σ_min (lower = better conditioned)
    - participation_ratio: (Σλ_i)² / Σλ_i² (higher = more uniform spectrum)
    - effective_dim: how many directions carry significant energy

    Returns dict with diagnostic values.
    """
    z_sub = z_test[:n_images].to(device)
    x_base = decoder(z_sub)
    x_flat = x_base.reshape(n_images, -1)

    B, C, H, W = z_sub.shape

    responses = []
    for _ in range(n_probes):
        v = torch.randn_like(z_sub)
        v = v / (v.reshape(n_images, -1).norm(dim=1, keepdim=True).reshape(
            n_images, 1, 1, 1) + 1e-8)
        x_pert = decoder(z_sub + eps * v)
        resp = (x_pert.reshape(n_images, -1) - x_flat) / eps
        responses.append(resp)

    R = torch.stack(responses, dim=1)  # [B, n_probes, D_out]
    G = torch.bmm(R, R.transpose(1, 2))  # [B, n_probes, n_probes]

    # SVD of Gram matrices
    svals_all = []
    for i in range(n_images):
        svals = torch.linalg.svdvals(G[i])
        svals_all.append(svals)
    svals = torch.stack(svals_all)  # [B, n_probes]

    # Condition number (per image, then average)
    cond = svals[:, 0] / (svals[:, -1] + 1e-10)

    # Participation ratio = (Σλ)² / Σλ²
    pr = (svals.sum(dim=1) ** 2) / ((svals ** 2).sum(dim=1) + 1e-10)

    # Effective dimension (number of svals > 1% of max)
    threshold = 0.01 * svals[:, 0:1]
    eff_dim = (svals > threshold).float().sum(dim=1)

    return {
        'condition_number': cond.mean().item(),
        'condition_number_median': cond.median().item(),
        'participation_ratio': pr.mean().item(),
        'effective_dim': eff_dim.mean().item(),
        'sval_max': svals[:, 0].mean().item(),
        'sval_min': svals[:, -1].mean().item(),
    }


# ============================================================================
# GRADIENT NORM UTILITY
# ============================================================================

def compute_grad_norm(loss, params, retain_graph=True):
    """Compute gradient norm for auto-normalization."""
    grads = torch.autograd.grad(loss, params, create_graph=False,
                                retain_graph=retain_graph, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.data.norm().item() ** 2
    return total ** 0.5


# ============================================================================
# ICC-2 ADC TRAINING — curriculum + spectral + aug_inv + vicreg
# ============================================================================

def train_adc_icc2(encoder, decoder, train_x, device,
                   spectral_eq=False, aug_inv=False, vicreg=False,
                   epochs=40, bs=64, warmup_epochs=10):
    """Train ADC with ICC-2 regularization.

    Curriculum design:
      - Epochs 1-warmup: reconstruction only (establish good baseline)
      - Epochs warmup+1 to end: add spectral/aug/vicreg losses

    Auto-normalization: λ computed on first regularized batch, scaled to 0.1×
    of reconstruction gradient norm (conservative, learned from ICC-1 failure
    where auto-norm was too aggressive).

    Args:
        spectral_eq: enable spectral equalization (decoder conditioning)
        aug_inv: enable augmentation invariance (encoder semantics)
        vicreg: enable VICReg channel decorrelation
        warmup_epochs: epochs of reconstruction-only warmup
    """
    params_enc = list(encoder.parameters())
    params_dec = list(decoder.parameters())
    params_all = params_enc + params_dec

    opt = torch.optim.Adam(params_all, lr=1e-3)

    lambda_spec = None
    lambda_aug = None
    lambda_vic = None

    for epoch in tqdm(range(epochs), desc="ADC"):
        encoder.train(); decoder.train()
        if hasattr(encoder, 'set_temperature'):
            encoder.set_temperature(1.0 + (0.3 - 1.0) * epoch / max(epochs - 1, 1))

        in_warmup = epoch < warmup_epochs
        perm = torch.randperm(len(train_x))
        tl_recon, tl_spec, tl_aug, tl_vic, nb = 0., 0., 0., 0., 0

        for i in range(0, len(train_x), bs):
            x = train_x[perm[i:i+bs]].to(device)
            opt.zero_grad()

            z, logits = encoder(x)
            xh = decoder(z)

            # L_recon (always on)
            loss_recon = F.mse_loss(xh, x) + 0.5 * F.binary_cross_entropy(xh, x)
            loss = loss_recon

            if not in_warmup:
                # Spectral equalization (fixes decoder conditioning)
                if spectral_eq:
                    # Detach z for spectral loss — only train decoder
                    l_spec = compute_spectral_loss(decoder, z.detach(), n_probes=4, eps=0.01)
                    if lambda_spec is None:
                        g_recon = compute_grad_norm(loss_recon, params_dec)
                        g_spec = compute_grad_norm(l_spec, params_dec)
                        # Conservative: 10% of recon gradient scale
                        lambda_spec = 0.1 * g_recon / max(g_spec, 1e-8)
                        lambda_spec = min(lambda_spec, 2.0)
                        print(f"    λ_spec auto = {lambda_spec:.4f} "
                              f"(g_recon={g_recon:.4f}, g_spec={g_spec:.4f})")
                    loss = loss + lambda_spec * l_spec
                    tl_spec += l_spec.item()

                # Augmentation invariance (fixes encoder semantics)
                if aug_inv:
                    x_aug = augment_batch(x)
                    l_aug = compute_aug_inv_loss(encoder, x, x_aug)
                    if lambda_aug is None:
                        g_recon_enc = compute_grad_norm(loss_recon, params_enc)
                        g_aug = compute_grad_norm(l_aug, params_enc)
                        lambda_aug = 0.1 * g_recon_enc / max(g_aug, 1e-8)
                        lambda_aug = min(lambda_aug, 2.0)
                        print(f"    λ_aug auto = {lambda_aug:.4f} "
                              f"(g_recon={g_recon_enc:.4f}, g_aug={g_aug:.4f})")
                    loss = loss + lambda_aug * l_aug
                    tl_aug += l_aug.item()

                # VICReg channel decorrelation
                if vicreg:
                    l_vic = compute_vicreg_loss(z)
                    if lambda_vic is None:
                        lambda_vic = 0.01  # Fixed small weight
                        print(f"    λ_vic = {lambda_vic:.4f}")
                    loss = loss + lambda_vic * l_vic
                    tl_vic += l_vic.item()

            loss.backward()
            opt.step()
            tl_recon += loss_recon.item()
            nb += 1

        if (epoch + 1) % 10 == 0 or epoch == warmup_epochs - 1:
            parts = [f"recon={tl_recon/nb:.4f}"]
            if not in_warmup:
                if spectral_eq: parts.append(f"spec={tl_spec/nb:.4f}")
                if aug_inv: parts.append(f"aug={tl_aug/nb:.4f}")
                if vicreg: parts.append(f"vic={tl_vic/nb:.4f}")
            else:
                parts.append("(warmup)")
            print(f"    Epoch {epoch+1}/{epochs}: {', '.join(parts)}")

    encoder.eval(); decoder.eval()
    return {
        'lambda_spec': lambda_spec,
        'lambda_aug': lambda_aug,
        'lambda_vic': lambda_vic,
    }


# ============================================================================
# DEAD-BIT / ENTROPY DIAGNOSTICS (from ICC-1)
# ============================================================================

@torch.no_grad()
def measure_dead_bits(encoder, decoder, test_x, device,
                      n_probes=200, threshold=1e-5, n_images=100):
    """Measure fraction of z-bits with negligible decoder influence."""
    x_sub = test_x[:n_images]
    z_list = []
    for i in range(0, len(x_sub), 32):
        batch = x_sub[i:i+32].to(device)
        z, _ = encoder(batch)
        z_list.append(z.cpu())
    z_all = torch.cat(z_list)

    x_orig_list = []
    for i in range(0, len(z_all), 32):
        x_orig_list.append(decoder(z_all[i:i+32].to(device)).cpu())
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
            x_flip_list.append(decoder(z_flip[i:i+32].to(device)).cpu())
        x_flip = torch.cat(x_flip_list)
        influences.append((x_orig - x_flip).pow(2).mean().item())

    influences = np.array(influences)
    return {
        'dead_ratio': float((influences < threshold).mean()),
        'mean_influence': float(influences.mean()),
        'p10_influence': float(np.percentile(influences, 10)),
    }


@torch.no_grad()
def measure_effective_entropy(encoder, data, device, bs=32):
    """Measure per-position entropy of z across the dataset."""
    logits_list = []
    for i in range(0, len(data), bs):
        _, logits = encoder(data[i:i+bs].to(device))
        logits_list.append(logits.cpu())
    all_logits = torch.cat(logits_list)
    p = torch.sigmoid(all_logits).clamp(1e-6, 1 - 1e-6)
    h = -(p * p.log() + (1 - p) * (1 - p).log())
    return {
        'mean_entropy': h.mean().item(),
        'entropy_below_01': float((h < 0.1).float().mean().item()),
    }


# ============================================================================
# REPAIR / GEN / CLASSIFY (standard three-mode eval, from ICC-1)
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
def sample_flow_gen(step_fn, e_core, n, K, H, W, device, T=10, dt=0.5):
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
    parser.add_argument('--output_dir', default='outputs/exp_icc2_spectral')
    parser.add_argument('--n_gen', type=int, default=256)
    parser.add_argument('--T', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    N_BITS = 16

    print("=" * 100)
    print("ICC-2: SPECTRAL EQUALIZATION + AUGMENTATION INVARIANCE")
    print("=" * 100)
    print(f"Device: {device}  |  Seed: {args.seed}  |  T: {args.T}  |  n_bits: {N_BITS}")
    print(f"\nHypothesis: Spectral gap (560K×) is the bottleneck, not rank.")
    print(f"Fix decoder conditioning (D-optimal) + encoder semantics (aug-inv).")
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

    # ========== CONFIGS ==========
    configs = [
        {
            'name': 'ICC2A_base',
            'spectral_eq': False, 'aug_inv': False, 'vicreg': False,
            'desc': 'Baseline 16bit (reproduce ICC1A)',
        },
        {
            'name': 'ICC2B_spectral',
            'spectral_eq': True, 'aug_inv': False, 'vicreg': False,
            'desc': '+Spectral eq (D-optimal decoder Jacobian)',
        },
        {
            'name': 'ICC2C_auginv',
            'spectral_eq': False, 'aug_inv': True, 'vicreg': False,
            'desc': '+Aug invariance (flip + color jitter + noise)',
        },
        {
            'name': 'ICC2D_spec_aug',
            'spectral_eq': True, 'aug_inv': True, 'vicreg': False,
            'desc': '+Spectral eq + Aug invariance (decoder + encoder)',
        },
        {
            'name': 'ICC2E_full',
            'spectral_eq': True, 'aug_inv': True, 'vicreg': True,
            'desc': '+Spectral eq + Aug inv + VICReg (all three)',
        },
    ]

    all_results = {}

    for cfg in configs:
        cfg_name = cfg['name']
        print(f"\n{'='*80}")
        print(f"CONFIG: {cfg_name}")
        print(f"  {cfg['desc']}")
        print(f"  spectral={cfg['spectral_eq']}  aug_inv={cfg['aug_inv']}  vicreg={cfg['vicreg']}")
        print("=" * 80)

        result = {k: v for k, v in cfg.items()}

        # --- Build models ---
        torch.manual_seed(args.seed)
        enc = Encoder16(N_BITS).to(device)
        dec = Decoder16(N_BITS).to(device)

        # --- Train ADC/DAC with ICC-2 ---
        adc_info = train_adc_icc2(
            enc, dec, train_x, device,
            spectral_eq=cfg['spectral_eq'],
            aug_inv=cfg['aug_inv'],
            vicreg=cfg['vicreg'],
            epochs=40, bs=64, warmup_epochs=10)
        result['adc'] = adc_info

        # --- Spectral diagnostics ---
        print(f"\n  [SPECTRAL] Decoder Jacobian diagnostics...")
        z_test_small = encode_all(enc, test_x[:100], device, bs=32)
        spec_diag = spectral_diagnostics(dec, z_test_small, device,
                                          n_probes=32, n_images=100)
        print(f"    cond_number={spec_diag['condition_number']:.1f}  "
              f"participation_ratio={spec_diag['participation_ratio']:.2f}  "
              f"eff_dim={spec_diag['effective_dim']:.1f}/{32}")
        result['spectral'] = spec_diag

        # --- Dead-bit diagnostic ---
        print(f"  [O] Dead-bit ratio...")
        dead_info = measure_dead_bits(enc, dec, test_x, device,
                                       n_probes=200, n_images=100)
        print(f"    Dead: {dead_info['dead_ratio']:.3f}  "
              f"mean_inf={dead_info['mean_influence']:.6f}")
        result['dead_bits'] = dead_info

        # --- Entropy diagnostic ---
        print(f"  [D] Effective entropy...")
        ent_info = measure_effective_entropy(enc, test_x, device)
        print(f"    mean_H={ent_info['mean_entropy']:.4f}  "
              f"near_deterministic={ent_info['entropy_below_01']:.3f}")
        result['entropy'] = ent_info

        # --- Encode all ---
        z_train = encode_all(enc, train_x, device, bs=32)
        z_test = encode_all(enc, test_x, device, bs=32)
        K, H, W = z_train.shape[1:]
        usage = z_train.float().mean().item()
        print(f"    z: {z_train.shape}, usage={usage:.3f}")

        # --- Train E_core ---
        e_core = DiffEnergyCore(N_BITS).to(device)
        train_ecore(e_core, z_train, device, epochs=15, bs=128)

        # --- Train StepFn ---
        step_fn = FlatStepFn_Norm(N_BITS).to(device)
        train_step_fn(step_fn, e_core, z_train, dec, device, epochs=30, bs=48)

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
                x_rep.append(dec(z_test_repaired[ri:ri+32].to(device)).cpu())
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

        with torch.no_grad():
            x_gen_list = []
            for gi in range(0, len(z_gen), 32):
                x_gen_list.append(dec(z_gen[gi:gi+32].to(device)).cpu())
            x_gen = torch.cat(x_gen_list)

        eval_result = evaluate(z_gen, dec, enc, e_core, z_train, test_x,
                               real_hf_coh, real_hf_noi, device, trajectory=agg_traj)
        if isinstance(eval_result, tuple):
            r_gen, _ = eval_result
        else:
            r_gen = eval_result
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

        del enc, dec, e_core, step_fn, probe
        torch.cuda.empty_cache()

    # ========== SUMMARY ==========
    print(f"\n{'='*100}")
    print("ICC-2 SUMMARY: SPECTRAL EQUALIZATION + AUGMENTATION INVARIANCE")
    print(f"Real: HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")
    print("=" * 100)

    # --- Spectral dashboard ---
    print(f"\n--- SPECTRAL DASHBOARD ---")
    sh = (f"{'config':<22} {'cond_num':>10} {'part_ratio':>10} {'eff_dim':>8} "
          f"{'dead%':>7} {'entropy':>8}")
    print(sh); print("-" * len(sh))
    for name, r in all_results.items():
        s = r['spectral']
        print(f"{name:<22} {s['condition_number']:>10.1f} "
              f"{s['participation_ratio']:>10.2f} "
              f"{s['effective_dim']:>8.1f} "
              f"{r['dead_bits']['dead_ratio']:>7.3f} "
              f"{r['entropy']['mean_entropy']:>8.4f}")

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
    a0 = all_results.get('ICC2A_base')
    if a0:
        print(f"\n--- DELTA vs ICC2A_base ---")
        dh = (f"{'config':<22} {'Δclean':>7} {'Δgap':>7} {'Δdiv':>7} "
              f"{'ΔHueV':>8} {'Δcond':>10} {'Δdead%':>7}")
        print(dh); print("-" * len(dh))
        for name, r in all_results.items():
            if name == 'ICC2A_base':
                continue
            print(f"{name:<22} "
                  f"{r['acc_clean']-a0['acc_clean']:>+7.3f} "
                  f"{r['gap']-a0['gap']:>+7.3f} "
                  f"{r['gen']['diversity']-a0['gen']['diversity']:>+7.3f} "
                  f"{r['gen'].get('hue_var',0)-a0['gen'].get('hue_var',0):>+8.4f} "
                  f"{r['spectral']['condition_number']-a0['spectral']['condition_number']:>+10.1f} "
                  f"{r['dead_bits']['dead_ratio']-a0['dead_bits']['dead_ratio']:>+7.3f}")

    # --- Gate check ---
    print(f"\n--- PRE-REGISTERED GATE CHECK ---")
    if a0:
        a0_cycle = a0['cycle_repair']
        a0_hue = a0['gen'].get('hue_var', 0)
        a0_cond = a0['spectral']['condition_number']

        for name, r in all_results.items():
            if name == 'ICC2A_base':
                continue
            hard_pass = (r['ham_unmasked'] < 0.001 and
                         r['cycle_repair'] <= a0_cycle + 0.02)

            criteria = 0
            d_clean = r['acc_clean'] - a0['acc_clean']
            d_div = r['gen']['diversity'] - a0['gen']['diversity']
            hue_ratio = r['gen'].get('hue_var', 0) / max(a0_hue, 1e-8)
            cond_ratio = r['spectral']['condition_number'] / max(a0_cond, 1e-8)

            if d_clean >= 0.02:
                criteria += 1
            if hue_ratio >= 2.0 or d_div >= 0.0:
                criteria += 1
            if cond_ratio <= 0.5:  # condition number halved
                criteria += 1

            verdict = "PASS" if hard_pass and criteria >= 2 else "FAIL"
            if not hard_pass:
                verdict = "HARD_FAIL"
            print(f"  {name}: hard={'PASS' if hard_pass else 'FAIL'}  "
                  f"criteria={criteria}/3  verdict={verdict}")

    # Save results
    save_results = {}
    for name, r in all_results.items():
        save_results[name] = {k: v for k, v in r.items()}
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\n{'='*100}")
    print("ICC-2 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
