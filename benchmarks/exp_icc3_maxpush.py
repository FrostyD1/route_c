#!/usr/bin/env python3
"""
ICC-3: MAXIMUM CLASSIFICATION PUSH
====================================

Combine ALL proven effective interventions for CIFAR-10 classification:
  1. ResBlock encoder/decoder at 32×32×16 (C2: +6.4% over plain encoder)
  2. Spectral equalization (ICC-2: +3.8% D-optimal Jacobian)
  3. 10K training data (3.3× more than previous experiments)
  4. 60 ADC epochs with cosine LR schedule
  5. Augmentation invariance (ICC-2: +1.0%)

Best individual results so far:
  - C2 flat_resblock 32×32×8: 51.5% (ResBlock + mixed probe)
  - ICC-2B spectral 16×16×16: 45.2% (spectral eq on small encoder)
  - C1 Z-C 32×32×16: 51.0% (bandwidth only)

Prediction: 57-61% if effects are additive.

4 configs:
  A: ResBlock baseline (32×32×16, 10K data, 60 epochs)
  B: +spectral_eq
  C: +spectral_eq + aug_inv
  D: +spectral_eq + aug_inv + stronger aug + lr_schedule

Target: 60%+ (vs supervised TinyCNN 61.6%)
All task-agnostic: no labels in training, no task-specific losses.

4GB GPU: 10K train, 1K test, bs=32 (larger models need smaller batches)
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

from exp_g2_protocol_density import encode_all, train_ecore, train_step_fn

from exp_e2a_global_prior import compute_hue_var, compute_marginal_kl


# ============================================================================
# RESBLOCK ENCODER/DECODER (from C2 exp_cifar10_staged_encoder.py)
# ============================================================================

class ResBlock(nn.Module):
    """Standard pre-activation ResBlock."""
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
    def forward(self, x): return x + self.net(x)


class ResBlockEncoder(nn.Module):
    """ResBlock encoder: 32×32 spatial, k bits per position."""
    def __init__(self, k=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, k, 3, padding=1),
        )
        self.q = GumbelSigmoid()

    def forward(self, x):
        logits = self.net(x)
        z = self.q(logits)
        return z, logits

    def set_temperature(self, tau):
        self.q.set_tau(tau)


class ResBlockDecoder(nn.Module):
    """ResBlock decoder: k bits → 3 channels, 32×32 spatial."""
    def __init__(self, k=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(k, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


# ============================================================================
# SPECTRAL EQUALIZATION — D-optimal design on decoder Jacobian
# ============================================================================

def compute_spectral_loss(decoder, z, n_probes=4, eps=0.01):
    """Maximize log det(J^T J + εI) via random Jacobian probing."""
    B, C, H, W = z.shape
    x_base = decoder(z)
    x_flat = x_base.reshape(B, -1)

    vs = torch.randn(n_probes, B, C, H, W, device=z.device)
    vs = vs / (vs.reshape(n_probes, B, -1).norm(dim=2, keepdim=True).reshape(
        n_probes, B, 1, 1, 1) + 1e-8)

    z_expanded = z.unsqueeze(0).expand(n_probes, -1, -1, -1, -1)
    z_pert = (z_expanded + eps * vs).reshape(n_probes * B, C, H, W)

    x_pert = decoder(z_pert)
    x_pert = x_pert.reshape(n_probes, B, -1)

    responses = (x_pert - x_flat.unsqueeze(0)) / eps
    R = responses.permute(1, 0, 2)
    G = torch.bmm(R, R.transpose(1, 2))
    G = G + 1e-6 * torch.eye(n_probes, device=G.device).unsqueeze(0)

    _, logdet = torch.linalg.slogdet(G)
    return -logdet.mean() / n_probes


# ============================================================================
# AUGMENTATION INVARIANCE
# ============================================================================

def augment_batch(x, strength='mild'):
    """Task-agnostic augmentations."""
    B = x.shape[0]
    x_aug = x.clone()

    # Random horizontal flip (50%)
    flip_mask = torch.rand(B, device=x.device) > 0.5
    if flip_mask.any():
        x_aug[flip_mask] = x_aug[flip_mask].flip(-1)

    if strength == 'strong':
        # Stronger color jitter (±30%)
        brightness = 1.0 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.6
        x_aug = (x_aug * brightness).clamp(0, 1)
        # Random crop-and-resize (4px padding)
        pad = 4
        x_padded = F.pad(x_aug, [pad]*4, mode='reflect')
        offsets_h = torch.randint(0, 2*pad, (B,))
        offsets_w = torch.randint(0, 2*pad, (B,))
        crops = []
        for b in range(B):
            crops.append(x_padded[b, :, offsets_h[b]:offsets_h[b]+32, offsets_w[b]:offsets_w[b]+32])
        x_aug = torch.stack(crops)
        # Gaussian noise (σ=0.05)
        x_aug = (x_aug + torch.randn_like(x_aug) * 0.05).clamp(0, 1)
    else:
        # Mild: brightness ±20%, noise σ=0.03
        brightness = 1.0 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.4
        x_aug = (x_aug * brightness).clamp(0, 1)
        x_aug = (x_aug + torch.randn_like(x_aug) * 0.03).clamp(0, 1)

    return x_aug


def compute_aug_inv_loss(encoder, x, x_aug):
    """Augmentation invariance: encoder(x_aug) should match encoder(x)."""
    _, logits_orig = encoder(x)
    _, logits_aug = encoder(x_aug)
    return F.mse_loss(logits_aug, logits_orig.detach())


# ============================================================================
# VICReg CHANNEL DECORRELATION
# ============================================================================

def compute_vicreg_loss(z):
    """VICReg-style: variance + covariance on z-channels."""
    B, C, Hz, Wz = z.shape
    z_chan = z.permute(0, 2, 3, 1).reshape(-1, C)
    std = z_chan.std(dim=0)
    var_loss = F.relu(0.3 - std).mean()
    z_centered = z_chan - z_chan.mean(dim=0, keepdim=True)
    N = z_centered.shape[0]
    cov = (z_centered.T @ z_centered) / max(N - 1, 1)
    mask = ~torch.eye(C, dtype=torch.bool, device=z.device)
    cov_loss = (cov[mask] ** 2).mean()
    return 1.0 * var_loss + 0.04 * cov_loss


# ============================================================================
# SPECTRAL DIAGNOSTICS
# ============================================================================

@torch.no_grad()
def spectral_diagnostics(decoder, z_test, device, n_probes=32, n_images=100,
                         eps=0.01):
    """Estimate decoder Jacobian spectral properties."""
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

    R = torch.stack(responses, dim=1)
    G = torch.bmm(R, R.transpose(1, 2))

    svals_all = []
    for i in range(n_images):
        svals = torch.linalg.svdvals(G[i])
        svals_all.append(svals)
    svals = torch.stack(svals_all)

    cond = svals[:, 0] / (svals[:, -1] + 1e-10)
    pr = (svals.sum(dim=1) ** 2) / ((svals ** 2).sum(dim=1) + 1e-10)
    threshold = 0.01 * svals[:, 0:1]
    eff_dim = (svals > threshold).float().sum(dim=1)

    return {
        'condition_number': cond.mean().item(),
        'participation_ratio': pr.mean().item(),
        'effective_dim': eff_dim.mean().item(),
    }


# ============================================================================
# GRADIENT NORM
# ============================================================================

def compute_grad_norm(loss, params, retain_graph=True):
    grads = torch.autograd.grad(loss, params, create_graph=False,
                                retain_graph=retain_graph, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.data.norm().item() ** 2
    return total ** 0.5


# ============================================================================
# ICC-3 ADC TRAINING
# ============================================================================

def train_adc_icc3(encoder, decoder, train_x, device,
                   spectral_eq=False, aug_inv=False, aug_strength='mild',
                   lr_schedule=False,
                   epochs=60, bs=32, warmup_epochs=15):
    """Train ADC with ICC-3 regularization: ResBlock + spectral + aug_inv.

    Changes from ICC-2:
    - Longer training (60 epochs, 15 warmup)
    - Cosine LR schedule option
    - Smaller batch size for larger model
    - Spectral probes: 4 directions (same as ICC-2)
    """
    params_enc = list(encoder.parameters())
    params_dec = list(decoder.parameters())
    params_all = params_enc + params_dec

    opt = torch.optim.Adam(params_all, lr=1e-3)

    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    else:
        scheduler = None

    lambda_spec = None
    lambda_aug = None

    for epoch in tqdm(range(epochs), desc="ADC"):
        encoder.train(); decoder.train()
        if hasattr(encoder, 'set_temperature'):
            encoder.set_temperature(1.0 + (0.3 - 1.0) * epoch / max(epochs - 1, 1))

        in_warmup = epoch < warmup_epochs
        perm = torch.randperm(len(train_x))
        tl_recon, tl_spec, tl_aug, nb = 0., 0., 0., 0

        for i in range(0, len(train_x), bs):
            x = train_x[perm[i:i+bs]].to(device)
            opt.zero_grad()

            z, logits = encoder(x)
            xh = decoder(z)

            loss_recon = F.mse_loss(xh, x) + 0.5 * F.binary_cross_entropy(xh, x)
            loss = loss_recon

            if not in_warmup:
                if spectral_eq:
                    l_spec = compute_spectral_loss(decoder, z.detach(), n_probes=4, eps=0.01)
                    if lambda_spec is None:
                        g_recon = compute_grad_norm(loss_recon, params_dec)
                        g_spec = compute_grad_norm(l_spec, params_dec)
                        lambda_spec = 0.1 * g_recon / max(g_spec, 1e-8)
                        lambda_spec = min(lambda_spec, 2.0)
                        print(f"    λ_spec auto = {lambda_spec:.4f} "
                              f"(g_recon={g_recon:.4f}, g_spec={g_spec:.4f})")
                    loss = loss + lambda_spec * l_spec
                    tl_spec += l_spec.item()

                if aug_inv:
                    x_aug = augment_batch(x, strength=aug_strength)
                    l_aug = compute_aug_inv_loss(encoder, x, x_aug)
                    if lambda_aug is None:
                        g_recon_enc = compute_grad_norm(loss_recon, params_enc)
                        g_aug = compute_grad_norm(l_aug, params_enc)
                        lambda_aug = 0.1 * g_recon_enc / max(g_aug, 1e-8)
                        lambda_aug = min(lambda_aug, 2.0)
                        print(f"    λ_aug auto = {lambda_aug:.4f}")
                    loss = loss + lambda_aug * l_aug
                    tl_aug += l_aug.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_all, max_norm=5.0)
            opt.step()
            tl_recon += loss_recon.item()
            nb += 1

        if scheduler:
            scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == warmup_epochs - 1:
            parts = [f"recon={tl_recon/nb:.4f}"]
            if not in_warmup:
                if spectral_eq: parts.append(f"spec={tl_spec/nb:.4f}")
                if aug_inv: parts.append(f"aug={tl_aug/nb:.4f}")
            else:
                parts.append("(warmup)")
            lr_str = f"  lr={opt.param_groups[0]['lr']:.6f}" if scheduler else ""
            print(f"    Epoch {epoch+1}/{epochs}: {', '.join(parts)}{lr_str}")

    encoder.eval(); decoder.eval()
    return {'lambda_spec': lambda_spec, 'lambda_aug': lambda_aug}


# ============================================================================
# DEAD-BIT / ENTROPY DIAGNOSTICS
# ============================================================================

@torch.no_grad()
def measure_dead_bits(encoder, decoder, test_x, device,
                      n_probes=200, threshold=1e-5, n_images=100):
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
    }


@torch.no_grad()
def measure_effective_entropy(encoder, data, device, bs=32):
    logits_list = []
    for i in range(0, len(data), bs):
        _, logits = encoder(data[i:i+bs].to(device))
        logits_list.append(logits.cpu())
    all_logits = torch.cat(logits_list)
    p = torch.sigmoid(all_logits).clamp(1e-6, 1 - 1e-6)
    h = -(p * p.log() + (1 - p) * (1 - p).log())
    return {'mean_entropy': h.mean().item()}


# ============================================================================
# REPAIR / GEN / CLASSIFY
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


class DeepConvProbe(nn.Module):
    """Slightly deeper probe for larger z-grids."""
    def __init__(self, n_bits, H, W, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(128 * 4 * 4, n_classes))
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
    parser.add_argument('--output_dir', default='outputs/exp_icc3_maxpush')
    parser.add_argument('--n_gen', type=int, default=256)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_test', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    N_BITS = 16

    print("=" * 100)
    print("ICC-3: MAXIMUM CLASSIFICATION PUSH")
    print("=" * 100)
    print(f"Device: {device}  |  Seed: {args.seed}  |  T: {args.T}  |  n_bits: {N_BITS}")
    print(f"Train: {args.n_train}  |  Test: {args.n_test}")
    print(f"\nCombining: ResBlock encoder + spectral eq + 10K data + 60 epochs")
    print(f"Target: 60%+ (supervised TinyCNN=61.6%)")
    print()

    # ========== DATA ==========
    print("[1] Loading CIFAR-10...")
    from torchvision import datasets, transforms
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(args.seed)

    train_idx = rng.choice(len(train_ds), args.n_train, replace=False)
    test_idx = rng.choice(len(test_ds), args.n_test, replace=False)
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
            'name': 'ICC3A_resblock',
            'spectral_eq': False, 'aug_inv': False, 'aug_strength': 'mild',
            'lr_schedule': False,
            'desc': 'ResBlock 32×32×16 baseline (10K data, 60 epochs)',
        },
        {
            'name': 'ICC3B_spectral',
            'spectral_eq': True, 'aug_inv': False, 'aug_strength': 'mild',
            'lr_schedule': False,
            'desc': '+Spectral eq (D-optimal decoder Jacobian)',
        },
        {
            'name': 'ICC3C_spec_aug',
            'spectral_eq': True, 'aug_inv': True, 'aug_strength': 'mild',
            'lr_schedule': False,
            'desc': '+Spectral eq + mild aug-inv',
        },
        {
            'name': 'ICC3D_full',
            'spectral_eq': True, 'aug_inv': True, 'aug_strength': 'strong',
            'lr_schedule': True,
            'desc': '+Spectral eq + strong aug + cosine LR',
        },
    ]

    all_results = {}

    for cfg in configs:
        cfg_name = cfg['name']
        print(f"\n{'='*80}")
        print(f"CONFIG: {cfg_name}")
        print(f"  {cfg['desc']}")
        print(f"  spectral={cfg['spectral_eq']}  aug={cfg['aug_inv']}  "
              f"aug_str={cfg['aug_strength']}  lr_sched={cfg['lr_schedule']}")
        print("=" * 80)

        result = {k: v for k, v in cfg.items()}

        # --- Build models ---
        torch.manual_seed(args.seed)
        enc = ResBlockEncoder(N_BITS).to(device)
        dec = ResBlockDecoder(N_BITS).to(device)
        n_params_enc = sum(p.numel() for p in enc.parameters())
        n_params_dec = sum(p.numel() for p in dec.parameters())
        print(f"  Encoder: {n_params_enc:,} params, Decoder: {n_params_dec:,} params")
        result['n_params'] = n_params_enc + n_params_dec

        # --- Train ADC/DAC ---
        adc_info = train_adc_icc3(
            enc, dec, train_x, device,
            spectral_eq=cfg['spectral_eq'],
            aug_inv=cfg['aug_inv'],
            aug_strength=cfg['aug_strength'],
            lr_schedule=cfg['lr_schedule'],
            epochs=60, bs=32, warmup_epochs=15)
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
        print(f"    mean_H={ent_info['mean_entropy']:.4f}")
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
        train_step_fn(step_fn, e_core, z_train, dec, device, epochs=30, bs=32)

        # ==================================================
        # REPAIR
        # ==================================================
        print(f"\n  --- REPAIR (center mask) ---")
        mask_center_test = make_center_mask(len(z_test), K, H, W, device='cpu')
        z_rep_list = []
        for ri in range(0, len(z_test), 16):
            nb = min(16, len(z_test) - ri)
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
        for gi in range(0, args.n_gen, 16):
            nb = min(16, args.n_gen - gi)
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
        # CLASSIFICATION (two probes: standard + deep)
        # ==================================================
        print(f"\n  --- CLASSIFICATION ---")
        mask_center_train = make_center_mask(1, K, H, W, device='cpu')
        z_train_rep_list = []
        for ri in range(0, len(z_train), 16):
            nb = min(16, len(z_train) - ri)
            z_batch = z_train[ri:ri+nb].to(device)
            m_batch = mask_center_train.expand(nb, -1, -1, -1).to(device)
            z_rep = repair_flow(step_fn, e_core, z_batch, m_batch, device,
                                T=args.T, dt=0.5)
            z_train_rep_list.append(z_rep.cpu())
        z_train_repaired = torch.cat(z_train_rep_list)

        # Standard ConvProbe
        probe = ConvProbe(N_BITS, H, W).to(device)
        train_probe_mixed(probe, z_train, z_train_repaired, train_y, device,
                          epochs=30, bs=128)
        acc_clean = eval_probe(probe, z_test, test_y, device)
        acc_repair = eval_probe(probe, z_test_repaired, test_y, device)
        gap = acc_clean - acc_repair
        result['acc_clean'] = acc_clean
        result['acc_repair'] = acc_repair
        result['gap'] = gap
        print(f"  [ConvProbe] acc_clean={acc_clean:.3f}  acc_repair={acc_repair:.3f}  gap={gap:.3f}")

        # Deep ConvProbe (may extract more from 32×32 grid)
        deep_probe = DeepConvProbe(N_BITS, H, W).to(device)
        train_probe_mixed(deep_probe, z_train, z_train_repaired, train_y, device,
                          epochs=30, bs=128)
        acc_clean_deep = eval_probe(deep_probe, z_test, test_y, device)
        acc_repair_deep = eval_probe(deep_probe, z_test_repaired, test_y, device)
        gap_deep = acc_clean_deep - acc_repair_deep
        result['acc_clean_deep'] = acc_clean_deep
        result['acc_repair_deep'] = acc_repair_deep
        result['gap_deep'] = gap_deep
        print(f"  [DeepProbe] acc_clean={acc_clean_deep:.3f}  acc_repair={acc_repair_deep:.3f}  gap={gap_deep:.3f}")

        all_results[cfg_name] = result

        del enc, dec, e_core, step_fn, probe, deep_probe
        torch.cuda.empty_cache()

    # ========== SUMMARY ==========
    print(f"\n{'='*100}")
    print("ICC-3 SUMMARY: MAXIMUM CLASSIFICATION PUSH")
    print(f"Real: HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")
    print(f"Target: 60%+ (supervised TinyCNN=61.6%)")
    print("=" * 100)

    # --- Spectral dashboard ---
    print(f"\n--- SPECTRAL DASHBOARD ---")
    sh = (f"{'config':<22} {'params':>8} {'cond':>8} {'part_r':>8} "
          f"{'dead%':>7} {'entropy':>8}")
    print(sh); print("-" * len(sh))
    for name, r in all_results.items():
        s = r['spectral']
        print(f"{name:<22} {r['n_params']:>8,} {s['condition_number']:>8.1f} "
              f"{s['participation_ratio']:>8.2f} "
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
    ch = f"{'config':<22} {'conv_cl':>7} {'conv_rp':>7} {'gap':>7} {'deep_cl':>8} {'deep_rp':>8} {'gap_d':>7}"
    print(ch); print("-" * len(ch))
    for name, r in all_results.items():
        print(f"{name:<22} {r['acc_clean']:>7.3f} {r['acc_repair']:>7.3f} {r['gap']:>7.3f} "
              f"{r['acc_clean_deep']:>8.3f} {r['acc_repair_deep']:>8.3f} {r['gap_deep']:>7.3f}")

    # --- Delta table ---
    a0 = all_results.get('ICC3A_resblock')
    if a0:
        print(f"\n--- DELTA vs ICC3A_resblock ---")
        dh = f"{'config':<22} {'Δconv':>7} {'Δdeep':>7} {'Δgap':>7} {'Δdiv':>7} {'Δdead':>7}"
        print(dh); print("-" * len(dh))
        for name, r in all_results.items():
            if name == 'ICC3A_resblock':
                continue
            print(f"{name:<22} "
                  f"{r['acc_clean']-a0['acc_clean']:>+7.3f} "
                  f"{r['acc_clean_deep']-a0['acc_clean_deep']:>+7.3f} "
                  f"{r['gap']-a0['gap']:>+7.3f} "
                  f"{r['gen']['diversity']-a0['gen']['diversity']:>+7.3f} "
                  f"{r['dead_bits']['dead_ratio']-a0['dead_bits']['dead_ratio']:>+7.3f}")

    # --- 60% gate check ---
    print(f"\n--- 60% TARGET CHECK ---")
    for name, r in all_results.items():
        best = max(r['acc_clean'], r['acc_clean_deep'])
        status = "✓ PASS" if best >= 0.60 else f"MISS ({best:.1%})"
        print(f"  {name}: best={best:.3f}  {status}")

    # Save results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump({n: {k: v for k, v in r.items()} for n, r in all_results.items()},
                  f, indent=2, default=str)

    print(f"\n{'='*100}")
    print("ICC-3 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
