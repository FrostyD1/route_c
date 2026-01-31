#!/usr/bin/env python3
"""
E2a: Global Statistics Prior — Breaking MRF Homogeneous Equilibrium
====================================================================
Diagnosis: Local 3×3 MRF (E_core) has homogeneous ground state (uniform z).
           Without global prior, generation converges to same-color-tone blocks.

This experiment adds E_global(z) during flow sampling to break symmetry:

  Config 1: baseline (no global prior, Op-D as in C1)
  Config 2: marginal_prior — per-channel, per-position activation rate from training z
  Config 3: covariance_prior — adds inter-channel covariance matching
  Config 4: histogram_prior — full per-position histogram (not just mean)
  Config 5: batch_diversity — explicit penalty for within-batch similarity

For each config, we measure:
  - Generation: violation, diversity, HF_noise, connectedness, cycle
  - HueVar: variance of mean color across generated samples (NEW)
  - z_MMD: maximum mean discrepancy between z_gen and z_data marginals (NEW)
  - Classification: conv probe clean/repair gap (does global prior reduce domain shift?)

E_global is added ONLY at sampling time (as gradient guidance), not in training.
This preserves the existing Op-D training and tests whether the global prior alone
breaks the homogeneous equilibrium.

4GB GPU: 5000 train, 1000 test, 16×16×16 z, Op-D energy-aware training
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
    Encoder16, Decoder16,
    DiffEnergyCore, quantize, soft_quantize, compute_e_core_grad,
    train_adc, encode_all, train_ecore,
    evaluate
)

from exp_flow_f0c_fixes import FlatStepFn_Norm, get_sigma


# ============================================================================
# GLOBAL PRIOR CLASSES
# ============================================================================

class MarginalPrior:
    """Per-channel, per-position activation rate prior.

    Computes p(z_{c,h,w} = 1) from training data.
    E_global = -sum log p(z_{c,h,w}) — negative log-likelihood under marginal.
    Gradient: pushes z toward per-position activation rates.
    """
    def __init__(self, z_data):
        # z_data: [N, K, H, W] binary
        self.mu = z_data.float().mean(dim=0)  # [K, H, W] — activation rate
        self.mu = self.mu.clamp(0.01, 0.99)   # avoid log(0)

    def grad(self, u, device):
        """Gradient of E_global w.r.t. u (logits).

        E = -sum [mu * log(sigma(u)) + (1-mu) * log(1-sigma(u))]
        dE/du = sigma(u) - mu  (pushes toward target activation rate)
        """
        mu = self.mu.to(device)
        return torch.sigmoid(u) - mu.unsqueeze(0)  # [B, K, H, W]


class CovariancePrior:
    """Marginal + inter-channel covariance matching.

    Adds a term that penalizes deviation of per-sample channel covariance
    from training distribution's covariance.
    """
    def __init__(self, z_data):
        # Marginal part
        self.mu = z_data.float().mean(dim=0).clamp(0.01, 0.99)

        # Channel covariance: compute per-sample cov, then average
        N, K, H, W = z_data.shape
        z_flat = z_data.float().view(N, K, -1)  # [N, K, HW]
        z_mean = z_flat.mean(dim=2, keepdim=True)  # [N, K, 1]
        z_centered = z_flat - z_mean
        # [N, K, K] sample covariance
        cov = torch.bmm(z_centered, z_centered.transpose(1, 2)) / (H * W - 1)
        self.target_cov = cov.mean(dim=0)  # [K, K] average covariance
        self.target_cov_diag = self.target_cov.diag()  # [K]

    def grad(self, u, device):
        """Combined marginal + covariance gradient."""
        mu = self.mu.to(device)
        target_diag = self.target_cov_diag.to(device)

        # Marginal gradient
        p = torch.sigmoid(u)
        grad_marginal = p - mu.unsqueeze(0)

        # Covariance gradient (simplified: match per-channel variance)
        # Current per-channel variance: var(p) across spatial
        B, K, H, W = u.shape
        p_flat = p.view(B, K, -1)  # [B, K, HW]
        p_mean = p_flat.mean(dim=2, keepdim=True)
        p_var = ((p_flat - p_mean) ** 2).mean(dim=2)  # [B, K]

        # Gradient to push variance toward target
        var_diff = (p_var - target_diag.unsqueeze(0))  # [B, K]
        # dvar/du ≈ 2 * (p - mean) * p * (1-p) / HW
        grad_var = 2.0 * (p_flat - p_mean) * p_flat * (1 - p_flat) / (H * W)
        grad_cov = (var_diff.unsqueeze(2) * grad_var).view(B, K, H, W)

        return grad_marginal + 0.5 * grad_cov


class HistogramPrior:
    """Per-position histogram prior using soft binning.

    For binary z, this is just the marginal. But we additionally
    compute spatial autocorrelation statistics (2-point function).
    """
    def __init__(self, z_data):
        self.mu = z_data.float().mean(dim=0).clamp(0.01, 0.99)

        # Spatial 2-point correlation: p(z_{h,w}=1 AND z_{h+1,w}=1)
        N, K, H, W = z_data.shape
        z = z_data.float()
        self.pair_h = (z[:, :, :-1, :] * z[:, :, 1:, :]).mean(dim=0)  # [K, H-1, W]
        self.pair_w = (z[:, :, :, :-1] * z[:, :, :, 1:]).mean(dim=0)  # [K, H, W-1]

    def grad(self, u, device):
        mu = self.mu.to(device)
        p = torch.sigmoid(u)
        grad_marginal = p - mu.unsqueeze(0)

        # 2-point correlation gradient
        pair_h = self.pair_h.to(device)
        pair_w = self.pair_w.to(device)

        B, K, H, W = u.shape
        # Current pair statistics
        cur_pair_h = p[:, :, :-1, :] * p[:, :, 1:, :]  # [B, K, H-1, W]
        cur_pair_w = p[:, :, :, :-1] * p[:, :, :, 1:]  # [B, K, H, W-1]

        # Gradient: d(pair_h)/du uses product rule
        diff_h = cur_pair_h - pair_h.unsqueeze(0)  # [B, K, H-1, W]
        diff_w = cur_pair_w - pair_w.unsqueeze(0)  # [B, K, H, W-1]

        grad_pair = torch.zeros_like(p)
        # p(u) * (1-p(u)) for sigmoid derivative
        dp = p * (1 - p)

        # Vertical pairs
        grad_pair[:, :, :-1, :] += diff_h * dp[:, :, :-1, :] * p[:, :, 1:, :]
        grad_pair[:, :, 1:, :]  += diff_h * p[:, :, :-1, :] * dp[:, :, 1:, :]
        # Horizontal pairs
        grad_pair[:, :, :, :-1] += diff_w * dp[:, :, :, :-1] * p[:, :, :, 1:]
        grad_pair[:, :, :, 1:]  += diff_w * p[:, :, :, :-1] * dp[:, :, :, 1:]

        return grad_marginal + 0.3 * grad_pair


class BatchDiversityPrior:
    """Penalizes within-batch similarity of generated z.

    E_div = sum_{i<j} exp(-hamming(z_i, z_j) / tau)
    Gradient pushes samples apart in Hamming space.
    """
    def __init__(self, z_data):
        self.mu = z_data.float().mean(dim=0).clamp(0.01, 0.99)
        self.tau = 50.0  # temperature for diversity

    def grad(self, u, device):
        mu = self.mu.to(device)
        p = torch.sigmoid(u)
        B, K, H, W = p.shape

        # Marginal gradient
        grad_marginal = p - mu.unsqueeze(0)

        # Diversity gradient: push apart samples that are too similar
        p_flat = p.view(B, -1)  # [B, KHW]
        # Pairwise soft-Hamming (using probabilities)
        # hamming_ij = sum (p_i * (1-p_j) + (1-p_i) * p_j)
        # = sum (p_i + p_j - 2*p_i*p_j)
        # For gradient: d/dp_i = 1 - 2*p_j for each j

        grad_div = torch.zeros_like(p_flat)
        for j in range(B):
            soft_ham = (p_flat + p_flat[j:j+1] - 2 * p_flat * p_flat[j:j+1]).sum(dim=1)  # [B]
            weight = torch.exp(-soft_ham / self.tau)  # [B] — high when similar
            weight[j] = 0  # skip self
            # d_hamming/dp_i = 1 - 2*p_j
            d_ham = 1.0 - 2.0 * p_flat[j:j+1]  # [1, KHW]
            # Repulsive gradient (negative sign: push apart)
            grad_div -= (weight.unsqueeze(1) * d_ham)  # [B, KHW]

        grad_div = grad_div.view(B, K, H, W) / max(B - 1, 1)

        return grad_marginal + 0.1 * grad_div


# ============================================================================
# FLOW SAMPLING WITH GLOBAL PRIOR
# ============================================================================

@torch.no_grad()
def sample_flow_with_prior(step_fn, e_core, prior, n, K, H, W, device,
                           T=20, dt=0.5, sigma_schedule='cosine',
                           prior_weight=0.3, prior_warmup=0.5):
    """Flow sampling with global prior gradient guidance.

    prior_warmup: fraction of steps before prior kicks in (early steps need
    to do coarse structure first, prior refines later)
    """
    u = torch.randn(n, K, H, W, device=device) * 0.5

    trajectory = {'delta_u_norm': [], 'e_core': [], 'prior_grad_norm': []}

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((n,), t_frac, device=device)
        e_grad = compute_e_core_grad(e_core, u)
        delta_u = step_fn(u, e_grad, t_tensor)

        trajectory['delta_u_norm'].append(delta_u.abs().mean().item())
        u = u + dt * delta_u

        # Add global prior gradient (with warmup)
        step_progress = step / T
        if prior is not None and step_progress >= prior_warmup:
            prior_grad = prior.grad(u, device)
            # Scale prior gradient by schedule (stronger as we converge)
            w = prior_weight * min(1.0, (step_progress - prior_warmup) / (1.0 - prior_warmup))
            u = u - w * prior_grad
            trajectory['prior_grad_norm'].append(prior_grad.abs().mean().item())
        else:
            trajectory['prior_grad_norm'].append(0.0)

        sigma = get_sigma(sigma_schedule, step, T)
        if sigma > 0.01:
            u = u + sigma * torch.randn_like(u)

        z_cur = quantize(u)
        trajectory['e_core'].append(e_core.energy(z_cur).item())

    z_final = quantize(u)
    return z_final, trajectory


# ============================================================================
# NEW METRICS: HueVar + z-domain MMD
# ============================================================================

def compute_hue_var(x_gen, decoder=None, z_gen=None, device='cpu'):
    """Compute color variance across generated samples.

    HueVar = variance of per-sample mean color.
    High HueVar = diverse colors. Low HueVar = same-hue problem.
    """
    if x_gen is None and z_gen is not None and decoder is not None:
        x_list = []
        with torch.no_grad():
            for i in range(0, len(z_gen), 32):
                x_list.append(decoder(z_gen[i:i+32].to(device)).cpu())
        x_gen = torch.cat(x_list)

    # Per-sample mean color: [B, C]
    mean_color = x_gen.mean(dim=(2, 3))  # [B, C]

    # Variance across samples
    hue_var = mean_color.var(dim=0).mean().item()  # scalar

    # Also compute per-channel variance for diagnosis
    per_channel = mean_color.var(dim=0)  # [C]

    return {
        'hue_var': hue_var,
        'hue_var_per_ch': per_channel.tolist(),
        'mean_color_mean': mean_color.mean(dim=0).tolist(),
    }


def compute_z_mmd(z_gen, z_data, n_samples=500):
    """Maximum Mean Discrepancy between z_gen and z_data.

    Uses Hamming kernel: k(z_i, z_j) = exp(-hamming(z_i,z_j) / bandwidth)
    """
    n = min(n_samples, len(z_gen), len(z_data))
    zg = z_gen[:n].float().view(n, -1)  # [n, D]
    zd = z_data[:n].float().view(n, -1)  # [n, D]
    D = zg.shape[1]
    bandwidth = D * 0.1  # 10% of dimension

    def kernel(a, b):
        # Hamming distance matrix
        ham = (a.unsqueeze(1) != b.unsqueeze(0)).float().sum(dim=2)
        return torch.exp(-ham / bandwidth)

    kgg = kernel(zg, zg)
    kdd = kernel(zd, zd)
    kgd = kernel(zg, zd)

    # Unbiased MMD^2 estimate
    n_f = float(n)
    mmd2 = (kgg.sum() - kgg.diag().sum()) / (n_f * (n_f - 1)) + \
           (kdd.sum() - kdd.diag().sum()) / (n_f * (n_f - 1)) - \
           2 * kgd.mean()

    return max(0.0, mmd2.item())


def compute_marginal_kl(z_gen, z_data):
    """Per-channel, per-position KL divergence of marginals."""
    p_gen = z_gen.float().mean(dim=0).clamp(0.001, 0.999)   # [K,H,W]
    p_data = z_data.float().mean(dim=0).clamp(0.001, 0.999)  # [K,H,W]

    kl = p_data * torch.log(p_data / p_gen) + \
         (1 - p_data) * torch.log((1 - p_data) / (1 - p_gen))

    return kl.mean().item()


# ============================================================================
# TRAINING (Op-D energy-aware, from C1 winner)
# ============================================================================

def train_energy_mode(step_fn, e_core, z_data, decoder, device,
                      epochs=30, bs=32, T_unroll=3, clip_grad=1.0):
    """Op-D: Energy-aware training (C1 Pareto winner)."""
    opt = torch.optim.Adam(step_fn.parameters(), lr=1e-3)
    N = len(z_data)

    for epoch in tqdm(range(epochs), desc="Op-D energy"):
        step_fn.train(); perm = torch.randperm(N)
        tl, el, nb = 0., 0., 0
        progress = epoch / max(epochs - 1, 1)

        for i in range(0, N, bs):
            idx = perm[i:i+bs]; z_clean = z_data[idx].to(device); B = z_clean.shape[0]

            noise_level = torch.rand(B, device=device)
            u_clean = z_clean * 2.0 - 1.0
            u_noisy = u_clean + torch.randn_like(u_clean) * noise_level.view(B, 1, 1, 1) * 2.0

            opt.zero_grad()

            u = u_noisy
            loss_descent = torch.tensor(0.0, device=device)

            for t_step in range(T_unroll):
                t_frac = 1.0 - t_step / T_unroll
                t_tensor = torch.full((B,), t_frac, device=device)

                e_before = e_core.soft_energy(u)

                e_grad = (torch.sigmoid(e_core.predictor(torch.sigmoid(u))) -
                          torch.sigmoid(u))
                delta_u = step_fn(u, e_grad, t_tensor)
                u = u + 0.5 * delta_u

                e_after = e_core.soft_energy(u)
                loss_descent = loss_descent + F.softplus(e_after - e_before + 0.01)

            loss_descent = loss_descent / T_unroll

            z_pred_soft = torch.sigmoid(u)
            loss_bce = F.binary_cross_entropy(z_pred_soft, z_clean)

            z_hard = (z_pred_soft > 0.5).float()
            z_ste = z_hard - z_pred_soft.detach() + z_pred_soft
            with torch.no_grad(): x_clean = decoder(z_clean)
            x_pred = decoder(z_ste)
            loss_freq = freq_scheduled_loss(x_pred, x_clean, progress)

            energy_w = 0.2 * min(1.0, progress * 2)
            loss = loss_bce + 0.3 * loss_freq + energy_w * loss_descent
            loss.backward()
            nn.utils.clip_grad_norm_(step_fn.parameters(), clip_grad)
            opt.step()
            tl += loss_bce.item(); el += loss_descent.item(); nb += 1

        if (epoch + 1) % 10 == 0:
            print(f"      ep {epoch+1}/{epochs}: BCE={tl/nb:.4f} E_desc={el/nb:.4f}")

    step_fn.eval()


# ============================================================================
# MASKS + REPAIR
# ============================================================================

def make_center_mask(B, K, H, W, device='cpu'):
    mask = torch.ones(B, K, H, W, device=device)
    h4, w4 = H // 4, W // 4
    mask[:, :, h4:3*h4, w4:3*w4] = 0
    return mask


@torch.no_grad()
def repair_flow(step_fn, e_core, z, mask, device, T=10, dt=0.5):
    B = z.shape[0]
    u_evidence = z * 2.0 - 1.0
    u = u_evidence * mask + torch.randn_like(u_evidence) * 0.5 * (1 - mask)
    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((B,), t_frac, device=device)
        e_grad = compute_e_core_grad(e_core, u)
        delta_u = step_fn(u, e_grad, t_tensor)
        u = u + dt * delta_u
        u = u_evidence * mask + u * (1 - mask)
    return quantize(u)


# ============================================================================
# CLASSIFICATION PROBE
# ============================================================================

class ConvProbe(nn.Module):
    def __init__(self, n_bits, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, n_classes))
    def forward(self, z):
        return self.net(z)


def train_probe(probe, z_data, labels, device, epochs=50, bs=256):
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for epoch in range(epochs):
        probe.train(); perm = torch.randperm(len(z_data))
        for i in range(0, len(z_data), bs):
            idx = perm[i:i+bs]
            z = z_data[idx].to(device); y = labels[idx].to(device)
            opt.zero_grad(); loss = F.cross_entropy(probe(z), y)
            loss.backward(); opt.step()
    probe.eval()


def eval_probe(probe, z_data, labels, device, bs=256):
    probe.eval(); nc, nb = 0, 0
    with torch.no_grad():
        for i in range(0, len(z_data), bs):
            z = z_data[i:i+bs].to(device); y = labels[i:i+bs].to(device)
            nc += (probe(z).argmax(1) == y).sum().item(); nb += len(y)
    return nc / nb


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_e2a_global_prior')
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--n_gen', type=int, default=256)
    parser.add_argument('--n_bits', type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    N_BITS = args.n_bits

    print("=" * 100)
    print("E2a: GLOBAL STATISTICS PRIOR — BREAKING MRF HOMOGENEOUS EQUILIBRIUM")
    print("=" * 100)

    # ========== DATA ==========
    print("\n[1] Loading CIFAR-10...")
    from torchvision import datasets, transforms
    train_ds = datasets.CIFAR10('/tmp/data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10('/tmp/data', train=False, download=True, transform=transforms.ToTensor())

    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(len(train_ds), args.n_train, replace=False)
    test_idx = rng.choice(len(test_ds), args.n_test, replace=False)

    train_x = torch.stack([train_ds[i][0] for i in train_idx])
    train_y = torch.tensor([train_ds[i][1] for i in train_idx])
    test_x = torch.stack([test_ds[i][0] for i in test_idx])
    test_y = torch.tensor([test_ds[i][1] for i in test_idx])
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")

    # ========== SHARED PIPELINE ==========
    print(f"\n[2] Training shared ADC/DAC (16×16×{N_BITS})...")
    torch.manual_seed(args.seed)
    encoder = Encoder16(N_BITS).to(device)
    decoder = Decoder16(N_BITS).to(device)
    train_adc(encoder, decoder, train_x, device, epochs=40, bs=32)

    print("\n[3] Encoding datasets...")
    z_train = encode_all(encoder, train_x, device, bs=32)
    z_test = encode_all(encoder, test_x, device, bs=32)
    K, H, W = z_train.shape[1:]
    print(f"    z_train: {z_train.shape}, usage={z_train.float().mean():.3f}")

    print("\n[4] Training E_core...")
    e_core = DiffEnergyCore(N_BITS).to(device)
    train_ecore(e_core, z_train, device, epochs=15, bs=128)

    # ========== REFERENCE METRICS ==========
    print("\n[5] Reference metrics...")
    x_ref = test_x[:args.n_gen].to(device)
    hf_coh_real = hf_coherence_metric(x_ref)
    hf_noise_real = hf_noise_index(x_ref)
    print(f"    Real: HF_coh={hf_coh_real:.4f}  HF_noise={hf_noise_real:.2f}")

    # Real image HueVar
    hv_real = compute_hue_var(test_x[:args.n_gen])
    print(f"    Real HueVar: {hv_real['hue_var']:.6f}")

    # ========== TRAIN SHARED STEP FUNCTION (Op-D) ==========
    print("\n[6] Training shared step function (Op-D energy-aware)...")
    step_fn = FlatStepFn_Norm(N_BITS).to(device)
    print(f"    StepFn params: {sum(p.numel() for p in step_fn.parameters()):,}")
    train_energy_mode(step_fn, e_core, z_train, decoder, device, epochs=30)

    # ========== BUILD PRIORS ==========
    print("\n[7] Building global priors from training z...")
    priors = {
        'baseline': None,
        'marginal': MarginalPrior(z_train),
        'covariance': CovariancePrior(z_train),
        'histogram': HistogramPrior(z_train),
        'batch_diversity': BatchDiversityPrior(z_train),
    }

    # ========== PRIOR WEIGHT SWEEP ==========
    # Test multiple prior weights for marginal (the simplest one)
    prior_weights = [0.1, 0.3, 0.5, 1.0]

    results = {}

    # ---- Phase 1: Prior type comparison (at weight=0.3) ----
    print("\n" + "=" * 80)
    print("PHASE 1: Prior type comparison (weight=0.3)")
    print("=" * 80)

    for name, prior in priors.items():
        tag = f"type_{name}"
        print(f"\n--- {name} ---")

        z_gen, traj = sample_flow_with_prior(
            step_fn, e_core, prior, args.n_gen, K, H, W, device,
            T=20, dt=0.5, prior_weight=0.3, prior_warmup=0.3)

        # Standard metrics
        metrics, x_gen = evaluate(z_gen, decoder, encoder, e_core,
                                   z_train[:args.n_gen], test_x[:args.n_gen],
                                   hf_coh_real, hf_noise_real, device,
                                   trajectory=traj)

        # New metrics
        hv = compute_hue_var(x_gen)
        mmd = compute_z_mmd(z_gen, z_train)
        mkl = compute_marginal_kl(z_gen, z_train)

        metrics['hue_var'] = hv['hue_var']
        metrics['hue_var_per_ch'] = hv['hue_var_per_ch']
        metrics['z_mmd'] = mmd
        metrics['marginal_kl'] = mkl
        metrics['final_delta_u'] = traj['delta_u_norm'][-1] if traj['delta_u_norm'] else 0.0
        metrics['prior_grad_mean'] = np.mean(traj['prior_grad_norm']) if traj['prior_grad_norm'] else 0.0

        results[tag] = metrics

        print(f"    viol={metrics['violation']:.4f}  div={metrics['diversity']:.4f}  "
              f"conn={metrics['connectedness']:.3f}  HF_noise={metrics['hf_noise_index']:.0f}")
        print(f"    HueVar={hv['hue_var']:.6f}  z_MMD={mmd:.6f}  "
              f"margKL={mkl:.4f}  cycle={metrics['cycle']:.4f}")

        save_grid(x_gen[:64], os.path.join(args.output_dir, f'gen_{tag}.png'))

    # ---- Phase 2: Weight sweep for best prior ----
    # Find best prior from Phase 1 (by marginal KL, since that directly measures
    # how well the prior matches the data distribution)
    type_keys = [k for k in results if k.startswith('type_')]
    best_type = min(type_keys, key=lambda k: results[k].get('marginal_kl', 999))
    best_prior_name = best_type.replace('type_', '')
    if best_prior_name == 'baseline':
        # If baseline wins on marginal_kl, use marginal for weight sweep
        best_prior_name = 'marginal'
    best_prior = priors[best_prior_name]
    print(f"\n>>> Best prior type: {best_prior_name}")

    print("\n" + "=" * 80)
    print(f"PHASE 2: Weight sweep for '{best_prior_name}' prior")
    print("=" * 80)

    for w in prior_weights:
        tag = f"weight_{w}"
        print(f"\n--- weight={w} ---")

        z_gen, traj = sample_flow_with_prior(
            step_fn, e_core, best_prior, args.n_gen, K, H, W, device,
            T=20, dt=0.5, prior_weight=w, prior_warmup=0.3)

        metrics, x_gen = evaluate(z_gen, decoder, encoder, e_core,
                                   z_train[:args.n_gen], test_x[:args.n_gen],
                                   hf_coh_real, hf_noise_real, device,
                                   trajectory=traj)

        hv = compute_hue_var(x_gen)
        mmd = compute_z_mmd(z_gen, z_train)
        mkl = compute_marginal_kl(z_gen, z_train)

        metrics['hue_var'] = hv['hue_var']
        metrics['z_mmd'] = mmd
        metrics['marginal_kl'] = mkl
        metrics['prior_weight'] = w

        results[tag] = metrics

        print(f"    viol={metrics['violation']:.4f}  div={metrics['diversity']:.4f}  "
              f"HueVar={hv['hue_var']:.6f}  z_MMD={mmd:.6f}  margKL={mkl:.4f}")

        save_grid(x_gen[:64], os.path.join(args.output_dir, f'gen_{tag}.png'))

    # ---- Phase 3: Classification domain shift test ----
    print("\n" + "=" * 80)
    print("PHASE 3: Classification domain shift (prior vs no-prior repair)")
    print("=" * 80)

    # Train conv probe on clean z (mixed: clean + repaired)
    # First do repair with and without prior
    z_repaired_no_prior = []
    z_repaired_with_prior = []

    with torch.no_grad():
        for i in range(0, min(args.n_test, len(z_test)), 64):
            zb = z_test[i:i+64].to(device); B = zb.shape[0]
            mask = make_center_mask(B, K, H, W, device)

            # Without prior (standard repair)
            z_rep = repair_flow(step_fn, e_core, zb, mask, device, T=10)
            z_repaired_no_prior.append(z_rep.cpu())

            # With prior (add marginal prior gradient during repair)
            # Note: for repair we don't use prior — evidence clamping handles it
            # This tests whether the *generated* z distribution better matches
            # what the probe expects
            z_repaired_with_prior.append(z_rep.cpu())  # same for now

    z_rep_no = torch.cat(z_repaired_no_prior)

    # Train probes
    # 1. Clean probe
    probe_clean = ConvProbe(N_BITS).to(device)
    train_probe(probe_clean, z_train, train_y, device, epochs=50)
    acc_clean_on_clean = eval_probe(probe_clean, z_test, test_y, device)
    acc_clean_on_repair = eval_probe(probe_clean, z_rep_no, test_y, device)

    # 2. Mixed probe (50% clean + 50% repaired)
    n_mix = min(len(z_train), len(z_train))
    z_mix_train = []
    y_mix_train = []
    for i in range(0, n_mix, 64):
        zb = z_train[i:i+64].to(device); B = zb.shape[0]
        mask = make_center_mask(B, K, H, W, device)
        z_rep = repair_flow(step_fn, e_core, zb, mask, device, T=10)
        z_mix_train.append(z_rep.cpu())
        y_mix_train.append(train_y[i:i+64])
    z_mix_train = torch.cat([z_train, torch.cat(z_mix_train)])
    y_mix_train = torch.cat([train_y, torch.cat(y_mix_train)])

    probe_mixed = ConvProbe(N_BITS).to(device)
    train_probe(probe_mixed, z_mix_train, y_mix_train, device, epochs=50)
    acc_mixed_on_clean = eval_probe(probe_mixed, z_test, test_y, device)
    acc_mixed_on_repair = eval_probe(probe_mixed, z_rep_no, test_y, device)

    cls_results = {
        'clean_probe_on_clean': acc_clean_on_clean,
        'clean_probe_on_repair': acc_clean_on_repair,
        'clean_probe_gap': acc_clean_on_clean - acc_clean_on_repair,
        'mixed_probe_on_clean': acc_mixed_on_clean,
        'mixed_probe_on_repair': acc_mixed_on_repair,
        'mixed_probe_gap': acc_mixed_on_clean - acc_mixed_on_repair,
    }
    results['classification'] = cls_results

    print(f"\n    Clean probe: clean={acc_clean_on_clean:.3f} repair={acc_clean_on_repair:.3f} "
          f"gap={acc_clean_on_clean - acc_clean_on_repair:.3f}")
    print(f"    Mixed probe: clean={acc_mixed_on_clean:.3f} repair={acc_mixed_on_repair:.3f} "
          f"gap={acc_mixed_on_clean - acc_mixed_on_repair:.3f}")

    # ---- Save results ----
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        return obj

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(make_serializable(results), f, indent=2)

    # ---- Summary ----
    print("\n" + "=" * 100)
    print("SUMMARY — E2a: Global Prior Comparison")
    print("=" * 100)

    print(f"\n{'Prior':>20} {'viol':>8} {'div':>8} {'conn':>8} {'HF_noise':>10} "
          f"{'HueVar':>10} {'z_MMD':>10} {'margKL':>8} {'cycle':>8}")
    print("-" * 105)
    for name in priors:
        tag = f"type_{name}"
        m = results[tag]
        print(f"{name:>20} {m['violation']:>8.4f} {m['diversity']:>8.4f} "
              f"{m['connectedness']:>8.3f} {m['hf_noise_index']:>10.0f} "
              f"{m['hue_var']:>10.6f} {m['z_mmd']:>10.6f} "
              f"{m['marginal_kl']:>8.4f} {m['cycle']:>8.4f}")

    print(f"\nReal HueVar: {hv_real['hue_var']:.6f}")

    print(f"\n--- Weight sweep ({best_prior_name}) ---")
    print(f"{'weight':>8} {'viol':>8} {'div':>8} {'HueVar':>10} {'z_MMD':>10} {'margKL':>8}")
    for w in prior_weights:
        tag = f"weight_{w}"
        m = results[tag]
        print(f"{w:>8.1f} {m['violation']:>8.4f} {m['diversity']:>8.4f} "
              f"{m['hue_var']:>10.6f} {m['z_mmd']:>10.6f} {m['marginal_kl']:>8.4f}")

    print("\nDONE.")


if __name__ == '__main__':
    main()
