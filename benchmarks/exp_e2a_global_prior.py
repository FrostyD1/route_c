#!/usr/bin/env python3
"""
E2a: Global Statistics Energy — Fixing "Same Color" Generation
===============================================================
Root cause diagnosis (GPT + Route C evidence):
  - E_core is local 3×3 MRF → only enforces local consistency
  - Homogeneous local MRF → lowest energy = "same phase" (constant field)
  - This produces same-color blocks in generation
  - Diffusion avoids this via learned score at large σ (global structure)

Fix: Add E_global(z) = task-agnostic global prior energy term.
Not a class label, not a task head — just statistical constraints on z.

Four approaches tested (increasing sophistication):
  A) marginal_prior: per-bit activation rate matching (cheapest, no learning)
  B) channel_stats: per-channel mean+var matching (captures color distribution)
  C) spatial_cov: cross-position covariance matching (captures global layout)
  D) learned_prior: small CNN that models log p(z) (most expressive)

Integration with flow: E_global gradient added to e_grad input of step_fn.
λ_global controls balance between E_core (local) and E_global (global).

Also combines best global prior with 24-bit bandwidth (G2 finding).

Key metrics:
  - HueVar: variance of dominant hue across generated samples (should be HIGH)
  - color_KL: KL divergence of per-channel activation rates (should be LOW)
  - Standard: violation, diversity, HF_noise, connectedness

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
    GumbelSigmoid, Encoder16, Decoder16,
    DiffEnergyCore,
    quantize, compute_e_core_grad,
    train_adc, encode_all, train_ecore,
    evaluate
)

from exp_flow_f0c_fixes import FlatStepFn_Norm, get_sigma


# ============================================================================
# GLOBAL PRIOR ENERGIES
# ============================================================================

class MarginalPrior(nn.Module):
    """Per-bit activation rate prior.
    E_global(z) = -Σ [p_i·log(σ(u_i)) + (1-p_i)·log(1-σ(u_i))]
    where p_i is the empirical activation rate at each bit position.
    """
    def __init__(self, z_data):
        super().__init__()
        # Compute per-position activation rate: [K, H, W]
        rates = z_data.float().mean(dim=0)
        rates = rates.clamp(0.01, 0.99)  # avoid log(0)
        self.register_buffer('rates', rates)

    def energy_grad(self, u):
        """Gradient of E_global w.r.t. u (in logit space)."""
        # E = -[p·log(σ(u)) + (1-p)·log(1-σ(u))]
        # dE/du = -(p - σ(u)) = σ(u) - p
        soft_z = torch.sigmoid(u)
        return soft_z - self.rates.unsqueeze(0)

    def energy(self, z):
        """Energy value for monitoring."""
        z_f = z.float().clamp(1e-6, 1 - 1e-6)
        bce = -(self.rates * torch.log(z_f) + (1 - self.rates) * torch.log(1 - z_f))
        return bce.mean()


class ChannelStatsPrior(nn.Module):
    """Per-channel mean + variance matching.
    Penalizes samples whose channel statistics deviate from real data.
    Operates per-sample (not per-batch).
    """
    def __init__(self, z_data):
        super().__init__()
        # Per-channel-position mean: [K, H, W]
        self.register_buffer('ch_mean', z_data.float().mean(dim=0))
        # Per-channel spatial mean + var: [K]
        ch_spatial_mean = z_data.float().mean(dim=[0, 2, 3])  # [K]
        ch_spatial_var = z_data.float().var(dim=[0, 2, 3])    # [K]
        self.register_buffer('target_mean', ch_spatial_mean)
        self.register_buffer('target_var', ch_spatial_var.clamp(min=0.01))

    def energy_grad(self, u):
        """Gradient encouraging channel stats to match real distribution."""
        soft_z = torch.sigmoid(u)
        B, K, H, W = soft_z.shape
        # Per-sample channel mean
        sample_mean = soft_z.mean(dim=[2, 3])  # [B, K]
        # Gradient: pull mean toward target
        mean_grad = (sample_mean - self.target_mean.unsqueeze(0))  # [B, K]
        # Broadcast back to spatial
        grad = mean_grad.unsqueeze(-1).unsqueeze(-1).expand_as(u) / (H * W)
        # Also penalize variance deviation
        sample_var = soft_z.var(dim=[2, 3])  # [B, K]
        var_diff = sample_var - self.target_var.unsqueeze(0)  # [B, K]
        # d(var)/du at position (i,j) ≈ 2·(z_ij - mean)·σ'(u_ij) / (HW)
        deviation = soft_z - sample_mean.unsqueeze(-1).unsqueeze(-1)
        sig_prime = soft_z * (1 - soft_z)
        var_grad = 2 * var_diff.unsqueeze(-1).unsqueeze(-1) * deviation * sig_prime / (H * W)
        return grad * sig_prime + 0.5 * var_grad

    def energy(self, z):
        z_f = z.float()
        sample_mean = z_f.mean(dim=[2, 3])
        mean_loss = ((sample_mean - self.target_mean) ** 2).mean()
        sample_var = z_f.var(dim=[2, 3])
        var_loss = ((sample_var - self.target_var) ** 2).mean()
        return mean_loss + 0.5 * var_loss


class SpatialCovPrior(nn.Module):
    """Cross-position covariance matching.
    Captures which spatial positions tend to co-activate.
    Uses low-rank approximation to keep memory manageable.
    """
    def __init__(self, z_data, rank=16):
        super().__init__()
        B, K, H, W = z_data.shape
        # Flatten spatial: [B, K*H*W]
        z_flat = z_data.float().reshape(B, -1)
        D = K * H * W
        # Compute mean
        self.register_buffer('z_mean', z_flat.mean(0))  # [D]
        # Low-rank SVD of centered data
        z_centered = z_flat - self.z_mean.unsqueeze(0)
        # Use random projection for efficiency
        if D > 512:
            # Random projection to rank dimensions
            proj = torch.randn(D, rank) / np.sqrt(rank)
            z_proj = z_centered @ proj  # [B, rank]
            # Covariance in projected space
            cov_proj = (z_proj.T @ z_proj) / B  # [rank, rank]
            self.register_buffer('proj', proj)
            self.register_buffer('cov_target', cov_proj)
            self.use_proj = True
        else:
            cov = (z_centered.T @ z_centered) / B
            self.register_buffer('cov_target', cov)
            self.use_proj = False
        self.K, self.H, self.W = K, H, W

    def energy_grad(self, u):
        """Approximate gradient via projected covariance deviation."""
        soft_z = torch.sigmoid(u)
        B, K, H, W = soft_z.shape
        z_flat = soft_z.reshape(B, -1)  # [B, D]
        z_centered = z_flat - self.z_mean.unsqueeze(0)

        if self.use_proj:
            z_proj = z_centered @ self.proj  # [B, rank]
            cov_sample = (z_proj.T @ z_proj) / B
            cov_diff = cov_sample - self.cov_target  # [rank, rank]
            # Gradient: d/dz_flat (||cov_sample - cov_target||^2)
            # ≈ 2/B * z_centered @ proj @ cov_diff @ proj^T
            grad_flat = 2.0 / B * z_centered @ self.proj @ cov_diff @ self.proj.T
        else:
            cov_sample = (z_centered.T @ z_centered) / B
            cov_diff = cov_sample - self.cov_target
            grad_flat = 2.0 / B * z_centered @ cov_diff

        # Chain rule: σ'(u)
        sig_prime = soft_z * (1 - soft_z)
        grad = grad_flat.reshape(B, K, H, W) * sig_prime
        return grad

    def energy(self, z):
        z_flat = z.float().reshape(z.shape[0], -1)
        z_centered = z_flat - self.z_mean.unsqueeze(0)
        if self.use_proj:
            z_proj = z_centered @ self.proj
            cov_sample = (z_proj.T @ z_proj) / z.shape[0]
            return ((cov_sample - self.cov_target) ** 2).mean()
        else:
            cov_sample = (z_centered.T @ z_centered) / z.shape[0]
            return ((cov_sample - self.cov_target) ** 2).mean()


class LearnedPrior(nn.Module):
    """Small CNN that models log p(z) directly.
    Trained on real z via maximum likelihood (minimize -log p(z_real)).
    """
    def __init__(self, n_bits):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),  # 16→8
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 8→1
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, z):
        """Returns -log p(z) ∝ energy."""
        return self.net(z.float()).squeeze(-1)

    def energy(self, z):
        return self.forward(z).mean()

    def energy_grad(self, u):
        """Gradient via autograd through soft z."""
        u_var = u.detach().requires_grad_(True)
        soft_z = torch.sigmoid(u_var)
        e = self.net(soft_z).sum()
        e.backward()
        return u_var.grad.detach()


def train_learned_prior(prior, z_data, device, epochs=20, bs=128):
    """Train learned prior via noise-contrastive estimation.
    Real z = low energy, shuffled z = high energy.
    """
    opt = torch.optim.Adam(prior.parameters(), lr=1e-3)
    N = len(z_data)

    for ep in tqdm(range(epochs), desc="Prior"):
        prior.train(); perm = torch.randperm(N)
        tl, nb = 0., 0
        for i in range(0, N, bs):
            z_real = z_data[perm[i:i+bs]].to(device)
            B = z_real.shape[0]
            # Generate fake by shuffling channels + adding noise
            z_fake = z_real[torch.randperm(B)]
            # Random bit flips (30% flip rate)
            flip_mask = (torch.rand_like(z_fake.float()) < 0.3).float()
            z_fake = (z_fake.float() * (1 - flip_mask) + (1 - z_fake.float()) * flip_mask)
            z_fake = (z_fake > 0.5).float()

            opt.zero_grad()
            e_real = prior(z_real)  # should be low
            e_fake = prior(z_fake)  # should be high
            # Contrastive: real energy < fake energy by margin
            loss = F.relu(e_real - e_fake + 1.0).mean() + 0.1 * e_real.mean()
            loss.backward(); opt.step()
            tl += loss.item(); nb += 1

    prior.eval()
    return tl / nb


# ============================================================================
# NEW METRICS
# ============================================================================

def hue_variance(x_gen):
    """Compute variance of dominant hue across generated images.
    Higher = more color diversity (good).
    """
    # Convert RGB to rough hue via atan2 on (G-B, R-mean)
    R, G, B = x_gen[:, 0], x_gen[:, 1], x_gen[:, 2]
    mean_rgb = x_gen.mean(dim=1)
    # Per-image mean color
    R_mean = R.mean(dim=[1, 2])  # [B]
    G_mean = G.mean(dim=[1, 2])
    B_mean = B.mean(dim=[1, 2])
    # Rough hue angle
    hue = torch.atan2(G_mean - B_mean, R_mean - (G_mean + B_mean) / 2)
    return hue.var().item()


def color_kl(x_gen, x_real):
    """KL divergence of per-channel color histograms.
    Lower = more realistic color distribution.
    """
    kl_total = 0.0
    for c in range(3):
        # Histogram (64 bins)
        gen_hist = torch.histc(x_gen[:, c].float(), bins=64, min=0, max=1) + 1e-8
        real_hist = torch.histc(x_real[:, c].float(), bins=64, min=0, max=1) + 1e-8
        gen_hist = gen_hist / gen_hist.sum()
        real_hist = real_hist / real_hist.sum()
        kl_total += (real_hist * (real_hist / gen_hist).log()).sum().item()
    return kl_total / 3


def activation_rate_kl(z_gen, z_real):
    """KL divergence of per-channel activation rates."""
    gen_rates = z_gen.float().mean(dim=[0, 2, 3]).clamp(1e-6, 1 - 1e-6)
    real_rates = z_real.float().mean(dim=[0, 2, 3]).clamp(1e-6, 1 - 1e-6)
    kl = (real_rates * (real_rates / gen_rates).log() +
          (1 - real_rates) * ((1 - real_rates) / (1 - gen_rates)).log())
    return kl.mean().item()


# ============================================================================
# FLOW SAMPLING WITH GLOBAL PRIOR
# ============================================================================

@torch.no_grad()
def sample_flow_global(step_fn, e_core, global_prior, n, K, H, W, device,
                       T=20, dt=0.5, sigma_schedule='cosine',
                       lambda_global=1.0, prior_type='marginal'):
    """Flow sampling with E_core + E_global guidance."""
    u = torch.randn(n, K, H, W, device=device) * 0.5

    trajectory = {'e_core': [], 'e_global': [], 'delta_u_norm': []}

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((n,), t_frac, device=device)

        # E_core gradient
        e_grad_core = compute_e_core_grad(e_core, u)

        # E_global gradient
        if prior_type == 'learned':
            # Learned prior needs grad mode
            with torch.enable_grad():
                e_grad_global = global_prior.energy_grad(u)
        else:
            e_grad_global = global_prior.energy_grad(u)

        # Combined gradient (with annealing: more global early, less late)
        # At start (t_frac=1.0): λ_global * 1.0
        # At end (t_frac→0): λ_global * 0.3
        lambda_t = lambda_global * (0.3 + 0.7 * t_frac)
        e_grad = e_grad_core + lambda_t * e_grad_global

        delta_u = step_fn(u, e_grad, t_tensor)
        trajectory['delta_u_norm'].append(delta_u.abs().mean().item())

        u = u + dt * delta_u

        sigma = get_sigma(sigma_schedule, step, T)
        if sigma > 0.01:
            u = u + sigma * torch.randn_like(u)

        z_cur = quantize(u)
        trajectory['e_core'].append(e_core.energy(z_cur).item())
        trajectory['e_global'].append(global_prior.energy(z_cur).item())

    return quantize(u), trajectory


# ============================================================================
# TRAINING (same as G2/C1)
# ============================================================================

def train_step_fn(step_fn, e_core, z_data, decoder, device,
                  epochs=30, bs=32, T_unroll=3, clip_grad=1.0):
    opt = torch.optim.Adam(step_fn.parameters(), lr=1e-3)
    N = len(z_data)

    for epoch in tqdm(range(epochs), desc="StepFn"):
        step_fn.train(); perm = torch.randperm(N)
        tl, fl, nb = 0., 0., 0
        progress = epoch / max(epochs - 1, 1)

        for i in range(0, N, bs):
            idx = perm[i:i+bs]; z_clean = z_data[idx].to(device); B = z_clean.shape[0]
            noise_level = torch.rand(B, device=device)
            u_clean = z_clean * 2.0 - 1.0
            u_noisy = u_clean + torch.randn_like(u_clean) * noise_level.view(B, 1, 1, 1) * 2.0
            opt.zero_grad()

            u = u_noisy
            for t_step in range(T_unroll):
                t_frac = 1.0 - t_step / T_unroll
                t_tensor = torch.full((B,), t_frac, device=device)
                e_grad = (torch.sigmoid(e_core.predictor(torch.sigmoid(u))) -
                          torch.sigmoid(u))
                delta_u = step_fn(u, e_grad, t_tensor)
                u = u + 0.5 * delta_u

            z_pred_soft = torch.sigmoid(u)
            loss_bce = F.binary_cross_entropy(z_pred_soft, z_clean)

            z_hard = (z_pred_soft > 0.5).float()
            z_ste = z_hard - z_pred_soft.detach() + z_pred_soft
            with torch.no_grad():
                x_clean = decoder(z_clean)
            x_pred = decoder(z_ste)
            loss_freq = freq_scheduled_loss(x_pred, x_clean, progress)

            loss = loss_bce + 0.3 * loss_freq
            loss.backward()
            nn.utils.clip_grad_norm_(step_fn.parameters(), clip_grad)
            opt.step()
            tl += loss_bce.item(); fl += loss_freq.item(); nb += 1

        if (epoch + 1) % 10 == 0:
            print(f"      ep {epoch+1}/{epochs}: BCE={tl/nb:.4f} freq={fl/nb:.4f}")

    step_fn.eval()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_e2a_global_prior')
    parser.add_argument('--n_samples', type=int, default=256)
    parser.add_argument('--T', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("E2a: GLOBAL STATISTICS ENERGY")
    print("=" * 100)

    # ========== DATA ==========
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
    real_hue_var = hue_variance(test_x[:200])
    print(f"    Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")
    save_grid(test_x[:64], os.path.join(args.output_dir, 'real_samples.png'))

    # ========== SHARED TRAINING ==========
    print("\n[3] Training shared infrastructure...")
    N_BITS = 16

    torch.manual_seed(args.seed)
    enc = Encoder16(N_BITS).to(device)
    dec = Decoder16(N_BITS).to(device)
    adc_loss = train_adc(enc, dec, train_x, device, epochs=40, bs=32)
    print(f"    ADC loss: {adc_loss:.4f}")

    z_data = encode_all(enc, train_x, device, bs=32)
    z_test = encode_all(enc, test_x, device, bs=32)
    K, H, W = z_data.shape[1:]
    print(f"    z: {z_data.shape}, usage={z_data.mean():.3f}")

    e_core = DiffEnergyCore(N_BITS).to(device)
    train_ecore(e_core, z_data, device, epochs=15, bs=128)

    step_fn = FlatStepFn_Norm(N_BITS).to(device)
    train_step_fn(step_fn, e_core, z_data, dec, device, epochs=30, bs=32, T_unroll=3)

    # ========== BUILD PRIORS ==========
    print("\n[4] Building global priors...")

    priors = {}
    priors['marginal'] = MarginalPrior(z_data).to(device)
    print(f"    marginal: E(real)={priors['marginal'].energy(z_data[:100].to(device)):.4f}")

    priors['channel_stats'] = ChannelStatsPrior(z_data).to(device)
    print(f"    channel_stats: E(real)={priors['channel_stats'].energy(z_data[:100].to(device)):.4f}")

    priors['spatial_cov'] = SpatialCovPrior(z_data, rank=16).to(device)
    print(f"    spatial_cov: E(real)={priors['spatial_cov'].energy(z_data[:100].to(device)):.4f}")

    prior_learned = LearnedPrior(N_BITS).to(device)
    train_learned_prior(prior_learned, z_data, device, epochs=20, bs=128)
    priors['learned'] = prior_learned
    print(f"    learned: E(real)={priors['learned'].energy(z_data[:100].to(device)):.4f}")

    # ========== BASELINE (no global prior) ==========
    all_results = {}

    print(f"\n{'='*80}")
    print("CONFIG: baseline (no global prior)")
    print("=" * 80)

    torch.manual_seed(args.seed + 100)
    z_gen_list, all_traj = [], []
    for gi in range(0, args.n_samples, 32):
        nb = min(32, args.n_samples - gi)
        # Standard sampling (no global prior)
        u = torch.randn(nb, K, H, W, device=device) * 0.5
        traj = {'e_core': [], 'delta_u_norm': []}
        for step in range(args.T):
            t_frac = 1.0 - step / args.T
            t_tensor = torch.full((nb,), t_frac, device=device)
            e_grad = compute_e_core_grad(e_core, u)
            delta_u = step_fn(u, e_grad, t_tensor)
            traj['delta_u_norm'].append(delta_u.abs().mean().item())
            u = u + 0.5 * delta_u
            sigma = get_sigma('cosine', step, args.T)
            if sigma > 0.01:
                u = u + sigma * torch.randn_like(u)
            z_cur = quantize(u)
            traj['e_core'].append(e_core.energy(z_cur).item())
        z_gen_list.append(quantize(u).cpu())
        all_traj.append(traj)
    z_gen = torch.cat(z_gen_list)

    agg_traj = {
        key: [np.mean([t[key][s] for t in all_traj])
              for s in range(len(all_traj[0][key]))]
        for key in all_traj[0].keys()
    }

    r, x_gen = evaluate(z_gen, dec, enc, e_core, z_data, test_x,
                        real_hf_coh, real_hf_noi, device, trajectory=agg_traj)
    r['hue_var'] = hue_variance(x_gen)
    r['color_kl'] = color_kl(x_gen, test_x[:len(x_gen)])
    r['act_rate_kl'] = activation_rate_kl(z_gen, z_data)
    all_results['baseline'] = r

    save_grid(x_gen, os.path.join(args.output_dir, 'gen_baseline.png'))
    print(f"    viol={r['violation']:.4f}  div={r['diversity']:.4f}  "
          f"HF_noise={r['hf_noise_index']:.2f}  hue_var={r['hue_var']:.4f}  "
          f"color_kl={r['color_kl']:.4f}")

    # ========== SWEEP PRIORS × LAMBDA ==========
    prior_names = ['marginal', 'channel_stats', 'spatial_cov', 'learned']
    lambdas = [0.3, 1.0, 3.0]

    for prior_name in prior_names:
        for lam in lambdas:
            cfg_name = f"{prior_name}_lam{lam}"
            print(f"\n{'='*80}")
            print(f"CONFIG: {cfg_name}")
            print("=" * 80)

            prior = priors[prior_name]
            p_type = 'learned' if prior_name == 'learned' else 'stats'

            torch.manual_seed(args.seed + 100)
            z_gen_list, all_traj = [], []
            for gi in range(0, args.n_samples, 32):
                nb = min(32, args.n_samples - gi)
                z_batch, traj = sample_flow_global(
                    step_fn, e_core, prior, nb, K, H, W, device,
                    T=args.T, dt=0.5, sigma_schedule='cosine',
                    lambda_global=lam, prior_type=p_type)
                z_gen_list.append(z_batch.cpu())
                all_traj.append(traj)
            z_gen = torch.cat(z_gen_list)

            agg_traj = {
                key: [np.mean([t[key][s] for t in all_traj])
                      for s in range(len(all_traj[0][key]))]
                for key in all_traj[0].keys()
            }

            r, x_gen = evaluate(z_gen, dec, enc, e_core, z_data, test_x,
                                real_hf_coh, real_hf_noi, device, trajectory=agg_traj)
            r['hue_var'] = hue_variance(x_gen)
            r['color_kl'] = color_kl(x_gen, test_x[:len(x_gen)])
            r['act_rate_kl'] = activation_rate_kl(z_gen, z_data)
            r['prior_type'] = prior_name
            r['lambda_global'] = lam
            if 'e_global' in agg_traj:
                r['e_global_start'] = agg_traj['e_global'][0]
                r['e_global_end'] = agg_traj['e_global'][-1]
            all_results[cfg_name] = r

            save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))
            print(f"    viol={r['violation']:.4f}  div={r['diversity']:.4f}  "
                  f"HF_noise={r['hf_noise_index']:.2f}  hue_var={r['hue_var']:.4f}  "
                  f"color_kl={r['color_kl']:.4f}  act_kl={r['act_rate_kl']:.4f}")

    # ========== SUMMARY ==========
    print(f"\n{'='*100}")
    print("E2a SUMMARY: GLOBAL PRIOR ENERGY")
    print(f"Real: HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")
    print("=" * 100)

    header = (f"{'config':<28} {'viol':>7} {'div':>7} {'HFnoi':>7} "
              f"{'hue_v':>7} {'col_kl':>7} {'act_kl':>7} {'conn':>7}")
    print(header); print("-" * len(header))
    for name, r in all_results.items():
        print(f"{name:<28} {r['violation']:>7.4f} {r['diversity']:>7.4f} "
              f"{r['hf_noise_index']:>7.2f} {r.get('hue_var', 0):>7.4f} "
              f"{r.get('color_kl', 0):>7.4f} {r.get('act_rate_kl', 0):>7.4f} "
              f"{r['connectedness']:>7.4f}")

    # Global prior effect analysis
    print(f"\n--- GLOBAL PRIOR EFFECT (vs baseline) ---")
    bl = all_results['baseline']
    for name, r in all_results.items():
        if name == 'baseline':
            continue
        hue_delta = r.get('hue_var', 0) - bl.get('hue_var', 0)
        col_delta = r.get('color_kl', 0) - bl.get('color_kl', 0)
        div_delta = r['diversity'] - bl['diversity']
        hf_delta = r['hf_noise_index'] - bl['hf_noise_index']
        print(f"  {name:<28} Δhue={hue_delta:>+.4f}  Δcol_kl={col_delta:>+.4f}  "
              f"Δdiv={div_delta:>+.4f}  ΔHF={hf_delta:>+.1f}")

    # Best config
    valid = {k: v for k, v in all_results.items()
             if k != 'baseline' and v['diversity'] > 0.1}
    if valid:
        best_hue = max(valid.items(), key=lambda kv: kv[1].get('hue_var', 0))
        best_col = min(valid.items(), key=lambda kv: kv[1].get('color_kl', 999))
        print(f"\n  Best HueVar: {best_hue[0]} ({best_hue[1].get('hue_var', 0):.4f})")
        print(f"  Best ColorKL: {best_col[0]} ({best_col[1].get('color_kl', 0):.4f})")

    # Save
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {kk: vv for kk, vv in v.items()
                           if not isinstance(vv, (list, np.ndarray))}
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n{'='*100}")
    print("E2a COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
