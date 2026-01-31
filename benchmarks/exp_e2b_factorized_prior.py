#!/usr/bin/env python3
"""
E2b-light: Factorized Prior Sampling Start
============================================
Tests whether "same-hue" generation comes from the Gaussian noise starting point.

Instead of u ~ N(0, 0.25), start from a factorized prior:
  - z0 ~ Bernoulli(p_{i,c})  where p_{i,c} = empirical activation rate
  - Then u0 = z0 * 2 - 1 + small noise
  - Refine with flow (same Op-D operator)

Also tests mixture-of-factorized priors (K=8 clusters based on z statistics).

Configs:
  1. gaussian_start — baseline: u ~ N(0, 0.25)
  2. marginal_start — u from Bernoulli(p) + noise
  3. mixture_K8 — cluster z into 8 groups, sample z0 from cluster-specific marginals
  4. mixture_K16 — 16 clusters
  5. spatial_cov_start — combine factorized start + spatial_cov guidance (E2a winner)

4GB GPU: 5000 train, 1000 test, 16×16×16 z, Op-D energy-aware
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

# Import priors from E2a
from exp_e2a_global_prior import (
    CovariancePrior, compute_hue_var, compute_z_mmd, compute_marginal_kl
)


# ============================================================================
# FACTORIZED PRIOR: STARTING POINT GENERATORS
# ============================================================================

class FactorizedStart:
    """Sample z0 from per-position Bernoulli marginals."""
    def __init__(self, z_data):
        self.mu = z_data.float().mean(dim=0).clamp(0.01, 0.99)  # [K, H, W]

    def sample(self, n, device):
        mu = self.mu.to(device)
        z0 = (torch.rand(n, *mu.shape, device=device) < mu.unsqueeze(0)).float()
        u0 = z0 * 2.0 - 1.0 + torch.randn(n, *mu.shape, device=device) * 0.1
        return u0


class MixtureStart:
    """Sample z0 from mixture of K factorized priors.

    Clusters training z by global activation pattern, then samples
    from cluster-specific marginals.
    """
    def __init__(self, z_data, K=8, seed=42):
        N, C, H, W = z_data.shape
        self.K = K

        # Compute per-sample global features for clustering
        z_flat = z_data.float().view(N, -1)  # [N, CHW]
        # Simple k-means-like clustering on channel means
        z_ch_mean = z_data.float().mean(dim=(2, 3))  # [N, C]

        rng = np.random.default_rng(seed)
        # Initialize centroids
        idx = rng.choice(N, K, replace=False)
        centroids = z_ch_mean[idx]  # [K, C]

        # Run k-means for 20 iterations
        for _ in range(20):
            dists = torch.cdist(z_ch_mean, centroids)  # [N, K]
            labels = dists.argmin(dim=1)  # [N]
            for k in range(K):
                mask_k = (labels == k)
                if mask_k.sum() > 0:
                    centroids[k] = z_ch_mean[mask_k].mean(dim=0)

        self.labels = labels
        self.cluster_mu = []  # Per-cluster marginals
        self.cluster_sizes = []

        for k in range(K):
            mask_k = (labels == k)
            if mask_k.sum() > 0:
                mu_k = z_data[mask_k].float().mean(dim=0).clamp(0.01, 0.99)
                self.cluster_mu.append(mu_k)
                self.cluster_sizes.append(mask_k.sum().item())
            else:
                # Empty cluster: use global marginal
                self.cluster_mu.append(z_data.float().mean(dim=0).clamp(0.01, 0.99))
                self.cluster_sizes.append(0)

        # Cluster weights
        total = sum(self.cluster_sizes)
        self.weights = [s / total for s in self.cluster_sizes]

    def sample(self, n, device):
        # Sample cluster assignments
        clusters = np.random.choice(self.K, size=n, p=self.weights)

        u_list = []
        for i in range(n):
            k = clusters[i]
            mu_k = self.cluster_mu[k].to(device)
            z0 = (torch.rand_like(mu_k) < mu_k).float()
            u0 = z0 * 2.0 - 1.0 + torch.randn_like(mu_k) * 0.1
            u_list.append(u0.unsqueeze(0))

        return torch.cat(u_list, dim=0)


# ============================================================================
# FLOW SAMPLING WITH CONFIGURABLE START
# ============================================================================

@torch.no_grad()
def sample_flow_with_start(step_fn, e_core, start_fn, prior, n, K, H, W, device,
                            T=20, dt=0.5, sigma_schedule='cosine',
                            prior_weight=0.0, prior_warmup=0.3):
    """Flow sampling with configurable starting point and optional prior guidance."""
    if start_fn is not None:
        u = start_fn.sample(n, device)
    else:
        u = torch.randn(n, K, H, W, device=device) * 0.5

    trajectory = {'delta_u_norm': [], 'e_core': []}

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((n,), t_frac, device=device)
        e_grad = compute_e_core_grad(e_core, u)
        delta_u = step_fn(u, e_grad, t_tensor)

        trajectory['delta_u_norm'].append(delta_u.abs().mean().item())
        u = u + dt * delta_u

        # Optional prior guidance
        step_progress = step / T
        if prior is not None and prior_weight > 0 and step_progress >= prior_warmup:
            prior_grad = prior.grad(u, device)
            w = prior_weight * min(1.0, (step_progress - prior_warmup) / (1.0 - prior_warmup))
            u = u - w * prior_grad

        sigma = get_sigma(sigma_schedule, step, T)
        if sigma > 0.01:
            u = u + sigma * torch.randn_like(u)

        z_cur = quantize(u)
        trajectory['e_core'].append(e_core.energy(z_cur).item())

    z_final = quantize(u)
    return z_final, trajectory


# ============================================================================
# TRAINING (Op-D from C1)
# ============================================================================

def train_energy_mode(step_fn, e_core, z_data, decoder, device,
                      epochs=30, bs=32, T_unroll=3, clip_grad=1.0):
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
                e_grad = (torch.sigmoid(e_core.predictor(torch.sigmoid(u))) - torch.sigmoid(u))
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
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_e2b_factorized_prior')
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
    print("E2b-LIGHT: FACTORIZED PRIOR SAMPLING START")
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
    test_x = torch.stack([test_ds[i][0] for i in test_idx])
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

    # Reference metrics
    print("\n[5] Reference metrics...")
    x_ref = test_x[:args.n_gen].to(device)
    hf_coh_real = hf_coherence_metric(x_ref)
    hf_noise_real = hf_noise_index(x_ref)
    hv_real = compute_hue_var(test_x[:args.n_gen])
    print(f"    Real: HF_noise={hf_noise_real:.2f}  HueVar={hv_real['hue_var']:.6f}")

    # ========== TRAIN SHARED STEP FUNCTION ==========
    print("\n[6] Training shared step function (Op-D)...")
    step_fn = FlatStepFn_Norm(N_BITS).to(device)
    train_energy_mode(step_fn, e_core, z_train, decoder, device, epochs=30)

    # ========== BUILD STARTING POINTS ==========
    print("\n[7] Building sampling starts...")
    factorized = FactorizedStart(z_train)
    mixture_k8 = MixtureStart(z_train, K=8)
    mixture_k16 = MixtureStart(z_train, K=16)
    spatial_cov_prior = CovariancePrior(z_train)

    print(f"    Mixture K=8 cluster sizes: {mixture_k8.cluster_sizes}")
    print(f"    Mixture K=16 cluster sizes: {mixture_k16.cluster_sizes}")

    configs = {
        'gaussian_start': (None, None, 0.0),
        'marginal_start': (factorized, None, 0.0),
        'mixture_K8': (mixture_k8, None, 0.0),
        'mixture_K16': (mixture_k16, None, 0.0),
        'gaussian_plus_cov': (None, spatial_cov_prior, 0.3),
        'marginal_plus_cov': (factorized, spatial_cov_prior, 0.3),
        'mixture_K8_plus_cov': (mixture_k8, spatial_cov_prior, 0.3),
    }

    results = {}

    print("\n" + "=" * 80)
    print("SAMPLING EXPERIMENTS")
    print("=" * 80)

    for name, (start_fn, prior, pw) in configs.items():
        print(f"\n--- {name} ---")

        z_gen, traj = sample_flow_with_start(
            step_fn, e_core, start_fn, prior, args.n_gen, K, H, W, device,
            T=20, dt=0.5, prior_weight=pw, prior_warmup=0.3)

        metrics, x_gen = evaluate(z_gen, decoder, encoder, e_core,
                                   z_train[:args.n_gen], test_x[:args.n_gen],
                                   hf_coh_real, hf_noise_real, device,
                                   trajectory=traj)

        hv = compute_hue_var(x_gen)
        mmd = compute_z_mmd(z_gen.cpu(), z_train.cpu())
        mkl = compute_marginal_kl(z_gen.cpu(), z_train.cpu())

        metrics['hue_var'] = hv['hue_var']
        metrics['z_mmd'] = mmd
        metrics['marginal_kl'] = mkl
        metrics['final_delta_u'] = traj['delta_u_norm'][-1] if traj['delta_u_norm'] else 0.0

        results[name] = metrics

        print(f"    viol={metrics['violation']:.4f}  div={metrics['diversity']:.4f}  "
              f"conn={metrics['connectedness']:.3f}  HF_noise={metrics['hf_noise_index']:.0f}")
        print(f"    HueVar={hv['hue_var']:.6f}  z_MMD={mmd:.6f}  "
              f"margKL={mkl:.4f}  cycle={metrics['cycle']:.4f}")

        save_grid(x_gen[:64], os.path.join(args.output_dir, f'gen_{name}.png'))

    # ---- Save ----
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
    print("SUMMARY — E2b: Factorized Prior Start")
    print("=" * 100)

    print(f"\nReal: HF_noise={hf_noise_real:.0f}  HueVar={hv_real['hue_var']:.4f}")
    print(f"\n{'Config':>25} {'viol':>8} {'div':>8} {'HFnoi':>8} {'HueVar':>10} "
          f"{'z_MMD':>10} {'margKL':>8} {'cycle':>8} {'conn':>8}")
    print("-" * 110)

    for name in configs:
        m = results[name]
        print(f"{name:>25} {m['violation']:>8.4f} {m['diversity']:>8.4f} "
              f"{m['hf_noise_index']:>8.0f} {m['hue_var']:>10.4f} "
              f"{m['z_mmd']:>10.6f} {m['marginal_kl']:>8.4f} "
              f"{m['cycle']:>8.4f} {m['connectedness']:>8.3f}")

    # Effect vs baseline
    base = results['gaussian_start']
    print(f"\n--- Δ vs gaussian_start ---")
    for name in configs:
        if name == 'gaussian_start': continue
        m = results[name]
        print(f"  {name:>25}: ΔHue={m['hue_var']-base['hue_var']:>+.4f}  "
              f"Δdiv={m['diversity']-base['diversity']:>+.4f}  "
              f"ΔHF={m['hf_noise_index']-base['hf_noise_index']:>+.0f}  "
              f"ΔmargKL={m['marginal_kl']-base['marginal_kl']:>+.4f}")

    print("\nDONE.")


if __name__ == '__main__':
    main()
