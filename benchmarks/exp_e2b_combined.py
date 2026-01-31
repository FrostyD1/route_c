#!/usr/bin/env python3
"""
E2b: Combined Fix — Spatial Covariance Prior + 24-bit Bandwidth
================================================================
Combines the two most impactful fixes found so far:
  1. G2: 24-bit/position → HF_noise drops from 921 to 231 (real=264)
  2. E2a: spatial_cov prior → HueVar jumps from 0.044 to 2.785 (real=2.44)

Each alone has tradeoffs:
  - 24-bit: great HF_noise, but same-color problem persists
  - spatial_cov: great color diversity, but HF_noise increases (399→542)

Hypothesis: Combined → both good HF_noise AND color diversity.

Tests:
  1. 24bit_baseline: 24-bit bandwidth, no global prior
  2. 24bit_spatial_cov_λ0.3: 24-bit + spatial_cov (mild)
  3. 24bit_spatial_cov_λ1.0: 24-bit + spatial_cov (medium)
  4. 16bit_spatial_cov_λ0.3: 16-bit + spatial_cov (reference from E2a)

Also: compare to best single-fix results.

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

# Import from G2 (Encoder/Decoder with variable n_bits)
from exp_g2_protocol_density import (
    Encoder16, Decoder16,
    train_adc, encode_all, train_ecore, train_step_fn
)

# Import from E2a (priors + metrics)
from exp_e2a_global_prior import (
    CovariancePrior as SpatialCovPrior,
    compute_hue_var,
    compute_marginal_kl
)


def hue_variance(x):
    """Wrapper: return scalar hue_var from compute_hue_var dict."""
    result = compute_hue_var(x)
    return result['hue_var']


def color_kl(x_gen, x_real):
    """KL divergence of per-channel mean distributions."""
    gen_means = x_gen.mean(dim=(2, 3))   # [N, C]
    real_means = x_real.mean(dim=(2, 3))  # [N, C]
    kl_total = 0.0
    for c in range(gen_means.shape[1]):
        g_hist = torch.histc(gen_means[:, c], bins=50, min=0, max=1) + 1e-8
        r_hist = torch.histc(real_means[:, c], bins=50, min=0, max=1) + 1e-8
        g_hist = g_hist / g_hist.sum()
        r_hist = r_hist / r_hist.sum()
        kl_total += (g_hist * (g_hist / r_hist).log()).sum().item()
    return kl_total / gen_means.shape[1]


def activation_rate_kl(z_gen, z_data):
    """KL divergence of per-position activation rates."""
    p_gen = z_gen.float().mean(dim=0).clamp(1e-6, 1 - 1e-6)
    p_data = z_data.float().mean(dim=0).clamp(1e-6, 1 - 1e-6)
    kl = p_data * (p_data / p_gen).log() + (1 - p_data) * ((1 - p_data) / (1 - p_gen)).log()
    return kl.mean().item()


# ============================================================================
# FLOW SAMPLING WITH GLOBAL PRIOR (from E2a, adapted for variable n_bits)
# ============================================================================

@torch.no_grad()
def sample_flow_combined(step_fn, e_core, global_prior, n, K, H, W, device,
                         T=20, dt=0.5, sigma_schedule='cosine',
                         lambda_global=0.0):
    """Flow sampling with optional global prior."""
    u = torch.randn(n, K, H, W, device=device) * 0.5
    trajectory = {'e_core': [], 'delta_u_norm': []}

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((n,), t_frac, device=device)
        e_grad_core = compute_e_core_grad(e_core, u)

        if global_prior is not None and lambda_global > 0:
            e_grad_global = global_prior.grad(u, device)
            lambda_t = lambda_global * (0.3 + 0.7 * t_frac)
            e_grad = e_grad_core + lambda_t * e_grad_global
        else:
            e_grad = e_grad_core

        delta_u = step_fn(u, e_grad, t_tensor)
        trajectory['delta_u_norm'].append(delta_u.abs().mean().item())
        u = u + dt * delta_u

        sigma = get_sigma(sigma_schedule, step, T)
        if sigma > 0.01:
            u = u + sigma * torch.randn_like(u)

        z_cur = quantize(u)
        trajectory['e_core'].append(e_core.energy(z_cur).item())

    return quantize(u), trajectory


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_e2b_combined')
    parser.add_argument('--n_samples', type=int, default=256)
    parser.add_argument('--T', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("E2b: COMBINED — 24-BIT BANDWIDTH + SPATIAL COV PRIOR")
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

    all_results = {}

    # ========== CONFIG SWEEP ==========
    configs = [
        ('16bit_baseline',           16, 0.0),
        ('16bit_spatial_cov_0.3',    16, 0.3),
        ('24bit_baseline',           24, 0.0),
        ('24bit_spatial_cov_0.3',    24, 0.3),
        ('24bit_spatial_cov_1.0',    24, 1.0),
        ('24bit_spatial_cov_3.0',    24, 3.0),
    ]

    for cfg_name, n_bits, lam in configs:
        print(f"\n{'='*80}")
        print(f"CONFIG: {cfg_name} (n_bits={n_bits}, λ_global={lam})")
        print("=" * 80)

        # Train ADC/DAC for this bit depth
        torch.manual_seed(args.seed)
        enc = Encoder16(n_bits).to(device)
        dec = Decoder16(n_bits).to(device)
        adc_loss = train_adc(enc, dec, train_x, device, epochs=40, bs=32)
        print(f"    ADC loss: {adc_loss:.4f}")

        with torch.no_grad():
            tb = test_x[:32].to(device); zo, _ = enc(tb)
            oracle_mse = F.mse_loss(dec(zo), tb).item()
        print(f"    Oracle MSE: {oracle_mse:.4f}")

        z_data = encode_all(enc, train_x, device, bs=32)
        K, H, W = z_data.shape[1:]
        print(f"    z: {z_data.shape}, usage={z_data.mean():.3f}")

        # Train E_core
        e_core = DiffEnergyCore(n_bits).to(device)
        train_ecore(e_core, z_data, device, epochs=15, bs=128)

        # Train StepFn
        step_fn = FlatStepFn_Norm(n_bits).to(device)
        train_step_fn(step_fn, e_core, z_data, dec, device, epochs=30, bs=32)

        # Build spatial covariance prior
        global_prior = None
        if lam > 0:
            global_prior = SpatialCovPrior(z_data)
            print(f"    Spatial cov prior built")

        # Generate
        torch.manual_seed(args.seed + 100)
        z_gen_list, all_traj = [], []
        for gi in range(0, args.n_samples, 32):
            nb = min(32, args.n_samples - gi)
            z_batch, traj = sample_flow_combined(
                step_fn, e_core, global_prior, nb, K, H, W, device,
                T=args.T, dt=0.5, lambda_global=lam)
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
        r['n_bits'] = n_bits
        r['lambda_global'] = lam
        r['oracle_mse'] = oracle_mse
        all_results[cfg_name] = r

        save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))
        print(f"    viol={r['violation']:.4f}  div={r['diversity']:.4f}  "
              f"HF_noise={r['hf_noise_index']:.2f}")
        print(f"    HueVar={r['hue_var']:.4f}  ColorKL={r['color_kl']:.4f}  "
              f"ActKL={r['act_rate_kl']:.4f}")
        print(f"    conn={r['connectedness']:.4f}  cycle={r['cycle']:.4f}")

        del enc, dec, e_core, step_fn, global_prior
        torch.cuda.empty_cache()

    # ========== SUMMARY ==========
    print(f"\n{'='*100}")
    print("E2b SUMMARY: COMBINED FIX (BANDWIDTH + GLOBAL PRIOR)")
    print(f"Real: HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")
    print("=" * 100)

    header = (f"{'config':<28} {'bits':>5} {'λ':>5} {'viol':>7} {'div':>7} "
              f"{'HFnoi':>7} {'hue_v':>7} {'col_kl':>7} {'conn':>7} {'cycle':>7}")
    print(header); print("-" * len(header))
    for name, r in all_results.items():
        print(f"{name:<28} {r['n_bits']:>5} {r['lambda_global']:>5.1f} "
              f"{r['violation']:>7.4f} {r['diversity']:>7.4f} "
              f"{r['hf_noise_index']:>7.2f} {r.get('hue_var', 0):>7.4f} "
              f"{r.get('color_kl', 0):>7.4f} {r['connectedness']:>7.4f} "
              f"{r['cycle']:>7.4f}")

    # Analysis: did the combination work?
    print(f"\n--- COMBINATION ANALYSIS ---")
    if '16bit_baseline' in all_results and '24bit_spatial_cov_0.3' in all_results:
        bl16 = all_results['16bit_baseline']
        combined = all_results['24bit_spatial_cov_0.3']
        print(f"  16bit_baseline:          HF_noise={bl16['hf_noise_index']:.1f}  "
              f"HueVar={bl16.get('hue_var', 0):.4f}")
        print(f"  24bit+spatial_cov(0.3):  HF_noise={combined['hf_noise_index']:.1f}  "
              f"HueVar={combined.get('hue_var', 0):.4f}")
        hf_improved = combined['hf_noise_index'] < bl16['hf_noise_index']
        hue_improved = combined.get('hue_var', 0) > bl16.get('hue_var', 0) * 5
        print(f"  HF_noise improved: {hf_improved}  "
              f"Color diversity improved: {hue_improved}")
        if hf_improved and hue_improved:
            print(f"  >>> COMBINATION WORKS: Both fixes are complementary! <<<")
        elif hf_improved:
            print(f"  >>> HF_noise fixed but color diversity not improved <<<")
        elif hue_improved:
            print(f"  >>> Color diversity fixed but HF_noise not improved <<<")
        else:
            print(f"  >>> Neither improved — fixes may be conflicting <<<")

    # Pareto frontier
    print(f"\n--- PARETO FRONTIER (closest to real on both metrics) ---")
    for name, r in sorted(all_results.items(),
                          key=lambda kv: (abs(kv[1]['hf_noise_index'] - real_hf_noi) +
                                          abs(kv[1].get('hue_var', 0) - real_hue_var) * 100)):
        hf_delta = r['hf_noise_index'] - real_hf_noi
        hue_delta = r.get('hue_var', 0) - real_hue_var
        print(f"  {name:<28} HF_noise={r['hf_noise_index']:>7.2f}(Δ={hf_delta:>+.1f})  "
              f"HueVar={r.get('hue_var', 0):.4f}(Δ={hue_delta:>+.4f})  "
              f"div={r['diversity']:.3f}")

    # Save
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {kk: vv for kk, vv in v.items()
                           if not isinstance(vv, (list, np.ndarray))}
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n{'='*100}")
    print("E2b COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
