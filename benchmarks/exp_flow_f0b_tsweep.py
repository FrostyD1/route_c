#!/usr/bin/env python3
"""
Flow F0b: T-Sweep — How Many Steps Does the Descent Operator Need?
====================================================================
F0 showed:
  - Flat flow converges at step 6/10 (余量)
  - U-Net flow converges at step 10/10 (刚好用完)
  - U-Net visually messier than flat despite better metrics

Hypothesis: U-Net needs more steps (T=20-50) to fully converge.
More steps → lower HF_noise, cleaner texture, while keeping diversity.

Design: Train ONCE (best config = UNet + energy hinge), sample with T=5,10,15,20,30,50.
Also test flat_flow at same T values for comparison.

This is cheap: training is shared, only sampling varies.

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

# Import everything from F0
from exp_flow_f0 import (
    dct2d, idct2d, get_freq_masks, decompose_bands, freq_scheduled_loss,
    per_band_energy_distance, hf_coherence_metric, hf_noise_index,
    connectedness_proxy, compute_diversity, save_grid,
    GumbelSigmoid, Encoder16, Decoder16,
    DiffEnergyCore, FlatStepFn, UNetStepFn, SelfAttention,
    quantize, soft_quantize, compute_e_core_grad,
    train_adc, encode_all, train_ecore,
    train_step_fn_base, train_step_fn_energy,
    evaluate
)


@torch.no_grad()
def sample_flow_detailed(step_fn, e_core, n, K, H, W, device,
                         T=10, dt=0.5, sigma_schedule='cosine',
                         use_langevin=True):
    """Flow sampling with detailed per-step trajectory logging."""
    u = torch.randn(n, K, H, W, device=device) * 0.5

    trajectory = {
        'e_core': [], 'e_soft': [],
        'u_mean': [], 'u_std': [],
        'delta_u_norm': [],
    }

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.tensor([t_frac], device=device).expand(n)

        e_grad = compute_e_core_grad(e_core, u)
        delta_u = step_fn(u, e_grad, t_tensor)

        # Track delta_u magnitude
        trajectory['delta_u_norm'].append(delta_u.abs().mean().item())

        u = u + dt * delta_u

        if use_langevin:
            sigma = 0.3 * np.cos(np.pi * step / (2 * T))
            if sigma > 0.01:
                u = u + sigma * torch.randn_like(u)

        z_cur = quantize(u)
        trajectory['e_core'].append(e_core.energy(z_cur).item())
        trajectory['e_soft'].append(e_core.soft_energy(u).item())
        trajectory['u_mean'].append(u.mean().item())
        trajectory['u_std'].append(u.std().item())

    z_final = quantize(u)
    return z_final, trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_flow_f0b')
    parser.add_argument('--n_samples', type=int, default=256)
    parser.add_argument('--n_bits', type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    N_BITS = args.n_bits

    print("=" * 100)
    print("FLOW F0b: T-SWEEP — CONVERGENCE STEPS")
    print("=" * 100)

    # ========== SHARED SETUP (same as F0) ==========
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

    print("\n[4] Encoding training set...")
    z_data = encode_all(encoder, train_x, device, bs=32)
    K, H, W = z_data.shape[1:]
    print(f"    z: {z_data.shape}, usage={z_data.mean():.3f}")

    print("\n[5] Training differentiable E_core...")
    e_core = DiffEnergyCore(N_BITS).to(device)
    train_ecore(e_core, z_data, device, epochs=15, bs=128)

    # ========== TRAIN BOTH MODELS ONCE ==========
    print("\n[6] Training flat step function...")
    torch.manual_seed(args.seed)
    flat_step = FlatStepFn(N_BITS).to(device)
    train_step_fn_base(flat_step, e_core, z_data, decoder, device, epochs=30, bs=32, T_unroll=3)

    print("\n[7] Training U-Net step function (energy hinge)...")
    torch.manual_seed(args.seed)
    unet_step = UNetStepFn(N_BITS, base_ch=48).to(device)
    train_step_fn_energy(unet_step, e_core, z_data, decoder, device, epochs=30, bs=32, T_unroll=3)

    # ========== T-SWEEP ==========
    T_values = [5, 10, 15, 20, 30, 50]
    models = {
        'flat_flow': flat_step,
        'unet_energy': unet_step,
    }

    all_results = {}

    for model_name, step_fn in models.items():
        for T in T_values:
            cfg_name = f"{model_name}_T{T}"
            print(f"\n{'='*80}")
            print(f"CONFIG: {cfg_name}")
            print("=" * 80)

            torch.manual_seed(args.seed + 100)
            gen_bs = 32
            z_gen_list = []; all_traj = []

            for gi in range(0, args.n_samples, gen_bs):
                nb = min(gen_bs, args.n_samples - gi)
                z_batch, traj = sample_flow_detailed(
                    step_fn, e_core, nb, K, H, W, device, T=T, dt=0.5)
                z_gen_list.append(z_batch.cpu())
                all_traj.append(traj)

            z_gen = torch.cat(z_gen_list)

            # Aggregate trajectory
            agg_traj = {
                key: [np.mean([t[key][s] for t in all_traj])
                      for s in range(len(all_traj[0][key]))]
                for key in all_traj[0].keys()
            }

            # Evaluate
            r, x_gen = evaluate(z_gen, decoder, encoder, e_core, z_data, test_x,
                                real_hf_coh, real_hf_noi, device, trajectory=agg_traj)
            r['oracle_mse'] = oracle_mse
            r['model'] = model_name
            r['T'] = T

            # Add trajectory details
            r['e_core_trajectory'] = agg_traj['e_core']
            r['delta_u_trajectory'] = agg_traj['delta_u_norm']
            r['final_delta_u'] = agg_traj['delta_u_norm'][-1]

            all_results[cfg_name] = r

            save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))

            print(f"    viol={r['violation']:.4f}  div={r['diversity']:.4f}  "
                  f"cycle={r['cycle']:.4f}  conn={r['connectedness']:.4f}")
            print(f"    HF_coh={r['hf_coherence']:.4f}  HF_noi={r['hf_noise_index']:.2f}")
            print(f"    E_gap: L={r['energy_gap_low']:.4f}  M={r['energy_gap_mid']:.4f}  "
                  f"H={r['energy_gap_high']:.4f}")
            if 'energy_mono_rate' in r:
                print(f"    mono={r['energy_mono_rate']:.3f}  E_drop={r.get('e_core_drop', 0):.4f}  "
                      f"converge={r.get('converge_step', '?')}")
            print(f"    final_delta_u={r['final_delta_u']:.6f}")

            # Print energy trajectory (sampled)
            et = agg_traj['e_core']
            step_indices = list(range(0, len(et), max(1, len(et)//8))) + [len(et)-1]
            step_indices = sorted(set(step_indices))
            traj_str = ' → '.join(f'{et[i]:.4f}' for i in step_indices)
            print(f"    E_core: {traj_str}")

            torch.cuda.empty_cache()

    # ========== SUMMARY ==========
    print(f"\n{'='*100}")
    print("T-SWEEP SUMMARY")
    print(f"Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}")
    print("=" * 100)

    for model_name in models.keys():
        print(f"\n--- {model_name} ---")
        header = f"{'T':>4} {'viol':>7} {'div':>7} {'cycle':>7} {'HFcoh':>7} {'HFnoi':>7} {'EgL':>7} {'EgH':>7} {'mono':>6} {'Edrop':>7} {'conv':>5} {'dU_end':>8}"
        print(header); print("-" * len(header))
        for T in T_values:
            cfg = f"{model_name}_T{T}"
            if cfg not in all_results: continue
            r = all_results[cfg]
            mono = r.get('energy_mono_rate', -1)
            edrop = r.get('e_core_drop', 0)
            conv = r.get('converge_step', -1)
            print(f"{T:>4} {r['violation']:>7.4f} {r['diversity']:>7.4f} "
                  f"{r['cycle']:>7.4f} {r['hf_coherence']:>7.4f} "
                  f"{r['hf_noise_index']:>7.2f} {r['energy_gap_low']:>7.4f} "
                  f"{r['energy_gap_high']:>7.4f} {mono:>6.3f} {edrop:>7.4f} "
                  f"{conv:>5} {r['final_delta_u']:>8.6f}")

    # Key comparison: T=10 vs T=30 vs T=50 for UNet
    print(f"\n{'='*100}")
    print("KEY COMPARISONS")
    print("=" * 100)

    for t_pair in [(10, 30), (10, 50), (30, 50)]:
        t1, t2 = t_pair
        k1, k2 = f"unet_energy_T{t1}", f"unet_energy_T{t2}"
        if k1 in all_results and k2 in all_results:
            r1, r2 = all_results[k1], all_results[k2]
            print(f"\nU-Net energy T={t1} → T={t2}:")
            print(f"  HF_noise: {r1['hf_noise_index']:.1f} → {r2['hf_noise_index']:.1f}")
            print(f"  diversity: {r1['diversity']:.4f} → {r2['diversity']:.4f}")
            print(f"  violation: {r1['violation']:.4f} → {r2['violation']:.4f}")
            print(f"  E_gap_H: {r1['energy_gap_high']:.4f} → {r2['energy_gap_high']:.4f}")

    # Save results (strip trajectory arrays for JSON)
    save_results = {}
    for k, v in all_results.items():
        sv = {kk: vv for kk, vv in v.items()
              if not isinstance(vv, list)}
        save_results[k] = sv
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n{'='*100}")
    print("F0b T-SWEEP COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
