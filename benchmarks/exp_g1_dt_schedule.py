#!/usr/bin/env python3
"""
G1-lite: dt Schedule Sweep
===========================
All prior experiments use dt=0.5 (never swept).
This experiment systematically tests dt × T × σ_schedule combinations
for both generation and repair quality.

Key question: Is dt=0.5 optimal, or is there a better dt/T trade-off?

Configs:
  Phase 1: dt sweep at fixed T=20 (gen-first operator)
    dt ∈ {0.1, 0.25, 0.5, 0.75, 1.0}
  Phase 2: T sweep at best dt (find convergence sweet spot)
    T ∈ {5, 10, 15, 20, 30, 50}
  Phase 3: dt-schedule (adaptive dt that decreases over steps)
    linear_decay: dt_t = dt_max * (1 - t/T)
    cosine_decay: dt_t = dt_max * cos(π*t / 2T)
    warmup: dt_t = dt_max * min(1, 2t/T)

Metrics: generation (violation, diversity, HF_noise, cycle, conn, delta_u)
         repair (hamming_masked, hamming_unmasked, cycle_repair)

Uses Op-D (energy-aware, C1 Pareto winner) training.
4GB GPU: 3000 train, 500 test, 16×16×16 z
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
# DT SCHEDULES
# ============================================================================

def get_dt(schedule, step, T, dt_max=0.5):
    """Compute dt for given schedule at step t."""
    if schedule == 'constant':
        return dt_max
    elif schedule == 'linear_decay':
        return dt_max * (1.0 - step / T)
    elif schedule == 'cosine_decay':
        return dt_max * np.cos(np.pi * step / (2 * T))
    elif schedule == 'warmup':
        return dt_max * min(1.0, 2.0 * step / T)
    elif schedule == 'warmup_decay':
        # Warmup first 1/4, then linear decay
        if step < T // 4:
            return dt_max * (4.0 * step / T)
        else:
            return dt_max * (1.0 - (step - T//4) / (T - T//4))
    else:
        return dt_max


# ============================================================================
# FLOW SAMPLING WITH CONFIGURABLE DT
# ============================================================================

@torch.no_grad()
def sample_flow_gen_dt(step_fn, e_core, n, K, H, W, device,
                       T=20, dt=0.5, dt_schedule='constant',
                       sigma_schedule='cosine'):
    """Flow sampling with configurable dt schedule."""
    u = torch.randn(n, K, H, W, device=device) * 0.5

    trajectory = {'delta_u_norm': [], 'e_core': [], 'dt_used': []}

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((n,), t_frac, device=device)
        e_grad = compute_e_core_grad(e_core, u)
        delta_u = step_fn(u, e_grad, t_tensor)

        # Get dt for this step
        dt_t = get_dt(dt_schedule, step, T, dt_max=dt)
        trajectory['delta_u_norm'].append(delta_u.abs().mean().item())
        trajectory['dt_used'].append(dt_t)

        u = u + dt_t * delta_u

        sigma = get_sigma(sigma_schedule, step, T)
        if sigma > 0.01:
            u = u + sigma * torch.randn_like(u)

        z_cur = quantize(u)
        trajectory['e_core'].append(e_core.energy(z_cur).item())

    z_final = quantize(u)
    return z_final, trajectory


@torch.no_grad()
def repair_flow_dt(step_fn, e_core, z, mask, device,
                   T=10, dt=0.5, dt_schedule='constant'):
    """Repair with configurable dt schedule."""
    B = z.shape[0]
    u_evidence = z * 2.0 - 1.0
    u = u_evidence * mask + torch.randn_like(u_evidence) * 0.5 * (1 - mask)

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((B,), t_frac, device=device)
        e_grad = compute_e_core_grad(e_core, u)
        delta_u = step_fn(u, e_grad, t_tensor)
        dt_t = get_dt(dt_schedule, step, T, dt_max=dt)
        u = u + dt_t * delta_u
        u = u_evidence * mask + u * (1 - mask)

    return quantize(u)


# ============================================================================
# MASKS
# ============================================================================

def make_center_mask(B, K, H, W, device='cpu'):
    mask = torch.ones(B, K, H, W, device=device)
    h4, w4 = H // 4, W // 4
    mask[:, :, h4:3*h4, w4:3*w4] = 0
    return mask

def make_training_mask(B, K, H, W, device='cpu'):
    n_center = B // 2
    masks = []
    if n_center > 0:
        masks.append(make_center_mask(n_center, K, H, W, device))
    n_random = B - n_center
    if n_random > 0:
        m = (torch.rand(n_random, 1, H, W, device=device) < 0.5).float()
        masks.append(m.expand(n_random, K, H, W))
    return torch.cat(masks, dim=0)


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
# EVALUATION
# ============================================================================

def eval_generation(step_fn, e_core, decoder, encoder, z_train, x_test,
                    hf_coh_real, hf_noise_real, device,
                    T, dt, dt_schedule, sigma_schedule, n_gen=256, tag=''):
    """Evaluate generation quality at given dt/T/schedule."""
    K, H, W = z_train.shape[1:]

    z_gen, traj = sample_flow_gen_dt(
        step_fn, e_core, n_gen, K, H, W, device,
        T=T, dt=dt, dt_schedule=dt_schedule, sigma_schedule=sigma_schedule)

    metrics, _ = evaluate(z_gen, decoder, encoder, e_core, z_train[:n_gen].to(device),
                           x_test[:n_gen].to(device), hf_coh_real, hf_noise_real, device)

    # Add trajectory info
    metrics['final_delta_u'] = traj['delta_u_norm'][-1] if traj['delta_u_norm'] else 0.0
    metrics['converge_step'] = next(
        (i for i, d in enumerate(traj['delta_u_norm']) if d < 1.0), T)
    metrics['e_core_start'] = traj['e_core'][0] if traj['e_core'] else 0.0
    metrics['e_core_end'] = traj['e_core'][-1] if traj['e_core'] else 0.0
    metrics['e_core_drop'] = metrics['e_core_start'] - metrics['e_core_end']

    # Effective total step size
    metrics['total_dt'] = sum(traj['dt_used'])

    return metrics


def eval_repair(step_fn, e_core, z_test, device,
                T, dt, dt_schedule, bs=64):
    """Evaluate repair quality at given dt/T/schedule."""
    K, H, W = z_test.shape[1:]
    z_repaired_list = []
    with torch.no_grad():
        for i in range(0, len(z_test), bs):
            z_batch = z_test[i:i+bs].to(device)
            B = z_batch.shape[0]
            mask = make_center_mask(B, K, H, W, device)
            z_rep = repair_flow_dt(step_fn, e_core, z_batch, mask, device,
                                   T=T, dt=dt, dt_schedule=dt_schedule)
            z_repaired_list.append(z_rep.cpu())
    z_repaired = torch.cat(z_repaired_list)

    mask_cpu = make_center_mask(len(z_test), K, H, W)
    masked = (mask_cpu == 0).float()
    unmasked = (mask_cpu == 1).float()

    ham_masked = ((z_repaired != z_test).float() * masked).sum() / masked.sum()
    ham_unmasked = ((z_repaired != z_test).float() * unmasked).sum() / unmasked.sum()

    # Cycle: encode→decode→encode stability of repaired z
    return {
        'hamming_masked': ham_masked.item(),
        'hamming_unmasked': ham_unmasked.item(),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_samples', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    n_gen = args.n_samples
    out_dir = os.path.join(PROJECT_ROOT, 'outputs', 'exp_g1_dt_schedule')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 100)
    print("G1-LITE: DT SCHEDULE SWEEP")
    print("=" * 100)

    # ---- Load CIFAR-10 ----
    print("\n[1] Loading CIFAR-10...")
    from torchvision import datasets, transforms
    tf = transforms.ToTensor()
    ds_train = datasets.CIFAR10('/tmp/data', train=True, download=True, transform=tf)
    ds_test = datasets.CIFAR10('/tmp/data', train=False, download=True, transform=tf)

    N_TRAIN, N_TEST = 3000, 500
    x_train = torch.stack([ds_train[i][0] for i in range(N_TRAIN)])
    y_train = torch.tensor([ds_train[i][1] for i in range(N_TRAIN)])
    x_test = torch.stack([ds_test[i][0] for i in range(N_TEST)])
    y_test = torch.tensor([ds_test[i][1] for i in range(N_TEST)])
    print(f"    Train: {x_train.shape}, Test: {x_test.shape}")

    # ---- Reference HF metrics ----
    print("\n[2] Reference metrics...")
    x_ref = x_test[:n_gen].to(device)
    hf_coh_real = hf_coherence_metric(x_ref)
    hf_noise_real = hf_noise_index(x_ref)
    print(f"    Real: HF_coh={hf_coh_real:.4f}  HF_noise={hf_noise_real:.2f}")

    # ---- Train shared ADC + E_core + StepFn ----
    print("\n[3] Training shared pipeline...")
    K = 16
    encoder = Encoder16(in_ch=3, n_bits=K).to(device)
    decoder = Decoder16(out_ch=3, n_bits=K).to(device)

    train_adc(encoder, decoder, x_train.to(device), device, epochs=40)
    z_train = encode_all(encoder, x_train.to(device), device)
    z_test_enc = encode_all(encoder, x_test.to(device), device)
    print(f"    z: {z_train.shape}, usage={z_train.float().mean():.3f}")

    e_core = DiffEnergyCore(K).to(device)
    train_ecore(e_core, z_train, device, epochs=15)

    step_fn = FlatStepFn_Norm(K).to(device)
    print(f"    StepFn params: {sum(p.numel() for p in step_fn.parameters()):,}")

    train_energy_mode(step_fn, e_core, z_train, decoder, device, epochs=30)

    # ---- Collect results ----
    results = {}

    # ================================================================
    # PHASE 1: dt sweep at T=20
    # ================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: dt sweep (T=20, σ=cosine, dt_schedule=constant)")
    print("=" * 80)

    dt_values = [0.1, 0.25, 0.5, 0.75, 1.0]
    T_fixed = 20

    for dt in dt_values:
        tag = f"dt{dt}"
        print(f"\n--- dt={dt}, T={T_fixed} ---")

        gen = eval_generation(step_fn, e_core, decoder, encoder, z_train, x_test,
                              hf_coh_real, hf_noise_real, device,
                              T=T_fixed, dt=dt, dt_schedule='constant',
                              sigma_schedule='cosine', n_gen=n_gen, tag=tag)

        rep = eval_repair(step_fn, e_core, z_test_enc, device,
                          T=T_fixed // 2, dt=dt, dt_schedule='constant')

        results[tag] = {'gen': gen, 'repair': rep}
        print(f"    GEN: viol={gen['violation']:.4f}  div={gen['diversity']:.4f}  "
              f"conn={gen['connectedness']:.3f}  HF_noise={gen['hf_noise_index']:.0f}  "
              f"delta_u={gen['final_delta_u']:.1f}  total_dt={gen['total_dt']:.1f}")
        print(f"    REP: ham_masked={rep['hamming_masked']:.4f}  "
              f"ham_unmasked={rep['hamming_unmasked']:.4f}")

        # Save grid for best candidates
        K_z, H_z, W_z = z_train.shape[1:]
        z_vis, _ = sample_flow_gen_dt(step_fn, e_core, 64, K_z, H_z, W_z, device,
                                      T=T_fixed, dt=dt, dt_schedule='constant',
                                      sigma_schedule='cosine')
        with torch.no_grad():
            x_vis = decoder(z_vis.to(device))
        save_grid(x_vis, os.path.join(out_dir, f'gen_{tag}.png'))

    # Find best dt from Phase 1
    best_dt_tag = min(results.keys(),
                      key=lambda k: results[k]['gen']['violation'])
    best_dt = float(best_dt_tag.replace('dt', ''))
    print(f"\n>>> Best dt (by violation): {best_dt}")

    # ================================================================
    # PHASE 2: T sweep at best dt
    # ================================================================
    print("\n" + "=" * 80)
    print(f"PHASE 2: T sweep (dt={best_dt}, σ=cosine, dt_schedule=constant)")
    print("=" * 80)

    T_values = [5, 10, 15, 20, 30, 50]

    for T in T_values:
        tag = f"T{T}_dt{best_dt}"
        print(f"\n--- T={T}, dt={best_dt} ---")

        gen = eval_generation(step_fn, e_core, decoder, encoder, z_train, x_test,
                              hf_coh_real, hf_noise_real, device,
                              T=T, dt=best_dt, dt_schedule='constant',
                              sigma_schedule='cosine', n_gen=n_gen, tag=tag)

        rep = eval_repair(step_fn, e_core, z_test_enc, device,
                          T=max(T // 2, 3), dt=best_dt, dt_schedule='constant')

        results[tag] = {'gen': gen, 'repair': rep}
        print(f"    GEN: viol={gen['violation']:.4f}  div={gen['diversity']:.4f}  "
              f"conn={gen['connectedness']:.3f}  HF_noise={gen['hf_noise_index']:.0f}  "
              f"delta_u={gen['final_delta_u']:.1f}  converge={gen['converge_step']}")
        print(f"    REP: ham_masked={rep['hamming_masked']:.4f}  "
              f"ham_unmasked={rep['hamming_unmasked']:.4f}")

    # Find best T from Phase 2
    phase2_keys = [k for k in results if k.startswith('T')]
    best_T_tag = min(phase2_keys,
                     key=lambda k: results[k]['gen']['violation'])
    best_T = int(best_T_tag.split('_')[0].replace('T', ''))
    print(f"\n>>> Best T (by violation): {best_T}")

    # ================================================================
    # PHASE 3: dt schedules at best dt × best T
    # ================================================================
    print("\n" + "=" * 80)
    print(f"PHASE 3: dt schedules (dt_max={best_dt}, T={best_T})")
    print("=" * 80)

    dt_schedules = ['constant', 'linear_decay', 'cosine_decay', 'warmup', 'warmup_decay']

    for sched in dt_schedules:
        tag = f"sched_{sched}"
        print(f"\n--- schedule={sched}, dt_max={best_dt}, T={best_T} ---")

        gen = eval_generation(step_fn, e_core, decoder, encoder, z_train, x_test,
                              hf_coh_real, hf_noise_real, device,
                              T=best_T, dt=best_dt, dt_schedule=sched,
                              sigma_schedule='cosine', n_gen=n_gen, tag=tag)

        rep = eval_repair(step_fn, e_core, z_test_enc, device,
                          T=max(best_T // 2, 3), dt=best_dt, dt_schedule=sched)

        results[tag] = {'gen': gen, 'repair': rep}
        print(f"    GEN: viol={gen['violation']:.4f}  div={gen['diversity']:.4f}  "
              f"conn={gen['connectedness']:.3f}  HF_noise={gen['hf_noise_index']:.0f}  "
              f"delta_u={gen['final_delta_u']:.1f}  total_dt={gen['total_dt']:.1f}")
        print(f"    REP: ham_masked={rep['hamming_masked']:.4f}  "
              f"ham_unmasked={rep['hamming_unmasked']:.4f}")

        z_vis, _ = sample_flow_gen_dt(step_fn, e_core, 64, K_z, H_z, W_z, device,
                                      T=best_T, dt=best_dt, dt_schedule=sched,
                                      sigma_schedule='cosine')
        with torch.no_grad():
            x_vis = decoder(z_vis.to(device))
        save_grid(x_vis, os.path.join(out_dir, f'gen_{tag}.png'))

    # ================================================================
    # PHASE 4: σ schedule interaction with best dt/T
    # ================================================================
    print("\n" + "=" * 80)
    print(f"PHASE 4: σ schedule sweep (dt_max={best_dt}, T={best_T}, best dt_sched)")
    print("=" * 80)

    # Find best dt_schedule from Phase 3
    phase3_keys = [k for k in results if k.startswith('sched_')]
    best_sched_tag = min(phase3_keys,
                         key=lambda k: results[k]['gen']['violation'])
    best_sched = best_sched_tag.replace('sched_', '')
    print(f"    Best dt_schedule: {best_sched}")

    sigma_schedules = ['cosine', 'cosine_fast', 'exp_decay', 'anneal_50', 'none']

    for sigma_s in sigma_schedules:
        tag = f"sigma_{sigma_s}"
        print(f"\n--- σ={sigma_s}, dt_sched={best_sched}, dt={best_dt}, T={best_T} ---")

        gen = eval_generation(step_fn, e_core, decoder, encoder, z_train, x_test,
                              hf_coh_real, hf_noise_real, device,
                              T=best_T, dt=best_dt, dt_schedule=best_sched,
                              sigma_schedule=sigma_s, n_gen=n_gen, tag=tag)

        results[tag] = {'gen': gen}
        print(f"    GEN: viol={gen['violation']:.4f}  div={gen['diversity']:.4f}  "
              f"conn={gen['connectedness']:.3f}  HF_noise={gen['hf_noise_index']:.0f}  "
              f"delta_u={gen['final_delta_u']:.1f}")

    # ---- Save results ----
    # Convert non-serializable types
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(make_serializable(results), f, indent=2)

    # ---- Summary table ----
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print("\n--- Phase 1: dt sweep (T=20) ---")
    print(f"{'dt':>6} {'viol':>8} {'div':>8} {'conn':>8} {'HF_noise':>10} {'delta_u':>9} {'total_dt':>9} {'ham_m':>8}")
    for dt in dt_values:
        tag = f"dt{dt}"
        g = results[tag]['gen']; r = results[tag]['repair']
        print(f"{dt:>6.2f} {g['violation']:>8.4f} {g['diversity']:>8.4f} "
              f"{g['connectedness']:>8.3f} {g['hf_noise_index']:>10.0f} "
              f"{g['final_delta_u']:>9.1f} {g['total_dt']:>9.1f} {r['hamming_masked']:>8.4f}")

    print(f"\n--- Phase 2: T sweep (dt={best_dt}) ---")
    print(f"{'T':>6} {'viol':>8} {'div':>8} {'conn':>8} {'HF_noise':>10} {'delta_u':>9} {'converge':>9} {'ham_m':>8}")
    for T in T_values:
        tag = f"T{T}_dt{best_dt}"
        g = results[tag]['gen']; r = results[tag]['repair']
        print(f"{T:>6} {g['violation']:>8.4f} {g['diversity']:>8.4f} "
              f"{g['connectedness']:>8.3f} {g['hf_noise_index']:>10.0f} "
              f"{g['final_delta_u']:>9.1f} {g['converge_step']:>9} {r['hamming_masked']:>8.4f}")

    print(f"\n--- Phase 3: dt schedules (dt_max={best_dt}, T={best_T}) ---")
    print(f"{'schedule':>15} {'viol':>8} {'div':>8} {'conn':>8} {'HF_noise':>10} {'delta_u':>9} {'total_dt':>9} {'ham_m':>8}")
    for sched in dt_schedules:
        tag = f"sched_{sched}"
        g = results[tag]['gen']; r = results[tag]['repair']
        print(f"{sched:>15} {g['violation']:>8.4f} {g['diversity']:>8.4f} "
              f"{g['connectedness']:>8.3f} {g['hf_noise_index']:>10.0f} "
              f"{g['final_delta_u']:>9.1f} {g['total_dt']:>9.1f} {r['hamming_masked']:>8.4f}")

    print(f"\n--- Phase 4: σ schedules (dt_sched={best_sched}) ---")
    print(f"{'σ_schedule':>15} {'viol':>8} {'div':>8} {'conn':>8} {'HF_noise':>10} {'delta_u':>9}")
    for sigma_s in sigma_schedules:
        tag = f"sigma_{sigma_s}"
        g = results[tag]['gen']
        print(f"{sigma_s:>15} {g['violation']:>8.4f} {g['diversity']:>8.4f} "
              f"{g['connectedness']:>8.3f} {g['hf_noise_index']:>10.0f} "
              f"{g['final_delta_u']:>9.1f}")

    print(f"\n>>> BEST CONFIG: dt={best_dt}, T={best_T}, dt_sched={best_sched}")
    print("DONE.")


if __name__ == '__main__':
    main()
