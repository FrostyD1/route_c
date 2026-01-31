#!/usr/bin/env python3
"""
Flow F0c: Fix Divergence + Langevin Schedule
==============================================
F0b showed:
  - Flat flow T>10: delta_u explodes (14→5.5M→4.5e13), diversity collapses
  - U-Net stable (delta_u ≈ 3) but HF_noise rises with T
  - HF_noise ∝ T for BOTH models → Langevin noise accumulates

Two fixes to test:
  Fix 1: Output normalization for flat step function
    - Tanh output (bounded Δu)
    - LayerNorm before output
    - Gradient clipping during training
  Fix 2: Aggressive Langevin schedule
    - 'cosine_fast': σ decays in first T/3 then zero
    - 'exp_decay': σ = 0.3 * exp(-4*step/T)
    - 'anneal_50': σ = 0.3 * cos(π·step/(2·T/2)) for first half, zero after

Design: 3×3 grid (3 flat variants × 3 σ schedules) + best σ on U-Net
All tested at T=20 (U-Net T=20 was optimal E_gap in F0b)

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

# Import from F0
from exp_flow_f0 import (
    dct2d, idct2d, get_freq_masks, decompose_bands, freq_scheduled_loss,
    per_band_energy_distance, hf_coherence_metric, hf_noise_index,
    connectedness_proxy, compute_diversity, save_grid,
    GumbelSigmoid, Encoder16, Decoder16,
    DiffEnergyCore, UNetStepFn, SelfAttention,
    quantize, soft_quantize, compute_e_core_grad,
    train_adc, encode_all, train_ecore,
    train_step_fn_energy,
    evaluate
)


# ============================================================================
# FIXED FLAT STEP FUNCTIONS
# ============================================================================

class FlatStepFn_Tanh(nn.Module):
    """Flat step with tanh output — bounded Δu in [-1, 1]."""
    def __init__(self, n_bits):
        super().__init__(); self.n_bits = n_bits
        hid = 64
        self.net = nn.Sequential(
            nn.Conv2d(n_bits * 2 + 1, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, n_bits, 3, padding=1),
            nn.Tanh())  # <-- bounded output
        self.skip = nn.Conv2d(n_bits, n_bits, 1)
        self.scale = nn.Parameter(torch.tensor(0.5))  # learnable scaling

    def forward(self, u, e_grad, t_scalar):
        B = u.shape[0]
        t_map = t_scalar.view(B, 1, 1, 1).expand(-1, 1, u.shape[2], u.shape[3])
        inp = torch.cat([u, e_grad, t_map], dim=1)
        return self.scale * self.net(inp) + self.skip(u)


class FlatStepFn_Norm(nn.Module):
    """Flat step with LayerNorm before output — prevents magnitude drift."""
    def __init__(self, n_bits):
        super().__init__(); self.n_bits = n_bits
        hid = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_bits * 2 + 1, hid, 3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU())
        self.norm = nn.GroupNorm(8, hid)
        self.out = nn.Conv2d(hid, n_bits, 3, padding=1)
        self.skip = nn.Conv2d(n_bits, n_bits, 1)

    def forward(self, u, e_grad, t_scalar):
        B = u.shape[0]
        t_map = t_scalar.view(B, 1, 1, 1).expand(-1, 1, u.shape[2], u.shape[3])
        inp = torch.cat([u, e_grad, t_map], dim=1)
        h = self.conv1(inp)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.norm(h)
        return self.out(h) + self.skip(u)


class FlatStepFn_TanhSkip(nn.Module):
    """Flat step with tanh on BOTH trunk and skip — no unbounded path."""
    def __init__(self, n_bits):
        super().__init__(); self.n_bits = n_bits
        hid = 64
        self.net = nn.Sequential(
            nn.Conv2d(n_bits * 2 + 1, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, n_bits, 3, padding=1))
        self.skip = nn.Conv2d(n_bits, n_bits, 1)

    def forward(self, u, e_grad, t_scalar):
        B = u.shape[0]
        t_map = t_scalar.view(B, 1, 1, 1).expand(-1, 1, u.shape[2], u.shape[3])
        inp = torch.cat([u, e_grad, t_map], dim=1)
        return torch.tanh(self.net(inp) + self.skip(u))


# ============================================================================
# IMPROVED SIGMA SCHEDULES
# ============================================================================

def get_sigma(schedule, step, T):
    """Compute Langevin noise magnitude for given schedule."""
    if schedule == 'cosine':
        # Original: decays over full trajectory
        return 0.3 * np.cos(np.pi * step / (2 * T))
    elif schedule == 'cosine_fast':
        # Fast decay: zero after T/3
        cutoff = T // 3
        if step >= cutoff:
            return 0.0
        return 0.3 * np.cos(np.pi * step / (2 * cutoff))
    elif schedule == 'exp_decay':
        # Exponential: fast initial decay
        return 0.3 * np.exp(-4.0 * step / T)
    elif schedule == 'anneal_50':
        # Noise only in first half
        if step >= T // 2:
            return 0.0
        return 0.3 * np.cos(np.pi * step / T)
    elif schedule == 'none':
        return 0.0
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


@torch.no_grad()
def sample_flow_f0c(step_fn, e_core, n, K, H, W, device,
                    T=20, dt=0.5, sigma_schedule='cosine',
                    use_langevin=True):
    """Flow sampling with configurable sigma schedule."""
    u = torch.randn(n, K, H, W, device=device) * 0.5

    trajectory = {
        'e_core': [], 'e_soft': [],
        'delta_u_norm': [],
    }

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.tensor([t_frac], device=device).expand(n)

        e_grad = compute_e_core_grad(e_core, u)
        delta_u = step_fn(u, e_grad, t_tensor)

        trajectory['delta_u_norm'].append(delta_u.abs().mean().item())

        u = u + dt * delta_u

        if use_langevin:
            sigma = get_sigma(sigma_schedule, step, T)
            if sigma > 0.01:
                u = u + sigma * torch.randn_like(u)

        z_cur = quantize(u)
        trajectory['e_core'].append(e_core.energy(z_cur).item())
        trajectory['e_soft'].append(e_core.soft_energy(u).item())

    z_final = quantize(u)
    return z_final, trajectory


def train_step_fn_base_clipped(step_fn, e_core, z_data, decoder, device,
                               epochs=30, bs=32, T_unroll=3, clip_grad=1.0):
    """Train step function with gradient clipping."""
    opt = torch.optim.Adam(step_fn.parameters(), lr=1e-3)
    N = len(z_data)

    for epoch in tqdm(range(epochs), desc="StepFn"):
        step_fn.train(); perm = torch.randperm(N)
        tl, fl, nb = 0., 0., 0
        progress = epoch / max(epochs - 1, 1)

        for i in range(0, N, bs):
            idx = perm[i:i+bs]; z_clean = z_data[idx].to(device); B = z_clean.shape[0]

            noise_level = torch.rand(B, device=device)
            flip_prob = noise_level.view(B, 1, 1, 1)
            u_clean = z_clean * 2.0 - 1.0
            u_noisy = u_clean + torch.randn_like(u_clean) * flip_prob * 2.0

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
            with torch.no_grad(): x_clean = decoder(z_clean)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_flow_f0c')
    parser.add_argument('--n_samples', type=int, default=256)
    parser.add_argument('--n_bits', type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    N_BITS = args.n_bits

    print("=" * 100)
    print("FLOW F0c: FIX DIVERGENCE + LANGEVIN SCHEDULE")
    print("=" * 100)

    # ========== SHARED SETUP ==========
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

    # ========== TRAIN FLAT VARIANTS ==========
    flat_variants = {}

    print("\n[6a] Training flat_tanh...")
    torch.manual_seed(args.seed)
    flat_tanh = FlatStepFn_Tanh(N_BITS).to(device)
    train_step_fn_base_clipped(flat_tanh, e_core, z_data, decoder, device,
                               epochs=30, bs=32, T_unroll=3, clip_grad=1.0)
    flat_variants['flat_tanh'] = flat_tanh

    print("\n[6b] Training flat_norm...")
    torch.manual_seed(args.seed)
    flat_norm = FlatStepFn_Norm(N_BITS).to(device)
    train_step_fn_base_clipped(flat_norm, e_core, z_data, decoder, device,
                               epochs=30, bs=32, T_unroll=3, clip_grad=1.0)
    flat_variants['flat_norm'] = flat_norm

    print("\n[6c] Training flat_tanh_skip...")
    torch.manual_seed(args.seed)
    flat_ts = FlatStepFn_TanhSkip(N_BITS).to(device)
    train_step_fn_base_clipped(flat_ts, e_core, z_data, decoder, device,
                               epochs=30, bs=32, T_unroll=3, clip_grad=1.0)
    flat_variants['flat_tanhskip'] = flat_ts

    # ========== TRAIN U-NET (energy hinge) ==========
    print("\n[7] Training U-Net step function (energy hinge)...")
    torch.manual_seed(args.seed)
    unet_step = UNetStepFn(N_BITS, base_ch=48).to(device)
    train_step_fn_energy(unet_step, e_core, z_data, decoder, device, epochs=30, bs=32, T_unroll=3)

    # ========== EVALUATION GRID ==========
    sigma_schedules = ['cosine', 'cosine_fast', 'exp_decay', 'anneal_50']
    T_test = 20

    all_results = {}

    # Phase 1: Test 3 flat variants × 4 sigma schedules (12 configs)
    for model_name, step_fn in flat_variants.items():
        for sigma_sch in sigma_schedules:
            cfg_name = f"{model_name}_T{T_test}_{sigma_sch}"
            print(f"\n{'='*80}")
            print(f"CONFIG: {cfg_name}")
            print("=" * 80)

            torch.manual_seed(args.seed + 100)
            gen_bs = 32
            z_gen_list = []; all_traj = []

            for gi in range(0, args.n_samples, gen_bs):
                nb = min(gen_bs, args.n_samples - gi)
                z_batch, traj = sample_flow_f0c(
                    step_fn, e_core, nb, K, H, W, device,
                    T=T_test, dt=0.5, sigma_schedule=sigma_sch)
                z_gen_list.append(z_batch.cpu())
                all_traj.append(traj)

            z_gen = torch.cat(z_gen_list)

            agg_traj = {
                key: [np.mean([t[key][s] for t in all_traj])
                      for s in range(len(all_traj[0][key]))]
                for key in all_traj[0].keys()
            }

            r, x_gen = evaluate(z_gen, decoder, encoder, e_core, z_data, test_x,
                                real_hf_coh, real_hf_noi, device, trajectory=agg_traj)
            r['oracle_mse'] = oracle_mse
            r['model'] = model_name
            r['T'] = T_test
            r['sigma_schedule'] = sigma_sch
            r['final_delta_u'] = agg_traj['delta_u_norm'][-1]
            r['max_delta_u'] = max(agg_traj['delta_u_norm'])
            r['e_core_trajectory'] = agg_traj['e_core']

            all_results[cfg_name] = r

            save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))

            print(f"    viol={r['violation']:.4f}  div={r['diversity']:.4f}  "
                  f"cycle={r['cycle']:.4f}  conn={r['connectedness']:.4f}")
            print(f"    HF_coh={r['hf_coherence']:.4f}  HF_noi={r['hf_noise_index']:.2f}")
            print(f"    delta_u: final={r['final_delta_u']:.4f}  max={r['max_delta_u']:.4f}")

            et = agg_traj['e_core']
            step_indices = list(range(0, len(et), max(1, len(et)//6))) + [len(et)-1]
            step_indices = sorted(set(step_indices))
            print(f"    E_core: {' → '.join(f'{et[i]:.4f}' for i in step_indices)}")

            torch.cuda.empty_cache()

    # Phase 2: Test U-Net with all sigma schedules at T=20
    for sigma_sch in sigma_schedules:
        cfg_name = f"unet_energy_T{T_test}_{sigma_sch}"
        print(f"\n{'='*80}")
        print(f"CONFIG: {cfg_name}")
        print("=" * 80)

        torch.manual_seed(args.seed + 100)
        gen_bs = 32
        z_gen_list = []; all_traj = []

        for gi in range(0, args.n_samples, gen_bs):
            nb = min(gen_bs, args.n_samples - gi)
            z_batch, traj = sample_flow_f0c(
                unet_step, e_core, nb, K, H, W, device,
                T=T_test, dt=0.5, sigma_schedule=sigma_sch)
            z_gen_list.append(z_batch.cpu())
            all_traj.append(traj)

        z_gen = torch.cat(z_gen_list)

        agg_traj = {
            key: [np.mean([t[key][s] for t in all_traj])
                  for s in range(len(all_traj[0][key]))]
            for key in all_traj[0].keys()
        }

        r, x_gen = evaluate(z_gen, decoder, encoder, e_core, z_data, test_x,
                            real_hf_coh, real_hf_noi, device, trajectory=agg_traj)
        r['oracle_mse'] = oracle_mse
        r['model'] = 'unet_energy'
        r['T'] = T_test
        r['sigma_schedule'] = sigma_sch
        r['final_delta_u'] = agg_traj['delta_u_norm'][-1]
        r['max_delta_u'] = max(agg_traj['delta_u_norm'])
        r['e_core_trajectory'] = agg_traj['e_core']

        all_results[cfg_name] = r

        save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))

        print(f"    viol={r['violation']:.4f}  div={r['diversity']:.4f}  "
              f"cycle={r['cycle']:.4f}  conn={r['connectedness']:.4f}")
        print(f"    HF_coh={r['hf_coherence']:.4f}  HF_noi={r['hf_noise_index']:.2f}")
        print(f"    delta_u: final={r['final_delta_u']:.4f}  max={r['max_delta_u']:.4f}")

        et = agg_traj['e_core']
        step_indices = list(range(0, len(et), max(1, len(et)//6))) + [len(et)-1]
        step_indices = sorted(set(step_indices))
        print(f"    E_core: {' → '.join(f'{et[i]:.4f}' for i in step_indices)}")

        torch.cuda.empty_cache()

    # Phase 3: Best flat + best sigma at T=10,20,30,50
    # Identify best flat variant (lowest delta_u_max with reasonable diversity)
    best_flat_name = None
    best_flat_score = float('inf')
    for name, step_fn in flat_variants.items():
        key = f"{name}_T{T_test}_cosine_fast"
        if key in all_results:
            r = all_results[key]
            # Score: low delta_u_max + reasonable div (>0.3)
            if r['diversity'] > 0.3 and r['max_delta_u'] < best_flat_score:
                best_flat_score = r['max_delta_u']
                best_flat_name = name

    if best_flat_name:
        print(f"\n{'='*80}")
        print(f"BEST FLAT: {best_flat_name} — testing at T=10,20,30,50 with cosine_fast")
        print("=" * 80)

        for T in [10, 20, 30, 50]:
            cfg_name = f"best_{best_flat_name}_T{T}_cosine_fast"
            print(f"\n  CONFIG: {cfg_name}")

            torch.manual_seed(args.seed + 100)
            z_gen_list = []; all_traj = []
            for gi in range(0, args.n_samples, 32):
                nb = min(32, args.n_samples - gi)
                z_batch, traj = sample_flow_f0c(
                    flat_variants[best_flat_name], e_core, nb, K, H, W, device,
                    T=T, dt=0.5, sigma_schedule='cosine_fast')
                z_gen_list.append(z_batch.cpu())
                all_traj.append(traj)

            z_gen = torch.cat(z_gen_list)
            agg_traj = {
                key: [np.mean([t[key][s] for t in all_traj])
                      for s in range(len(all_traj[0][key]))]
                for key in all_traj[0].keys()
            }

            r, x_gen = evaluate(z_gen, decoder, encoder, e_core, z_data, test_x,
                                real_hf_coh, real_hf_noi, device, trajectory=agg_traj)
            r['oracle_mse'] = oracle_mse
            r['model'] = f'best_{best_flat_name}'
            r['T'] = T
            r['sigma_schedule'] = 'cosine_fast'
            r['final_delta_u'] = agg_traj['delta_u_norm'][-1]
            r['max_delta_u'] = max(agg_traj['delta_u_norm'])
            all_results[cfg_name] = r

            save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))
            print(f"    viol={r['violation']:.4f}  div={r['diversity']:.4f}  "
                  f"HF_noi={r['hf_noise_index']:.2f}  delta_u: max={r['max_delta_u']:.4f}")

    # ========== SUMMARY ==========
    print(f"\n{'='*100}")
    print("F0c SUMMARY")
    print(f"Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}")
    print("=" * 100)

    # Group by model
    models_seen = set()
    for k, v in all_results.items():
        models_seen.add(v['model'])

    for model in sorted(models_seen):
        print(f"\n--- {model} ---")
        header = f"{'sigma':>12} {'T':>3} {'viol':>7} {'div':>7} {'HFnoi':>7} {'conn':>7} {'dU_max':>8} {'dU_end':>8}"
        print(header); print("-" * len(header))
        for k, v in sorted(all_results.items()):
            if v['model'] != model: continue
            print(f"{v.get('sigma_schedule','?'):>12} {v['T']:>3} "
                  f"{v['violation']:>7.4f} {v['diversity']:>7.4f} "
                  f"{v['hf_noise_index']:>7.2f} {v['connectedness']:>7.4f} "
                  f"{v['max_delta_u']:>8.4f} {v['final_delta_u']:>8.4f}")

    # Save results
    save_results = {}
    for k, v in all_results.items():
        sv = {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}
        save_results[k] = sv
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n{'='*100}")
    print("F0c COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
