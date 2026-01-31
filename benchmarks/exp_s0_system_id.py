#!/usr/bin/env python3
"""
S0: System Identification Diagnostic — Empirical Gramian Rank Measurement
=========================================================================
First-principles motivation (state-space theory):

  State equation:    z_{t+1} = F(z_t, u)      (flow operator)
  Output equation:   x = D(z)                  (decoder / DAC)

  Observability Gramian:  W_o = J_D^T J_D  where J_D = ∂D/∂z
    rank(W_o) = number of "testable" state dimensions
    null(W_o) = dead bits (unobservable)

  Controllability Gramian: W_c = [B, FB, F²B, ...]
    rank(W_c) = number of reachable state dimensions
    If some bits are unreachable from any control input → uncontrollable

This is a DIAGNOSTIC experiment — no training changes, pure measurement
on existing baseline models. We train a standard baseline, then probe it.

Measurements:
  S0-A: Observability probe
    - Perturb single z-bits (K random positions), decode, collect output vectors
    - Form matrix M_obs ∈ R^{K × D_out}, compute SVD
    - Effective rank = #singular values > threshold
    - Spectral gap = σ_1 / σ_k (condition number proxy)

  S0-B: Decoder Jacobian (continuous approximation)
    - Finite-difference Jacobian: ∂D/∂z_i ≈ (D(z+εe_i) - D(z-εe_i)) / 2ε
    - Full Jacobian too large → sample random directions, estimate rank via SVD

  S0-C: Controllability probe
    - Generate from different initial conditions (noise seeds)
    - Collect terminal z-states, form matrix M_ctrl ∈ R^{K × dim(z)}
    - SVD → effective rank of reachable set
    - Also: repair from different masks → reachable repair set

  S0-D: Cross-config comparison
    - Run S0-A through S0-C on both 16-bit and 24-bit baselines
    - Compare effective ranks → does bandwidth increase observability?

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
    GumbelSigmoid, DiffEnergyCore,
    quantize, compute_e_core_grad,
    evaluate, hf_coherence_metric, hf_noise_index,
    connectedness_proxy, compute_diversity, save_grid,
)
from exp_flow_f0c_fixes import FlatStepFn_Norm, get_sigma
from exp_g2_protocol_density import (
    Encoder16, Decoder16,
    encode_all, train_ecore, train_step_fn
)
from exp_e2a_global_prior import compute_hue_var


# ============================================================================
# S0-A: OBSERVABILITY PROBE — Discrete Jacobian Column Sampling
# ============================================================================

@torch.no_grad()
def observability_probe(encoder, decoder, test_x, device,
                        n_positions=512, n_images=50):
    """Measure observability via discrete Jacobian column sampling.

    For each random z-position (k,h,w), flip that bit across n_images,
    record the output change vector. Stack into matrix M_obs, compute SVD.

    Returns:
        effective_rank: number of singular values > 1% of σ_max
        spectral_gap: σ_1 / σ_min_nonzero
        singular_values: full spectrum (sorted descending)
        dead_ratio: fraction of positions with negligible output change
    """
    # Encode images
    x_sub = test_x[:n_images]
    z_list, xh_list = [], []
    for i in range(0, len(x_sub), 32):
        batch = x_sub[i:i+32].to(device)
        z, _ = encoder(batch)
        xh = decoder(z)
        z_list.append(z.cpu())
        xh_list.append(xh.cpu())
    z_all = torch.cat(z_list)     # (N, K, H, W)
    xh_all = torch.cat(xh_list)   # (N, C, Ho, Wo)

    B, K, Hz, Wz = z_all.shape
    _, C, Ho, Wo = xh_all.shape
    D_out = C * Ho * Wo  # output dimensionality

    # Flatten outputs for matrix construction
    xh_flat = xh_all.reshape(B, D_out)  # (N, D_out)

    # Sample random positions
    positions = []
    for _ in range(n_positions):
        k = torch.randint(K, (1,)).item()
        h = torch.randint(Hz, (1,)).item()
        w = torch.randint(Wz, (1,)).item()
        positions.append((k, h, w))

    # For each position, flip that bit, decode, compute output change
    # Average over images → one "Jacobian column" per position
    jacobian_cols = []  # will be (n_positions, D_out)
    influence_norms = []

    for k, h, w in tqdm(positions, desc="ObsProbe", leave=False):
        z_flip = z_all.clone()
        z_flip[:, k, h, w] = 1.0 - z_flip[:, k, h, w]

        xh_flip_list = []
        for i in range(0, B, 32):
            xh_flip_list.append(decoder(z_flip[i:i+32].to(device)).cpu())
        xh_flip = torch.cat(xh_flip_list)
        xh_flip_flat = xh_flip.reshape(B, D_out)

        # Average output change across images
        delta = (xh_flip_flat - xh_flat).mean(dim=0)  # (D_out,)
        jacobian_cols.append(delta)
        influence_norms.append(delta.norm().item())

    # Form matrix M_obs: (n_positions, D_out)
    M_obs = torch.stack(jacobian_cols)

    # SVD
    # Use smaller dimension for efficiency
    if M_obs.shape[0] > M_obs.shape[1]:
        # More rows than columns — SVD on M_obs^T M_obs
        U, S, Vh = torch.linalg.svd(M_obs, full_matrices=False)
    else:
        U, S, Vh = torch.linalg.svd(M_obs, full_matrices=False)

    S_np = S.numpy()
    s_max = S_np[0] if len(S_np) > 0 else 1.0

    # Effective rank: singular values > 1% of max
    threshold_01 = 0.01 * s_max
    threshold_10 = 0.10 * s_max
    eff_rank_01 = int((S_np > threshold_01).sum())
    eff_rank_10 = int((S_np > threshold_10).sum())

    # Spectral gap
    nonzero_s = S_np[S_np > 1e-10]
    spectral_gap = float(s_max / nonzero_s[-1]) if len(nonzero_s) > 1 else float('inf')

    # Cumulative energy
    total_energy = (S_np ** 2).sum()
    cum_energy = np.cumsum(S_np ** 2) / (total_energy + 1e-10)
    rank_90 = int((cum_energy < 0.90).sum()) + 1
    rank_99 = int((cum_energy < 0.99).sum()) + 1

    # Dead ratio (positions with negligible influence)
    influence_arr = np.array(influence_norms)
    dead_ratio = float((influence_arr < 1e-4).mean())

    return {
        'effective_rank_1pct': eff_rank_01,
        'effective_rank_10pct': eff_rank_10,
        'rank_90_energy': rank_90,
        'rank_99_energy': rank_99,
        'max_dimension': min(n_positions, D_out),
        'spectral_gap': spectral_gap,
        'top10_singular': S_np[:10].tolist(),
        'dead_ratio': dead_ratio,
        'mean_influence': float(influence_arr.mean()),
        'median_influence': float(np.median(influence_arr)),
        'p10_influence': float(np.percentile(influence_arr, 10)),
        'p90_influence': float(np.percentile(influence_arr, 90)),
        'singular_values': S_np.tolist(),
    }


# ============================================================================
# S0-B: CONTINUOUS JACOBIAN APPROXIMATION (finite difference)
# ============================================================================

@torch.no_grad()
def continuous_jacobian_probe(decoder, z_sample, device,
                              n_directions=256, epsilon=0.01):
    """Approximate decoder Jacobian rank via random direction probing.

    Instead of full Jacobian (too large: K*H*W × C*Ho*Wo), probe
    random directions in z-space and measure output change.

    J_D · v ≈ (D(z + εv) - D(z - εv)) / 2ε

    Collect n_directions such columns, SVD → rank estimate.
    """
    z = z_sample[0:1].to(device).float()  # (1, K, H, W)
    B, K, Hz, Wz = z.shape
    z_dim = K * Hz * Wz

    xh_center = decoder(z)
    _, C, Ho, Wo = xh_center.shape
    D_out = C * Ho * Wo

    # Random directions in z-space (normalized)
    directions = torch.randn(n_directions, K, Hz, Wz, device=device)
    directions = directions / (directions.reshape(n_directions, -1).norm(dim=1, keepdim=True).reshape(n_directions, 1, 1, 1) + 1e-8)

    # Finite-difference Jacobian-vector products
    jvp_list = []
    for i in range(0, n_directions, 32):
        batch_dirs = directions[i:i+32]
        nb = len(batch_dirs)

        z_plus = (z + epsilon * batch_dirs).clamp(0, 1)
        z_minus = (z - epsilon * batch_dirs).clamp(0, 1)

        xh_plus = decoder(z_plus)
        xh_minus = decoder(z_minus)

        jvp = ((xh_plus - xh_minus) / (2 * epsilon)).reshape(nb, D_out)
        jvp_list.append(jvp.cpu())

    M_jac = torch.cat(jvp_list)  # (n_directions, D_out)

    U, S, Vh = torch.linalg.svd(M_jac, full_matrices=False)
    S_np = S.numpy()
    s_max = S_np[0] if len(S_np) > 0 else 1.0

    threshold_01 = 0.01 * s_max
    eff_rank = int((S_np > threshold_01).sum())

    total_energy = (S_np ** 2).sum()
    cum_energy = np.cumsum(S_np ** 2) / (total_energy + 1e-10)
    rank_90 = int((cum_energy < 0.90).sum()) + 1

    return {
        'eff_rank_1pct': eff_rank,
        'rank_90_energy': rank_90,
        'max_dimension': min(n_directions, D_out),
        'spectral_gap': float(s_max / (S_np[S_np > 1e-10][-1] if (S_np > 1e-10).sum() > 0 else 1)),
        'top10_singular': S_np[:10].tolist(),
        'singular_values': S_np.tolist(),
    }


# ============================================================================
# S0-C: CONTROLLABILITY PROBE — Reachable Set Analysis
# ============================================================================

@torch.no_grad()
def controllability_probe_generation(step_fn, e_core, K, Hz, Wz, device,
                                     n_seeds=256, T=10, dt=0.5):
    """Controllability via generation: how diverse are terminal states?

    Start from n_seeds different noise initializations, run flow to completion.
    Collect terminal z-states, flatten, SVD → effective rank of reachable set.

    High rank = many distinct z-configurations reachable = high controllability.
    Low rank = mode collapse = low controllability.
    """
    z_terminals = []
    for si in range(0, n_seeds, 32):
        nb = min(32, n_seeds - si)
        torch.manual_seed(si * 7919)  # different seeds
        u = torch.randn(nb, K, Hz, Wz, device=device) * 0.5

        for step in range(T):
            t_frac = 1.0 - step / T
            t_tensor = torch.full((nb,), t_frac, device=device)
            e_grad = compute_e_core_grad(e_core, u)
            delta_u = step_fn(u, e_grad, t_tensor)
            u = u + dt * delta_u
            sigma = get_sigma('cosine', step, T)
            if sigma > 0.01:
                u = u + sigma * torch.randn_like(u)

        z_term = quantize(u)
        z_terminals.append(z_term.cpu())

    z_all = torch.cat(z_terminals)  # (n_seeds, K, H, W)
    z_flat = z_all.reshape(n_seeds, -1).float()  # (n_seeds, K*H*W)

    # Center before SVD
    z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)

    U, S, Vh = torch.linalg.svd(z_centered, full_matrices=False)
    S_np = S.numpy()
    s_max = S_np[0] if len(S_np) > 0 else 1.0

    eff_rank = int((S_np > 0.01 * s_max).sum())

    total_energy = (S_np ** 2).sum()
    cum_energy = np.cumsum(S_np ** 2) / (total_energy + 1e-10)
    rank_90 = int((cum_energy < 0.90).sum()) + 1
    rank_99 = int((cum_energy < 0.99).sum()) + 1

    # Bit usage: fraction of bits that are not constant
    usage_per_bit = z_flat.std(dim=0)  # (K*H*W,)
    active_bits = float((usage_per_bit > 0.01).float().mean())

    return {
        'eff_rank_1pct': eff_rank,
        'rank_90_energy': rank_90,
        'rank_99_energy': rank_99,
        'max_dimension': min(n_seeds, z_flat.shape[1]),
        'active_bits_ratio': active_bits,
        'top10_singular': S_np[:10].tolist(),
        'singular_values': S_np[:50].tolist(),  # top 50 only
    }


@torch.no_grad()
def controllability_probe_repair(step_fn, e_core, z_test, device,
                                 n_masks=8, T=10, dt=0.5, n_images=50):
    """Controllability via repair: how many distinct repair states are reachable
    from different mask configurations on the same images?

    For each image, apply n_masks different masks, repair, collect results.
    SVD over mask-dimension → effective rank of the repair-reachable set.
    """
    z_sub = z_test[:n_images]
    B, K, Hz, Wz = z_sub.shape

    # Generate diverse masks
    masks = []
    # center mask
    m = torch.ones(1, 1, Hz, Wz)
    h4, w4 = Hz // 4, Wz // 4
    m[:, :, h4:3*h4, w4:3*w4] = 0.0
    masks.append(m)

    # quadrant masks
    for qi in range(4):
        m = torch.ones(1, 1, Hz, Wz)
        r, c = qi // 2, qi % 2
        m[:, :, r*Hz//2:(r+1)*Hz//2, c*Wz//2:(c+1)*Wz//2] = 0.0
        masks.append(m)

    # random masks
    for ri in range(n_masks - 5):
        torch.manual_seed(ri * 31337)
        m = (torch.rand(1, 1, Hz, Wz) > 0.5).float()
        masks.append(m)

    masks = masks[:n_masks]

    # For each mask, repair all images
    repair_results = []  # (n_masks, B, K*H*W)
    for mi, mask in enumerate(masks):
        z_rep_list = []
        for i in range(0, B, 32):
            nb = min(32, B - i)
            z_batch = z_sub[i:i+nb].to(device)
            m_batch = mask.expand(nb, K, -1, -1).to(device)

            u_clean = torch.where(z_batch > 0.5,
                                  torch.tensor(2.0, device=device),
                                  torch.tensor(-2.0, device=device))
            u_noise = torch.randn(nb, K, Hz, Wz, device=device) * 0.3
            u = m_batch * u_clean + (1 - m_batch) * u_noise

            for step in range(T):
                t_frac = 1.0 - step / T
                t_tensor = torch.full((nb,), t_frac, device=device)
                e_grad = compute_e_core_grad(e_core, u)
                delta_u = step_fn(u, e_grad, t_tensor)
                u = u + 0.5 * delta_u
                sigma = get_sigma('cosine', step, T)
                if sigma > 0.01:
                    u = u + sigma * torch.randn_like(u) * (1 - m_batch)
                u = m_batch * u_clean + (1 - m_batch) * u

            z_rep = m_batch * z_batch + (1 - m_batch) * quantize(u)
            z_rep_list.append(z_rep.cpu())

        z_repaired = torch.cat(z_rep_list)  # (B, K, H, W)
        repair_results.append(z_repaired.reshape(B, -1).float())

    # Stack: (n_masks, B, z_dim)
    R = torch.stack(repair_results)  # (n_masks, B, z_dim)

    # Per-image: SVD over mask dimension
    # Average effective rank across images
    eff_ranks = []
    for b in range(B):
        M = R[:, b, :]  # (n_masks, z_dim)
        M_centered = M - M.mean(dim=0, keepdim=True)
        try:
            _, S, _ = torch.linalg.svd(M_centered, full_matrices=False)
            S_np = S.numpy()
            s_max = S_np[0] if len(S_np) > 0 else 1.0
            eff_ranks.append(int((S_np > 0.01 * s_max).sum()))
        except:
            eff_ranks.append(0)

    # Also: global SVD (flatten masks × images)
    R_flat = R.reshape(n_masks * B, -1)
    R_flat_centered = R_flat - R_flat.mean(dim=0, keepdim=True)
    _, S_global, _ = torch.linalg.svd(R_flat_centered, full_matrices=False)
    S_g = S_global.numpy()
    s_max_g = S_g[0] if len(S_g) > 0 else 1.0
    global_eff_rank = int((S_g > 0.01 * s_max_g).sum())

    total_energy = (S_g ** 2).sum()
    cum_energy = np.cumsum(S_g ** 2) / (total_energy + 1e-10)
    rank_90 = int((cum_energy < 0.90).sum()) + 1

    return {
        'per_image_eff_rank_mean': float(np.mean(eff_ranks)),
        'per_image_eff_rank_std': float(np.std(eff_ranks)),
        'global_eff_rank': global_eff_rank,
        'global_rank_90_energy': rank_90,
        'n_masks': n_masks,
        'top10_singular': S_g[:10].tolist(),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_s0_system_id')
    parser.add_argument('--T', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("S0: SYSTEM IDENTIFICATION — EMPIRICAL GRAMIAN RANK MEASUREMENT")
    print("=" * 100)
    print(f"Device: {device}  |  Seed: {args.seed}  |  T: {args.T}")
    print(f"\nState-space equations:")
    print(f"  z_{{t+1}} = F(z_t, u)      — flow operator (transition)")
    print(f"  x       = D(z)           — decoder (output)")
    print(f"\nObservability  = rank(W_o) = rank(J_D^T J_D)")
    print(f"Controllability = rank(W_c) = rank(reachable set)")
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

    # ========== CONFIGS ==========
    configs = [
        ('S0_16bit', 16),
        ('S0_24bit', 24),
    ]

    all_results = {}

    for cfg_name, n_bits in configs:
        print(f"\n{'='*80}")
        print(f"CONFIG: {cfg_name}  (n_bits={n_bits})")
        print("=" * 80)

        result = {'name': cfg_name, 'n_bits': n_bits}

        # --- Train baseline ADC/DAC ---
        print("\n  [1] Training ADC/DAC (baseline, no regularizer)...")
        torch.manual_seed(args.seed)
        enc = Encoder16(n_bits).to(device)
        dec = Decoder16(n_bits).to(device)
        params = list(enc.parameters()) + list(dec.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)

        for epoch in tqdm(range(40), desc="ADC"):
            enc.train(); dec.train()
            if hasattr(enc, 'set_temperature'):
                enc.set_temperature(1.0 + (0.3 - 1.0) * epoch / 39)
            perm = torch.randperm(len(train_x))
            for i in range(0, len(train_x), 32):
                x = train_x[perm[i:i+32]].to(device)
                opt.zero_grad()
                z, logits = enc(x)
                xh = dec(z)
                loss = F.mse_loss(xh, x) + 0.5 * F.binary_cross_entropy(xh, x)
                loss.backward()
                opt.step()
        enc.eval(); dec.eval()
        print(f"    ADC training complete")

        # --- Encode ---
        z_train = encode_all(enc, train_x, device, bs=32)
        z_test = encode_all(enc, test_x, device, bs=32)
        K, Hz, Wz = z_train.shape[1:]
        usage = z_train.float().mean().item()
        print(f"    z: {z_train.shape}, usage={usage:.3f}")
        result['z_usage'] = usage
        result['z_dim'] = K * Hz * Wz

        # --- Train E_core ---
        print("\n  [2] Training E_core...")
        e_core = DiffEnergyCore(n_bits).to(device)
        train_ecore(e_core, z_train, device, epochs=15, bs=128)

        # --- Train StepFn ---
        print("\n  [3] Training StepFn...")
        step_fn = FlatStepFn_Norm(n_bits).to(device)
        train_step_fn(step_fn, e_core, z_train, dec, device, epochs=30, bs=32)

        # ==================================================
        # S0-A: OBSERVABILITY PROBE
        # ==================================================
        print(f"\n  [4] S0-A: Observability Probe (discrete Jacobian)")
        obs_result = observability_probe(enc, dec, test_x, device,
                                         n_positions=512, n_images=50)
        result['observability'] = {k: v for k, v in obs_result.items()
                                    if k != 'singular_values'}
        result['obs_spectrum'] = obs_result['singular_values']

        print(f"    Effective rank (1%): {obs_result['effective_rank_1pct']} / {obs_result['max_dimension']}")
        print(f"    Effective rank (10%): {obs_result['effective_rank_10pct']} / {obs_result['max_dimension']}")
        print(f"    Rank for 90% energy: {obs_result['rank_90_energy']}")
        print(f"    Rank for 99% energy: {obs_result['rank_99_energy']}")
        print(f"    Spectral gap: {obs_result['spectral_gap']:.1f}")
        print(f"    Dead ratio: {obs_result['dead_ratio']:.3f}")
        print(f"    Influence: mean={obs_result['mean_influence']:.6f}  "
              f"median={obs_result['median_influence']:.6f}  "
              f"p10={obs_result['p10_influence']:.6f}")
        print(f"    Top-5 σ: {[f'{s:.4f}' for s in obs_result['top10_singular'][:5]]}")

        # ==================================================
        # S0-B: CONTINUOUS JACOBIAN PROBE
        # ==================================================
        print(f"\n  [5] S0-B: Continuous Jacobian Probe (finite difference)")
        jac_result = continuous_jacobian_probe(dec, z_test, device,
                                               n_directions=256, epsilon=0.01)
        result['jacobian'] = {k: v for k, v in jac_result.items()
                               if k != 'singular_values'}
        result['jac_spectrum'] = jac_result['singular_values']

        print(f"    Effective rank (1%): {jac_result['eff_rank_1pct']} / {jac_result['max_dimension']}")
        print(f"    Rank for 90% energy: {jac_result['rank_90_energy']}")
        print(f"    Spectral gap: {jac_result['spectral_gap']:.1f}")
        print(f"    Top-5 σ: {[f'{s:.4f}' for s in jac_result['top10_singular'][:5]]}")

        # ==================================================
        # S0-C: CONTROLLABILITY — GENERATION
        # ==================================================
        print(f"\n  [6] S0-C1: Controllability Probe (generation)")
        ctrl_gen = controllability_probe_generation(
            step_fn, e_core, K, Hz, Wz, device,
            n_seeds=256, T=args.T, dt=0.5)
        result['ctrl_generation'] = {k: v for k, v in ctrl_gen.items()
                                      if k != 'singular_values'}

        print(f"    Effective rank (1%): {ctrl_gen['eff_rank_1pct']} / {ctrl_gen['max_dimension']}")
        print(f"    Rank for 90% energy: {ctrl_gen['rank_90_energy']}")
        print(f"    Active bits ratio: {ctrl_gen['active_bits_ratio']:.3f}")
        print(f"    Top-5 σ: {[f'{s:.4f}' for s in ctrl_gen['top10_singular'][:5]]}")

        # ==================================================
        # S0-C2: CONTROLLABILITY — REPAIR
        # ==================================================
        print(f"\n  [7] S0-C2: Controllability Probe (repair)")
        ctrl_rep = controllability_probe_repair(
            step_fn, e_core, z_test, device,
            n_masks=8, T=args.T, dt=0.5, n_images=50)
        result['ctrl_repair'] = ctrl_rep

        print(f"    Per-image eff rank: {ctrl_rep['per_image_eff_rank_mean']:.1f} "
              f"± {ctrl_rep['per_image_eff_rank_std']:.1f} (of {ctrl_rep['n_masks']} masks)")
        print(f"    Global eff rank: {ctrl_rep['global_eff_rank']}")
        print(f"    Global rank 90% energy: {ctrl_rep['global_rank_90_energy']}")

        all_results[cfg_name] = result

        del enc, dec, e_core, step_fn
        torch.cuda.empty_cache()

    # ========== SUMMARY ==========
    print(f"\n{'='*100}")
    print("S0 SUMMARY: SYSTEM IDENTIFICATION")
    print("=" * 100)

    # --- Observability ---
    print(f"\n--- OBSERVABILITY (decoder Jacobian rank) ---")
    oh = (f"{'config':<15} {'bits':>4} {'z_dim':>7} "
          f"{'rank_1%':>8} {'rank_10%':>8} {'r90_E':>7} {'r99_E':>7} "
          f"{'gap':>8} {'dead%':>6} {'mean_inf':>9}")
    print(oh); print("-" * len(oh))
    for name, r in all_results.items():
        o = r['observability']
        print(f"{name:<15} {r['n_bits']:>4} {r['z_dim']:>7} "
              f"{o['effective_rank_1pct']:>8} {o['effective_rank_10pct']:>8} "
              f"{o['rank_90_energy']:>7} {o['rank_99_energy']:>7} "
              f"{o['spectral_gap']:>8.1f} {o['dead_ratio']:>6.3f} "
              f"{o['mean_influence']:>9.6f}")

    # --- Continuous Jacobian ---
    print(f"\n--- CONTINUOUS JACOBIAN ---")
    jh = f"{'config':<15} {'rank_1%':>8} {'r90_E':>7} {'gap':>8}"
    print(jh); print("-" * len(jh))
    for name, r in all_results.items():
        j = r['jacobian']
        print(f"{name:<15} {j['eff_rank_1pct']:>8} "
              f"{j['rank_90_energy']:>7} {j['spectral_gap']:>8.1f}")

    # --- Controllability ---
    print(f"\n--- CONTROLLABILITY (reachable set rank) ---")
    ch = (f"{'config':<15} {'gen_rank_1%':>11} {'gen_r90_E':>10} "
          f"{'active%':>8} {'rep_rank':>9} {'rep_r90_E':>10}")
    print(ch); print("-" * len(ch))
    for name, r in all_results.items():
        cg = r['ctrl_generation']
        cr = r['ctrl_repair']
        print(f"{name:<15} {cg['eff_rank_1pct']:>11} "
              f"{cg['rank_90_energy']:>10} {cg['active_bits_ratio']:>8.3f} "
              f"{cr['global_eff_rank']:>9} {cr['global_rank_90_energy']:>10}")

    # --- State-space interpretation ---
    print(f"\n--- STATE-SPACE INTERPRETATION ---")
    for name, r in all_results.items():
        z_dim = r['z_dim']
        obs_rank = r['observability']['effective_rank_1pct']
        ctrl_rank = r['ctrl_generation']['eff_rank_1pct']
        dead = r['observability']['dead_ratio']
        active = r['ctrl_generation']['active_bits_ratio']

        obs_ratio = obs_rank / max(r['observability']['max_dimension'], 1)
        ctrl_ratio = ctrl_rank / max(r['ctrl_generation']['max_dimension'], 1)

        print(f"\n  {name} (z_dim={z_dim}):")
        print(f"    Observability:    rank={obs_rank}  ({obs_ratio:.1%} of probed)")
        print(f"    Controllability:  rank={ctrl_rank}  ({ctrl_ratio:.1%} of probed)")
        print(f"    Dead bits:        {dead:.1%}")
        print(f"    Active bits:      {active:.1%}")

        if obs_ratio < 0.5:
            print(f"    → LOW OBSERVABILITY: decoder ignores >{(1-obs_ratio):.0%} of z-space")
        if ctrl_ratio < 0.5:
            print(f"    → LOW CONTROLLABILITY: flow only reaches {ctrl_ratio:.0%} of z-space")
        if dead > 0.5:
            print(f"    → DEAD BIT CRISIS: {dead:.0%} of bits have no visible effect")

    # Save
    save_results = {}
    for name, r in all_results.items():
        save_r = {k: v for k, v in r.items()
                  if k not in ('obs_spectrum', 'jac_spectrum')}
        save_results[name] = save_r

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    # Save spectra separately (for plotting)
    for name, r in all_results.items():
        np.savez(os.path.join(args.output_dir, f'spectra_{name}.npz'),
                 obs_spectrum=np.array(r.get('obs_spectrum', [])),
                 jac_spectrum=np.array(r.get('jac_spectrum', [])))

    print(f"\n{'='*100}")
    print("S0 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
