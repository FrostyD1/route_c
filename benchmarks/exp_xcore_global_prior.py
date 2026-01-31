#!/usr/bin/env python3
"""
X-CORE: Unified E_prior for Repair + Generation
=================================================================
GPT-proposed cross-task experiment: test whether the SAME E_prior module
simultaneously improves:
  1. Repair distribution shift (KL_marg, Δcov, probe gap)
  2. Generation quality (HF_noise, HueVar, ColorKL, diversity)

Key novelty: Flow-based REPAIR with E_prior gradient guidance.
  - Unmasked positions clamped (evidence), masked positions updated by flow
  - E_prior gradient added during repair flow steps on masked region only
  - This tests the causal chain: prior on → KL_marg/Δcov ↓ → probe gap ↓

Design (2 × 3 matrix):
  Bits: {16, 24}
  Prior: {none, pos-marg, spatial_cov(λ=0.3)}
  Modes: repair + generation (same pipeline)

Metrics:
  Contract: ham_unmasked, ham_masked, cycle, violation
  Distribution-shift: KL_marg, Δcov (new!)
  Generation: HF_noise, HueVar, ColorKL, diversity, connectedness
  Readout: conv probe mixed acc_clean/acc_repair/gap

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

from exp_g2_protocol_density import (
    Encoder16, Decoder16,
    train_adc, encode_all, train_ecore, train_step_fn
)

from exp_e2a_global_prior import (
    MarginalPrior,
    CovariancePrior,
    compute_hue_var,
    compute_marginal_kl
)


# ============================================================================
# DISTRIBUTION-SHIFT METRICS (NEW)
# ============================================================================

def compute_kl_marg(z_test, z_ref):
    """KL divergence of per-position marginals: KL(p_test || p_ref).
    Measures how much z_test's activation rates deviate from z_ref's.
    """
    p_test = z_test.float().mean(dim=0).clamp(1e-6, 1 - 1e-6)  # [K, H, W]
    p_ref = z_ref.float().mean(dim=0).clamp(1e-6, 1 - 1e-6)
    kl = p_ref * (p_ref / p_test).log() + (1 - p_ref) * ((1 - p_ref) / (1 - p_test)).log()
    return kl.mean().item()


def compute_delta_cov(z_test, z_ref):
    """Frobenius norm of covariance difference between z_test and z_ref.
    Measures how much the channel covariance structure shifted.
    """
    def get_cov(z):
        N, K, H, W = z.shape
        z_flat = z.float().view(N, K, -1)  # [N, K, HW]
        z_mean = z_flat.mean(dim=2, keepdim=True)
        z_centered = z_flat - z_mean
        cov = torch.bmm(z_centered, z_centered.transpose(1, 2)) / (H * W - 1)
        return cov.mean(dim=0)  # [K, K]

    cov_test = get_cov(z_test)
    cov_ref = get_cov(z_ref)
    return (cov_test - cov_ref).norm().item()


def compute_hamming_stats(z_clean, z_repaired, mask):
    """Compute hamming distance on masked/unmasked regions."""
    diff = (z_clean != z_repaired).float()
    # mask: 1=evidence(unmasked), 0=repair
    unmasked_bits = mask.sum()
    masked_bits = (1 - mask).sum()

    ham_unmasked = (diff * mask).sum() / max(unmasked_bits.item(), 1)
    ham_masked = (diff * (1 - mask)).sum() / max(masked_bits.item(), 1)
    return ham_unmasked.item(), ham_masked.item()


# ============================================================================
# FLOW-BASED REPAIR WITH E_PRIOR (KEY NOVELTY)
# ============================================================================

@torch.no_grad()
def repair_flow_with_prior(step_fn, e_core, z_clean, mask, device,
                           global_prior=None, lambda_global=0.0,
                           T=10, dt=0.5, sigma_schedule='cosine'):
    """Flow-based repair: update masked positions using E_core + E_prior.

    Args:
        z_clean: [B, K, H, W] binary — original clean encoding
        mask: [B, 1, H, W] or [B, K, H, W] — 1=evidence, 0=repair
        global_prior: prior object with .grad(u, device) method
        lambda_global: prior strength

    Returns:
        z_repaired: [B, K, H, W] binary — repaired, unmasked clamped
    """
    B, K, H, W = z_clean.shape
    if mask.shape[1] == 1:
        mask = mask.expand(-1, K, -1, -1)

    # Initialize: unmasked from z_clean logits, masked from noise
    u_clean = torch.where(z_clean > 0.5,
                          torch.tensor(2.0, device=device),
                          torch.tensor(-2.0, device=device))
    u_noise = torch.randn(B, K, H, W, device=device) * 0.3
    u = mask * u_clean + (1 - mask) * u_noise

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((B,), t_frac, device=device)

        # E_core gradient
        e_grad = compute_e_core_grad(e_core, u)

        # E_prior gradient (only on masked region)
        if global_prior is not None and lambda_global > 0:
            e_grad_prior = global_prior.grad(u, device)
            lambda_t = lambda_global * (0.3 + 0.7 * t_frac)
            # Only apply prior to masked positions
            e_grad = e_grad + lambda_t * e_grad_prior * (1 - mask)

        delta_u = step_fn(u, e_grad, t_tensor)
        u = u + dt * delta_u

        # Noise injection (anneal)
        sigma = get_sigma(sigma_schedule, step, T)
        if sigma > 0.01:
            u = u + sigma * torch.randn_like(u) * (1 - mask)

        # Clamp evidence: unmasked positions stay fixed
        u = mask * u_clean + (1 - mask) * u

    return mask * z_clean + (1 - mask) * quantize(u)


# ============================================================================
# MASK GENERATION
# ============================================================================

def make_masks(B, K, H, W, device):
    """Generate center, stripes, multi_hole masks. Returns dict."""
    masks = {}

    # Center mask: mask out center 50% area
    mask_center = torch.ones(B, 1, H, W, device=device)
    h4, w4 = H // 4, W // 4
    mask_center[:, :, h4:3*h4, w4:3*w4] = 0.0
    masks['center'] = mask_center

    # Stripes mask: horizontal stripes (every other row)
    mask_stripes = torch.ones(B, 1, H, W, device=device)
    mask_stripes[:, :, ::2, :] = 0.0
    masks['stripes'] = mask_stripes

    # Multi-hole mask: random 30% masked
    mask_multi = (torch.rand(B, 1, H, W, device=device) > 0.3).float()
    masks['multi_hole'] = mask_multi

    return masks


# ============================================================================
# CONV PROBE (mixed training)
# ============================================================================

class ConvProbe(nn.Module):
    def __init__(self, n_bits, H, W, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, n_classes)
        )

    def forward(self, z):
        return self.net(z.float())


def train_probe_mixed(probe, z_clean, z_repair, labels, device, epochs=30, bs=128):
    """Train probe on 50/50 mix of clean and repaired z."""
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    N = len(z_clean)
    for ep in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            # 50/50 mix
            use_repair = torch.rand(len(idx)) < 0.5
            z_batch = torch.where(
                use_repair.view(-1, 1, 1, 1),
                z_repair[idx],
                z_clean[idx]
            ).to(device)
            y = labels[idx].to(device)
            logits = probe(z_batch)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
    return total_loss / max(1, N // bs)


@torch.no_grad()
def eval_probe(probe, z, labels, device, bs=128):
    correct = 0; total = 0
    for i in range(0, len(z), bs):
        z_b = z[i:i+bs].to(device)
        y_b = labels[i:i+bs].to(device)
        pred = probe(z_b).argmax(dim=1)
        correct += (pred == y_b).sum().item()
        total += len(y_b)
    return correct / total


# ============================================================================
# GENERATION (reuse from E2b)
# ============================================================================

@torch.no_grad()
def sample_flow_gen(step_fn, e_core, global_prior, n, K, H, W, device,
                    T=20, dt=0.5, sigma_schedule='cosine', lambda_global=0.0):
    """Unconditional generation with optional E_prior."""
    u = torch.randn(n, K, H, W, device=device) * 0.5
    trajectory = {'e_core': [], 'delta_u_norm': []}

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((n,), t_frac, device=device)
        e_grad = compute_e_core_grad(e_core, u)

        if global_prior is not None and lambda_global > 0:
            e_grad_global = global_prior.grad(u, device)
            lambda_t = lambda_global * (0.3 + 0.7 * t_frac)
            e_grad = e_grad + lambda_t * e_grad_global

        delta_u = step_fn(u, e_grad, t_tensor)
        trajectory['delta_u_norm'].append(delta_u.abs().mean().item())
        u = u + dt * delta_u

        sigma = get_sigma(sigma_schedule, step, T)
        if sigma > 0.01:
            u = u + sigma * torch.randn_like(u)

        z_cur = quantize(u)
        trajectory['e_core'].append(e_core.energy(z_cur).item())

    return quantize(u), trajectory


def hue_variance(x):
    result = compute_hue_var(x)
    return result['hue_var']


def color_kl(x_gen, x_real):
    gen_means = x_gen.mean(dim=(2, 3))
    real_means = x_real.mean(dim=(2, 3))
    kl_total = 0.0
    for c in range(gen_means.shape[1]):
        g_hist = torch.histc(gen_means[:, c], bins=50, min=0, max=1) + 1e-8
        r_hist = torch.histc(real_means[:, c], bins=50, min=0, max=1) + 1e-8
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
    parser.add_argument('--output_dir', default='outputs/exp_xcore_global_prior')
    parser.add_argument('--n_gen', type=int, default=256)
    parser.add_argument('--T', type=int, default=15)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("X-CORE: UNIFIED E_PRIOR FOR REPAIR + GENERATION")
    print("=" * 100)

    # ========== DATA ==========
    print("\n[1] Loading CIFAR-10...")
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

    # Reference metrics
    print("\n[2] Reference metrics...")
    real_hf_coh = hf_coherence_metric(test_x[:200], device)
    real_hf_noi = hf_noise_index(test_x[:200], device)
    real_hue_var = hue_variance(test_x[:200])
    print(f"    Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")

    all_results = {}

    # ========== CONFIG SWEEP ==========
    configs = [
        # (name, n_bits, prior_type, lambda)
        ('16bit_none',        16, 'none',        0.0),
        ('16bit_pos_marg',    16, 'pos_marg',    0.3),
        ('16bit_spatial_cov', 16, 'spatial_cov',  0.3),
        ('24bit_none',        24, 'none',        0.0),
        ('24bit_pos_marg',    24, 'pos_marg',    0.3),
        ('24bit_spatial_cov', 24, 'spatial_cov',  0.3),
    ]

    for cfg_name, n_bits, prior_type, lam in configs:
        print(f"\n{'='*80}")
        print(f"CONFIG: {cfg_name} (n_bits={n_bits}, prior={prior_type}, λ={lam})")
        print("=" * 80)

        # --- Train ADC/DAC ---
        torch.manual_seed(args.seed)
        enc = Encoder16(n_bits).to(device)
        dec = Decoder16(n_bits).to(device)
        adc_loss = train_adc(enc, dec, train_x, device, epochs=40, bs=32)
        print(f"    ADC loss: {adc_loss:.4f}")

        # --- Encode ---
        z_train = encode_all(enc, train_x, device, bs=32)
        z_test = encode_all(enc, test_x, device, bs=32)
        K, H, W = z_train.shape[1:]
        print(f"    z: {z_train.shape}, usage={z_train.float().mean():.3f}")

        # --- Train E_core ---
        e_core = DiffEnergyCore(n_bits).to(device)
        train_ecore(e_core, z_train, device, epochs=15, bs=128)

        # --- Train StepFn ---
        step_fn = FlatStepFn_Norm(n_bits).to(device)
        train_step_fn(step_fn, e_core, z_train, dec, device, epochs=30, bs=32)

        # --- Build prior ---
        global_prior = None
        if prior_type == 'pos_marg':
            global_prior = MarginalPrior(z_train)
            print(f"    Built MarginalPrior")
        elif prior_type == 'spatial_cov':
            global_prior = CovariancePrior(z_train)
            print(f"    Built CovariancePrior")

        result = {
            'n_bits': n_bits, 'prior_type': prior_type, 'lambda': lam,
        }

        # ==================================================
        # PART A: REPAIR (the novel part)
        # ==================================================
        print(f"\n  --- REPAIR MODE ---")
        masks_dict = make_masks(len(z_test), K, H, W, device='cpu')

        repair_results = {}
        for mask_name, mask_cpu in masks_dict.items():
            z_clean_gpu = z_test.to(device)
            mask_gpu = mask_cpu.to(device)

            # Apply mask to z
            z_masked = z_clean_gpu * mask_gpu.expand(-1, K, -1, -1)

            # Flow-based repair with prior
            z_rep_list = []
            for ri in range(0, len(z_test), 32):
                nb = min(32, len(z_test) - ri)
                z_batch = z_masked[ri:ri+nb]
                m_batch = mask_gpu[ri:ri+nb]
                z_c_batch = z_clean_gpu[ri:ri+nb]
                z_rep = repair_flow_with_prior(
                    step_fn, e_core, z_c_batch, m_batch, device,
                    global_prior=global_prior, lambda_global=lam,
                    T=args.T, dt=0.5)
                z_rep_list.append(z_rep.cpu())
            z_repaired = torch.cat(z_rep_list)

            # Contract metrics
            mask_expanded = mask_cpu.expand(-1, K, -1, -1)
            ham_unmasked, ham_masked = compute_hamming_stats(
                z_test, z_repaired, mask_expanded)

            # Distribution shift metrics
            kl_marg = compute_kl_marg(z_repaired, z_test)
            delta_cov = compute_delta_cov(z_repaired, z_test)

            repair_results[mask_name] = {
                'ham_unmasked': ham_unmasked,
                'ham_masked': ham_masked,
                'kl_marg': kl_marg,
                'delta_cov': delta_cov,
            }

            print(f"    {mask_name}: ham_unmask={ham_unmasked:.4f}  "
                  f"ham_mask={ham_masked:.4f}  "
                  f"KL_marg={kl_marg:.4f}  Δcov={delta_cov:.4f}")

        result['repair'] = repair_results

        # Probe: use center mask repair for classification test
        z_rep_center_list = []
        mask_center = masks_dict['center']
        for ri in range(0, len(z_train), 32):
            nb = min(32, len(z_train) - ri)
            z_batch = z_train[ri:ri+nb].to(device)
            m_batch = mask_center[0:1].expand(nb, -1, -1, -1).to(device)
            z_masked_b = z_batch * m_batch.expand(-1, K, -1, -1)
            z_rep = repair_flow_with_prior(
                step_fn, e_core, z_batch, m_batch, device,
                global_prior=global_prior, lambda_global=lam,
                T=args.T, dt=0.5)
            z_rep_center_list.append(z_rep.cpu())
        z_train_repaired = torch.cat(z_rep_center_list)

        # Also repair test set with center mask
        z_test_rep_list = []
        for ri in range(0, len(z_test), 32):
            nb = min(32, len(z_test) - ri)
            z_batch = z_test[ri:ri+nb].to(device)
            m_batch = mask_center[0:1].expand(nb, -1, -1, -1).to(device)
            z_rep = repair_flow_with_prior(
                step_fn, e_core, z_batch, m_batch, device,
                global_prior=global_prior, lambda_global=lam,
                T=args.T, dt=0.5)
            z_test_rep_list.append(z_rep.cpu())
        z_test_repaired = torch.cat(z_test_rep_list)

        # Train mixed probe
        probe = ConvProbe(n_bits, H, W).to(device)
        train_probe_mixed(probe, z_train, z_train_repaired, train_y, device,
                          epochs=30, bs=128)

        acc_clean = eval_probe(probe, z_test, test_y, device)
        acc_repair = eval_probe(probe, z_test_repaired, test_y, device)
        gap = acc_clean - acc_repair

        result['probe_clean'] = acc_clean
        result['probe_repair'] = acc_repair
        result['probe_gap'] = gap
        print(f"    Probe: clean={acc_clean:.3f}  repair={acc_repair:.3f}  gap={gap:.3f}")

        # ==================================================
        # PART B: GENERATION
        # ==================================================
        print(f"\n  --- GENERATION MODE ---")
        torch.manual_seed(args.seed + 100)
        z_gen_list, all_traj = [], []
        for gi in range(0, args.n_gen, 32):
            nb = min(32, args.n_gen - gi)
            z_batch, traj = sample_flow_gen(
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

        r_gen, x_gen = evaluate(z_gen, dec, enc, e_core, z_train, test_x,
                                real_hf_coh, real_hf_noi, device, trajectory=agg_traj)
        r_gen['hue_var'] = hue_variance(x_gen)
        r_gen['color_kl'] = color_kl(x_gen, test_x[:len(x_gen)])

        result['gen'] = {k: v for k, v in r_gen.items()
                         if not isinstance(v, (list, np.ndarray))}

        save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))

        print(f"    viol={r_gen['violation']:.4f}  div={r_gen['diversity']:.4f}  "
              f"HF_noise={r_gen['hf_noise_index']:.2f}")
        print(f"    HueVar={r_gen['hue_var']:.4f}  ColorKL={r_gen['color_kl']:.4f}  "
              f"conn={r_gen['connectedness']:.4f}")

        all_results[cfg_name] = result

        del enc, dec, e_core, step_fn, global_prior, probe
        torch.cuda.empty_cache()

    # ========== SUMMARY ==========
    print(f"\n{'='*100}")
    print("X-CORE SUMMARY: UNIFIED E_PRIOR FOR REPAIR + GENERATION")
    print(f"Real: HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")
    print("=" * 100)

    # Repair summary
    print(f"\n--- REPAIR: DISTRIBUTION SHIFT ---")
    header = (f"{'config':<22} {'bits':>5} {'prior':>12} "
              f"{'ham_un':>7} {'KL_m_c':>7} {'KL_m_s':>7} {'KL_m_mh':>7} "
              f"{'Δcov_c':>7} {'Δcov_s':>7} {'Δcov_mh':>7} "
              f"{'p_cln':>6} {'p_rep':>6} {'gap':>6}")
    print(header); print("-" * len(header))
    for name, r in all_results.items():
        rep = r['repair']
        print(f"{name:<22} {r['n_bits']:>5} {r['prior_type']:>12} "
              f"{rep['center']['ham_unmasked']:>7.4f} "
              f"{rep['center']['kl_marg']:>7.4f} "
              f"{rep['stripes']['kl_marg']:>7.4f} "
              f"{rep['multi_hole']['kl_marg']:>7.4f} "
              f"{rep['center']['delta_cov']:>7.4f} "
              f"{rep['stripes']['delta_cov']:>7.4f} "
              f"{rep['multi_hole']['delta_cov']:>7.4f} "
              f"{r['probe_clean']:>6.3f} {r['probe_repair']:>6.3f} "
              f"{r['probe_gap']:>6.3f}")

    # Generation summary
    print(f"\n--- GENERATION QUALITY ---")
    header2 = (f"{'config':<22} {'bits':>5} {'prior':>12} "
               f"{'viol':>7} {'div':>7} {'HFnoi':>7} "
               f"{'HueV':>7} {'ColKL':>7} {'conn':>7}")
    print(header2); print("-" * len(header2))
    for name, r in all_results.items():
        g = r['gen']
        print(f"{name:<22} {r['n_bits']:>5} {r['prior_type']:>12} "
              f"{g['violation']:>7.4f} {g['diversity']:>7.4f} "
              f"{g['hf_noise_index']:>7.2f} "
              f"{g.get('hue_var', 0):>7.4f} {g.get('color_kl', 0):>7.4f} "
              f"{g['connectedness']:>7.4f}")

    # Causal chain analysis
    print(f"\n--- CAUSAL CHAIN: prior → KL_marg ↓ → gap ↓ ? ---")
    for bits in [16, 24]:
        none_name = f'{bits}bit_none'
        for prior in ['pos_marg', 'spatial_cov']:
            prior_name = f'{bits}bit_{prior}'
            if none_name in all_results and prior_name in all_results:
                r_none = all_results[none_name]
                r_prior = all_results[prior_name]
                kl_none = r_none['repair']['center']['kl_marg']
                kl_prior = r_prior['repair']['center']['kl_marg']
                gap_none = r_none['probe_gap']
                gap_prior = r_prior['probe_gap']
                kl_improved = kl_prior < kl_none
                gap_improved = gap_prior < gap_none
                print(f"  {bits}bit {prior}: "
                      f"KL_marg {kl_none:.4f}→{kl_prior:.4f} ({'↓' if kl_improved else '↑'})  "
                      f"gap {gap_none:.3f}→{gap_prior:.3f} ({'↓' if gap_improved else '↑'})  "
                      f"{'CAUSAL ✓' if kl_improved and gap_improved else 'NO CAUSAL'}")

    # Save
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = v
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\n{'='*100}")
    print("X-CORE COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
