#!/usr/bin/env python3
"""
O0: Observability Upgrade — Make Every Bit "Testable"
======================================================
First-principles motivation (state-space theory):

  State equation:    z_{t+1} = F(z_t, u)      (flow operator)
  Output equation:   x = D(z)                  (decoder / DAC)

  Observability matrix analog: O = ∂D/∂z  (decoder Jacobian)
  If rank(O) < dim(z), some state dimensions are UNOBSERVABLE:
    changing those bits has no visible effect on the output.

  This is exactly "dead bits" — the root cause of:
    - Same-color generation (HueVar collapse): many z-configs decode to same mode
    - Classification ceiling: unobservable bits carry no usable information
    - Poor generation diversity: decoder maps large z-regions to same output

  In EDA terms: unobservable nodes can't be tested (DFT), can't be debugged,
  and don't contribute to circuit function. Same for z-bits.

Solution: Observability Floor Regularizer
  During ADC training, perturb logits u → u+ε, measure ||D(σ(u+ε)) - D(σ(u))||².
  Penalize when this sensitivity is too low.
  This is equivalent to maximizing effective rank of decoder Jacobian.

  Deployment: obs_floor is a DESIGN-TIME constraint (like DFT insertion).
  The trained system operates purely in discrete z-space.

Configs (3):
  O0-A: baseline 16bit         — control
  O0-B: +obs_floor 16bit       — observability regularizer only
  O0-C: +obs_floor 24bit       — combined with bandwidth (known to help HF_noise)

New metrics:
  - dead_bit_ratio: fraction of bits with influence < threshold
  - mean_bit_influence: average ||D(z⊕δ) - D(z)|| per single-bit flip
  - state_identifiability: consistency of repair across random seeds

Pre-registered gates:
  Hard: ham_unmasked == 0.000, cycle ≤ baseline + 0.02
  Success (any 2 of 3):
    - dead_bit_ratio decreased ≥ 20%
    - HueVar increased ≥ 2×  OR  classify acc_clean +2%
    - div ≥ baseline - 0.03

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
    encode_all, train_ecore, train_step_fn
)

from exp_e2a_global_prior import compute_hue_var, compute_marginal_kl


# ============================================================================
# OBSERVABILITY FLOOR REGULARIZER
# ============================================================================

def compute_obs_loss(decoder, z_hard, logits, sigma=0.3):
    """Observability floor: penalize decoder insensitivity to z-changes.

    Perturb logits u → u+ε, decode both, measure output sensitivity.
    Loss = -log(sensitivity + eps) — naturally acts as soft floor:
      - sensitivity → 0 (dead bits): loss → ∞ (strong penalty)
      - sensitivity large: loss → small (diminishing returns, no instability)

    This maximizes the effective rank of the decoder Jacobian ∂D/∂z,
    ensuring every z-bit maps to a visible output difference.

    Args:
        decoder: DAC module
        z_hard: quantized z from encoder (B, K, H, W)
        logits: pre-quantization logits u (B, K, H, W)
        sigma: perturbation scale in logit space

    Returns:
        obs_loss: scalar — lower = more observable
        sensitivity: scalar — average output sensitivity (diagnostic)
    """
    # Perturb logits (Gaussian noise in logit space)
    noise = torch.randn_like(logits) * sigma
    logits_pert = logits + noise

    # STE quantization of perturbed logits
    z_soft_pert = torch.sigmoid(logits_pert)
    z_hard_pert = (z_soft_pert > 0.5).float()
    z_ste_pert = z_hard_pert - z_soft_pert.detach() + z_soft_pert

    # Decode original and perturbed
    x_orig = decoder(z_hard)
    x_pert = decoder(z_ste_pert)

    # Per-sample output change (MSE)
    delta_x = (x_orig - x_pert).pow(2).mean(dim=(1, 2, 3))  # (B,)

    # Per-sample input change (in logit space, for normalization)
    delta_u = noise.pow(2).mean(dim=(1, 2, 3))  # (B,)

    # Sensitivity = output_change / input_change
    sensitivity = delta_x / (delta_u + 1e-8)  # (B,)

    # Loss: -log(sensitivity) — soft floor, no threshold to tune
    obs_loss = -torch.log(sensitivity + 1e-6).mean()

    return obs_loss, sensitivity.mean().item()


def compute_grad_norm(loss, params):
    """Compute gradient norm for auto-normalization."""
    grads = torch.autograd.grad(loss, params, create_graph=False,
                                retain_graph=True, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.data.norm().item() ** 2
    return total ** 0.5


# ============================================================================
# MODIFIED ADC TRAINING WITH OBSERVABILITY FLOOR
# ============================================================================

def train_adc_obs(encoder, decoder, train_x, device,
                  use_obs_floor=False, obs_sigma=0.3,
                  epochs=40, bs=32):
    """Train ADC/DAC with optional observability floor regularizer.

    When use_obs_floor=True, adds -log(sensitivity) loss to maximize
    the effective rank of the decoder Jacobian ∂D/∂z.

    λ_obs auto-normalized on first batch: λ = grad_norm(L_recon) / grad_norm(L_obs)
    """
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    lambda_obs = None

    for epoch in tqdm(range(epochs), desc="ADC"):
        encoder.train(); decoder.train()
        if hasattr(encoder, 'set_temperature'):
            encoder.set_temperature(1.0 + (0.3 - 1.0) * epoch / max(epochs - 1, 1))
        perm = torch.randperm(len(train_x))
        tl_recon, tl_obs, nb = 0., 0., 0
        sens_accum = 0.

        for i in range(0, len(train_x), bs):
            x = train_x[perm[i:i+bs]].to(device)
            opt.zero_grad()

            z, logits = encoder(x)
            xh = decoder(z)
            loss_recon = F.mse_loss(xh, x) + 0.5 * F.binary_cross_entropy(xh, x)

            if use_obs_floor:
                obs_loss, sens = compute_obs_loss(decoder, z, logits, sigma=obs_sigma)

                # Auto-normalize λ on first batch
                if lambda_obs is None:
                    g_recon = compute_grad_norm(loss_recon, params)
                    g_obs = compute_grad_norm(obs_loss, params)
                    lambda_obs = g_recon / max(g_obs, 1e-8)
                    lambda_obs = min(lambda_obs, 5.0)  # cap
                    print(f"    Auto-norm λ_obs = {lambda_obs:.4f} "
                          f"(g_recon={g_recon:.4f}, g_obs={g_obs:.4f})")

                loss = loss_recon + lambda_obs * obs_loss
                tl_obs += obs_loss.item()
                sens_accum += sens
            else:
                loss = loss_recon

            loss.backward()
            opt.step()
            tl_recon += loss_recon.item()
            nb += 1

    encoder.eval(); decoder.eval()
    return {
        'recon_loss': tl_recon / max(nb, 1),
        'obs_loss': tl_obs / max(nb, 1),
        'lambda_obs': lambda_obs,
        'avg_sensitivity': sens_accum / max(nb, 1) if use_obs_floor else None,
    }


# ============================================================================
# OBSERVABILITY DIAGNOSTICS — "dead bit" analysis
# ============================================================================

@torch.no_grad()
def measure_dead_bits(encoder, decoder, test_x, device,
                      n_probes=200, threshold=1e-5, n_images=100):
    """Measure fraction of z-bits that are 'dead' (flipping has no visible effect).

    This is the discrete analog of checking rank(∂D/∂z):
    each probe tests one column of the Jacobian.

    Returns:
        dead_ratio: fraction of probed bits with influence < threshold
        mean_influence: average per-bit pixel change
        influence_histogram: for analysis
    """
    # Encode subset of test images
    x_sub = test_x[:n_images]
    z_list = []
    for i in range(0, len(x_sub), 32):
        z, _ = encoder(x_sub[i:i+32].to(device))
        z_list.append(z.cpu())
    z_all = torch.cat(z_list)  # (N, K, H, W)

    # Decode originals
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

        # Per-bit influence: average pixel change across images
        influence = (x_orig - x_flip).pow(2).mean().item()
        influences.append(influence)

    influences = np.array(influences)
    return {
        'dead_ratio': float((influences < threshold).mean()),
        'mean_influence': float(influences.mean()),
        'min_influence': float(influences.min()),
        'median_influence': float(np.median(influences)),
        'p10_influence': float(np.percentile(influences, 10)),
        'p90_influence': float(np.percentile(influences, 90)),
    }


# ============================================================================
# STATE IDENTIFIABILITY — repair consistency across random seeds
# ============================================================================

@torch.no_grad()
def measure_state_identifiability(step_fn, e_core, z_test, device,
                                   n_runs=5, T=10, n_images=100):
    """Measure how deterministic repair is across random initializations.

    Same mask + same z_clean, but different random init for masked region.
    High identifiability = observations fully determine the repaired state.

    Returns:
        identifiability: [0,1] — 1 = all runs agree on every masked bit
    """
    z_sub = z_test[:n_images]
    B, K, H, W = z_sub.shape

    # Center mask
    mask = torch.ones(1, 1, H, W)
    h4, w4 = H // 4, W // 4
    mask[:, :, h4:3*h4, w4:3*w4] = 0.0
    mask_exp = mask.expand(B, K, -1, -1)

    z_repairs = []
    for run in range(n_runs):
        torch.manual_seed(run * 12345)
        z_rep_list = []
        for i in range(0, B, 32):
            nb = min(32, B - i)
            z_batch = z_sub[i:i+nb].to(device)
            m_batch = mask.expand(nb, -1, -1, -1).to(device)

            # Evidence clamping repair
            u_clean = torch.where(z_batch > 0.5,
                                  torch.tensor(2.0, device=device),
                                  torch.tensor(-2.0, device=device))
            m_k = m_batch.expand(-1, K, -1, -1)
            u_noise = torch.randn(nb, K, H, W, device=device) * 0.3
            u = m_k * u_clean + (1 - m_k) * u_noise

            for step in range(T):
                t_frac = 1.0 - step / T
                t_tensor = torch.full((nb,), t_frac, device=device)
                e_grad = compute_e_core_grad(e_core, u)
                delta_u = step_fn(u, e_grad, t_tensor)
                u = u + 0.5 * delta_u
                sigma = get_sigma('cosine', step, T)
                if sigma > 0.01:
                    u = u + sigma * torch.randn_like(u) * (1 - m_k)
                u = m_k * u_clean + (1 - m_k) * u

            z_rep = m_k * z_batch + (1 - m_k) * quantize(u)
            z_rep_list.append(z_rep.cpu())
        z_repairs.append(torch.cat(z_rep_list))

    # Stack: (n_runs, B, K, H, W)
    z_stack = torch.stack(z_repairs).float()

    # Per-bit agreement: fraction of runs where bit=1
    p1 = z_stack.mean(dim=0)  # (B, K, H, W)
    # Certainty: distance from 0.5 (0=random, 1=deterministic)
    certainty = (2 * (p1 - 0.5)).abs()

    # Only on masked region
    mask_region = (1 - mask_exp).bool()
    identifiability = certainty[mask_region].mean().item()

    return {
        'identifiability': identifiability,
        'certainty_mean': certainty.mean().item(),
    }


# ============================================================================
# REPAIR / GENERATION / CLASSIFICATION (reuse from previous experiments)
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
    parser.add_argument('--output_dir', default='outputs/exp_o0_observability')
    parser.add_argument('--n_gen', type=int, default=256)
    parser.add_argument('--T', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("O0: OBSERVABILITY UPGRADE — DECODER JACOBIAN RANK MAXIMIZATION")
    print("=" * 100)
    print(f"Device: {device}  |  Seed: {args.seed}  |  T: {args.T}")
    print(f"\nFirst-principles: Observability = rank(∂D/∂z)")
    print(f"Dead bits = null space of decoder Jacobian")
    print(f"obs_floor = DFT insertion at design time\n")

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

    # ========== REFERENCE ==========
    print("\n[2] Reference metrics...")
    real_hf_coh = hf_coherence_metric(test_x[:200], device)
    real_hf_noi = hf_noise_index(test_x[:200], device)
    real_hue_var = hue_variance(test_x[:200])
    print(f"    Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")

    # ========== CONFIGS ==========
    configs = [
        # (name, n_bits, use_obs_floor)
        ('O0A_baseline_16bit',   16, False),
        ('O0B_obs_floor_16bit',  16, True),
        ('O0C_obs_floor_24bit',  24, True),
    ]

    all_results = {}

    for cfg_name, n_bits, use_obs in configs:
        print(f"\n{'='*80}")
        print(f"CONFIG: {cfg_name}  (n_bits={n_bits}, obs_floor={use_obs})")
        print("=" * 80)

        result = {'name': cfg_name, 'n_bits': n_bits, 'obs_floor': use_obs}

        # --- Train ADC/DAC ---
        torch.manual_seed(args.seed)
        enc = Encoder16(n_bits).to(device)
        dec = Decoder16(n_bits).to(device)
        adc_info = train_adc_obs(enc, dec, train_x, device,
                                 use_obs_floor=use_obs, obs_sigma=0.3,
                                 epochs=40, bs=32)
        print(f"    ADC: recon={adc_info['recon_loss']:.4f}  "
              f"obs={adc_info['obs_loss']:.4f}  λ={adc_info['lambda_obs']}  "
              f"sens={adc_info['avg_sensitivity']}")
        result['adc'] = adc_info

        # --- Dead-bit diagnostic ---
        print("\n  [Diagnostic] Measuring dead-bit ratio...")
        dead_info = measure_dead_bits(enc, dec, test_x, device,
                                      n_probes=200, n_images=100)
        print(f"    Dead ratio: {dead_info['dead_ratio']:.3f}  "
              f"Mean influence: {dead_info['mean_influence']:.6f}  "
              f"Min: {dead_info['min_influence']:.6f}  "
              f"P10: {dead_info['p10_influence']:.6f}")
        result['dead_bits'] = dead_info

        # --- Encode ---
        z_train = encode_all(enc, train_x, device, bs=32)
        z_test = encode_all(enc, test_x, device, bs=32)
        K, H, W = z_train.shape[1:]
        usage = z_train.float().mean().item()
        print(f"    z: {z_train.shape}, usage={usage:.3f}")
        result['z_usage'] = usage

        # --- Train E_core ---
        e_core = DiffEnergyCore(n_bits).to(device)
        train_ecore(e_core, z_train, device, epochs=15, bs=128)

        # --- Train StepFn ---
        step_fn = FlatStepFn_Norm(n_bits).to(device)
        train_step_fn(step_fn, e_core, z_train, dec, device, epochs=30, bs=32)

        # --- State identifiability ---
        print("\n  [Diagnostic] Measuring state identifiability...")
        ident_info = measure_state_identifiability(
            step_fn, e_core, z_test, device, n_runs=5, T=args.T, n_images=100)
        print(f"    Identifiability: {ident_info['identifiability']:.4f}")
        result['identifiability'] = ident_info

        # ==================================================
        # PART A: REPAIR
        # ==================================================
        print(f"\n  --- REPAIR (center mask) ---")
        mask_center_test = make_center_mask(len(z_test), K, H, W, device='cpu')
        z_rep_test_list = []
        for ri in range(0, len(z_test), 32):
            nb = min(32, len(z_test) - ri)
            z_batch = z_test[ri:ri+nb].to(device)
            m_batch = mask_center_test[0:1].expand(nb, -1, -1, -1).to(device)
            z_rep = repair_flow(step_fn, e_core, z_batch, m_batch, device,
                                T=args.T, dt=0.5)
            z_rep_test_list.append(z_rep.cpu())
        z_test_repaired = torch.cat(z_rep_test_list)

        # Contract metrics
        mask_exp = mask_center_test[0:1].expand(len(z_test), K, -1, -1)
        diff = (z_test != z_test_repaired).float()
        ham_unmasked = (diff * mask_exp).sum() / max(mask_exp.sum().item(), 1)
        ham_masked = (diff * (1 - mask_exp)).sum() / max((1 - mask_exp).sum().item(), 1)

        # Cycle
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
        # PART B: GENERATION
        # ==================================================
        print(f"\n  --- GENERATION ---")
        torch.manual_seed(args.seed + 100)
        z_gen_list, all_traj = [], []
        for gi in range(0, args.n_gen, 32):
            nb = min(32, args.n_gen - gi)
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

        # ==================================================
        # PART C: CLASSIFICATION
        # ==================================================
        print(f"\n  --- CLASSIFICATION (conv probe, mixed) ---")
        mask_center_train = make_center_mask(1, K, H, W, device='cpu')
        z_train_rep_list = []
        for ri in range(0, len(z_train), 32):
            nb = min(32, len(z_train) - ri)
            z_batch = z_train[ri:ri+nb].to(device)
            m_batch = mask_center_train.expand(nb, -1, -1, -1).to(device)
            z_rep = repair_flow(step_fn, e_core, z_batch, m_batch, device,
                                T=args.T, dt=0.5)
            z_train_rep_list.append(z_rep.cpu())
        z_train_repaired = torch.cat(z_train_rep_list)

        probe = ConvProbe(n_bits, H, W).to(device)
        train_probe_mixed(probe, z_train, z_train_repaired, train_y, device,
                          epochs=30, bs=128)

        acc_clean = eval_probe(probe, z_test, test_y, device)
        acc_repair = eval_probe(probe, z_test_repaired, test_y, device)
        gap = acc_clean - acc_repair

        result['acc_clean'] = acc_clean
        result['acc_repair'] = acc_repair
        result['gap'] = gap
        print(f"    acc_clean={acc_clean:.3f}  acc_repair={acc_repair:.3f}  gap={gap:.3f}")

        all_results[cfg_name] = result

        del enc, dec, e_core, step_fn, probe
        torch.cuda.empty_cache()

    # ========== SUMMARY ==========
    print(f"\n{'='*100}")
    print("O0 SUMMARY: OBSERVABILITY UPGRADE")
    print(f"Real: HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")
    print("=" * 100)

    # --- Observability diagnostics ---
    print(f"\n--- OBSERVABILITY DIAGNOSTICS (decoder Jacobian rank proxy) ---")
    obs_header = (f"{'config':<25} {'bits':>4} {'dead%':>6} {'mean_inf':>9} "
                  f"{'p10_inf':>9} {'median':>9} {'ident':>6}")
    print(obs_header); print("-" * len(obs_header))
    for name, r in all_results.items():
        db = r['dead_bits']
        ident = r['identifiability']['identifiability']
        print(f"{name:<25} {r['n_bits']:>4} {db['dead_ratio']:>6.3f} "
              f"{db['mean_influence']:>9.6f} {db['p10_influence']:>9.6f} "
              f"{db['median_influence']:>9.6f} {ident:>6.4f}")

    # --- Repair ---
    print(f"\n--- REPAIR CONTRACT ---")
    rep_header = f"{'config':<25} {'ham_un':>7} {'ham_m':>7} {'cycle':>7}"
    print(rep_header); print("-" * len(rep_header))
    for name, r in all_results.items():
        print(f"{name:<25} {r['ham_unmasked']:>7.4f} {r['ham_masked']:>7.4f} "
              f"{r['cycle_repair']:>7.4f}")

    # --- Generation ---
    print(f"\n--- GENERATION QUALITY ---")
    gen_header = (f"{'config':<25} {'viol':>7} {'div':>7} {'HFnoi':>7} "
                  f"{'HueV':>7} {'ColKL':>7} {'conn':>7}")
    print(gen_header); print("-" * len(gen_header))
    for name, r in all_results.items():
        g = r['gen']
        print(f"{name:<25} {g['violation']:>7.4f} {g['diversity']:>7.4f} "
              f"{g['hf_noise_index']:>7.2f} {g.get('hue_var', 0):>7.4f} "
              f"{g.get('color_kl', 0):>7.4f} {g['connectedness']:>7.4f}")

    # --- Classification ---
    print(f"\n--- CLASSIFICATION ---")
    cls_header = f"{'config':<25} {'clean':>7} {'repair':>7} {'gap':>7}"
    print(cls_header); print("-" * len(cls_header))
    for name, r in all_results.items():
        print(f"{name:<25} {r['acc_clean']:>7.3f} {r['acc_repair']:>7.3f} {r['gap']:>7.3f}")

    # --- Delta table ---
    a0 = all_results.get('O0A_baseline_16bit')
    if a0:
        print(f"\n--- DELTA vs O0A_baseline ---")
        dh = (f"{'config':<25} {'Δclean':>7} {'Δgap':>7} {'Δdiv':>7} "
              f"{'ΔHFnoi':>7} {'ΔHueV':>8} {'Δdead%':>7}")
        print(dh); print("-" * len(dh))
        for name, r in all_results.items():
            if name == 'O0A_baseline_16bit':
                continue
            d_clean = r['acc_clean'] - a0['acc_clean']
            d_gap = r['gap'] - a0['gap']
            d_div = r['gen']['diversity'] - a0['gen']['diversity']
            d_hf = r['gen']['hf_noise_index'] - a0['gen']['hf_noise_index']
            d_hue = r['gen'].get('hue_var', 0) - a0['gen'].get('hue_var', 0)
            d_dead = r['dead_bits']['dead_ratio'] - a0['dead_bits']['dead_ratio']
            print(f"{name:<25} {d_clean:>+7.3f} {d_gap:>+7.3f} {d_div:>+7.3f} "
                  f"{d_hf:>+7.1f} {d_hue:>+8.4f} {d_dead:>+7.3f}")

    # --- Gate check ---
    print(f"\n--- PRE-REGISTERED GATE CHECK ---")
    a0_cycle = a0['cycle_repair'] if a0 else 0
    a0_dead = a0['dead_bits']['dead_ratio'] if a0 else 1
    for name, r in all_results.items():
        if name == 'O0A_baseline_16bit':
            continue
        hard_pass = (r['ham_unmasked'] < 0.001 and
                     r['cycle_repair'] <= a0_cycle + 0.02)
        d_dead = r['dead_bits']['dead_ratio'] - a0_dead
        d_clean = r['acc_clean'] - a0['acc_clean']
        d_div = r['gen']['diversity'] - a0['gen']['diversity']
        hue_ratio = r['gen'].get('hue_var', 0) / max(a0['gen'].get('hue_var', 0), 1e-8)

        criteria = 0
        if d_dead <= -0.20 * a0_dead:
            criteria += 1
        if hue_ratio >= 2.0 or d_clean >= 0.02:
            criteria += 1
        if d_div >= -0.03:
            criteria += 1

        verdict = "PASS" if hard_pass and criteria >= 2 else "FAIL"
        if not hard_pass:
            verdict = "HARD_FAIL"
        print(f"  {name}: hard={'PASS' if hard_pass else 'FAIL'}  "
              f"criteria={criteria}/3  verdict={verdict}")

    # Save
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*100}")
    print("O0 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
