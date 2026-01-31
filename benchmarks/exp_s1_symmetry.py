#!/usr/bin/env python3
"""
S1/S2: Symmetry-Compiled Loss — Equivariance & Invariance Gate Test
====================================================================
Core hypothesis: Encoder equivariance (E_sym) is a "physical ADC constraint"
that improves both classification readability and generation multi-modality,
without breaking the repair contract or cycle stability.

Background:
  - Phase 6 (#14): D4 constraint on E_core had limited effect
    Root cause: encoder itself lacks equivariance → constraint can't propagate
  - S1 fixes this: constrain the encoder mapping z(Tx) ≈ P_T(z(x))
  - P_T is pure index remap (flip/rot on z-grid), netlist-friendly
  - This is "geometric fidelity" (physics), not "semantic injection" (representation learning)

S2 is a one-time gate test: does encoder INVARIANCE (same z for augmented views)
significantly boost classification? If yes → "pay the price"; if no → close the door.

Configs (5):
  A0: baseline          — recon only
  A1: sym_flip          — +E_sym (flipH, flipV)
  A2: sym_d4            — +E_sym (flipH, flipV, rot90, rot180)
  B1: inv_weak          — +L_inv (color_jitter 0.2, gauss 0.05) [RED LINE TEST]
  B2: inv_strong        — +L_inv (color_jitter 0.4, gauss 0.1)  [RED LINE TEST]

Pre-registered gates:
  Hard (must pass):
    - ham_unmasked == 0.000
    - cycle_repair ≤ baseline + 0.02
  Success (any 2 of 3):
    - classify: acc_clean +2% OR gap -1%
    - generation: HueVar ×2 OR ColorKL -15%
    - diversity: div ≥ baseline - 0.03

Fixed: 16×16×16 z, FlatStepFn_Norm energy hinge, T=10, CIFAR-10 3000/500
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
# D4 TRANSFORMS — pure index remap on z-grid (netlist = wire routing)
# ============================================================================

def flip_h(t):
    """Horizontal flip: z[:,:,:,j] → z[:,:,:,W-1-j]"""
    return t.flip(3)

def flip_v(t):
    """Vertical flip: z[:,:,i,:] → z[:,:,H-1-i,:]"""
    return t.flip(2)

def rot90_ccw(t):
    """90° counter-clockwise: z[:,:,i,j] → z[:,:,j,H-1-i]"""
    return t.rot90(1, [2, 3])

def rot180(t):
    """180°: z[:,:,i,j] → z[:,:,H-1-i,W-1-j]"""
    return t.rot90(2, [2, 3])


TRANSFORMS_FLIP = [
    ('flipH', flip_h, flip_h),   # (name, T_img, P_z) — same op for both
    ('flipV', flip_v, flip_v),
]

TRANSFORMS_D4 = TRANSFORMS_FLIP + [
    ('rot90',  rot90_ccw, rot90_ccw),
    ('rot180', rot180, rot180),
]


# ============================================================================
# AUGMENTATION FUNCTIONS FOR S2 (invariance, not equivariance)
# ============================================================================

def make_color_jitter(strength):
    """Color jitter: random brightness/contrast shift."""
    def aug(x):
        B = x.shape[0]
        # Random brightness shift per image
        shift = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 2 * strength
        return (x + shift).clamp(0, 1)
    return aug

def make_gauss_noise(sigma):
    """Add Gaussian noise."""
    def aug(x):
        return (x + torch.randn_like(x) * sigma).clamp(0, 1)
    return aug


INV_AUGMENTS_WEAK = [
    ('color_jitter', lambda: make_color_jitter(0.2)),
    ('gauss_noise',  lambda: make_gauss_noise(0.05)),
]

INV_AUGMENTS_STRONG = [
    ('color_jitter', lambda: make_color_jitter(0.4)),
    ('gauss_noise',  lambda: make_gauss_noise(0.1)),
]


# ============================================================================
# MODIFIED ADC TRAINING WITH E_SYM / L_INV
# ============================================================================

def compute_grad_norm(loss, params):
    """Compute gradient norm without stepping optimizer."""
    grads = torch.autograd.grad(loss, params, create_graph=False,
                                retain_graph=True, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.data.norm().item() ** 2
    return total ** 0.5


def train_adc_sym(encoder, decoder, train_x, device,
                  sym_transforms=None, inv_augments=None,
                  epochs=40, bs=32):
    """Train ADC/DAC with optional equivariance (E_sym) or invariance (L_inv) loss.

    E_sym: MSE(logits(Tx), P_T(logits(x))) — both sides get gradients
    L_inv: MSE(logits(aug(x)), logits(x).detach()) — only aug side gets gradients

    λ auto-normalized on first batch: λ = grad_norm(L_recon) / grad_norm(L_extra)
    """
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    lambda_extra = None  # auto-normalized

    # Build aug functions if needed
    aug_fns = None
    if inv_augments is not None:
        aug_fns = [fn_factory() for _, fn_factory in inv_augments]

    for epoch in tqdm(range(epochs), desc="ADC"):
        encoder.train(); decoder.train()
        if hasattr(encoder, 'set_temperature'):
            encoder.set_temperature(1.0 + (0.3 - 1.0) * epoch / max(epochs - 1, 1))
        perm = torch.randperm(len(train_x))
        tl, tl_extra, nb = 0., 0., 0

        for i in range(0, len(train_x), bs):
            x = train_x[perm[i:i+bs]].to(device)
            opt.zero_grad()

            z, logits = encoder(x)
            xh = decoder(z)
            loss_recon = F.mse_loss(xh, x) + 0.5 * F.binary_cross_entropy(xh, x)

            # ---- E_sym: equivariance ----
            if sym_transforms is not None:
                loss_sym = 0.0
                for name, T_img, P_z in sym_transforms:
                    x_t = T_img(x)
                    _, logits_t = encoder(x_t)
                    target = P_z(logits)
                    loss_sym += F.mse_loss(logits_t, target)
                loss_sym = loss_sym / len(sym_transforms)

                # Auto-normalize λ on first batch
                if lambda_extra is None:
                    g_recon = compute_grad_norm(loss_recon, params)
                    g_sym = compute_grad_norm(loss_sym, params)
                    lambda_extra = g_recon / max(g_sym, 1e-8)
                    lambda_extra = min(lambda_extra, 10.0)  # cap at 10
                    print(f"    Auto-norm λ_sym = {lambda_extra:.3f} "
                          f"(g_recon={g_recon:.4f}, g_sym={g_sym:.4f})")

                loss = loss_recon + lambda_extra * loss_sym
                tl_extra += loss_sym.item()

            # ---- L_inv: invariance ----
            elif aug_fns is not None:
                loss_inv = 0.0
                logits_detached = logits.detach()  # stop gradient on original
                for aug_fn in aug_fns:
                    x_aug = aug_fn(x)
                    _, logits_aug = encoder(x_aug)
                    loss_inv += F.mse_loss(logits_aug, logits_detached)
                loss_inv = loss_inv / len(aug_fns)

                # Auto-normalize λ on first batch
                if lambda_extra is None:
                    g_recon = compute_grad_norm(loss_recon, params)
                    g_inv = compute_grad_norm(loss_inv, params)
                    lambda_extra = g_recon / max(g_inv, 1e-8)
                    lambda_extra = min(lambda_extra, 10.0)
                    print(f"    Auto-norm λ_inv = {lambda_extra:.3f} "
                          f"(g_recon={g_recon:.4f}, g_inv={g_inv:.4f})")

                loss = loss_recon + lambda_extra * loss_inv
                tl_extra += loss_inv.item()

            # ---- Baseline: recon only ----
            else:
                loss = loss_recon

            loss.backward()
            opt.step()
            tl += loss_recon.item()
            nb += 1

    encoder.eval(); decoder.eval()
    avg_recon = tl / max(nb, 1)
    avg_extra = tl_extra / max(nb, 1)
    return avg_recon, avg_extra, lambda_extra


# ============================================================================
# EQUIVARIANCE DIAGNOSTIC — measure actual equivariance error
# ============================================================================

@torch.no_grad()
def measure_equivariance(encoder, test_x, transforms, device, bs=32):
    """Measure Hamming(z(Tx), P_T(z(x))) for each transform."""
    results = {}
    for name, T_img, P_z in transforms:
        total_hamming = 0.0
        total_bits = 0
        for i in range(0, len(test_x), bs):
            x = test_x[i:i+bs].to(device)
            z, _ = encoder(x)
            z_t, _ = encoder(T_img(x))
            z_target = P_z(z)
            hamming = (z_t != z_target).float().sum().item()
            total_hamming += hamming
            total_bits += z.numel()
        results[name] = total_hamming / total_bits
    return results


# ============================================================================
# REPAIR (from X-CORE)
# ============================================================================

def make_center_mask(B, K, H, W, device):
    mask = torch.ones(B, 1, H, W, device=device)
    h4, w4 = H // 4, W // 4
    mask[:, :, h4:3*h4, w4:3*w4] = 0.0
    return mask


@torch.no_grad()
def repair_flow(step_fn, e_core, z_clean, mask, device, T=10, dt=0.5):
    """Flow-based repair with evidence clamping."""
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

        # Evidence clamping
        u = mask * u_clean + (1 - mask) * u

    return mask * z_clean + (1 - mask) * quantize(u)


# ============================================================================
# GENERATION (from X-CORE)
# ============================================================================

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


# ============================================================================
# CLASSIFICATION PROBE (from X-CORE)
# ============================================================================

class ConvProbe(nn.Module):
    def __init__(self, n_bits, H, W, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
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
            z_batch = torch.where(
                use_repair.view(-1, 1, 1, 1),
                z_repair[idx], z_clean[idx]).to(device)
            y = labels[idx].to(device)
            loss = F.cross_entropy(probe(z_batch), y)
            opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def eval_probe(probe, z, labels, device, bs=128):
    correct, total = 0, 0
    for i in range(0, len(z), bs):
        z_b = z[i:i+bs].to(device)
        y_b = labels[i:i+bs].to(device)
        pred = probe(z_b).argmax(dim=1)
        correct += (pred == y_b).sum().item()
        total += len(y_b)
    return correct / total


# ============================================================================
# COLOR METRICS (from X-CORE)
# ============================================================================

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
    parser.add_argument('--output_dir', default='outputs/exp_s1_symmetry')
    parser.add_argument('--n_gen', type=int, default=256)
    parser.add_argument('--T', type=int, default=10)
    args = parser.parse_args()

    N_BITS = 16
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("S1/S2: SYMMETRY-COMPILED LOSS — EQUIVARIANCE & INVARIANCE GATE TEST")
    print("=" * 100)
    print(f"Device: {device}  |  Seed: {args.seed}  |  n_bits: {N_BITS}  |  T: {args.T}")

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

    # ========== REFERENCE METRICS ==========
    print("\n[2] Reference metrics...")
    real_hf_coh = hf_coherence_metric(test_x[:200], device)
    real_hf_noi = hf_noise_index(test_x[:200], device)
    real_hue_var = hue_variance(test_x[:200])
    print(f"    Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")

    # ========== CONFIGS ==========
    configs = [
        # (name, sym_transforms, inv_augments)
        ('A0_baseline',   None,              None),
        ('A1_sym_flip',   TRANSFORMS_FLIP,   None),
        ('A2_sym_d4',     TRANSFORMS_D4,     None),
        ('B1_inv_weak',   None,              INV_AUGMENTS_WEAK),
        ('B2_inv_strong', None,              INV_AUGMENTS_STRONG),
    ]

    all_results = {}

    for cfg_name, sym_transforms, inv_augments in configs:
        print(f"\n{'='*80}")
        print(f"CONFIG: {cfg_name}")
        print(f"  sym: {[t[0] for t in sym_transforms] if sym_transforms else 'none'}")
        print(f"  inv: {[a[0] for a in inv_augments] if inv_augments else 'none'}")
        print("=" * 80)

        result = {'name': cfg_name}

        # --- Train ADC/DAC ---
        torch.manual_seed(args.seed)
        enc = Encoder16(N_BITS).to(device)
        dec = Decoder16(N_BITS).to(device)
        recon_loss, extra_loss, lam = train_adc_sym(
            enc, dec, train_x, device,
            sym_transforms=sym_transforms,
            inv_augments=inv_augments,
            epochs=40, bs=32)
        print(f"    ADC: recon={recon_loss:.4f}  extra={extra_loss:.4f}  λ={lam}")
        result['recon_loss'] = recon_loss
        result['extra_loss'] = extra_loss
        result['lambda'] = lam

        # --- Equivariance diagnostic ---
        eqv = measure_equivariance(enc, test_x, TRANSFORMS_D4, device)
        print(f"    Equivariance error (Hamming fraction):")
        for tname, err in eqv.items():
            print(f"      {tname}: {err:.4f}")
        result['equivariance'] = eqv

        # --- Encode ---
        z_train = encode_all(enc, train_x, device, bs=32)
        z_test = encode_all(enc, test_x, device, bs=32)
        K, H, W = z_train.shape[1:]
        print(f"    z: {z_train.shape}, usage={z_train.float().mean():.3f}")

        # --- Train E_core ---
        e_core = DiffEnergyCore(N_BITS).to(device)
        train_ecore(e_core, z_train, device, epochs=15, bs=128)

        # --- Train StepFn ---
        step_fn = FlatStepFn_Norm(N_BITS).to(device)
        train_step_fn(step_fn, e_core, z_train, dec, device, epochs=30, bs=32)

        # ========================
        # PART A: REPAIR
        # ========================
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

        # Cycle: encode(decode(z_repaired))
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

        # ========================
        # PART B: GENERATION
        # ========================
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

        # ========================
        # PART C: CLASSIFICATION
        # ========================
        print(f"\n  --- CLASSIFICATION (conv probe, mixed) ---")

        # Repair train set for mixed training
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

        probe = ConvProbe(N_BITS, H, W).to(device)
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

    # ========== SUMMARY TABLES ==========
    print(f"\n{'='*100}")
    print("S1/S2 SUMMARY: SYMMETRY-COMPILED LOSS")
    print(f"Real: HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")
    print("=" * 100)

    # --- Equivariance diagnostic ---
    print(f"\n--- EQUIVARIANCE ERROR (Hamming fraction, lower = more equivariant) ---")
    eqv_header = f"{'config':<18} {'flipH':>8} {'flipV':>8} {'rot90':>8} {'rot180':>8}"
    print(eqv_header); print("-" * len(eqv_header))
    for name, r in all_results.items():
        eqv = r['equivariance']
        print(f"{name:<18} {eqv['flipH']:>8.4f} {eqv['flipV']:>8.4f} "
              f"{eqv['rot90']:>8.4f} {eqv['rot180']:>8.4f}")

    # --- Repair ---
    print(f"\n--- REPAIR CONTRACT ---")
    rep_header = f"{'config':<18} {'ham_un':>7} {'ham_m':>7} {'cycle':>7}"
    print(rep_header); print("-" * len(rep_header))
    for name, r in all_results.items():
        print(f"{name:<18} {r['ham_unmasked']:>7.4f} {r['ham_masked']:>7.4f} "
              f"{r['cycle_repair']:>7.4f}")

    # --- Generation ---
    print(f"\n--- GENERATION QUALITY ---")
    gen_header = f"{'config':<18} {'viol':>7} {'div':>7} {'HFnoi':>7} {'HueV':>7} {'ColKL':>7} {'conn':>7}"
    print(gen_header); print("-" * len(gen_header))
    for name, r in all_results.items():
        g = r['gen']
        print(f"{name:<18} {g['violation']:>7.4f} {g['diversity']:>7.4f} "
              f"{g['hf_noise_index']:>7.2f} {g.get('hue_var', 0):>7.4f} "
              f"{g.get('color_kl', 0):>7.4f} {g['connectedness']:>7.4f}")

    # --- Classification ---
    print(f"\n--- CLASSIFICATION (conv probe, mixed training) ---")
    cls_header = f"{'config':<18} {'clean':>7} {'repair':>7} {'gap':>7}"
    print(cls_header); print("-" * len(cls_header))
    for name, r in all_results.items():
        print(f"{name:<18} {r['acc_clean']:>7.3f} {r['acc_repair']:>7.3f} {r['gap']:>7.3f}")

    # --- Delta table (relative to A0) ---
    a0 = all_results.get('A0_baseline')
    if a0:
        print(f"\n--- DELTA vs A0_baseline ---")
        delta_header = (f"{'config':<18} {'Δclean':>7} {'Δrepair':>7} {'Δgap':>7} "
                        f"{'Δdiv':>7} {'ΔHFnoi':>7} {'ΔHueV':>7} {'ΔColKL':>7}")
        print(delta_header); print("-" * len(delta_header))
        for name, r in all_results.items():
            if name == 'A0_baseline':
                continue
            d_clean = r['acc_clean'] - a0['acc_clean']
            d_repair = r['acc_repair'] - a0['acc_repair']
            d_gap = r['gap'] - a0['gap']
            d_div = r['gen']['diversity'] - a0['gen']['diversity']
            d_hf = r['gen']['hf_noise_index'] - a0['gen']['hf_noise_index']
            d_hue = r['gen'].get('hue_var', 0) - a0['gen'].get('hue_var', 0)
            d_ckl = r['gen'].get('color_kl', 0) - a0['gen'].get('color_kl', 0)
            print(f"{name:<18} {d_clean:>+7.3f} {d_repair:>+7.3f} {d_gap:>+7.3f} "
                  f"{d_div:>+7.3f} {d_hf:>+7.1f} {d_hue:>+7.4f} {d_ckl:>+7.2f}")

    # --- Gate check ---
    print(f"\n--- PRE-REGISTERED GATE CHECK ---")
    a0_cycle = a0['cycle_repair'] if a0 else 0
    for name, r in all_results.items():
        if name == 'A0_baseline':
            continue
        hard_pass = (r['ham_unmasked'] < 0.001 and
                     r['cycle_repair'] <= a0_cycle + 0.02)
        d_clean = r['acc_clean'] - a0['acc_clean']
        d_gap = r['gap'] - a0['gap']
        d_div = r['gen']['diversity'] - a0['gen']['diversity']
        d_hue = r['gen'].get('hue_var', 0) - a0['gen'].get('hue_var', 0)
        d_ckl = r['gen'].get('color_kl', 0) - a0['gen'].get('color_kl', 0)
        hue_ratio = r['gen'].get('hue_var', 0) / max(a0['gen'].get('hue_var', 0), 1e-8)

        criteria_met = 0
        if d_clean >= 0.02 or d_gap <= -0.01:
            criteria_met += 1
        if hue_ratio >= 2.0 or (d_ckl <= -0.15 * abs(a0['gen'].get('color_kl', 1))):
            criteria_met += 1
        if d_div >= -0.03:
            criteria_met += 1

        verdict = "PASS" if hard_pass and criteria_met >= 2 else "FAIL"
        if not hard_pass:
            verdict = "HARD_FAIL"
        print(f"  {name}: hard={'PASS' if hard_pass else 'FAIL'}  "
              f"criteria={criteria_met}/3  verdict={verdict}")

    # Save
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*100}")
    print("S1/S2 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
