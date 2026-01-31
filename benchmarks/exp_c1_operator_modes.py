#!/usr/bin/env python3
"""
C1: Unified Operator Three-Mode Compatibility
==============================================
Tests flat_norm (F0c winner) across repair / generation / classification.

4 operator configs:
  Op-A: repair-first (mask → denoise flow training)
  Op-B: gen-first (noise → z flow training, same as F0c)
  Op-C: balanced (50% repair + 50% generation per batch)
  Op-D: energy-aware (flow + E_core descent hinge)

Evaluation for each operator:
  GENERATION: violation, diversity, HF_noise, connectedness, cycle, delta_u
  REPAIR: unmasked_hamming, repair_hamming, E_obs_drop, cycle_after_repair
  CLASSIFICATION: conv probe on clean z, repaired z, gap
  COST: INT4/INT8 activation quantization degradation on generation

Three hard deployment gates:
  1. Cost: operator works under INT4/INT8 activation quant
  2. Contract: repair unmasked_change ≈ 0, cycle bounded
  3. Mode-switch: repair not biased by gen training (and vice versa)

4GB GPU: 5000 train, 1000 test, batch_size=32, 16×16×16 z
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
    DiffEnergyCore, quantize, soft_quantize, compute_e_core_grad,
    train_adc, encode_all, train_ecore,
    evaluate
)

from exp_flow_f0c_fixes import FlatStepFn_Norm, get_sigma


# ============================================================================
# REPAIR WITH FLOW OPERATOR
# ============================================================================

@torch.no_grad()
def repair_flow(step_fn, e_core, z, mask, device, T=10, dt=0.5):
    """Repair corrupted z using flow operator.

    z: binary [B, K, H, W] — original z (evidence in unmasked region)
    mask: [B, K, H, W] — 1=keep (evidence), 0=repair
    Returns: repaired z (binary)
    """
    B = z.shape[0]
    u_evidence = z * 2.0 - 1.0  # {0,1} → {-1,+1}
    u = u_evidence * mask + torch.randn_like(u_evidence) * 0.5 * (1 - mask)

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((B,), t_frac, device=device)
        e_grad = compute_e_core_grad(e_core, u)
        delta_u = step_fn(u, e_grad, t_tensor)
        u = u + dt * delta_u
        # Clamp evidence: preserve unmasked region
        u = u_evidence * mask + u * (1 - mask)

    return quantize(u)


# ============================================================================
# GENERATION WITH FLOW OPERATOR
# ============================================================================

@torch.no_grad()
def sample_flow_gen(step_fn, e_core, n, K, H, W, device,
                    T=20, dt=0.5, sigma_schedule='cosine'):
    """Flow sampling from noise (generation mode)."""
    u = torch.randn(n, K, H, W, device=device) * 0.5

    trajectory = {'delta_u_norm': [], 'e_core': []}

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((n,), t_frac, device=device)
        e_grad = compute_e_core_grad(e_core, u)
        delta_u = step_fn(u, e_grad, t_tensor)

        trajectory['delta_u_norm'].append(delta_u.abs().mean().item())
        u = u + dt * delta_u

        sigma = get_sigma(sigma_schedule, step, T)
        if sigma > 0.01:
            u = u + sigma * torch.randn_like(u)

        z_cur = quantize(u)
        trajectory['e_core'].append(e_core.energy(z_cur).item())

    z_final = quantize(u)
    return z_final, trajectory


@torch.no_grad()
def sample_flow_gen_quantized(step_fn, e_core, n, K, H, W, device,
                               T=20, dt=0.5, sigma_schedule='cosine',
                               act_bits=8):
    """Flow sampling with activation quantization simulation.

    Quantizes u (intermediate state) to act_bits at each step.
    """
    def quant_act(x, bits):
        # Symmetric quantization of activations
        qmax = 2**(bits - 1) - 1
        scale = x.abs().max().clamp(min=1e-8) / qmax
        xq = (x / scale).round().clamp(-qmax, qmax) * scale
        return xq

    u = torch.randn(n, K, H, W, device=device) * 0.5

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((n,), t_frac, device=device)
        e_grad = compute_e_core_grad(e_core, u)
        delta_u = step_fn(u, e_grad, t_tensor)

        u = u + dt * delta_u
        # Quantize intermediate state
        u = quant_act(u, act_bits)

        sigma = get_sigma(sigma_schedule, step, T)
        if sigma > 0.01:
            u = u + sigma * torch.randn_like(u)

    return quantize(u)


# ============================================================================
# MASKS
# ============================================================================

def make_center_mask(B, K, H, W, device='cpu'):
    mask = torch.ones(B, K, H, W, device=device)
    h4, w4 = H // 4, W // 4
    mask[:, :, h4:3*h4, w4:3*w4] = 0
    return mask


def make_random_mask(B, K, H, W, device='cpu', keep_ratio=0.5):
    """Random per-position mask (each position kept independently)."""
    mask = (torch.rand(B, 1, H, W, device=device) < keep_ratio).float()
    return mask.expand(B, K, H, W)


def make_training_mask(B, K, H, W, device='cpu'):
    """Mixed mask for repair training (center 50%, random 50%)."""
    n_center = B // 2
    masks = []
    # Center masks
    if n_center > 0:
        masks.append(make_center_mask(n_center, K, H, W, device))
    # Random masks
    n_random = B - n_center
    if n_random > 0:
        masks.append(make_random_mask(n_random, K, H, W, device, keep_ratio=0.5))
    return torch.cat(masks, dim=0)


# ============================================================================
# TRAINING MODES
# ============================================================================

def train_repair_mode(step_fn, e_core, z_data, decoder, device,
                      epochs=30, bs=32, T_unroll=3, clip_grad=1.0):
    """Op-A: Repair-first training.

    Input: z_clean with mask applied → noise in masked region
    Target: reconstruct z_clean
    Evidence clamping at each step.
    """
    opt = torch.optim.Adam(step_fn.parameters(), lr=1e-3)
    N = len(z_data); K, H, W = z_data.shape[1:]

    for epoch in tqdm(range(epochs), desc="Op-A repair"):
        step_fn.train(); perm = torch.randperm(N)
        tl, nb = 0., 0
        progress = epoch / max(epochs - 1, 1)

        for i in range(0, N, bs):
            idx = perm[i:i+bs]; z_clean = z_data[idx].to(device); B = z_clean.shape[0]

            # Create masked input
            mask = make_training_mask(B, K, H, W, device)
            u_evidence = z_clean * 2.0 - 1.0
            u = u_evidence * mask + torch.randn_like(u_evidence) * (1 - mask)

            opt.zero_grad()

            for t_step in range(T_unroll):
                t_frac = 1.0 - t_step / T_unroll
                t_tensor = torch.full((B,), t_frac, device=device)
                e_grad = (torch.sigmoid(e_core.predictor(torch.sigmoid(u))) -
                          torch.sigmoid(u))
                delta_u = step_fn(u, e_grad, t_tensor)
                u = u + 0.5 * delta_u
                # Clamp evidence
                u = u_evidence * mask + u * (1 - mask)

            z_pred_soft = torch.sigmoid(u)
            loss = F.binary_cross_entropy(z_pred_soft, z_clean)

            loss.backward()
            nn.utils.clip_grad_norm_(step_fn.parameters(), clip_grad)
            opt.step()
            tl += loss.item(); nb += 1

        if (epoch + 1) % 10 == 0:
            print(f"      ep {epoch+1}/{epochs}: BCE={tl/nb:.4f}")

    step_fn.eval()


def train_gen_mode(step_fn, e_core, z_data, decoder, device,
                   epochs=30, bs=32, T_unroll=3, clip_grad=1.0):
    """Op-B: Generation-first training (same as F0c flat_norm)."""
    opt = torch.optim.Adam(step_fn.parameters(), lr=1e-3)
    N = len(z_data)

    for epoch in tqdm(range(epochs), desc="Op-B gen"):
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


def train_balanced_mode(step_fn, e_core, z_data, decoder, device,
                        epochs=30, bs=32, T_unroll=3, clip_grad=1.0):
    """Op-C: Balanced training (alternate repair / gen each batch)."""
    opt = torch.optim.Adam(step_fn.parameters(), lr=1e-3)
    N = len(z_data); K, H, W = z_data.shape[1:]

    for epoch in tqdm(range(epochs), desc="Op-C balanced"):
        step_fn.train(); perm = torch.randperm(N)
        tl, nb = 0., 0
        progress = epoch / max(epochs - 1, 1)
        batch_idx = 0

        for i in range(0, N, bs):
            idx = perm[i:i+bs]; z_clean = z_data[idx].to(device); B = z_clean.shape[0]

            opt.zero_grad()

            if batch_idx % 2 == 0:
                # REPAIR mode: evidence clamping
                mask = make_training_mask(B, K, H, W, device)
                u_evidence = z_clean * 2.0 - 1.0
                u = u_evidence * mask + torch.randn_like(u_evidence) * (1 - mask)

                for t_step in range(T_unroll):
                    t_frac = 1.0 - t_step / T_unroll
                    t_tensor = torch.full((B,), t_frac, device=device)
                    e_grad = (torch.sigmoid(e_core.predictor(torch.sigmoid(u))) -
                              torch.sigmoid(u))
                    delta_u = step_fn(u, e_grad, t_tensor)
                    u = u + 0.5 * delta_u
                    u = u_evidence * mask + u * (1 - mask)

                z_pred_soft = torch.sigmoid(u)
                loss = F.binary_cross_entropy(z_pred_soft, z_clean)
            else:
                # GENERATION mode: full noise
                noise_level = torch.rand(B, device=device)
                u_clean = z_clean * 2.0 - 1.0
                u_noisy = u_clean + torch.randn_like(u_clean) * noise_level.view(B, 1, 1, 1) * 2.0

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
            tl += loss.item(); nb += 1
            batch_idx += 1

        if (epoch + 1) % 10 == 0:
            print(f"      ep {epoch+1}/{epochs}: loss={tl/nb:.4f}")

    step_fn.eval()


def train_energy_mode(step_fn, e_core, z_data, decoder, device,
                      epochs=30, bs=32, T_unroll=3, clip_grad=1.0):
    """Op-D: Energy-aware training (gen + E_core descent hinge)."""
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
# CLASSIFICATION PROBE
# ============================================================================

class ConvProbe(nn.Module):
    def __init__(self, n_bits, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, n_classes))

    def forward(self, z):
        return self.net(z)


def train_probe(probe, z_data, labels, device, epochs=50, bs=256, lr=1e-3):
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for epoch in range(epochs):
        probe.train(); perm = torch.randperm(len(z_data))
        for i in range(0, len(z_data), bs):
            idx = perm[i:i+bs]
            z = z_data[idx].to(device); y = labels[idx].to(device)
            opt.zero_grad()
            loss = F.cross_entropy(probe(z), y)
            loss.backward(); opt.step()
    probe.eval()


def eval_probe(probe, z_data, labels, device, bs=256):
    probe.eval(); nc, nb = 0, 0
    with torch.no_grad():
        for i in range(0, len(z_data), bs):
            z = z_data[i:i+bs].to(device); y = labels[i:i+bs].to(device)
            nc += (probe(z).argmax(1) == y).sum().item(); nb += len(y)
    return nc / nb


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def eval_repair(step_fn, e_core, z_test, encoder, decoder, device, T=10, bs=32):
    """Evaluate repair quality on center mask."""
    K, H, W = z_test.shape[1:]
    z_repaired_list = []
    with torch.no_grad():
        for i in range(0, len(z_test), bs):
            z_batch = z_test[i:i+bs].to(device)
            B = z_batch.shape[0]
            mask = make_center_mask(B, K, H, W, device)
            z_rep = repair_flow(step_fn, e_core, z_batch, mask, device, T=T)
            z_repaired_list.append(z_rep.cpu())
    z_repaired = torch.cat(z_repaired_list)

    # Intervention stability: masked vs unmasked change
    B = len(z_test)
    mask_template = make_center_mask(B, K, H, W)
    diff = (z_test != z_repaired).float()
    masked_region = (1 - mask_template)
    unmasked_region = mask_template

    hamming_masked = (diff * masked_region).sum().item() / max(masked_region.sum().item(), 1)
    hamming_unmasked = (diff * unmasked_region).sum().item() / max(unmasked_region.sum().item(), 1)

    # Cycle test on repaired z
    cycle_list = []
    with torch.no_grad():
        for i in range(0, min(200, len(z_repaired)), bs):
            zr = z_repaired[i:i+bs].to(device)
            xr = decoder(zr); zcy, _ = encoder(xr)
            cycle_list.append((zr != zcy).float().mean().item())
    cycle_repair = np.mean(cycle_list)

    # E_obs: MSE between decoded z_repaired and decoded z_clean
    eobs_list = []
    with torch.no_grad():
        for i in range(0, min(200, len(z_repaired)), bs):
            zr = z_repaired[i:i+bs].to(device)
            zc = z_test[i:i+bs].to(device)
            xr = decoder(zr); xc = decoder(zc)
            eobs_list.append(F.mse_loss(xr, xc).item())
    eobs_drop = np.mean(eobs_list)

    return {
        'hamming_masked': hamming_masked,
        'hamming_unmasked': hamming_unmasked,
        'cycle_repair': cycle_repair,
        'eobs_drop': eobs_drop,
    }, z_repaired


def eval_gen_quantized(step_fn, e_core, encoder, decoder, z_data, test_x,
                       real_hf_coh, real_hf_noi, device, n_samples=256,
                       T=20, act_bits=8, seed=142):
    """Evaluate generation with activation quantization."""
    K, H, W = z_data.shape[1:]
    torch.manual_seed(seed)
    z_gen_list = []
    for gi in range(0, n_samples, 32):
        nb = min(32, n_samples - gi)
        z_batch = sample_flow_gen_quantized(
            step_fn, e_core, nb, K, H, W, device,
            T=T, act_bits=act_bits)
        z_gen_list.append(z_batch.cpu())
    z_gen = torch.cat(z_gen_list)

    r, _ = evaluate(z_gen, decoder, encoder, e_core, z_data, test_x,
                    real_hf_coh, real_hf_noi, device)
    return r


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_c1_operator_modes')
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--n_gen', type=int, default=256)
    parser.add_argument('--n_bits', type=int, default=16)
    parser.add_argument('--T_gen', type=int, default=20)
    parser.add_argument('--T_repair', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    N_BITS = args.n_bits

    print("=" * 100)
    print("C1: UNIFIED OPERATOR THREE-MODE COMPATIBILITY")
    print("=" * 100)

    # ========== DATA ==========
    print("\n[1] Loading CIFAR-10 with labels...")
    from torchvision import datasets, transforms
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())

    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(len(train_ds), args.n_train, replace=False)
    test_idx = rng.choice(len(test_ds), args.n_test, replace=False)

    train_x = torch.stack([train_ds[i][0] for i in train_idx])
    train_y = torch.tensor([train_ds[i][1] for i in train_idx])
    test_x = torch.stack([test_ds[i][0] for i in test_idx])
    test_y = torch.tensor([test_ds[i][1] for i in test_idx])
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")

    # ========== SHARED WORLD MODEL ==========
    print(f"\n[2] Training shared ADC/DAC (16×16×{N_BITS})...")
    torch.manual_seed(args.seed)
    encoder = Encoder16(N_BITS).to(device)
    decoder = Decoder16(N_BITS).to(device)
    adc_loss = train_adc(encoder, decoder, train_x, device, epochs=40, bs=32)
    print(f"    ADC loss: {adc_loss:.4f}")

    with torch.no_grad():
        tb = test_x[:32].to(device); zo, _ = encoder(tb)
        oracle_mse = F.mse_loss(decoder(zo), tb).item()
    print(f"    Oracle MSE: {oracle_mse:.4f}")

    print("\n[3] Encoding datasets...")
    z_train = encode_all(encoder, train_x, device, bs=32)
    z_test = encode_all(encoder, test_x, device, bs=32)
    K, H, W = z_train.shape[1:]
    print(f"    z_train: {z_train.shape}, z_test: {z_test.shape}, usage={z_train.mean():.3f}")

    print("\n[4] Training E_core...")
    e_core = DiffEnergyCore(N_BITS).to(device)
    train_ecore(e_core, z_train, device, epochs=15, bs=128)
    viol_train = e_core.violation_rate(z_train[:100].to(device))
    print(f"    Violation: {viol_train:.4f}")

    print("\n[5] Reference generation metrics...")
    real_hf_coh = hf_coherence_metric(test_x[:200], device)
    real_hf_noi = hf_noise_index(test_x[:200], device)
    print(f"    Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}")

    # Clean cycle baseline
    cycle_list = []
    with torch.no_grad():
        for i in range(0, min(200, len(z_test)), 32):
            zc = z_test[i:i+32].to(device)
            xc = decoder(zc); zcy, _ = encoder(xc)
            cycle_list.append((zc != zcy).float().mean().item())
    cycle_clean = np.mean(cycle_list)
    print(f"    Clean cycle: {cycle_clean:.4f}")

    # ========== TRAIN 4 OPERATORS ==========
    operator_configs = [
        ("Op-A_repair",   train_repair_mode),
        ("Op-B_gen",      train_gen_mode),
        ("Op-C_balanced", train_balanced_mode),
        ("Op-D_energy",   train_energy_mode),
    ]

    all_results = {}
    all_operators = {}

    for op_name, train_fn in operator_configs:
        print(f"\n{'='*80}")
        print(f"TRAINING: {op_name}")
        print("=" * 80)

        torch.manual_seed(args.seed)
        step_fn = FlatStepFn_Norm(N_BITS).to(device)
        n_params = sum(p.numel() for p in step_fn.parameters())
        print(f"    Params: {n_params:,}")

        train_fn(step_fn, e_core, z_train, decoder, device,
                 epochs=30, bs=32, T_unroll=3, clip_grad=1.0)
        all_operators[op_name] = step_fn

        # ========== EVAL GENERATION ==========
        print(f"\n  [GEN] Generating {args.n_gen} samples (T={args.T_gen})...")
        torch.manual_seed(args.seed + 100)
        z_gen_list = []; all_traj = []
        for gi in range(0, args.n_gen, 32):
            nb = min(32, args.n_gen - gi)
            z_batch, traj = sample_flow_gen(
                step_fn, e_core, nb, K, H, W, device,
                T=args.T_gen, dt=0.5, sigma_schedule='cosine')
            z_gen_list.append(z_batch.cpu())
            all_traj.append(traj)
        z_gen = torch.cat(z_gen_list)

        agg_traj = {
            key: [np.mean([t[key][s] for t in all_traj])
                  for s in range(len(all_traj[0][key]))]
            for key in all_traj[0].keys()
        }

        gen_r, x_gen = evaluate(z_gen, decoder, encoder, e_core, z_train, test_x,
                                real_hf_coh, real_hf_noi, device, trajectory=agg_traj)
        gen_r['final_delta_u'] = agg_traj['delta_u_norm'][-1]
        save_grid(x_gen, os.path.join(args.output_dir, f'gen_{op_name}.png'))
        print(f"    viol={gen_r['violation']:.4f}  div={gen_r['diversity']:.4f}  "
              f"cycle={gen_r['cycle']:.4f}  conn={gen_r['connectedness']:.4f}")
        print(f"    HF_noise={gen_r['hf_noise_index']:.2f}  delta_u={gen_r['final_delta_u']:.4f}")

        # ========== EVAL REPAIR ==========
        print(f"\n  [REPAIR] Center mask repair (T={args.T_repair})...")
        repair_r, z_test_repaired = eval_repair(
            step_fn, e_core, z_test, encoder, decoder, device,
            T=args.T_repair, bs=32)
        print(f"    ham_masked={repair_r['hamming_masked']:.4f}  "
              f"ham_unmasked={repair_r['hamming_unmasked']:.4f}  "
              f"cycle={repair_r['cycle_repair']:.4f}")

        # ========== EVAL CLASSIFICATION ==========
        print(f"\n  [CLASSIFY] Conv probe (mixed training)...")
        # Train mixed probe: clean + repaired z
        z_train_repaired_list = []
        with torch.no_grad():
            for i in range(0, len(z_train), 32):
                z_batch = z_train[i:i+32].to(device)
                B = z_batch.shape[0]
                mask = make_center_mask(B, K, H, W, device)
                z_rep = repair_flow(step_fn, e_core, z_batch, mask, device, T=args.T_repair)
                z_train_repaired_list.append(z_rep.cpu())
        z_train_repaired = torch.cat(z_train_repaired_list)

        z_train_mixed = torch.cat([z_train, z_train_repaired])
        train_y_mixed = torch.cat([train_y, train_y])

        torch.manual_seed(args.seed + 200)
        probe_mixed = ConvProbe(N_BITS).to(device)
        train_probe(probe_mixed, z_train_mixed, train_y_mixed, device, epochs=50)
        acc_clean = eval_probe(probe_mixed, z_test, test_y, device)
        acc_repair = eval_probe(probe_mixed, z_test_repaired, test_y, device)
        probe_gap = acc_repair - acc_clean
        print(f"    acc_clean={acc_clean:.3f}  acc_repair={acc_repair:.3f}  gap={probe_gap:+.3f}")
        del probe_mixed; torch.cuda.empty_cache()

        # ========== EVAL COST: INT8 / INT4 quantization ==========
        print(f"\n  [COST] Activation quantization...")
        quant_results = {}
        for act_bits in [8, 4]:
            qr = eval_gen_quantized(
                step_fn, e_core, encoder, decoder, z_train, test_x,
                real_hf_coh, real_hf_noi, device,
                n_samples=128, T=args.T_gen, act_bits=act_bits, seed=args.seed + 300)
            quant_results[f'INT{act_bits}'] = qr
            print(f"    INT{act_bits}: viol={qr['violation']:.4f}  div={qr['diversity']:.4f}  "
                  f"cycle={qr['cycle']:.4f}")

        # ========== STORE RESULTS ==========
        result = {
            'gen': {k: v for k, v in gen_r.items() if not isinstance(v, list)},
            'repair': repair_r,
            'classify': {
                'acc_clean': acc_clean, 'acc_repair': acc_repair, 'gap': probe_gap
            },
            'cost': {
                'INT8': {k: v for k, v in quant_results['INT8'].items() if not isinstance(v, list)},
                'INT4': {k: v for k, v in quant_results['INT4'].items() if not isinstance(v, list)},
            },
        }
        all_results[op_name] = result

        torch.cuda.empty_cache()

    # ========== SUMMARY ==========
    print(f"\n{'='*100}")
    print("C1 SUMMARY: THREE-MODE COMPATIBILITY")
    print(f"Baselines: clean_cycle={cycle_clean:.4f}  oracle_mse={oracle_mse:.4f}")
    print("=" * 100)

    # Generation table
    print(f"\n--- GENERATION ---")
    header = f"{'Operator':<18} {'viol':>7} {'div':>7} {'HFnoi':>7} {'conn':>7} {'cycle':>7} {'dU':>7}"
    print(header); print("-" * len(header))
    for name, r in all_results.items():
        g = r['gen']
        print(f"{name:<18} {g['violation']:>7.4f} {g['diversity']:>7.4f} "
              f"{g['hf_noise_index']:>7.2f} {g['connectedness']:>7.4f} "
              f"{g['cycle']:>7.4f} {g.get('final_delta_u', 0):>7.2f}")

    # Repair table
    print(f"\n--- REPAIR (center mask) ---")
    header = f"{'Operator':<18} {'ham_masked':>10} {'ham_unmask':>10} {'cycle_rep':>10} {'eobs_drop':>10}"
    print(header); print("-" * len(header))
    for name, r in all_results.items():
        rep = r['repair']
        print(f"{name:<18} {rep['hamming_masked']:>10.4f} {rep['hamming_unmasked']:>10.4f} "
              f"{rep['cycle_repair']:>10.4f} {rep['eobs_drop']:>10.4f}")

    # Classification table
    print(f"\n--- CLASSIFICATION (conv probe, mixed training) ---")
    header = f"{'Operator':<18} {'acc_clean':>10} {'acc_repair':>12} {'gap':>8}"
    print(header); print("-" * len(header))
    for name, r in all_results.items():
        c = r['classify']
        print(f"{name:<18} {c['acc_clean']:>10.3f} {c['acc_repair']:>12.3f} {c['gap']:>+8.3f}")

    # Cost table
    print(f"\n--- COST (activation quantization) ---")
    header = f"{'Operator':<18} {'FP32_viol':>10} {'INT8_viol':>10} {'INT4_viol':>10} {'FP32_div':>9} {'INT8_div':>9} {'INT4_div':>9}"
    print(header); print("-" * len(header))
    for name, r in all_results.items():
        gv = r['gen']['violation']; gd = r['gen']['diversity']
        i8v = r['cost']['INT8']['violation']; i8d = r['cost']['INT8']['diversity']
        i4v = r['cost']['INT4']['violation']; i4d = r['cost']['INT4']['diversity']
        print(f"{name:<18} {gv:>10.4f} {i8v:>10.4f} {i4v:>10.4f} "
              f"{gd:>9.4f} {i8d:>9.4f} {i4d:>9.4f}")

    # ========== DEPLOYMENT GATES ==========
    print(f"\n{'='*100}")
    print("DEPLOYMENT GATES")
    print("=" * 100)

    for name, r in all_results.items():
        g1_cost = (r['cost']['INT8']['diversity'] > 0.3 and
                   r['cost']['INT4']['diversity'] > 0.2)
        g2_contract = (r['repair']['hamming_unmasked'] < 0.01 and
                       r['repair']['cycle_repair'] < 0.1)
        g3_modeswitch = (abs(r['classify']['gap']) < 0.1)

        status = lambda b: "PASS" if b else "FAIL"
        print(f"\n  {name}:")
        print(f"    G1 Cost:       {status(g1_cost)}  (INT8_div={r['cost']['INT8']['diversity']:.3f}, INT4_div={r['cost']['INT4']['diversity']:.3f})")
        print(f"    G2 Contract:   {status(g2_contract)}  (ham_unmask={r['repair']['hamming_unmasked']:.4f}, cycle={r['repair']['cycle_repair']:.4f})")
        print(f"    G3 ModeSwitch: {status(g3_modeswitch)}  (probe_gap={r['classify']['gap']:+.3f})")

    # Save
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*100}")
    print("C1 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
