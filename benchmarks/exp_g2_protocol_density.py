#!/usr/bin/env python3
"""
G2-lite: Protocol Density / Layering
=====================================
Root cause of HF_noise: ADC/DAC pipeline (16×16 binary → 32×32 RGB).
Too few bits per spatial position cannot encode high-frequency texture.

Tests three approaches to increase protocol density:
  L1: More bits per position (n_bits = 16, 24, 32)
  L2: Main + residual codes (16 main + 8 residual, decoded separately and added)
  L3: INT4 tokens (4-bit integer per channel, quantized not binary)

All tested with flat_norm flow (F0c winner), T=20.
Generation-only evaluation (repair not tested here — that's C1's job).

Key metric: HF_noise_index (should approach real=264 for CIFAR-10).
Also: violation, diversity, cycle, connectedness, E_gap_high.

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


# ============================================================================
# L1: STANDARD ADC/DAC WITH VARIABLE BIT DEPTH
# ============================================================================

class Encoder16(nn.Module):
    """16×16 encoder with configurable n_bits."""
    def __init__(self, n_bits=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = self._res(64, 64)
        self.res2 = self._res(64, 64)
        self.head = nn.Conv2d(64, n_bits, 3, padding=1)
        self.q = GumbelSigmoid()

    def _res(self, ic, oc):
        return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(),
                             nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc))

    def forward(self, x):
        h = self.stem(x); h = F.relu(h + self.res1(h)); h = F.relu(h + self.res2(h))
        return self.q(self.head(h)), self.head(h)

    def set_temperature(self, tau): self.q.set_temperature(tau)


class Decoder16(nn.Module):
    """16×16 decoder with configurable n_bits."""
    def __init__(self, n_bits=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ConvTranspose2d(n_bits, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = self._res(64, 64)
        self.res2 = self._res(64, 64)
        self.head = nn.Sequential(nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid())

    def _res(self, ic, oc):
        return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(),
                             nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc))

    def forward(self, z):
        h = self.stem(z); h = F.relu(h + self.res1(h)); h = F.relu(h + self.res2(h))
        return self.head(h)


# ============================================================================
# L2: MAIN + RESIDUAL DECODER
# ============================================================================

class ResidualDecoder16(nn.Module):
    """Two-stage decoder: main (16 bits) + residual (8 bits).

    x_recon = Dec_main(z_main) + Dec_residual(z_residual)
    The residual decoder captures what main decoder misses (HF detail).
    """
    def __init__(self, n_bits_main=16, n_bits_residual=8):
        super().__init__()
        self.main_dec = Decoder16(n_bits_main)
        self.res_dec = nn.Sequential(
            nn.ConvTranspose2d(n_bits_residual, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh())  # residual can be negative
        self.scale = nn.Parameter(torch.tensor(0.1))  # small residual initially

    def forward(self, z_main, z_residual):
        x_main = self.main_dec(z_main)
        x_res = self.res_dec(z_residual) * self.scale
        return (x_main + x_res).clamp(0, 1)


class ResidualEncoder16(nn.Module):
    """Encode main (16 bits) + residual (8 bits).

    Stage 1: Encode image to z_main (like standard encoder)
    Stage 2: Compute pixel residual, encode to z_residual
    """
    def __init__(self, n_bits_main=16, n_bits_residual=8):
        super().__init__()
        self.main_enc = Encoder16(n_bits_main)
        # Residual encoder: takes (residual image, 3ch) → z_residual
        self.res_enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, n_bits_residual, 3, padding=1))
        self.res_q = GumbelSigmoid()

    def forward(self, x, main_decoder=None):
        z_main, logits_main = self.main_enc(x)
        if main_decoder is not None:
            with torch.no_grad():
                x_recon_main = main_decoder(z_main)
            residual = x - x_recon_main
        else:
            residual = torch.zeros_like(x)
        z_res = self.res_q(self.res_enc(residual))
        return z_main, z_res, logits_main

    def set_temperature(self, tau):
        self.main_enc.set_temperature(tau)
        self.res_q.temperature = tau


# ============================================================================
# L3: INT4 TOKEN ENCODER/DECODER
# ============================================================================

class INT4Quantizer(nn.Module):
    """Quantize continuous logits to INT4 (16 levels) with STE."""
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        # Map to [0, 15] range
        soft = torch.sigmoid(logits) * 15.0
        hard = soft.round().clamp(0, 15)
        # STE
        return hard - soft.detach() + soft

    def to_binary(self, int4_vals):
        """Convert INT4 [B, C, H, W] to binary [B, C*4, H, W] for E_core."""
        B, C, H, W = int4_vals.shape
        bits = []
        vals = int4_vals.long()
        for bit_pos in range(4):
            bits.append(((vals >> bit_pos) & 1).float())
        return torch.cat(bits, dim=1)  # [B, C*4, H, W]


class Encoder16_INT4(nn.Module):
    """Encoder that outputs INT4 tokens (16 levels per channel)."""
    def __init__(self, n_channels=4):
        super().__init__()
        # n_channels INT4 values = n_channels * 4 binary bits
        self.n_channels = n_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = self._res(64, 64)
        self.res2 = self._res(64, 64)
        self.head = nn.Conv2d(64, n_channels, 3, padding=1)
        self.q = INT4Quantizer()

    def _res(self, ic, oc):
        return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(),
                             nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc))

    def forward(self, x):
        h = self.stem(x); h = F.relu(h + self.res1(h)); h = F.relu(h + self.res2(h))
        logits = self.head(h)
        int4_vals = self.q(logits)
        return int4_vals, logits


class Decoder16_INT4(nn.Module):
    """Decoder that takes INT4 tokens."""
    def __init__(self, n_channels=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ConvTranspose2d(n_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = self._res(64, 64)
        self.res2 = self._res(64, 64)
        self.head = nn.Sequential(nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid())

    def _res(self, ic, oc):
        return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(),
                             nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc))

    def forward(self, z):
        # Normalize INT4 to [0, 1] for decoder input
        z_norm = z / 15.0
        h = self.stem(z_norm); h = F.relu(h + self.res1(h)); h = F.relu(h + self.res2(h))
        return self.head(h)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_adc(encoder, decoder, train_x, device, epochs=40, bs=32, is_residual=False,
              main_decoder=None):
    """Train ADC/DAC pair."""
    if is_residual:
        params = list(encoder.parameters()) + list(decoder.parameters())
    else:
        params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    for epoch in tqdm(range(epochs), desc="ADC"):
        encoder.train(); decoder.train()
        if hasattr(encoder, 'set_temperature'):
            encoder.set_temperature(1.0 + (0.3 - 1.0) * epoch / max(epochs - 1, 1))
        perm = torch.randperm(len(train_x)); tl, nb = 0., 0
        for i in range(0, len(train_x), bs):
            x = train_x[perm[i:i+bs]].to(device)
            opt.zero_grad()

            if is_residual:
                z_main, z_res, _ = encoder(x, main_decoder)
                xh = decoder(z_main, z_res)
            else:
                z, _ = encoder(x)
                xh = decoder(z)

            loss = F.mse_loss(xh, x) + 0.5 * F.binary_cross_entropy(xh, x)
            loss.backward(); opt.step(); tl += loss.item(); nb += 1

    encoder.eval(); decoder.eval()
    return tl / nb


def encode_all(encoder, data, device, bs=32, is_residual=False, main_decoder=None):
    zs = []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            if is_residual:
                z_main, z_res, _ = encoder(data[i:i+bs].to(device), main_decoder)
                # Concatenate main + residual for E_core / flow
                zs.append(torch.cat([z_main, z_res], dim=1).cpu())
            else:
                z, _ = encoder(data[i:i+bs].to(device))
                zs.append(z.cpu())
    return torch.cat(zs)


def encode_all_int4(encoder, data, device, bs=32):
    """Encode and convert INT4 to binary for E_core."""
    q = INT4Quantizer()
    zs_int4, zs_binary = [], []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            z_int4, _ = encoder(data[i:i+bs].to(device))
            z_binary = q.to_binary(z_int4)
            zs_int4.append(z_int4.cpu())
            zs_binary.append(z_binary.cpu())
    return torch.cat(zs_int4), torch.cat(zs_binary)


def train_ecore(e_core, z_data, device, epochs=15, bs=128):
    opt = torch.optim.Adam(e_core.parameters(), lr=1e-3)
    for ep in tqdm(range(epochs), desc="E_core"):
        e_core.train(); perm = torch.randperm(len(z_data))
        tl, nb = 0., 0
        for i in range(0, len(z_data), bs):
            z = z_data[perm[i:i+bs]].to(device)
            opt.zero_grad(); loss = e_core.energy(z)
            loss.backward(); opt.step(); tl += loss.item(); nb += 1
    e_core.eval()
    return tl / nb


def train_step_fn(step_fn, e_core, z_data, decoder, device,
                  epochs=30, bs=32, T_unroll=3, clip_grad=1.0,
                  is_residual=False, n_bits_main=16):
    """Train flat_norm step function (gen mode, same as F0c)."""
    opt = torch.optim.Adam(step_fn.parameters(), lr=1e-3)
    N = len(z_data)

    for epoch in tqdm(range(epochs), desc="StepFn"):
        step_fn.train(); perm = torch.randperm(N)
        tl, fl, nb = 0., 0., 0
        progress = epoch / max(epochs - 1, 1)

        for i in range(0, N, bs):
            idx = perm[i:i+bs]; z_clean = z_data[idx].to(device); B = z_clean.shape[0]

            noise_level = torch.rand(B, device=device)
            u_clean = z_clean * 2.0 - 1.0
            u_noisy = u_clean + torch.randn_like(u_clean) * noise_level.view(B, 1, 1, 1) * 2.0

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

            # Freq loss via decoder
            z_hard = (z_pred_soft > 0.5).float()
            z_ste = z_hard - z_pred_soft.detach() + z_pred_soft
            with torch.no_grad():
                if is_residual:
                    x_clean = decoder(z_clean[:, :n_bits_main], z_clean[:, n_bits_main:])
                else:
                    x_clean = decoder(z_clean)
            if is_residual:
                x_pred = decoder(z_ste[:, :n_bits_main], z_ste[:, n_bits_main:])
            else:
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


# ============================================================================
# FLOW SAMPLING
# ============================================================================

@torch.no_grad()
def sample_flow(step_fn, e_core, n, K, H, W, device,
                T=20, dt=0.5, sigma_schedule='cosine'):
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

    return quantize(u), trajectory


# ============================================================================
# CUSTOM EVALUATE FOR L2/L3
# ============================================================================

def evaluate_custom(z_gen, decode_fn, encoder_fn, e_core, z_data, test_x,
                    real_hf_coh, real_hf_noi, device, bs=32, trajectory=None):
    """Evaluate with custom decode/encode functions."""
    z_cpu = z_gen.cpu()
    x_gen_list = []
    with torch.no_grad():
        for gi in range(0, len(z_gen), bs):
            x_gen_list.append(decode_fn(z_gen[gi:gi+bs].to(device)).cpu())
    x_gen = torch.cat(x_gen_list)

    viol = e_core.violation_rate(z_cpu[:100].to(device))
    div = compute_diversity(z_cpu)

    # Cycle
    with torch.no_grad():
        zc = z_cpu[:100].to(device)
        xc = decode_fn(zc)
        zcy = encoder_fn(xc)
        cycle = (zc != zcy).float().mean().item()

    conn = connectedness_proxy(x_gen[:100])
    band = per_band_energy_distance(x_gen[:200], test_x[:200], device)
    hfc = hf_coherence_metric(x_gen[:200], device)
    hfn = hf_noise_index(x_gen[:200], device)

    result = {
        'violation': viol, 'diversity': div, 'cycle': cycle,
        'connectedness': conn, 'hf_coherence': hfc, 'hf_noise_index': hfn,
        'energy_gap_low': band['energy_gap_low'],
        'energy_gap_mid': band['energy_gap_mid'],
        'energy_gap_high': band['energy_gap_high'],
    }

    if trajectory and len(trajectory.get('e_core', [])) > 1:
        e_seq = trajectory['e_core']
        mono_count = sum(1 for i in range(1, len(e_seq)) if e_seq[i] < e_seq[i-1])
        result['energy_mono_rate'] = mono_count / (len(e_seq) - 1)
        result['e_core_start'] = e_seq[0]
        result['e_core_end'] = e_seq[-1]
        result['e_core_drop'] = e_seq[0] - e_seq[-1]

    return result, x_gen


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_g2_protocol_density')
    parser.add_argument('--n_samples', type=int, default=256)
    parser.add_argument('--T', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("G2-LITE: PROTOCOL DENSITY / LAYERING")
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
    print(f"    Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}")
    save_grid(test_x[:64], os.path.join(args.output_dir, 'real_samples.png'))

    all_results = {}

    # ================================================================
    # L1: VARYING BIT DEPTH (n_bits = 16, 24, 32)
    # ================================================================
    for n_bits in [16, 24, 32]:
        cfg_name = f"L1_bits{n_bits}"
        print(f"\n{'='*80}")
        print(f"CONFIG: {cfg_name} — {n_bits} binary bits/position, "
              f"total={16*16*n_bits} bits")
        print("=" * 80)

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

        e_core = DiffEnergyCore(n_bits).to(device)
        train_ecore(e_core, z_data, device, epochs=15, bs=128)

        step_fn = FlatStepFn_Norm(n_bits).to(device)
        n_params = sum(p.numel() for p in step_fn.parameters())
        print(f"    StepFn params: {n_params:,}")
        train_step_fn(step_fn, e_core, z_data, dec, device,
                      epochs=30, bs=32, T_unroll=3)

        # Generate
        torch.manual_seed(args.seed + 100)
        z_gen_list, all_traj = [], []
        for gi in range(0, args.n_samples, 32):
            nb = min(32, args.n_samples - gi)
            z_batch, traj = sample_flow(step_fn, e_core, nb, K, H, W, device,
                                        T=args.T)
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
        r['oracle_mse'] = oracle_mse
        r['n_bits'] = n_bits
        r['total_bits'] = 16 * 16 * n_bits
        r['final_delta_u'] = agg_traj['delta_u_norm'][-1]
        r['n_params_stepfn'] = n_params
        all_results[cfg_name] = r

        save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))
        print(f"    viol={r['violation']:.4f}  div={r['diversity']:.4f}  "
              f"cycle={r['cycle']:.4f}  conn={r['connectedness']:.4f}")
        print(f"    HF_noise={r['hf_noise_index']:.2f}  E_gap_H={r['energy_gap_high']:.4f}")

        del enc, dec, e_core, step_fn; torch.cuda.empty_cache()

    # ================================================================
    # L2: MAIN (16) + RESIDUAL (8) = 24 bits total
    # ================================================================
    cfg_name = "L2_main16_res8"
    print(f"\n{'='*80}")
    print(f"CONFIG: {cfg_name} — 16 main + 8 residual bits, total=24 bits/pos")
    print("=" * 80)

    N_MAIN, N_RES = 16, 8

    torch.manual_seed(args.seed)
    # Stage 1: Train main encoder/decoder
    main_enc = Encoder16(N_MAIN).to(device)
    main_dec_standalone = Decoder16(N_MAIN).to(device)
    train_adc(main_enc, main_dec_standalone, train_x, device, epochs=40, bs=32)
    main_enc.eval(); main_dec_standalone.eval()

    # Stage 2: Train residual encoder/decoder (conditioned on main decoder output)
    res_enc = ResidualEncoder16(N_MAIN, N_RES).to(device)
    res_dec = ResidualDecoder16(N_MAIN, N_RES).to(device)

    # Copy main encoder weights
    res_enc.main_enc.load_state_dict(main_enc.state_dict())
    res_dec.main_dec.load_state_dict(main_dec_standalone.state_dict())

    # Freeze main encoder/decoder, train only residual
    for p in res_enc.main_enc.parameters(): p.requires_grad = False
    for p in res_dec.main_dec.parameters(): p.requires_grad = False

    train_adc(res_enc, res_dec, train_x, device, epochs=40, bs=32,
              is_residual=True, main_decoder=main_dec_standalone)

    with torch.no_grad():
        tb = test_x[:32].to(device)
        z_m, z_r, _ = res_enc(tb, main_dec_standalone)
        oracle_mse = F.mse_loss(res_dec(z_m, z_r), tb).item()
    print(f"    Oracle MSE (main+res): {oracle_mse:.4f}")

    # Encode all data (concatenated)
    z_data = encode_all(res_enc, train_x, device, bs=32,
                        is_residual=True, main_decoder=main_dec_standalone)
    K_total = N_MAIN + N_RES
    print(f"    z: {z_data.shape}, K={K_total}")

    e_core = DiffEnergyCore(K_total).to(device)
    train_ecore(e_core, z_data, device, epochs=15, bs=128)

    step_fn = FlatStepFn_Norm(K_total).to(device)
    n_params = sum(p.numel() for p in step_fn.parameters())
    print(f"    StepFn params: {n_params:,}")

    # Custom decoder wrapper for training
    class L2DecWrapper(nn.Module):
        def __init__(self, res_dec, n_main):
            super().__init__()
            self.res_dec = res_dec
            self.n_main = n_main
        def forward(self, z_concat):
            return self.res_dec(z_concat[:, :self.n_main], z_concat[:, self.n_main:])

    dec_wrapper = L2DecWrapper(res_dec, N_MAIN).to(device)
    train_step_fn(step_fn, e_core, z_data, dec_wrapper, device,
                  epochs=30, bs=32, T_unroll=3, is_residual=False)

    # Generate
    _, H, W = z_data.shape[1], z_data.shape[2], z_data.shape[3]
    torch.manual_seed(args.seed + 100)
    z_gen_list, all_traj = [], []
    for gi in range(0, args.n_samples, 32):
        nb = min(32, args.n_samples - gi)
        z_batch, traj = sample_flow(step_fn, e_core, nb, K_total, H, W, device, T=args.T)
        z_gen_list.append(z_batch.cpu())
        all_traj.append(traj)
    z_gen = torch.cat(z_gen_list)
    agg_traj = {
        key: [np.mean([t[key][s] for t in all_traj])
              for s in range(len(all_traj[0][key]))]
        for key in all_traj[0].keys()
    }

    # Custom evaluate
    def l2_decode(z):
        return res_dec(z[:, :N_MAIN], z[:, N_MAIN:])

    def l2_encode(x):
        z_m, z_r, _ = res_enc(x, main_dec_standalone)
        return torch.cat([z_m, z_r], dim=1)

    r, x_gen = evaluate_custom(z_gen, l2_decode, l2_encode, e_core, z_data, test_x,
                               real_hf_coh, real_hf_noi, device, trajectory=agg_traj)
    r['oracle_mse'] = oracle_mse
    r['n_bits'] = K_total
    r['total_bits'] = 16 * 16 * K_total
    r['final_delta_u'] = agg_traj['delta_u_norm'][-1]
    r['n_params_stepfn'] = n_params
    all_results[cfg_name] = r

    save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))
    print(f"    viol={r['violation']:.4f}  div={r['diversity']:.4f}  "
          f"cycle={r['cycle']:.4f}  conn={r['connectedness']:.4f}")
    print(f"    HF_noise={r['hf_noise_index']:.2f}  E_gap_H={r['energy_gap_high']:.4f}")

    del main_enc, main_dec_standalone, res_enc, res_dec, e_core, step_fn
    torch.cuda.empty_cache()

    # ================================================================
    # L3: INT4 TOKENS (4 channels × 4 bits = 16 binary bits equivalent)
    # ================================================================
    for n_ch in [4, 8]:
        equiv_bits = n_ch * 4
        cfg_name = f"L3_int4_ch{n_ch}"
        print(f"\n{'='*80}")
        print(f"CONFIG: {cfg_name} — {n_ch} INT4 channels = {equiv_bits} binary equiv, "
              f"total={16*16*equiv_bits} binary bits")
        print("=" * 80)

        torch.manual_seed(args.seed)
        enc = Encoder16_INT4(n_ch).to(device)
        dec = Decoder16_INT4(n_ch).to(device)

        # Train INT4 ADC/DAC
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
        for epoch in tqdm(range(40), desc="INT4 ADC"):
            enc.train(); dec.train()
            perm = torch.randperm(len(train_x)); tl, nb = 0., 0
            for i in range(0, len(train_x), 32):
                x = train_x[perm[i:i+32]].to(device)
                opt.zero_grad()
                z_int4, _ = enc(x); xh = dec(z_int4)
                loss = F.mse_loss(xh, x) + 0.5 * F.binary_cross_entropy(xh, x)
                loss.backward(); opt.step(); tl += loss.item(); nb += 1
        enc.eval(); dec.eval()
        print(f"    INT4 ADC loss: {tl/nb:.4f}")

        with torch.no_grad():
            tb = test_x[:32].to(device); z4, _ = enc(tb)
            oracle_mse = F.mse_loss(dec(z4), tb).item()
        print(f"    Oracle MSE: {oracle_mse:.4f}")

        # Encode and convert to binary for E_core/flow
        z_int4_data, z_binary_data = encode_all_int4(enc, train_x, device, bs=32)
        K_bin = equiv_bits
        print(f"    z_binary: {z_binary_data.shape}, z_int4: {z_int4_data.shape}")

        e_core = DiffEnergyCore(K_bin).to(device)
        train_ecore(e_core, z_binary_data, device, epochs=15, bs=128)

        # Flow on binary representation
        step_fn = FlatStepFn_Norm(K_bin).to(device)
        n_params = sum(p.numel() for p in step_fn.parameters())
        print(f"    StepFn params: {n_params:,}")

        # For flow training, use binary z with decoder via INT4 conversion
        q = INT4Quantizer()

        class INT4DecWrapper(nn.Module):
            def __init__(self, dec, n_channels):
                super().__init__()
                self.dec = dec
                self.n_channels = n_channels
            def forward(self, z_binary):
                # Convert binary back to INT4 for decoder
                B, K, H, W = z_binary.shape
                n_ch = self.n_channels
                # Reshape: [B, n_ch*4, H, W] → [B, n_ch, 4, H, W]
                z_reshaped = z_binary.view(B, n_ch, 4, H, W)
                # Convert bits to int: sum 2^bit * bit_val
                powers = torch.tensor([1, 2, 4, 8], device=z_binary.device,
                                      dtype=z_binary.dtype).view(1, 1, 4, 1, 1)
                z_int4 = (z_reshaped * powers).sum(dim=2)  # [B, n_ch, H, W]
                return self.dec(z_int4)

        dec_wrapper = INT4DecWrapper(dec, n_ch).to(device)
        train_step_fn(step_fn, e_core, z_binary_data, dec_wrapper, device,
                      epochs=30, bs=32, T_unroll=3)

        # Generate
        K, H, W = z_binary_data.shape[1:]
        torch.manual_seed(args.seed + 100)
        z_gen_list, all_traj = [], []
        for gi in range(0, args.n_samples, 32):
            nb = min(32, args.n_samples - gi)
            z_batch, traj = sample_flow(step_fn, e_core, nb, K, H, W, device, T=args.T)
            z_gen_list.append(z_batch.cpu())
            all_traj.append(traj)
        z_gen = torch.cat(z_gen_list)
        agg_traj = {
            key: [np.mean([t[key][s] for t in all_traj])
                  for s in range(len(all_traj[0][key]))]
            for key in all_traj[0].keys()
        }

        def int4_decode(z_bin):
            return dec_wrapper(z_bin)

        def int4_encode(x):
            z4, _ = enc(x)
            return q.to_binary(z4)

        r, x_gen = evaluate_custom(z_gen, int4_decode, int4_encode, e_core,
                                   z_binary_data, test_x, real_hf_coh, real_hf_noi,
                                   device, trajectory=agg_traj)
        r['oracle_mse'] = oracle_mse
        r['n_bits'] = K_bin
        r['n_channels_int4'] = n_ch
        r['total_bits'] = 16 * 16 * K_bin
        r['final_delta_u'] = agg_traj['delta_u_norm'][-1]
        r['n_params_stepfn'] = n_params
        all_results[cfg_name] = r

        save_grid(x_gen, os.path.join(args.output_dir, f'gen_{cfg_name}.png'))
        print(f"    viol={r['violation']:.4f}  div={r['diversity']:.4f}  "
              f"cycle={r['cycle']:.4f}  conn={r['connectedness']:.4f}")
        print(f"    HF_noise={r['hf_noise_index']:.2f}  E_gap_H={r['energy_gap_high']:.4f}")

        del enc, dec, e_core, step_fn, dec_wrapper; torch.cuda.empty_cache()

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*100}")
    print("G2-LITE SUMMARY: PROTOCOL DENSITY")
    print(f"Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}")
    print("=" * 100)

    header = (f"{'config':<22} {'bits/pos':>8} {'total':>8} {'viol':>7} {'div':>7} "
              f"{'HFnoi':>7} {'conn':>7} {'cycle':>7} {'EgH':>7} {'MSE':>7}")
    print(header); print("-" * len(header))
    for name, r in sorted(all_results.items()):
        print(f"{name:<22} {r.get('n_bits', '?'):>8} {r.get('total_bits', '?'):>8} "
              f"{r['violation']:>7.4f} {r['diversity']:>7.4f} "
              f"{r['hf_noise_index']:>7.2f} {r['connectedness']:>7.4f} "
              f"{r['cycle']:>7.4f} {r['energy_gap_high']:>7.4f} "
              f"{r.get('oracle_mse', 0):>7.4f}")

    # HF_noise improvement analysis
    print(f"\n--- HF_NOISE ANALYSIS (target: real={real_hf_noi:.2f}) ---")
    baseline_hfn = all_results.get('L1_bits16', {}).get('hf_noise_index', 0)
    for name, r in sorted(all_results.items()):
        hfn = r['hf_noise_index']
        delta_from_real = hfn - real_hf_noi
        delta_from_base = hfn - baseline_hfn if baseline_hfn else 0
        gate = "CLOSER" if abs(delta_from_real) < abs(baseline_hfn - real_hf_noi) else "WORSE"
        print(f"  {name:<22} HF_noise={hfn:>7.2f}  Δ(real)={delta_from_real:>+8.2f}  "
              f"Δ(base16)={delta_from_base:>+8.2f}  {gate}")

    # Save
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n{'='*100}")
    print("G2-LITE COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
