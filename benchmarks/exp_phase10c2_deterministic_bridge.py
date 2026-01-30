#!/usr/bin/env python3
"""
Phase 10C-2: Deterministic Protocol Bridge
============================================
Phase 10C showed INT z_cycle=8-9% because STE fake-quant has per-batch scale drift
and stochastic rounding boundary.

Fixes (P0):
1. Deterministic quantization: calibrate scale once (EMA), fix during eval
2. Multi-cycle drift loss: train on 2-3 cycles, not just 1
3. Bit certainty: push pre-quant values away from rounding boundaries

Success: z_cycle ≤ 2%, drift5 < 2× drift1, agreement ≥ 95%.

Usage:
    python3 -u benchmarks/exp_phase10c2_deterministic_bridge.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os, sys, csv, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from benchmarks.exp_phase10b_bridge_v3 import (
    FMNISTTeacher, train_teacher, load_dataset
)


# ============================================================================
# DETERMINISTIC INT QUANTIZATION with fixed EMA scale
# ============================================================================

class DeterministicINTQuant(nn.Module):
    """Deterministic fake quantization with EMA-calibrated fixed scale.
    - Scale is calibrated via EMA during training (not recomputed per-batch)
    - Rounding is deterministic (round, not stochastic)
    - Train and eval use SAME quantization path
    """
    def __init__(self, n_levels=16, ema_decay=0.999):
        super().__init__()
        self.n_levels = n_levels
        self.half_levels = n_levels // 2
        self.ema_decay = ema_decay
        # Per-channel scale, initialized to 1
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('calibrated', torch.tensor(False))

    def calibrate(self, x):
        """Update EMA scale from data."""
        with torch.no_grad():
            # Per-channel max absolute value
            batch_scale = x.abs().amax() / self.half_levels
            batch_scale = batch_scale.clamp(min=1e-8)
            if not self.calibrated:
                self.scale.copy_(batch_scale)
                self.calibrated.fill_(True)
            else:
                self.scale.mul_(self.ema_decay).add_(batch_scale, alpha=1 - self.ema_decay)

    def forward(self, x):
        if self.training:
            self.calibrate(x)

        # Deterministic quantization with FIXED scale
        s = self.scale.clamp(min=1e-8)
        x_scaled = x / s
        # Deterministic round + clamp
        x_q = x_scaled.round().clamp(-self.half_levels, self.half_levels - 1)
        x_deq = x_q * s

        # STE: forward uses quantized, backward passes through
        return x + (x_deq - x).detach()


# ============================================================================
# INT BRIDGE with deterministic quant + boundary margin
# ============================================================================

class DeterministicINTBridge(nn.Module):
    """INT bridge with deterministic quantization and boundary margin."""
    def __init__(self, in_ch=64, k=32, quant_bits=4):
        super().__init__()
        n_levels = 2 ** quant_bits
        self.quant = DeterministicINTQuant(n_levels)
        self.Q = nn.Sequential(
            nn.Conv2d(in_ch, k, 3, padding=1), nn.ReLU(),
            nn.Conv2d(k, k, 3, padding=1))
        self.R = nn.Sequential(
            nn.Conv2d(k, in_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(in_ch, in_ch, 3, padding=1))
        self.skip_q = nn.Conv2d(in_ch, k, 1)
        self.skip_r = nn.Conv2d(k, in_ch, 1)

    def encode(self, h):
        pre_quant = self.Q(h) + self.skip_q(h)
        return self.quant(pre_quant), pre_quant

    def decode(self, z):
        return self.R(z) + self.skip_r(z)

    def forward(self, h):
        z, pre_quant = self.encode(h)
        h_hat = self.decode(z)
        return z, h_hat, pre_quant


def boundary_margin_loss(pre_quant, scale, n_levels):
    """Push pre-quant values away from rounding boundaries.
    Rounding boundaries are at (n+0.5)*scale for integer n.
    We want values to be NEAR integer multiples of scale, not halfway.
    """
    s = scale.clamp(min=1e-8)
    x_scaled = pre_quant / s
    # Distance from nearest integer (0 = on integer, 0.5 = on boundary)
    frac = x_scaled - x_scaled.round()
    # We want |frac| to be small (near integer → deterministic rounding)
    # Loss: mean(|frac|²) — penalizes being near boundary
    return (frac ** 2).mean()


# ============================================================================
# TRAINING with multi-cycle drift loss + boundary margin
# ============================================================================

def train_deterministic_bridge(bridge, teacher, train_x, device,
                                epochs=50, batch_size=64,
                                lam_func=1.0, lam_cycle=1.0, lam_feat=0.05,
                                lam_margin=0.1, n_train_cycles=3):
    """Train with:
    - KL distillation (functional)
    - Multi-cycle drift loss (protocol stability)
    - Boundary margin (deterministic quantization)
    - Weak feature regularizer
    """
    opt = torch.optim.Adam(bridge.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        bridge.train()
        tl_kl, tl_cyc, tl_feat, tl_margin, nb = 0., 0., 0., 0., 0

        for (x,) in loader:
            x = x.to(device); opt.zero_grad()
            with torch.no_grad():
                h = teacher.extract(x)
                logits_orig = teacher.head(h)
                probs_orig = F.softmax(logits_orig, dim=1)

            # Forward: h → z0 → ĥ
            z0, h_hat, pre_quant = bridge(h)

            # L_func: KL distillation
            logits_recon = teacher.head(h_hat)
            loss_kl = F.kl_div(F.log_softmax(logits_recon, dim=1),
                               probs_orig, reduction='batchmean')

            # L_cycle: multi-cycle drift
            # Unroll n_train_cycles and accumulate cycle error
            loss_cycle = 0.
            h_current = h_hat
            for c in range(n_train_cycles):
                z_next, h_next, _ = bridge(h_current.detach())
                # Cycle error: z_next should match z0
                loss_cycle += F.mse_loss(z_next, z0.detach()) / n_train_cycles
                h_current = h_next

            # L_margin: push away from rounding boundaries
            loss_margin = boundary_margin_loss(
                pre_quant, bridge.quant.scale, bridge.quant.n_levels)

            # L_feat: weak regularizer
            loss_feat = F.mse_loss(h_hat, h.detach())

            loss = (lam_func * loss_kl + lam_cycle * loss_cycle +
                    lam_margin * loss_margin + lam_feat * loss_feat)
            loss.backward(); opt.step()
            tl_kl += loss_kl.item()
            tl_cyc += loss_cycle.item()
            tl_feat += loss_feat.item()
            tl_margin += loss_margin.item()
            nb += 1

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{epochs}: KL={tl_kl/nb:.4f} "
                  f"cycle={tl_cyc/nb:.6f} margin={tl_margin/nb:.4f} "
                  f"feat={tl_feat/nb:.4f}")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_deterministic_bridge(bridge, teacher, test_x, device, batch_size=64):
    """Evaluate with proper z_cycle measurement for deterministic quant."""
    bridge.eval(); teacher.eval()

    agreements, kl_divs, cosines = [], [], []
    z_cycle_errors = []
    k_cycle_drifts = {1: [], 2: [], 3: [], 5: []}
    semantic_flips_when_z_stable = []

    with torch.no_grad():
        for i in range(0, len(test_x), batch_size):
            x = test_x[i:i+batch_size].to(device)
            h = teacher.extract(x)
            logits_orig = teacher.head(h)
            pred_orig = logits_orig.argmax(1)

            # Forward
            z0, h_hat, _ = bridge(h)
            z1, _, _ = bridge(h_hat)

            # z_cycle_error: exact match (deterministic quant should be exact!)
            z_match = (z0 == z1).float().mean(dim=(1,2,3))
            z_cycle_err = 1.0 - z_match
            z_cycle_errors.extend(z_cycle_err.cpu().tolist())

            # Also measure relative error as backup
            z_rel_err = ((z0 - z1).abs() / (z0.abs().mean() + 1e-8)).mean(dim=(1,2,3))

            # Functional
            logits_recon = teacher.head(h_hat)
            pred_recon = logits_recon.argmax(1)
            agree = (pred_orig == pred_recon)
            agreements.extend(agree.cpu().tolist())

            kl = F.kl_div(F.log_softmax(logits_recon, dim=1),
                          F.softmax(logits_orig, dim=1),
                          reduction='none').sum(1)
            kl_divs.extend(kl.cpu().tolist())

            h_flat = h.reshape(h.shape[0], -1)
            hh_flat = h_hat.reshape(h_hat.shape[0], -1)
            cosines.extend(
                F.cosine_similarity(h_flat, hh_flat, dim=1).cpu().tolist())

            # Semantic flip when z stable
            for s in range(len(z_cycle_err)):
                if z_cycle_err[s].item() < 0.01:
                    flip = not agree[s].item()
                    semantic_flips_when_z_stable.append(float(flip))

            # k-cycle drift (exact match)
            z_current = z0.clone()
            h_current = h_hat.clone()
            for c in range(5):
                z_next, h_next, _ = bridge(h_current)
                if (c + 1) in k_cycle_drifts:
                    mismatch = (z_next != z0).float().mean(dim=(1,2,3))
                    k_cycle_drifts[c+1].extend(mismatch.cpu().tolist())
                z_current = z_next
                h_current = h_next

    results = {
        'agreement': np.mean(agreements),
        'kl_div': np.mean(kl_divs),
        'cosine': np.mean(cosines),
        'z_cycle_error': np.mean(z_cycle_errors),
        'drift_1': np.mean(k_cycle_drifts[1]),
        'drift_2': np.mean(k_cycle_drifts[2]),
        'drift_3': np.mean(k_cycle_drifts[3]),
        'drift_5': np.mean(k_cycle_drifts[5]),
        'semantic_flip_when_z_stable': (np.mean(semantic_flips_when_z_stable)
                                        if semantic_flips_when_z_stable else float('nan')),
        'n_z_stable_samples': len(semantic_flips_when_z_stable),
    }
    results['drift_bounded'] = results['drift_5'] <= 2.0 * results['drift_1'] + 0.005
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_phase10c2')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 10C-2: DETERMINISTIC PROTOCOL BRIDGE")
    print("=" * 100)

    print("[1] Loading FMNIST...")
    train_x, train_y, test_x, test_y = load_dataset(4000, 500, args.seed)

    print("[2] Training same-domain teacher...")
    teacher = train_teacher(train_x, train_y, test_x, test_y, device, epochs=15)

    all_results = []

    # Sweep: INT4/INT8 × k=32/16 × λ_cycle sweep
    configs = [
        ('INT4_k32_cyc05', 4, 32, 0.5),
        ('INT4_k32_cyc10', 4, 32, 1.0),
        ('INT4_k32_cyc20', 4, 32, 2.0),
        ('INT4_k16_cyc10', 4, 16, 1.0),
        ('INT8_k32_cyc10', 8, 32, 1.0),
        ('INT8_k16_cyc10', 8, 16, 1.0),
    ]

    print(f"\n{'config':<22} {'agree%':>8} {'z_cycle':>8} {'drift1':>8} "
          f"{'drift5':>8} {'bound':>6} {'sem_flip':>9} {'KL':>7}")
    print("-" * 90)

    for name, qbits, k, lam_cyc in configs:
        print(f"\n  [{name}] Training (deterministic quant + {lam_cyc}× cycle)...")
        torch.manual_seed(args.seed)
        bridge = DeterministicINTBridge(in_ch=64, k=k, quant_bits=qbits).to(device)

        train_deterministic_bridge(bridge, teacher, train_x, device,
                                    epochs=50, lam_cycle=lam_cyc,
                                    lam_margin=0.1, n_train_cycles=3)

        r = evaluate_deterministic_bridge(bridge, teacher, test_x, device)
        r['config'] = name
        r['quant_bits'] = qbits
        r['k'] = k
        r['lam_cycle'] = lam_cyc
        all_results.append(r)

        sf = f"{r['semantic_flip_when_z_stable']:.1%}" if not np.isnan(r['semantic_flip_when_z_stable']) else "N/A"
        print(f"  {name:<22} {r['agreement']:>7.1%} {r['z_cycle_error']:>8.4f} "
              f"{r['drift_1']:>8.4f} {r['drift_5']:>8.4f} "
              f"{'Y' if r['drift_bounded'] else 'N':>6} {sf:>9} {r['kl_div']:>7.4f}")

        del bridge; torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 100)
    print("PHASE 10C-2 RESULTS")
    print("=" * 100)

    print(f"\n{'config':<22} {'agree%':>8} {'z_cyc%':>8} {'d1%':>8} {'d2%':>8} "
          f"{'d3%':>8} {'d5%':>8} {'bound':>6} {'flip':>8}")
    print("-" * 100)

    for r in sorted(all_results, key=lambda x: x['z_cycle_error']):
        sf = f"{r['semantic_flip_when_z_stable']:.1%}" if not np.isnan(r['semantic_flip_when_z_stable']) else "N/A"
        print(f"{r['config']:<22} {r['agreement']:>7.1%} {r['z_cycle_error']:>7.2%} "
              f"{r['drift_1']:>7.2%} {r['drift_2']:>7.2%} "
              f"{r['drift_3']:>7.2%} {r['drift_5']:>7.2%} "
              f"{'Y' if r['drift_bounded'] else 'N':>6} {sf:>8}")

    # Success check
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA")
    print("=" * 80)

    for r in all_results:
        c1 = r['agreement'] >= 0.95
        c2 = r['z_cycle_error'] <= 0.02
        c3 = r['drift_bounded']
        n = sum([c1, c2, c3])
        status = 'PROTOCOL-STABLE' if n == 3 else f'{n}/3'
        print(f"  {r['config']:<22}: agree≥95%={'P' if c1 else 'F'} "
              f"cycle≤2%={'P' if c2 else 'F'} "
              f"drift_bound={'P' if c3 else 'F'} → {status}")

    # vs Phase 10C
    print("\n" + "=" * 80)
    print("vs PHASE 10C (non-deterministic)")
    print("=" * 80)

    p10c_ref = {
        'INT4_k32': {'z_cycle': 0.0845, 'drift_5': 0.1898, 'agreement': 0.984},
        'INT4_k16': {'z_cycle': 0.0775, 'drift_5': 0.2157, 'agreement': 0.964},
        'INT8_k32': {'z_cycle': 0.0896, 'drift_5': 0.2533, 'agreement': 0.984},
        'INT8_k16': {'z_cycle': 0.0821, 'drift_5': 0.3026, 'agreement': 0.980},
    }

    for r in all_results:
        # Match to reference
        ref_key = f"INT{r['quant_bits']}_k{r['k']}"
        ref = p10c_ref.get(ref_key, {})
        if ref:
            print(f"  {r['config']:<22}: z_cycle {ref.get('z_cycle',0):.2%} → {r['z_cycle_error']:.2%} "
                  f"  drift5 {ref.get('drift_5',0):.2%} → {r['drift_5']:.2%} "
                  f"  agree {ref.get('agreement',0):.1%} → {r['agreement']:.1%}")

    # Best result
    best = min(all_results, key=lambda x: x['z_cycle_error'])
    print(f"\n  BEST: {best['config']} — z_cycle={best['z_cycle_error']:.4f}, "
          f"drift5={best['drift_5']:.4f}, agreement={best['agreement']:.1%}")

    n_stable = sum(1 for r in all_results
                   if r['agreement'] >= 0.95
                   and r['z_cycle_error'] <= 0.02
                   and r['drift_bounded'])
    print(f"\n  {n_stable}/{len(all_results)} configs are PROTOCOL-STABLE")

    # Save
    csv_path = os.path.join(args.output_dir, "phase10c2_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(set().union(*(r.keys() for r in all_results)))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    print("\n" + "=" * 100)
    print("Phase 10C-2 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
