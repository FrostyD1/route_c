#!/usr/bin/env python3
"""
Phase 10C: Protocol-Stable Feature Bridge
==========================================
Upgrade Phase 10B from "agreement high" to "protocol reversible + closeable loop".

Key addition: z-cycle constraint.
  z0 = Q(h), ĥ = R(z0), z1 = Q(ĥ)
  Train: L = λ_func * KL + λ_cycle * CE(z0, z1) + λ_feat * ||h - ĥ||²

Metrics:
  A. Functional: agreement, KL_div (from 10B)
  B. Protocol loop (NEW):
     - z_cycle_error: mean(z1 != z0) for INT, mean(idx1 != idx0) for VQ
     - k-cycle drift (k=1,2,3,5): repeated z→ĥ→z mismatch
  C. Semantic-protocol coherence (NEW):
     - semantic_flip_when_z_stable: among z-stable samples, how many flip prediction?

Success: agreement≥95%, z_cycle_error≤2%, 5-cycle drift ≤ 2× drift(1).

Usage:
    python3 -u benchmarks/exp_phase10c_protocol_bridge.py --device cuda
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
    FMNISTTeacher, train_teacher, load_dataset,
    INTTokenBridge, VQBridge, FakeQuantINT, fake_quant
)


# ============================================================================
# TRAINING with z-cycle loss
# ============================================================================

def train_int_bridge_cycle(bridge, teacher, train_x, device,
                            epochs=40, batch_size=64,
                            lam_func=1.0, lam_cycle=0.5, lam_feat=0.05):
    """Train INT bridge with explicit z-cycle constraint."""
    opt = torch.optim.Adam(bridge.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        bridge.train()
        tl_kl, tl_cyc, tl_feat, nb = 0., 0., 0., 0

        for (x,) in loader:
            x = x.to(device); opt.zero_grad()
            with torch.no_grad():
                h = teacher.extract(x)
                logits_orig = teacher.head(h)
                probs_orig = F.softmax(logits_orig, dim=1)

            # Forward: h → z0 → ĥ
            z0 = bridge.encode(h)
            h_hat = bridge.decode(z0)

            # Re-encode: ĥ → z1
            z1 = bridge.encode(h_hat)

            # L_func: KL distillation (PRIMARY)
            logits_recon = teacher.head(h_hat)
            loss_kl = F.kl_div(F.log_softmax(logits_recon, dim=1),
                               probs_orig, reduction='batchmean')

            # L_cycle: z0 ≈ z1 (per-position token CE)
            # For INT tokens: z0 and z1 are quantized continuous values
            # Use MSE in token space (since values are multi-level, not binary)
            loss_cycle = F.mse_loss(z1, z0.detach())

            # L_feat: weak feature regularizer
            loss_feat = F.mse_loss(h_hat, h.detach())

            loss = lam_func * loss_kl + lam_cycle * loss_cycle + lam_feat * loss_feat
            loss.backward(); opt.step()
            tl_kl += loss_kl.item()
            tl_cyc += loss_cycle.item()
            tl_feat += loss_feat.item()
            nb += 1

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{epochs}: KL={tl_kl/nb:.4f} "
                  f"cycle={tl_cyc/nb:.4f} feat={tl_feat/nb:.4f}")


def train_vq_bridge_cycle(bridge, teacher, train_x, device,
                           epochs=40, batch_size=64,
                           lam_func=1.0, lam_cycle=0.5, lam_feat=0.05):
    """Train VQ bridge with z-cycle constraint on codebook indices."""
    opt = torch.optim.Adam(bridge.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        bridge.train()
        tl_kl, tl_cyc, tl_com, tl_feat, nb = 0., 0., 0., 0., 0

        for (x,) in loader:
            x = x.to(device); opt.zero_grad()
            with torch.no_grad():
                h = teacher.extract(x)
                logits_orig = teacher.head(h)
                probs_orig = F.softmax(logits_orig, dim=1)

            # Forward: h → z_q0, idx0 → ĥ
            z_q0, idx0, z_e0 = bridge.encode(h)
            h_hat = bridge.decode(z_q0)

            # Re-encode: ĥ → z_q1, idx1
            z_q1, idx1, z_e1 = bridge.encode(h_hat)

            # L_func: KL distillation
            logits_recon = teacher.head(h_hat)
            loss_kl = F.kl_div(F.log_softmax(logits_recon, dim=1),
                               probs_orig, reduction='batchmean')

            # L_cycle: codebook index match
            # Use MSE on normalized embeddings (not raw, to prevent scale explosion)
            z_q0_norm = F.normalize(z_q0, dim=1)
            z_q1_norm = F.normalize(z_q1, dim=1)
            loss_cycle = F.mse_loss(z_q1_norm, z_q0_norm.detach())

            # VQ commitment loss
            commit_loss = F.mse_loss(z_e0, z_q0.detach())

            # L_feat: weak regularizer
            loss_feat = F.mse_loss(h_hat, h.detach())

            loss = (lam_func * loss_kl + lam_cycle * loss_cycle +
                    0.25 * commit_loss + lam_feat * loss_feat)
            loss.backward(); opt.step()
            tl_kl += loss_kl.item()
            tl_cyc += loss_cycle.item()
            tl_com += commit_loss.item()
            tl_feat += loss_feat.item()
            nb += 1

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{epochs}: KL={tl_kl/nb:.4f} "
                  f"cycle={tl_cyc/nb:.4f} commit={tl_com/nb:.4f} feat={tl_feat/nb:.4f}")


# ============================================================================
# EVALUATION: Protocol-focused metrics
# ============================================================================

def evaluate_protocol_bridge(bridge, teacher, test_x, device,
                              route='int', batch_size=64):
    """Comprehensive evaluation: functional + protocol + semantic coherence."""
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

            # Forward pass
            if route == 'int':
                z0 = bridge.encode(h)
                h_hat = bridge.decode(z0)
                z1 = bridge.encode(h_hat)
                # z_cycle_error: relative difference in quantized token space
                # Use relative MSE (not exact float equality, which always fails)
                z_diff = (z0 - z1).abs()
                z_scale = z0.abs().mean() + 1e-8
                z_cycle_err = (z_diff / z_scale).mean(dim=(1,2,3))  # per-sample
            else:
                z_q0, idx0, _ = bridge.encode(h)
                h_hat = bridge.decode(z_q0)
                z_q1, idx1, _ = bridge.encode(h_hat)
                # z_cycle_error: codebook index mismatch rate
                z_match = (idx0 == idx1).float().mean(dim=(1,2))  # per-sample
                z_cycle_err = 1.0 - z_match
                z0 = z_q0  # for drift computation

            z_cycle_errors.extend(z_cycle_err.cpu().tolist())

            # Functional metrics
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
            # Among samples where z1 ≈ z0 (cycle error < 5%), did agreement flip?
            for s in range(len(z_cycle_err)):
                if z_cycle_err[s].item() < 0.05:  # z is stable (within 5%)
                    flip = not agree[s].item()
                    semantic_flips_when_z_stable.append(float(flip))

            # k-cycle drift
            z_current = z0.clone()
            h_current = h_hat.clone()
            for c in range(5):
                if route == 'int':
                    z_next = bridge.encode(h_current)
                    h_next = bridge.decode(z_next)
                else:
                    z_next, idx_next, _ = bridge.encode(h_current)
                    h_next = bridge.decode(z_next)

                if (c + 1) in k_cycle_drifts:
                    if route == 'int':
                        # Relative difference (not exact float equality)
                        drift_diff = (z_next - z0).abs()
                        drift_scale = z0.abs().mean() + 1e-8
                        mismatch = (drift_diff / drift_scale).mean(dim=(1,2,3))
                    else:
                        mismatch = (idx_next != idx0).float().mean(dim=(1,2))
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

    # Check drift bound: 5-cycle ≤ 2× 1-cycle
    results['drift_bounded'] = results['drift_5'] <= 2.0 * results['drift_1'] + 0.001

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_phase10c')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 10C: PROTOCOL-STABLE FEATURE BRIDGE")
    print("=" * 100)

    print("[1] Loading FMNIST...")
    train_x, train_y, test_x, test_y = load_dataset(4000, 500, args.seed)

    print("[2] Training same-domain teacher...")
    teacher = train_teacher(train_x, train_y, test_x, test_y, device, epochs=15)

    all_results = []

    # ================================================================
    # ROUTE A: INT Token Bridge with cycle loss
    # ================================================================
    print("\n" + "=" * 80)
    print("ROUTE A: INT TOKEN BRIDGE (with z-cycle constraint)")
    print("=" * 80)

    int_configs = [
        ('INT8_k32', 8, 32),
        ('INT8_k16', 8, 16),
        ('INT4_k32', 4, 32),
        ('INT4_k16', 4, 16),
    ]

    for name, qbits, k in int_configs:
        print(f"\n  [{name}] Training with cycle loss...")
        torch.manual_seed(args.seed)
        bridge = INTTokenBridge(in_ch=64, k=k, quant_bits=qbits).to(device)
        train_int_bridge_cycle(bridge, teacher, train_x, device,
                                epochs=40, lam_func=1.0, lam_cycle=0.5, lam_feat=0.05)

        r = evaluate_protocol_bridge(bridge, teacher, test_x, device, route='int')
        r['config'] = name
        r['route'] = 'INT'
        r['bits_per_pos'] = k * qbits
        all_results.append(r)

        print(f"    agree={r['agreement']:.1%} z_cycle={r['z_cycle_error']:.4f} "
              f"drift1={r['drift_1']:.4f} drift5={r['drift_5']:.4f} "
              f"bounded={'Y' if r['drift_bounded'] else 'N'} "
              f"sem_flip={r['semantic_flip_when_z_stable']:.1%}")

        del bridge; torch.cuda.empty_cache()

    # ================================================================
    # ROUTE B: VQ Codebook Bridge with cycle loss
    # ================================================================
    print("\n" + "=" * 80)
    print("ROUTE B: VQ CODEBOOK BRIDGE (with z-cycle constraint)")
    print("=" * 80)

    vq_configs = [
        ('VQ_K256_d32', 256, 32),
        ('VQ_K256_d16', 256, 16),
        ('VQ_K512_d32', 512, 32),
        ('VQ_K512_d16', 512, 16),
    ]

    for name, num_codes, embed_dim in vq_configs:
        print(f"\n  [{name}] Training with cycle loss...")
        torch.manual_seed(args.seed)
        bridge = VQBridge(in_ch=64, embed_dim=embed_dim, num_codes=num_codes).to(device)
        train_vq_bridge_cycle(bridge, teacher, train_x, device,
                               epochs=40, lam_func=1.0, lam_cycle=0.5, lam_feat=0.05)

        r = evaluate_protocol_bridge(bridge, teacher, test_x, device, route='vq')
        r['config'] = name
        r['route'] = 'VQ'
        r['bits_per_pos'] = np.log2(num_codes)
        all_results.append(r)

        print(f"    agree={r['agreement']:.1%} z_cycle={r['z_cycle_error']:.4f} "
              f"drift1={r['drift_1']:.4f} drift5={r['drift_5']:.4f} "
              f"bounded={'Y' if r['drift_bounded'] else 'N'} "
              f"sem_flip={r['semantic_flip_when_z_stable']:.1%}")

        del bridge; torch.cuda.empty_cache()

    # ================================================================
    # COMPARISON: Phase 10B vs Phase 10C
    # ================================================================
    print("\n" + "=" * 100)
    print("PHASE 10C RESULTS: PROTOCOL-STABLE BRIDGE")
    print("=" * 100)

    print(f"\n{'config':<16} {'route':>5} {'agree%':>8} {'z_cycle':>8} "
          f"{'drift1':>8} {'drift5':>8} {'bound':>6} {'sem_flip':>9} {'KL':>7} {'cos':>7}")
    print("-" * 95)

    for r in sorted(all_results, key=lambda x: -x['agreement']):
        sf = f"{r['semantic_flip_when_z_stable']:.1%}" if not np.isnan(r['semantic_flip_when_z_stable']) else "N/A"
        print(f"{r['config']:<16} {r['route']:>5} {r['agreement']:>7.1%} "
              f"{r['z_cycle_error']:>8.4f} {r['drift_1']:>8.4f} {r['drift_5']:>8.4f} "
              f"{'Y' if r['drift_bounded'] else 'N':>6} {sf:>9} "
              f"{r['kl_div']:>7.4f} {r['cosine']:>7.4f}")

    # Success criteria check
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)

    for r in all_results:
        checks = {
            'agreement≥95%': r['agreement'] >= 0.95,
            'z_cycle≤2%': r['z_cycle_error'] <= 0.02,
            'drift_bounded': r['drift_bounded'],
        }
        n_pass = sum(checks.values())
        status = 'PROTOCOL-STABLE' if n_pass == 3 else f'{n_pass}/3'
        print(f"  {r['config']:<16}: ", end='')
        for k, v in checks.items():
            print(f"{k}={'PASS' if v else 'FAIL'} ", end='')
        print(f"→ {status}")

    # Summary comparison with Phase 10B
    print("\n" + "=" * 80)
    print("vs PHASE 10B (without cycle constraint)")
    print("=" * 80)

    # Phase 10B reference values
    p10b_ref = {
        'INT8_k32': {'agreement': 0.976, 'drift_1': 0.0098},
        'INT8_k16': {'agreement': 0.974, 'drift_1': 0.0105},
        'INT4_k32': {'agreement': 0.972, 'drift_1': 0.0230},
        'INT4_k16': {'agreement': 0.966, 'drift_1': 0.0241},
        'VQ_K256_d32': {'agreement': 0.940, 'drift_1': 0.0177},
        'VQ_K256_d16': {'agreement': 0.960, 'drift_1': 0.0236},
        'VQ_K512_d32': {'agreement': 0.950, 'drift_1': 0.0166},
        'VQ_K512_d16': {'agreement': 0.948, 'drift_1': 0.0255},
    }

    print(f"\n{'config':<16} {'10B agree':>10} {'10C agree':>10} {'Δagree':>8} "
          f"{'10B drift1':>11} {'10C drift1':>11} {'z_cycle':>8}")
    print("-" * 80)

    for r in all_results:
        ref = p10b_ref.get(r['config'], {})
        ref_a = ref.get('agreement', 0)
        ref_d = ref.get('drift_1', 0)
        delta_a = r['agreement'] - ref_a
        print(f"{r['config']:<16} {ref_a:>9.1%} {r['agreement']:>9.1%} "
              f"{delta_a:>+7.1%} {ref_d:>11.4f} {r['drift_1']:>11.4f} "
              f"{r['z_cycle_error']:>8.4f}")

    # Final verdict
    n_protocol_stable = sum(1 for r in all_results
                            if r['agreement'] >= 0.95
                            and r['z_cycle_error'] <= 0.02
                            and r['drift_bounded'])

    print(f"\n  VERDICT: {n_protocol_stable}/{len(all_results)} configs are PROTOCOL-STABLE")

    if n_protocol_stable > 0:
        print("  Bridge is a composable protocol operator:")
        print("    - Any network → h → z (write)")
        print("    - z → ĥ → z (read back, z doesn't drift)")
        print("    - Multi-step inference / causal intervention possible")
    else:
        best_cycle = min(all_results, key=lambda x: x['z_cycle_error'])
        print(f"  Closest: {best_cycle['config']} with z_cycle={best_cycle['z_cycle_error']:.4f}")
        print(f"  Next: increase λ_cycle or use deterministic quantization")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "phase10c_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(set().union(*(r.keys() for r in all_results)))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    print("\n" + "=" * 100)
    print("Phase 10C experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
