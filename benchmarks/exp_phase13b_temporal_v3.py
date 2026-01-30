#!/usr/bin/env python3
"""
Phase 13B: Temporal Dynamics v3 — Fix Collapse + Multi-Step Win
================================================================
Phase 13A collapsed because temporal smoothness loss was too strong.

Fixes (per GPT execution plan):
1. alpha_temp sweep with collapse monitoring (Var(z), Hamming)
2. Anti-collapse: bit entropy regularizer + stop-gradient on temporal loss
3. Multi-horizon loss: predict t+1..t+5 simultaneously
4. Energy projection: prediction-as-proposal + E_dyn/E_core refinement

Success criteria:
  - 1-step still beats baseline
  - 5-step avg Hamming AND MSE beat baseline
  - Var(z) doesn't collapse, drift stays sub-linear

Usage:
    python3 -u benchmarks/exp_phase13b_temporal_v3.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, csv, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from benchmarks.exp_phase13_temporal_dynamics import (
    MovingMNISTGenerator, GumbelSigmoid, VideoEncoder, VideoDecoder,
    TemporalDynamicsEnergy, evaluate_reconstruction, evaluate_e_dyn
)
from benchmarks.exp_phase13a_temporal_v2 import TemporalPredictorV2


# ============================================================================
# STEP 1: alpha_temp SWEEP with collapse detection
# ============================================================================

def check_collapse(encoder, test_seqs, device):
    """Check if encoder has collapsed (constant z)."""
    encoder.eval()
    z_vars, z_hams = [], []
    with torch.no_grad():
        for i in range(min(20, len(test_seqs))):
            seq = torch.tensor(test_seqs[i]).unsqueeze(1).to(device)
            z_seq, _ = encoder(seq)  # (T, k, H, W)
            # Var(z) across batch/spatial/bit
            z_vars.append(z_seq.var().item())
            # Hamming between consecutive frames
            for t in range(len(z_seq) - 1):
                ham = (z_seq[t].round() != z_seq[t+1].round()).float().mean().item()
                z_hams.append(ham)
    return {
        'var_z': np.mean(z_vars),
        'hamming_consecutive': np.mean(z_hams),
        'collapsed': np.mean(z_vars) < 0.01 or np.mean(z_hams) < 0.01
    }


def train_adc_dac_v3(encoder, decoder, train_seqs, device,
                      epochs=20, batch_size=8, alpha_temp=0.03,
                      alpha_entropy=1e-3):
    """Train encoder/decoder with:
    - Temporal predictability (not smoothness!) via stop-gradient
    - Bit entropy regularizer (anti-collapse)
    """
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    N, T, H, W = train_seqs.shape
    seqs_t = torch.tensor(train_seqs)

    for epoch in range(epochs):
        encoder.train(); decoder.train()
        tau = 1.0 + (0.3 - 1.0) * epoch / max(1, epochs - 1)
        encoder.set_temperature(tau)

        perm = torch.randperm(N)
        tl_bce, tl_temp, tl_ent, nb = 0., 0., 0., 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            seq = seqs_t[idx].to(device)
            B = seq.shape[0]
            opt.zero_grad()

            frames = seq.reshape(B * T, 1, H, W)
            z_all, logits_all = encoder(frames)
            x_hat = decoder(z_all)
            loss_bce = F.binary_cross_entropy(x_hat, frames)

            # Temporal predictability (stop-gradient on target!)
            # Key insight: don't penalize z change directly, penalize UNpredictable change
            # Use simple L1 between consecutive z, but with stop_grad on one side
            z_seq = z_all.reshape(B, T, *z_all.shape[1:])
            z_t = z_seq[:, :-1]
            z_tp1 = z_seq[:, 1:]
            # stop-grad on target: encoder learns to be smooth, but target doesn't
            # get pulled toward current — prevents collapse
            loss_temp = F.l1_loss(z_tp1, z_t.detach())

            # Bit entropy regularizer: encourage each bit to have ~50% usage
            # p(bit=1) for each bit channel, averaged over spatial/batch
            bit_probs = torch.sigmoid(logits_all).mean(dim=(0, 2, 3))  # (k,)
            # Entropy: -p*log(p) - (1-p)*log(1-p), maximize it
            eps = 1e-8
            bit_entropy = -(bit_probs * (bit_probs + eps).log() +
                           (1 - bit_probs) * (1 - bit_probs + eps).log()).mean()
            # We want to MAXIMIZE entropy, so MINIMIZE negative entropy
            loss_ent = -bit_entropy  # negative because we want to maximize

            loss = loss_bce + alpha_temp * loss_temp + alpha_entropy * loss_ent
            loss.backward(); opt.step()
            tl_bce += loss_bce.item()
            tl_temp += loss_temp.item()
            tl_ent += bit_entropy.item()
            nb += 1

        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{epochs}: BCE={tl_bce/nb:.4f} "
                  f"temp={tl_temp/nb:.4f} entropy={tl_ent/nb:.4f} τ={tau:.2f}")


# ============================================================================
# STEP 3: MULTI-HORIZON PREDICTOR TRAINING
# ============================================================================

def train_predictor_multihorizon(predictor, encoder, train_seqs, device,
                                  epochs=35, batch_size=8, max_horizon=5):
    """Train predictor with multi-horizon loss.
    Loss = Σ_{k=1}^{K} w_k * BCE(pred_k, true_{t+k})
    where w_k = 1/k (emphasize near-term but still learn far)
    Uses scheduled sampling from epoch 15+.
    """
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    # Pre-encode
    encoder.eval()
    N, T, H, W = train_seqs.shape
    all_z = []
    with torch.no_grad():
        for i in range(N):
            seq = torch.tensor(train_seqs[i]).unsqueeze(1).to(device)
            z_seq, _ = encoder(seq)
            all_z.append(z_seq.cpu())
    all_z = torch.stack(all_z)

    for epoch in range(epochs):
        predictor.train()
        tau = 1.0 + (0.5 - 1.0) * epoch / max(1, epochs - 1)
        predictor.set_temperature(tau)

        # Scheduled sampling ratio
        ss_ratio = min(0.5, max(0, (epoch - 15) / 20))

        perm = torch.randperm(N)
        tl, tl_ham, nb = 0., 0., 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            z_batch = all_z[idx].to(device)
            opt.zero_grad()

            # Random start point for multi-horizon
            max_start = T - max_horizon - 2
            if max_start < 2:
                max_start = 2
            start = torch.randint(2, max_start + 1, (1,)).item()

            total_loss = 0.
            total_ham = 0.
            n_steps = 0

            z_prev = z_batch[:, start - 2]
            z_curr = z_batch[:, start - 1]

            for k in range(max_horizon):
                t = start + k
                if t >= T:
                    break

                z_target = z_batch[:, t]
                z_pred, logits = predictor(z_prev, z_curr)

                # Weighted loss: 1/k prioritizes near-term
                weight = 1.0 / (k + 1)
                loss = weight * F.binary_cross_entropy_with_logits(
                    logits, z_target, reduction='mean')
                total_loss += loss

                with torch.no_grad():
                    ham = ((logits > 0).float() != z_target).float().mean()
                    total_ham += ham.item()

                # Scheduled sampling
                use_pred = torch.rand(1).item() < ss_ratio
                z_prev = z_curr
                z_curr = z_pred.detach() if use_pred else z_target.detach()
                n_steps += 1

            if n_steps > 0:
                (total_loss / n_steps).backward()
                opt.step()
                tl += (total_loss / n_steps).item()
                tl_ham += total_ham / n_steps
                nb += 1

        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{epochs}: loss={tl/nb:.4f} "
                  f"Hamming={tl_ham/nb:.4f} ss={ss_ratio:.2f} τ={tau:.2f}")


# ============================================================================
# STEP 4: ENERGY PROJECTION (prediction-as-proposal + refinement)
# ============================================================================

class EnergyProjection(nn.Module):
    """Project predicted z back to low-energy manifold.
    Uses E_dyn + local E_core-like consistency.
    Amortized: 1-2 forward passes through a small refinement net.
    """
    def __init__(self, n_bits=8):
        super().__init__()
        # Refinement net: takes z_pred + z_context → refined z
        self.refine = nn.Sequential(
            nn.Conv2d(3 * n_bits, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, n_bits, 3, padding=1),
        )
        self.skip = nn.Conv2d(n_bits, n_bits, 1)
        self.quantizer = GumbelSigmoid()
        self.n_bits = n_bits

    def forward(self, z_pred, z_prev, z_curr):
        """Refine z_pred using context from z_prev and z_curr."""
        inp = torch.cat([z_pred, z_prev, z_curr], dim=1)
        logits = self.refine(inp) + self.skip(z_pred)
        return self.quantizer(logits), logits

    def set_temperature(self, tau):
        self.quantizer.set_temperature(tau)


def train_energy_projection(projector, predictor, encoder, e_dyn,
                             train_seqs, device, epochs=15, batch_size=8):
    """Train energy projector to reduce E_dyn after prediction."""
    opt = torch.optim.Adam(projector.parameters(), lr=1e-3)

    encoder.eval(); predictor.eval(); e_dyn.eval()
    N, T, H, W = train_seqs.shape
    all_z = []
    with torch.no_grad():
        for i in range(N):
            seq = torch.tensor(train_seqs[i]).unsqueeze(1).to(device)
            z_seq, _ = encoder(seq)
            all_z.append(z_seq.cpu())
    all_z = torch.stack(all_z)

    for epoch in range(epochs):
        projector.train()
        tau = 1.0 + (0.5 - 1.0) * epoch / max(1, epochs - 1)
        projector.set_temperature(tau)

        perm = torch.randperm(N)
        tl_bce, tl_edyn, nb = 0., 0., 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            z_batch = all_z[idx].to(device)
            opt.zero_grad()

            total_loss = 0.
            n_steps = 0

            for t in range(2, min(T, 12)):
                z_prev = z_batch[:, t-2]
                z_curr = z_batch[:, t-1]
                z_target = z_batch[:, t]

                with torch.no_grad():
                    z_pred, _ = predictor(z_prev, z_curr)

                z_refined, ref_logits = projector(z_pred.detach(), z_prev.detach(), z_curr.detach())

                # Primary: match ground truth
                loss_bce = F.binary_cross_entropy_with_logits(
                    ref_logits, z_target, reduction='mean')

                # Secondary: lower E_dyn (temporal consistency)
                loss_edyn = e_dyn(z_curr.detach(), z_refined)

                loss = loss_bce + 0.1 * loss_edyn
                total_loss += loss
                tl_bce += loss_bce.item()
                tl_edyn += loss_edyn.item()
                n_steps += 1

            if n_steps > 0:
                (total_loss / n_steps).backward()
                opt.step()
                nb += 1

        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{epochs}: BCE={tl_bce/max(1,nb*n_steps):.4f} "
                  f"E_dyn={tl_edyn/max(1,nb*n_steps):.4f}")


# ============================================================================
# EVALUATION with optional energy projection
# ============================================================================

def evaluate_prediction_with_projection(predictor, encoder, decoder, test_seqs,
                                         device, projector=None,
                                         context_len=5, predict_len=10):
    """Evaluate multi-step prediction with optional energy projection."""
    predictor.eval(); encoder.eval(); decoder.eval()
    if projector:
        projector.eval()

    N, T, H, W = test_seqs.shape
    all_ham = {t: [] for t in range(predict_len)}
    all_mse = {t: [] for t in range(predict_len)}

    with torch.no_grad():
        for i in range(N):
            seq = torch.tensor(test_seqs[i]).unsqueeze(1).to(device)
            z_seq, _ = encoder(seq)

            z_prev = z_seq[context_len - 2].unsqueeze(0)
            z_curr = z_seq[context_len - 1].unsqueeze(0)

            for t in range(predict_len):
                target_t = context_len + t
                if target_t >= T:
                    break

                z_pred, _ = predictor(z_prev, z_curr)

                # Energy projection (if available)
                if projector is not None:
                    z_pred, _ = projector(z_pred, z_prev, z_curr)

                z_true = z_seq[target_t].unsqueeze(0)
                ham = (z_pred.round() != z_true.round()).float().mean().item()
                all_ham[t].append(ham)

                x_pred = decoder(z_pred)
                x_true = seq[target_t].unsqueeze(0)
                mse = F.mse_loss(x_pred, x_true).item()
                all_mse[t].append(mse)

                z_prev = z_curr
                z_curr = z_pred

    results = {}
    for t in range(predict_len):
        if all_ham[t]:
            results[f'ham_t{t+1}'] = np.mean(all_ham[t])
            results[f'mse_t{t+1}'] = np.mean(all_mse[t])
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_phase13b')
    parser.add_argument('--n_bits', type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 13B: TEMPORAL DYNAMICS v3 (FIX COLLAPSE + MULTI-STEP WIN)")
    print("=" * 100)

    # [1] Generate data
    print("[1] Generating Moving MNIST (500 train, 100 test)...")
    gen = MovingMNISTGenerator(canvas_size=64, n_digits=2, seq_len=20, seed=args.seed)
    gen.load_digits(n=1000)
    train_seqs = gen.generate_dataset(500)
    test_seqs = gen.generate_dataset(100)
    print(f"    Train: {train_seqs.shape}, Test: {test_seqs.shape}")

    # ================================================================
    # STEP 1: alpha_temp sweep (quick, 3 epochs each)
    # ================================================================
    print("\n" + "=" * 80)
    print("STEP 1: alpha_temp SWEEP (collapse detection)")
    print("=" * 80)

    alphas = [0.003, 0.01, 0.03, 0.1, 0.3]
    sweep_results = []

    for alpha in alphas:
        torch.manual_seed(args.seed)
        enc_test = VideoEncoder(args.n_bits).to(device)
        dec_test = VideoDecoder(args.n_bits).to(device)

        # Quick train (5 epochs)
        train_adc_dac_v3(enc_test, dec_test, train_seqs, device,
                         epochs=5, alpha_temp=alpha, alpha_entropy=1e-3)

        collapse = check_collapse(enc_test, test_seqs, device)
        recon = evaluate_reconstruction(enc_test, dec_test, test_seqs, device)

        sweep_results.append({
            'alpha': alpha, **collapse, 'mse': recon['mse_mean']
        })
        status = "COLLAPSE" if collapse['collapsed'] else "OK"
        print(f"    α={alpha:.3f}: Var(z)={collapse['var_z']:.4f} "
              f"Ham={collapse['hamming_consecutive']:.4f} MSE={recon['mse_mean']:.4f} → {status}")

        del enc_test, dec_test
        torch.cuda.empty_cache()

    # Find best alpha (largest non-collapsing)
    ok_alphas = [s for s in sweep_results if not s['collapsed']]
    if ok_alphas:
        best_alpha = max(ok_alphas, key=lambda s: s['alpha'])['alpha']
    else:
        best_alpha = alphas[0]  # fallback to smallest
    print(f"\n    Best alpha_temp: {best_alpha}")

    # ================================================================
    # STEP 2: Full training with best alpha + anti-collapse
    # ================================================================
    print("\n" + "=" * 80)
    print(f"STEP 2: FULL ADC/DAC TRAINING (α_temp={best_alpha}, entropy reg)")
    print("=" * 80)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    encoder = VideoEncoder(args.n_bits).to(device)
    decoder = VideoDecoder(args.n_bits).to(device)
    train_adc_dac_v3(encoder, decoder, train_seqs, device,
                      epochs=25, alpha_temp=best_alpha, alpha_entropy=1e-3)

    collapse = check_collapse(encoder, test_seqs, device)
    recon = evaluate_reconstruction(encoder, decoder, test_seqs, device)
    print(f"    Var(z)={collapse['var_z']:.4f} Ham={collapse['hamming_consecutive']:.4f} "
          f"MSE={recon['mse_mean']:.4f} collapsed={collapse['collapsed']}")

    # ================================================================
    # STEP 3: Multi-horizon predictor training
    # ================================================================
    print("\n" + "=" * 80)
    print("STEP 3: MULTI-HORIZON PREDICTOR (predict t+1..t+5)")
    print("=" * 80)

    predictor = TemporalPredictorV2(args.n_bits).to(device)
    train_predictor_multihorizon(predictor, encoder, train_seqs, device,
                                  epochs=40, max_horizon=5)

    # Evaluate WITHOUT projection
    pred_noproj = evaluate_prediction_with_projection(
        predictor, encoder, decoder, test_seqs, device,
        projector=None, context_len=5, predict_len=10)

    print(f"\n  [No projection]")
    print(f"  {'step':>6} {'Hamming':>10} {'MSE':>10}")
    print(f"  {'-'*30}")
    for t in range(10):
        k_h, k_m = f'ham_t{t+1}', f'mse_t{t+1}'
        if k_h in pred_noproj:
            print(f"  {t+1:>6} {pred_noproj[k_h]:>10.4f} {pred_noproj[k_m]:>10.4f}")

    # ================================================================
    # STEP 4: E_dyn + Energy Projection
    # ================================================================
    print("\n" + "=" * 80)
    print("STEP 4: ENERGY PROJECTION (prediction-as-proposal + refinement)")
    print("=" * 80)

    e_dyn = TemporalDynamicsEnergy(args.n_bits).to(device)
    from benchmarks.exp_phase13_temporal_dynamics import train_e_dyn
    train_e_dyn(e_dyn, encoder, train_seqs, device, epochs=15)

    projector = EnergyProjection(args.n_bits).to(device)
    train_energy_projection(projector, predictor, encoder, e_dyn,
                             train_seqs, device, epochs=20)

    # Evaluate WITH projection
    pred_proj = evaluate_prediction_with_projection(
        predictor, encoder, decoder, test_seqs, device,
        projector=projector, context_len=5, predict_len=10)

    print(f"\n  [With energy projection]")
    print(f"  {'step':>6} {'Hamming':>10} {'MSE':>10}")
    print(f"  {'-'*30}")
    for t in range(10):
        k_h, k_m = f'ham_t{t+1}', f'mse_t{t+1}'
        if k_h in pred_proj:
            print(f"  {t+1:>6} {pred_proj[k_h]:>10.4f} {pred_proj[k_m]:>10.4f}")

    # ================================================================
    # BASELINES + E_DYN
    # ================================================================
    print("\n" + "=" * 80)
    print("BASELINES + E_DYN")
    print("=" * 80)

    encoder.eval(); decoder.eval()
    baseline_mse, copy_z_ham = [], []
    with torch.no_grad():
        for i in range(min(len(test_seqs), 100)):
            seq = torch.tensor(test_seqs[i]).unsqueeze(1).to(device)
            z_seq, _ = encoder(seq)
            for t in range(5, min(15, 20)):
                baseline_mse.append(F.mse_loss(seq[t-1:t], seq[t:t+1]).item())
                copy_z_ham.append((z_seq[t-1].round() != z_seq[t].round()).float().mean().item())
    print(f"    Copy-last MSE: {np.mean(baseline_mse):.4f}")
    print(f"    Copy-z Hamming: {np.mean(copy_z_ham):.4f}")

    edyn_results = evaluate_e_dyn(e_dyn, encoder, test_seqs, device)
    print(f"    E_dyn real: {edyn_results['real_energy']:.4f}")
    print(f"    E_dyn shuffled: {edyn_results['shuffled_energy']:.4f}")
    print(f"    E_dyn gap: {edyn_results['energy_gap']:.4f}")

    # ================================================================
    # FINAL SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 100)
    print("PHASE 13B FINAL SUMMARY")
    print("=" * 100)

    bl_mse = np.mean(baseline_mse)
    bl_ham = np.mean(copy_z_ham)

    def avg_metric(results, prefix, steps=5):
        vals = [results[f'{prefix}_t{t+1}'] for t in range(steps)
                if f'{prefix}_t{t+1}' in results]
        return np.mean(vals) if vals else float('inf')

    avg5_ham_np = avg_metric(pred_noproj, 'ham')
    avg5_mse_np = avg_metric(pred_noproj, 'mse')
    avg5_ham_p = avg_metric(pred_proj, 'ham')
    avg5_mse_p = avg_metric(pred_proj, 'mse')

    t1_ham_np = pred_noproj.get('ham_t1', 0)
    t1_mse_np = pred_noproj.get('mse_t1', 0)
    t1_ham_p = pred_proj.get('ham_t1', 0)
    t1_mse_p = pred_proj.get('mse_t1', 0)

    print(f"\n  {'':>25} {'Baseline':>10} {'Phase13':>10} {'NoPrj':>10} {'+Proj':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Recon MSE':>25} {'—':>10} {'0.0077':>10} {recon['mse_mean']:>10.4f} {'—':>10}")
    print(f"  {'1-step Hamming':>25} {bl_ham:>10.4f} {'0.1392':>10} {t1_ham_np:>10.4f} {t1_ham_p:>10.4f}")
    print(f"  {'1-step MSE':>25} {bl_mse:>10.4f} {'0.0311':>10} {t1_mse_np:>10.4f} {t1_mse_p:>10.4f}")
    print(f"  {'5-step Hamming':>25} {bl_ham:>10.4f} {'0.1987':>10} {avg5_ham_np:>10.4f} {avg5_ham_p:>10.4f}")
    print(f"  {'5-step MSE':>25} {bl_mse:>10.4f} {'0.0481':>10} {avg5_mse_np:>10.4f} {avg5_mse_p:>10.4f}")
    print(f"  {'E_dyn gap':>25} {'—':>10} {'0.2693':>10} {edyn_results['energy_gap']:>10.4f} {'—':>10}")
    print(f"  {'Var(z)':>25} {'—':>10} {'—':>10} {collapse['var_z']:>10.4f} {'—':>10}")
    print(f"  {'z flip rate':>25} {'—':>10} {'—':>10} {collapse['hamming_consecutive']:>10.4f} {'—':>10}")

    # Verdicts
    checks = {}
    checks['1step_ham'] = t1_ham_p < bl_ham
    checks['1step_mse'] = t1_mse_p < bl_mse
    checks['5step_ham'] = avg5_ham_p < bl_ham
    checks['5step_mse'] = avg5_mse_p < bl_mse
    checks['edyn_disc'] = edyn_results['energy_gap'] > 0.01
    checks['no_collapse'] = not collapse['collapsed']

    # Projection gain
    proj_gain_ham = avg5_ham_np - avg5_ham_p
    proj_gain_mse = avg5_mse_np - avg5_mse_p

    print(f"\n  CHECKS:")
    for k, v in checks.items():
        print(f"    {k:>20}: {'PASS' if v else 'FAIL'}")
    n_pass = sum(checks.values())
    print(f"\n  Energy projection gain: Hamming {proj_gain_ham:+.4f}, MSE {proj_gain_mse:+.4f}")

    # Drift analysis
    if 'ham_t1' in pred_proj and 'ham_t5' in pred_proj:
        drift = pred_proj['ham_t5'] - pred_proj['ham_t1']
        print(f"  Drift (t1→t5): {drift:+.4f}")
        if 'ham_t10' in pred_proj:
            drift10 = pred_proj['ham_t10'] - pred_proj['ham_t1']
            linear_expected = pred_proj['ham_t1'] * 10
            print(f"  Drift (t1→t10): {drift10:+.4f}")
            print(f"  Drift type: {'sub-linear' if pred_proj['ham_t10'] < linear_expected else 'super-linear'}")

    print(f"\n  VERDICT: {n_pass}/6 checks passed → ", end='')
    if n_pass >= 5:
        print("TEMPORAL DYNAMICS CONFIRMED")
    elif n_pass >= 3:
        print("PARTIAL SUCCESS")
    else:
        print("NEEDS MORE ITERATION")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "phase13b_results.csv")
    all_res = {
        'alpha_temp': best_alpha,
        **{f'noproj_{k}': v for k, v in pred_noproj.items()},
        **{f'proj_{k}': v for k, v in pred_proj.items()},
        **collapse, **recon, **edyn_results,
        'baseline_mse': bl_mse, 'baseline_ham': bl_ham,
        'proj_gain_ham': proj_gain_ham, 'proj_gain_mse': proj_gain_mse,
        'checks_passed': n_pass,
    }
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted(all_res.keys()))
        w.writeheader()
        w.writerow(all_res)
    print(f"\nCSV saved to {csv_path}")

    print("\n" + "=" * 100)
    print("Phase 13B experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
