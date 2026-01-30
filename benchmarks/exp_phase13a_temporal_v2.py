#!/usr/bin/env python3
"""
Phase 13A: Temporal Dynamics v2 — Temporal-Aware Training
==========================================================
Phase 13 showed 1-step prediction beats baselines but multi-step degrades.

Fixes:
1. Temporal-aware encoder: add temporal smoothness loss during ADC/DAC training
   (consecutive frames should have similar z, since pixel changes are small)
2. More data: 500 train sequences (vs 200)
3. Deeper predictor with skip connections
4. Multi-step training: train predictor on its own rollout (scheduled sampling)
5. E_dyn as rollout regularizer

Goal: Multi-step prediction (5-step avg) beats copy-last baseline.

Usage:
    python3 -u benchmarks/exp_phase13a_temporal_v2.py --device cuda
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
    TemporalDynamicsEnergy, evaluate_reconstruction, evaluate_prediction,
    evaluate_e_dyn
)


# ============================================================================
# IMPROVED TEMPORAL PREDICTOR with skip connections
# ============================================================================

class TemporalPredictorV2(nn.Module):
    """Improved predictor: residual blocks + skip from z_curr."""
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        ch = 64

        self.input_proj = nn.Sequential(
            nn.Conv2d(2 * n_bits, ch, 3, padding=1), nn.ReLU())

        self.res1 = self._res_block(ch)
        self.res2 = self._res_block(ch)
        self.res3 = self._res_block(ch)

        self.output = nn.Conv2d(ch, n_bits, 3, padding=1)
        self.skip = nn.Conv2d(n_bits, n_bits, 1)  # skip from z_curr
        self.quantizer = GumbelSigmoid()

    def _res_block(self, ch):
        return nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1))

    def forward(self, z_prev, z_curr):
        inp = torch.cat([z_prev, z_curr], dim=1)
        h = self.input_proj(inp)
        h = h + self.res1(h)
        h = F.relu(h)
        h = h + self.res2(h)
        h = F.relu(h)
        h = h + self.res3(h)
        logits = self.output(F.relu(h)) + self.skip(z_curr)
        z_pred = self.quantizer(logits)
        return z_pred, logits

    def set_temperature(self, tau):
        self.quantizer.set_temperature(tau)


# ============================================================================
# TEMPORAL-AWARE ENCODER TRAINING
# ============================================================================

def train_adc_dac_temporal(encoder, decoder, train_seqs, device,
                           epochs=25, batch_size=8, alpha_temp=0.5):
    """Train encoder/decoder with temporal smoothness loss.
    L = BCE(x, x_hat) + α * ||z_t - z_{t+1}||_1 / (||x_t - x_{t+1}||_1 + ε)
    Idea: if pixel change is small, z change should also be small.
    """
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    N, T, H, W = train_seqs.shape
    seqs_t = torch.tensor(train_seqs)

    for epoch in range(epochs):
        encoder.train(); decoder.train()
        tau = 1.0 + (0.3 - 1.0) * epoch / max(1, epochs - 1)
        encoder.set_temperature(tau)

        perm = torch.randperm(N)
        tl_bce, tl_temp, nb = 0., 0., 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            seq = seqs_t[idx].to(device)  # (B, T, H, W)
            B = seq.shape[0]
            opt.zero_grad()

            # Reconstruction loss on all frames
            frames = seq.reshape(B * T, 1, H, W)
            z_all, _ = encoder(frames)
            x_hat = decoder(z_all)
            loss_bce = F.binary_cross_entropy(x_hat, frames)

            # Temporal smoothness on consecutive z
            z_seq = z_all.reshape(B, T, *z_all.shape[1:])
            z_t = z_seq[:, :-1].reshape(-1, *z_all.shape[1:])
            z_tp1 = z_seq[:, 1:].reshape(-1, *z_all.shape[1:])

            # Pixel change magnitude (normalize temporal loss)
            x_t = seq[:, :-1].reshape(-1, 1, H, W)
            x_tp1 = seq[:, 1:].reshape(-1, 1, H, W)
            pixel_change = (x_t - x_tp1).abs().mean(dim=(1,2,3), keepdim=True)

            # z change should be proportional to pixel change
            z_change = (z_t - z_tp1).abs().mean(dim=(1,2,3), keepdim=True)
            loss_temp = (z_change / (pixel_change.view(-1,1,1,1) + 0.01)).mean()

            loss = loss_bce + alpha_temp * loss_temp
            loss.backward(); opt.step()
            tl_bce += loss_bce.item()
            tl_temp += loss_temp.item()
            nb += 1

        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{epochs}: BCE={tl_bce/nb:.4f} "
                  f"temp={tl_temp/nb:.4f} τ={tau:.2f}")


# ============================================================================
# MULTI-STEP TRAINING with scheduled sampling
# ============================================================================

def train_predictor_multistep(predictor, encoder, train_seqs, device,
                               epochs=30, batch_size=8, rollout_steps=5):
    """Train predictor with multi-step rollout (scheduled sampling).
    Phase 1 (epoch 1-15): teacher forcing (use ground truth z for input)
    Phase 2 (epoch 16-30): scheduled sampling (mix GT and predicted z)
    """
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    # Pre-encode
    encoder.eval()
    N, T, H, W = train_seqs.shape
    all_z = []
    with torch.no_grad():
        for i in range(N):
            seq_frames = torch.tensor(train_seqs[i]).unsqueeze(1).to(device)
            z_seq, _ = encoder(seq_frames)
            all_z.append(z_seq.cpu())
    all_z = torch.stack(all_z)  # (N, T, k, H_z, W_z)

    for epoch in range(epochs):
        predictor.train()
        tau = 1.0 + (0.5 - 1.0) * epoch / max(1, epochs - 1)
        predictor.set_temperature(tau)

        # Scheduled sampling ratio: starts at 0 (all teacher), ends at 0.5
        ss_ratio = min(0.5, max(0, (epoch - 10) / 20))

        perm = torch.randperm(N)
        tl, tl_ham, nb = 0., 0., 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            z_batch = all_z[idx].to(device)  # (B, T, k, H, W)
            B = z_batch.shape[0]
            opt.zero_grad()

            total_loss = 0.
            total_ham = 0.
            n_steps = 0

            # Multi-step rollout
            z_prev = z_batch[:, 0]
            z_curr = z_batch[:, 1]

            for t in range(2, min(T, 2 + rollout_steps)):
                z_target = z_batch[:, t]
                z_pred, logits = predictor(z_prev, z_curr)

                loss = F.binary_cross_entropy_with_logits(
                    logits, z_target, reduction='mean')
                total_loss += loss

                with torch.no_grad():
                    ham = ((logits > 0).float() != z_target).float().mean()
                    total_ham += ham.item()

                # Scheduled sampling: use predicted z or GT z for next step
                use_pred = torch.rand(1).item() < ss_ratio
                z_prev = z_curr
                z_curr = z_pred.detach() if use_pred else z_target.detach()
                n_steps += 1

            (total_loss / n_steps).backward()
            opt.step()
            tl += (total_loss / n_steps).item()
            tl_ham += total_ham / n_steps
            nb += 1

        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{epochs}: BCE={tl/nb:.4f} "
                  f"Hamming={tl_ham/nb:.4f} ss={ss_ratio:.2f} τ={tau:.2f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_phase13a')
    parser.add_argument('--n_bits', type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 13A: TEMPORAL DYNAMICS v2 (TEMPORAL-AWARE + MULTI-STEP)")
    print("=" * 100)

    # [1] Generate data (more than Phase 13)
    print("[1] Generating Moving MNIST sequences (500 train, 100 test)...")
    gen = MovingMNISTGenerator(canvas_size=64, n_digits=2, seq_len=20, seed=args.seed)
    gen.load_digits(n=1000)
    train_seqs = gen.generate_dataset(500)
    test_seqs = gen.generate_dataset(100)
    print(f"    Train: {train_seqs.shape}, Test: {test_seqs.shape}")

    # [2] Train temporal-aware encoder
    print("\n[2] Training temporal-aware ADC/DAC...")
    encoder = VideoEncoder(args.n_bits).to(device)
    decoder = VideoDecoder(args.n_bits).to(device)
    train_adc_dac_temporal(encoder, decoder, train_seqs, device, epochs=25, alpha_temp=0.3)

    recon = evaluate_reconstruction(encoder, decoder, test_seqs, device)
    print(f"    Reconstruction: MSE={recon['mse_mean']:.4f}")

    # [3] Check temporal smoothness of learned z
    print("\n[3] Checking z temporal smoothness...")
    encoder.eval()
    z_diffs = []
    with torch.no_grad():
        for i in range(min(50, len(test_seqs))):
            seq = torch.tensor(test_seqs[i]).unsqueeze(1).to(device)
            z_seq, _ = encoder(seq)
            for t in range(len(z_seq) - 1):
                diff = (z_seq[t] != z_seq[t+1]).float().mean().item()
                z_diffs.append(diff)
    print(f"    Avg z flip rate between consecutive frames: {np.mean(z_diffs):.4f}")

    # [4] Train multi-step predictor
    print("\n[4] Training multi-step predictor (with scheduled sampling)...")
    predictor = TemporalPredictorV2(args.n_bits).to(device)
    train_predictor_multistep(predictor, encoder, train_seqs, device,
                               epochs=35, rollout_steps=5)

    # [5] Train E_dyn
    print("\n[5] Training E_dyn...")
    e_dyn = TemporalDynamicsEnergy(args.n_bits).to(device)
    from benchmarks.exp_phase13_temporal_dynamics import train_e_dyn
    train_e_dyn(e_dyn, encoder, train_seqs, device, epochs=15)

    # [6] Evaluate
    print("\n[6] Evaluating multi-step prediction...")
    pred_results = evaluate_prediction(predictor, encoder, decoder, test_seqs, device,
                                       context_len=5, predict_len=10)

    print(f"\n{'step':>6} {'Hamming':>10} {'MSE':>10}")
    print("-" * 30)
    for t in range(10):
        k_h, k_m = f'ham_t{t+1}', f'mse_t{t+1}'
        if k_h in pred_results:
            print(f"{t+1:>6} {pred_results[k_h]:>10.4f} {pred_results[k_m]:>10.4f}")

    # [7] E_dyn
    print("\n[7] E_dyn discrimination...")
    edyn_results = evaluate_e_dyn(e_dyn, encoder, test_seqs, device)
    print(f"    Real energy: {edyn_results['real_energy']:.4f}")
    print(f"    Shuffled energy: {edyn_results['shuffled_energy']:.4f}")
    print(f"    Gap: {edyn_results['energy_gap']:.4f}")

    # [8] Baselines
    print("\n[8] Baselines...")
    encoder.eval(); decoder.eval()
    baseline_mse = []
    with torch.no_grad():
        for i in range(min(len(test_seqs), 100)):
            seq = torch.tensor(test_seqs[i]).unsqueeze(1).to(device)
            for t in range(5, min(15, 20)):
                mse = F.mse_loss(seq[t-1:t], seq[t:t+1]).item()
                baseline_mse.append(mse)
    copy_z_ham = []
    with torch.no_grad():
        for i in range(min(len(test_seqs), 100)):
            seq = torch.tensor(test_seqs[i]).unsqueeze(1).to(device)
            z_seq, _ = encoder(seq)
            for t in range(5, min(15, 20)):
                ham = (z_seq[t-1].round() != z_seq[t].round()).float().mean().item()
                copy_z_ham.append(ham)
    print(f"    Copy-last MSE: {np.mean(baseline_mse):.4f}")
    print(f"    Copy-z Hamming: {np.mean(copy_z_ham):.4f}")

    # Summary
    print("\n" + "=" * 100)
    print("PHASE 13A COMPARISON (vs Phase 13)")
    print("=" * 100)

    avg5_ham = np.mean([pred_results[f'ham_t{t+1}'] for t in range(5)
                        if f'ham_t{t+1}' in pred_results])
    avg5_mse = np.mean([pred_results[f'mse_t{t+1}'] for t in range(5)
                        if f'mse_t{t+1}' in pred_results])

    pred_beats_copy_mse = avg5_mse < np.mean(baseline_mse)
    pred_beats_copy_ham = avg5_ham < np.mean(copy_z_ham)
    edyn_disc = edyn_results['energy_gap'] > 0.01

    t1_ham = pred_results.get('ham_t1', 0)
    t1_mse = pred_results.get('mse_t1', 0)

    print(f"\n  {'Metric':>25} {'Phase 13':>12} {'Phase 13A':>12} {'Baseline':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Recon MSE':>25} {'0.0077':>12} {recon['mse_mean']:>12.4f} {'—':>12}")
    print(f"  {'1-step Hamming':>25} {'0.1392':>12} {t1_ham:>12.4f} {np.mean(copy_z_ham):>12.4f}")
    print(f"  {'1-step MSE':>25} {'0.0311':>12} {t1_mse:>12.4f} {np.mean(baseline_mse):>12.4f}")
    print(f"  {'5-step avg Hamming':>25} {'0.1987':>12} {avg5_ham:>12.4f} {np.mean(copy_z_ham):>12.4f}")
    print(f"  {'5-step avg MSE':>25} {'0.0481':>12} {avg5_mse:>12.4f} {np.mean(baseline_mse):>12.4f}")
    print(f"  {'E_dyn gap':>25} {'0.2693':>12} {edyn_results['energy_gap']:>12.4f} {'—':>12}")
    print(f"  {'z flip rate':>25} {'—':>12} {np.mean(z_diffs):>12.4f} {'—':>12}")

    verdict = sum([pred_beats_copy_mse, pred_beats_copy_ham, edyn_disc])
    print(f"\n  5-step beats copy-last (MSE):  {'YES' if pred_beats_copy_mse else 'NO'}")
    print(f"  5-step beats copy-z (Hamming): {'YES' if pred_beats_copy_ham else 'NO'}")
    print(f"  E_dyn discriminates:            {'YES' if edyn_disc else 'NO'}")
    print(f"\n  VERDICT: {verdict}/3 → "
          f"{'TEMPORAL DYNAMICS WORK' if verdict >= 2 else 'PARTIAL' if verdict >= 1 else 'NEEDS ITERATION'}")

    # Save
    csv_path = os.path.join(args.output_dir, "phase13a_results.csv")
    all_res = {**recon, **pred_results, **edyn_results,
               'z_flip_rate': np.mean(z_diffs),
               'copy_last_mse': np.mean(baseline_mse),
               'copy_z_hamming': np.mean(copy_z_ham),
               'verdict': verdict}
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted(all_res.keys()))
        w.writeheader()
        w.writerow(all_res)
    print(f"\nCSV saved to {csv_path}")
    print("\n" + "=" * 100)
    print("Phase 13A experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
