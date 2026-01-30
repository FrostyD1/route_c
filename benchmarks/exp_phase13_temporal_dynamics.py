#!/usr/bin/env python3
"""
Phase 13: Temporal Dynamics on Moving MNIST
============================================
Core question: Can Route C's discrete core handle temporal/generative tasks?

Setup:
  1. Generate Moving MNIST sequences (2 digits bouncing in 64×64, T=20 frames)
  2. Train ADC/DAC on individual frames → z_t ∈ {0,1}^{k×H×W}
  3. Train temporal predictor in z-space: z_{t-1}, z_t → z_{t+1}
  4. Extend E_core with temporal neighborhood (E_dyn)
  5. Test: given first T_ctx frames, predict next T_pred frames

Metrics:
  - Frame reconstruction (MSE, SSIM proxy)
  - Temporal prediction accuracy (Hamming distance in z-space)
  - Multi-step rollout stability (drift over T steps)
  - E_dyn energy decrease during prediction

GPU: 4GB safe — 64×64 frames, small models, binary latent.

Usage:
    python3 -u benchmarks/exp_phase13_temporal_dynamics.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, csv, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# MOVING MNIST GENERATOR (procedural, no download needed)
# ============================================================================

class MovingMNISTGenerator:
    """Generate bouncing digit sequences procedurally from static MNIST."""
    def __init__(self, canvas_size=64, digit_size=28, n_digits=2, seq_len=20, seed=42):
        self.canvas_size = canvas_size
        self.digit_size = digit_size
        self.n_digits = n_digits
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def load_digits(self, n=1000):
        """Load MNIST digits for compositing."""
        from torchvision import datasets, transforms
        ds = datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.ToTensor())
        idx = self.rng.choice(len(ds), n, replace=False)
        self.digits = [ds[i][0].squeeze(0).numpy() for i in idx]  # 28×28 float

    def generate_sequence(self):
        """Generate one sequence of bouncing digits on canvas."""
        cs = self.canvas_size
        ds = self.digit_size
        max_pos = cs - ds

        frames = np.zeros((self.seq_len, cs, cs), dtype=np.float32)

        # Init positions and velocities for each digit
        positions = []
        velocities = []
        digit_imgs = []
        for _ in range(self.n_digits):
            x = self.rng.integers(0, max_pos + 1)
            y = self.rng.integers(0, max_pos + 1)
            # velocity: 2-5 pixels per frame
            speed = self.rng.uniform(2, 5)
            angle = self.rng.uniform(0, 2 * np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            positions.append([float(x), float(y)])
            velocities.append([vx, vy])
            digit_imgs.append(self.digits[self.rng.integers(0, len(self.digits))])

        for t in range(self.seq_len):
            canvas = np.zeros((cs, cs), dtype=np.float32)
            for d in range(self.n_digits):
                x, y = positions[d]
                vx, vy = velocities[d]

                # Bounce off walls
                if x < 0:
                    x = -x; vx = -vx
                if y < 0:
                    y = -y; vy = -vy
                if x > max_pos:
                    x = 2 * max_pos - x; vx = -vx
                if y > max_pos:
                    y = 2 * max_pos - y; vy = -vy

                ix, iy = int(round(x)), int(round(y))
                ix = max(0, min(ix, max_pos))
                iy = max(0, min(iy, max_pos))

                # Composite digit onto canvas (max blend)
                canvas[iy:iy+ds, ix:ix+ds] = np.maximum(
                    canvas[iy:iy+ds, ix:ix+ds], digit_imgs[d])

                # Update position
                positions[d] = [x + vx, y + vy]
                velocities[d] = [vx, vy]

            frames[t] = canvas

        return frames  # (T, 64, 64)

    def generate_dataset(self, n_sequences):
        """Generate n sequences."""
        all_seqs = []
        for _ in range(n_sequences):
            seq = self.generate_sequence()
            all_seqs.append(seq)
        return np.stack(all_seqs)  # (N, T, 64, 64)


# ============================================================================
# ENCODER / DECODER for 64×64 → 8×8 latent
# ============================================================================

class GumbelSigmoid(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    def forward(self, logits):
        if self.training:
            u = torch.rand_like(logits).clamp(1e-8, 1-1e-8)
            noisy = (logits - torch.log(-torch.log(u))) / self.temperature
        else:
            noisy = logits / self.temperature
        soft = torch.sigmoid(noisy)
        hard = (soft > 0.5).float()
        return hard - soft.detach() + soft
    def set_temperature(self, tau): self.temperature = tau


class VideoEncoder(nn.Module):
    """64×64 → 8×8×k binary latent."""
    def __init__(self, n_bits=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),   # 32×32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 16×16
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),  # 8×8
            nn.Conv2d(64, n_bits, 3, padding=1),                    # 8×8×k
        )
        self.quantizer = GumbelSigmoid()

    def forward(self, x):
        logits = self.conv(x)
        return self.quantizer(logits), logits

    def set_temperature(self, tau):
        self.quantizer.set_temperature(tau)


class VideoDecoder(nn.Module):
    """8×8×k binary → 64×64 reconstruction."""
    def __init__(self, n_bits=8):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, 64, 4, stride=2, padding=1), nn.ReLU(),  # 16×16
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),      # 32×32
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),      # 64×64
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(z)


# ============================================================================
# TEMPORAL PREDICTOR: z_{t-1}, z_t → z_{t+1} in discrete space
# ============================================================================

class TemporalPredictor(nn.Module):
    """Predict next z from two previous frames in z-space.
    Input: (2k, H, W) = concat(z_{t-1}, z_t)
    Output: (k, H, W) logits for z_{t+1}
    """
    def __init__(self, n_bits=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * n_bits, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, n_bits, 3, padding=1),
        )
        self.n_bits = n_bits
        self.quantizer = GumbelSigmoid()

    def forward(self, z_prev, z_curr):
        inp = torch.cat([z_prev, z_curr], dim=1)  # (B, 2k, H, W)
        logits = self.net(inp)
        z_pred = self.quantizer(logits)
        return z_pred, logits

    def set_temperature(self, tau):
        self.quantizer.set_temperature(tau)


# ============================================================================
# E_DYN: Temporal dynamics energy
# ============================================================================

class TemporalDynamicsEnergy(nn.Module):
    """E_dyn(z_t, z_{t+1}) = learned temporal consistency.
    Uses 3×3 spatiotemporal context to predict each bit.
    """
    def __init__(self, n_bits=8):
        super().__init__()
        # Predict z_{t+1} bits from z_t context
        self.predictor = nn.Sequential(
            nn.Conv2d(n_bits, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, n_bits, 3, padding=1),
        )

    def forward(self, z_t, z_next):
        """Return per-bit negative log-likelihood (energy contribution)."""
        logits = self.predictor(z_t)
        # BCE as temporal energy: how well does z_t predict z_{t+1}?
        energy = F.binary_cross_entropy_with_logits(
            logits, z_next, reduction='none')
        return energy.mean()


# ============================================================================
# TRAINING
# ============================================================================

def train_adc_dac(encoder, decoder, train_seqs, device, epochs=15, batch_size=32):
    """Train encoder/decoder on individual frames."""
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    # Flatten sequences to individual frames
    N, T, H, W = train_seqs.shape
    frames = train_seqs.reshape(N * T, 1, H, W)
    frames_t = torch.tensor(frames)

    for epoch in range(epochs):
        encoder.train(); decoder.train()
        tau = 1.0 + (0.3 - 1.0) * epoch / max(1, epochs - 1)
        encoder.set_temperature(tau)

        perm = torch.randperm(len(frames_t))
        tl, nb = 0., 0

        for i in range(0, len(frames_t), batch_size):
            idx = perm[i:i+batch_size]
            x = frames_t[idx].to(device)
            opt.zero_grad()

            z, _ = encoder(x)
            x_hat = decoder(z)
            loss = F.binary_cross_entropy(x_hat, x)
            loss.backward(); opt.step()
            tl += loss.item(); nb += 1

        if (epoch + 1) % 5 == 0:
            print(f"    ADC/DAC epoch {epoch+1}/{epochs}: BCE={tl/nb:.4f} τ={tau:.2f}")


def train_temporal_predictor(predictor, encoder, train_seqs, device,
                             epochs=20, batch_size=16):
    """Train temporal predictor on z-space sequences."""
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    # Pre-encode all frames
    encoder.eval()
    N, T, H, W = train_seqs.shape
    all_z = []
    with torch.no_grad():
        for i in range(N):
            seq_frames = torch.tensor(train_seqs[i]).unsqueeze(1).to(device)  # (T, 1, H, W)
            z_seq, _ = encoder(seq_frames)
            all_z.append(z_seq.cpu())
    all_z = torch.stack(all_z)  # (N, T, k, H_z, W_z)

    for epoch in range(epochs):
        predictor.train()
        tau = 1.0 + (0.5 - 1.0) * epoch / max(1, epochs - 1)
        predictor.set_temperature(tau)

        perm = torch.randperm(N)
        tl_bce, tl_ham, nb = 0., 0., 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            z_batch = all_z[idx].to(device)  # (B, T, k, H, W)
            opt.zero_grad()

            total_loss = 0.
            total_ham = 0.
            n_pairs = 0

            for t in range(2, T):
                z_prev = z_batch[:, t-2]
                z_curr = z_batch[:, t-1]
                z_target = z_batch[:, t]

                z_pred, logits = predictor(z_prev, z_curr)

                # BCE loss on prediction
                loss = F.binary_cross_entropy_with_logits(
                    logits, z_target, reduction='mean')
                total_loss += loss

                # Hamming distance (monitoring)
                with torch.no_grad():
                    hard_pred = (logits > 0).float()
                    ham = (hard_pred != z_target).float().mean()
                    total_ham += ham.item()
                n_pairs += 1

            (total_loss / n_pairs).backward()
            opt.step()
            tl_bce += (total_loss / n_pairs).item()
            tl_ham += total_ham / n_pairs
            nb += 1

        if (epoch + 1) % 5 == 0:
            print(f"    Pred epoch {epoch+1}/{epochs}: "
                  f"BCE={tl_bce/nb:.4f} Hamming={tl_ham/nb:.4f} τ={tau:.2f}")


def train_e_dyn(e_dyn, encoder, train_seqs, device, epochs=10, batch_size=16):
    """Train temporal energy model."""
    opt = torch.optim.Adam(e_dyn.parameters(), lr=1e-3)

    # Pre-encode
    encoder.eval()
    N, T, H, W = train_seqs.shape
    all_z = []
    with torch.no_grad():
        for i in range(N):
            seq_frames = torch.tensor(train_seqs[i]).unsqueeze(1).to(device)
            z_seq, _ = encoder(seq_frames)
            all_z.append(z_seq.cpu())
    all_z = torch.stack(all_z)

    for epoch in range(epochs):
        e_dyn.train()
        perm = torch.randperm(N)
        tl, nb = 0., 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            z_batch = all_z[idx].to(device)
            opt.zero_grad()

            total_loss = 0.
            for t in range(T - 1):
                loss = e_dyn(z_batch[:, t], z_batch[:, t+1])
                total_loss += loss
            (total_loss / (T-1)).backward()
            opt.step()
            tl += (total_loss / (T-1)).item(); nb += 1

        if (epoch + 1) % 5 == 0:
            print(f"    E_dyn epoch {epoch+1}/{epochs}: energy={tl/nb:.4f}")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_reconstruction(encoder, decoder, test_seqs, device):
    """Evaluate frame reconstruction quality."""
    encoder.eval(); decoder.eval()
    N, T, H, W = test_seqs.shape
    frames = torch.tensor(test_seqs.reshape(N * T, 1, H, W))

    mses, bit_usage = [], []
    with torch.no_grad():
        for i in range(0, len(frames), 16):
            x = frames[i:i+16].to(device)
            z, _ = encoder(x)
            x_hat = decoder(z)
            mse = F.mse_loss(x_hat, x, reduction='none').mean(dim=(1,2,3))
            mses.extend(mse.cpu().tolist())
            bit_usage.append(z.mean().item())

    return {
        'mse_mean': np.mean(mses),
        'mse_std': np.std(mses),
        'bit_usage': np.mean(bit_usage),
    }


def evaluate_prediction(predictor, encoder, decoder, test_seqs, device,
                        context_len=5, predict_len=10):
    """Evaluate multi-step temporal prediction."""
    predictor.eval(); encoder.eval(); decoder.eval()
    N, T, H, W = test_seqs.shape

    all_ham = {t: [] for t in range(predict_len)}
    all_mse = {t: [] for t in range(predict_len)}

    with torch.no_grad():
        for i in range(N):
            seq_frames = torch.tensor(test_seqs[i]).unsqueeze(1).to(device)  # (T, 1, H, W)
            z_seq, _ = encoder(seq_frames)  # (T, k, H_z, W_z)

            # Use first context_len frames, predict next predict_len
            z_prev = z_seq[context_len - 2].unsqueeze(0)
            z_curr = z_seq[context_len - 1].unsqueeze(0)

            for t in range(predict_len):
                target_t = context_len + t
                if target_t >= T:
                    break

                z_pred, _ = predictor(z_prev, z_curr)
                z_true = z_seq[target_t].unsqueeze(0)

                # Hamming distance
                ham = (z_pred.round() != z_true.round()).float().mean().item()
                all_ham[t].append(ham)

                # Pixel-space MSE
                x_pred = decoder(z_pred)
                x_true = seq_frames[target_t].unsqueeze(0)
                mse = F.mse_loss(x_pred, x_true).item()
                all_mse[t].append(mse)

                # Autoregressive: use prediction for next step
                z_prev = z_curr
                z_curr = z_pred

    results = {}
    for t in range(predict_len):
        if all_ham[t]:
            results[f'ham_t{t+1}'] = np.mean(all_ham[t])
            results[f'mse_t{t+1}'] = np.mean(all_mse[t])

    return results


def evaluate_e_dyn(e_dyn, encoder, test_seqs, device):
    """Evaluate temporal energy on real vs shuffled sequences."""
    e_dyn.eval(); encoder.eval()
    N, T, H, W = test_seqs.shape

    real_energies = []
    shuffled_energies = []

    with torch.no_grad():
        for i in range(min(N, 50)):
            seq_frames = torch.tensor(test_seqs[i]).unsqueeze(1).to(device)
            z_seq, _ = encoder(seq_frames)

            # Real sequence energy
            for t in range(T - 1):
                e = e_dyn(z_seq[t:t+1], z_seq[t+1:t+2]).item()
                real_energies.append(e)

            # Shuffled sequence energy (random temporal order)
            perm = torch.randperm(T)
            z_shuffled = z_seq[perm]
            for t in range(T - 1):
                e = e_dyn(z_shuffled[t:t+1], z_shuffled[t+1:t+2]).item()
                shuffled_energies.append(e)

    return {
        'real_energy': np.mean(real_energies),
        'shuffled_energy': np.mean(shuffled_energies),
        'energy_gap': np.mean(shuffled_energies) - np.mean(real_energies),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_phase13')
    parser.add_argument('--n_train', type=int, default=200)
    parser.add_argument('--n_test', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--n_bits', type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 13: TEMPORAL DYNAMICS ON MOVING MNIST")
    print("=" * 100)

    # [1] Generate data
    print("[1] Generating Moving MNIST sequences...")
    gen = MovingMNISTGenerator(canvas_size=64, n_digits=2,
                               seq_len=args.seq_len, seed=args.seed)
    gen.load_digits(n=500)
    train_seqs = gen.generate_dataset(args.n_train)  # (200, 20, 64, 64)
    test_seqs = gen.generate_dataset(args.n_test)    # (50, 20, 64, 64)
    print(f"    Train: {train_seqs.shape}, Test: {test_seqs.shape}")
    print(f"    Pixel range: [{train_seqs.min():.2f}, {train_seqs.max():.2f}]")

    # [2] Train ADC/DAC
    print("\n[2] Training ADC/DAC (64×64 → 8×8×{} binary)...".format(args.n_bits))
    encoder = VideoEncoder(args.n_bits).to(device)
    decoder = VideoDecoder(args.n_bits).to(device)
    train_adc_dac(encoder, decoder, train_seqs, device, epochs=20)

    recon = evaluate_reconstruction(encoder, decoder, test_seqs, device)
    print(f"    Reconstruction: MSE={recon['mse_mean']:.4f}±{recon['mse_std']:.4f}, "
          f"bit_usage={recon['bit_usage']:.3f}")

    # [3] Train temporal predictor
    print(f"\n[3] Training temporal predictor (z_{{t-1}}, z_t → z_{{t+1}})...")
    predictor = TemporalPredictor(args.n_bits).to(device)
    train_temporal_predictor(predictor, encoder, train_seqs, device, epochs=25)

    # [4] Train E_dyn
    print(f"\n[4] Training E_dyn (temporal dynamics energy)...")
    e_dyn = TemporalDynamicsEnergy(args.n_bits).to(device)
    train_e_dyn(e_dyn, encoder, train_seqs, device, epochs=15)

    # [5] Evaluate prediction
    print(f"\n[5] Evaluating multi-step prediction (context=5, predict=10)...")
    pred_results = evaluate_prediction(predictor, encoder, decoder, test_seqs, device,
                                       context_len=5, predict_len=10)

    print(f"\n{'step':>6} {'Hamming':>10} {'MSE':>10}")
    print("-" * 30)
    for t in range(10):
        key_h = f'ham_t{t+1}'
        key_m = f'mse_t{t+1}'
        if key_h in pred_results:
            print(f"{t+1:>6} {pred_results[key_h]:>10.4f} {pred_results[key_m]:>10.4f}")

    # [6] Evaluate E_dyn discrimination
    print(f"\n[6] E_dyn discrimination (real vs shuffled sequences)...")
    edyn_results = evaluate_e_dyn(e_dyn, encoder, test_seqs, device)
    print(f"    Real sequence energy:     {edyn_results['real_energy']:.4f}")
    print(f"    Shuffled sequence energy:  {edyn_results['shuffled_energy']:.4f}")
    print(f"    Energy gap (shuffle-real): {edyn_results['energy_gap']:.4f}")
    discriminates = edyn_results['energy_gap'] > 0.01
    print(f"    E_dyn discriminates:       {'YES' if discriminates else 'NO'}")

    # [7] Baseline comparison: copy-last-frame
    print(f"\n[7] Baseline: copy-last-frame prediction...")
    encoder.eval(); decoder.eval()
    baseline_mse = []
    with torch.no_grad():
        for i in range(min(len(test_seqs), 50)):
            seq_frames = torch.tensor(test_seqs[i]).unsqueeze(1).to(device)
            for t in range(5, min(15, args.seq_len)):
                x_last = seq_frames[t-1:t]
                x_true = seq_frames[t:t+1]
                mse = F.mse_loss(x_last, x_true).item()
                baseline_mse.append(mse)
    print(f"    Copy-last MSE: {np.mean(baseline_mse):.4f}")

    # Also: encode-copy baseline (copy z_{t-1} as prediction)
    copy_z_ham = []
    with torch.no_grad():
        for i in range(min(len(test_seqs), 50)):
            seq_frames = torch.tensor(test_seqs[i]).unsqueeze(1).to(device)
            z_seq, _ = encoder(seq_frames)
            for t in range(5, min(15, args.seq_len)):
                ham = (z_seq[t-1].round() != z_seq[t].round()).float().mean().item()
                copy_z_ham.append(ham)
    print(f"    Copy-z Hamming: {np.mean(copy_z_ham):.4f}")

    # Summary
    print("\n" + "=" * 100)
    print("PHASE 13 SUMMARY")
    print("=" * 100)

    # Compute averages
    avg_pred_ham = np.mean([pred_results[f'ham_t{t+1}'] for t in range(min(5, 10))
                           if f'ham_t{t+1}' in pred_results])
    avg_pred_mse = np.mean([pred_results[f'mse_t{t+1}'] for t in range(min(5, 10))
                           if f'mse_t{t+1}' in pred_results])

    print(f"\n  Reconstruction:     MSE={recon['mse_mean']:.4f}")
    print(f"  Prediction (1-5):   Hamming={avg_pred_ham:.4f}, MSE={avg_pred_mse:.4f}")
    print(f"  Copy-last baseline: MSE={np.mean(baseline_mse):.4f}")
    print(f"  Copy-z baseline:    Hamming={np.mean(copy_z_ham):.4f}")

    pred_beats_copy = avg_pred_mse < np.mean(baseline_mse)
    ham_beats_copy = avg_pred_ham < np.mean(copy_z_ham)

    print(f"\n  Predictor beats copy-last (MSE):     {'YES' if pred_beats_copy else 'NO'}")
    print(f"  Predictor beats copy-z (Hamming):     {'YES' if ham_beats_copy else 'NO'}")
    print(f"  E_dyn discriminates temporal order:    {'YES' if discriminates else 'NO'}")

    # Drift analysis
    if 'ham_t1' in pred_results and 'ham_t5' in pred_results:
        drift = pred_results['ham_t5'] - pred_results['ham_t1']
        print(f"\n  Prediction drift (t1→t5): {drift:+.4f} Hamming")
        if 'ham_t10' in pred_results:
            drift10 = pred_results['ham_t10'] - pred_results['ham_t1']
            print(f"  Prediction drift (t1→t10): {drift10:+.4f} Hamming")
            linear_drift = pred_results['ham_t1'] * 10 / 1
            print(f"  Drift type: {'sub-linear (stable)' if pred_results.get('ham_t10', 0) < linear_drift else 'super-linear (diverging)'}")

    verdict_pass = sum([pred_beats_copy, ham_beats_copy, discriminates])
    print(f"\n  VERDICT: {verdict_pass}/3 checks passed → "
          f"{'TEMPORAL DYNAMICS WORK' if verdict_pass >= 2 else 'NEEDS ITERATION'}")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "phase13_results.csv")
    all_results = {**recon, **pred_results, **edyn_results,
                   'copy_last_mse': np.mean(baseline_mse),
                   'copy_z_hamming': np.mean(copy_z_ham),
                   'pred_beats_copy': pred_beats_copy,
                   'ham_beats_copy': ham_beats_copy,
                   'e_dyn_discriminates': discriminates}
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted(all_results.keys()))
        w.writeheader()
        w.writerow(all_results)
    print(f"\nCSV saved to {csv_path}")

    print("\n" + "=" * 100)
    print("Phase 13 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
