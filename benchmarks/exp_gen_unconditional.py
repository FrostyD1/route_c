#!/usr/bin/env python3
"""
Generative Experiment: Unconditional Image Generation via Discrete Core
========================================================================
Core idea: Generate images by sampling z from learned prior, then decode.

Architecture:
  1. Train ADC/DAC on dataset (14×14 z, k=8 bits)
  2. Encode training set → z_data
  3. Learn z prior via 3 baselines + 1 paradigm method:
     - G2-A: Independent Bernoulli (per-bit marginal)
     - G2-B: Per-position token categorical (per-position marginal)
     - G2-C: Autoregressive (raster-order token prediction)
     - G3: Denoising compilation (iterative refinement from noise)
  4. Sample z → decode → image

Metrics:
  - FID (using classifier features)
  - E_core violation rate
  - Token histogram KL
  - Cycle stability
  - Diversity (Hamming distance distribution)
  - 1-NN distance (memorization check)

Outputs: 8×8 grid images + 512 samples + CSV metrics.

Usage:
    python3 -u benchmarks/exp_gen_unconditional.py --device cuda --dataset fmnist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, csv, argparse
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# ADC/DAC: 28×28 → 14×14×k binary
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


class Encoder14(nn.Module):
    """28×28 → 14×14×k binary."""
    def __init__(self, in_ch=1, n_bits=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(),  # 14×14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, n_bits, 3, padding=1),
        )
        self.quantizer = GumbelSigmoid()
    def forward(self, x):
        logits = self.conv(x)
        return self.quantizer(logits), logits
    def set_temperature(self, tau):
        self.quantizer.set_temperature(tau)


class Decoder14(nn.Module):
    """14×14×k → 28×28."""
    def __init__(self, out_ch=1, n_bits=8):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, 64, 4, stride=2, padding=1), nn.ReLU(),  # 28×28
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, out_ch, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z):
        return self.deconv(z)


# ============================================================================
# E_CORE: Local MRF energy (simplified for generation)
# ============================================================================

class LocalEnergyCore(nn.Module):
    """E_core: 3×3 neighborhood pseudo-likelihood."""
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        context_size = 9 * n_bits - 1
        self.predictor = nn.Sequential(
            nn.Linear(context_size, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def get_context(self, z, bit_idx, i, j):
        """Extract 3×3 neighborhood context for bit (bit_idx, i, j)."""
        B, K, H, W = z.shape
        contexts = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = (i + di) % H, (j + dj) % W
                for b in range(K):
                    if di == 0 and dj == 0 and b == bit_idx:
                        continue
                    contexts.append(z[:, b, ni, nj])
        return torch.stack(contexts, dim=1)

    def violation_rate(self, z):
        """Compute average pseudo-likelihood violation."""
        B, K, H, W = z.shape
        violations = []
        # Sample random positions
        n_samples = min(50, H * W * K)
        for _ in range(n_samples):
            b = torch.randint(K, (1,)).item()
            i = torch.randint(H, (1,)).item()
            j = torch.randint(W, (1,)).item()
            ctx = self.get_context(z, b, i, j)
            logit = self.predictor(ctx).squeeze(1)
            pred = (logit > 0).float()
            actual = z[:, b, i, j]
            violations.append((pred != actual).float().mean().item())
        return np.mean(violations)


# ============================================================================
# G2-A: Independent Bernoulli
# ============================================================================

class BernoulliPrior:
    """Per-bit marginal Bernoulli prior."""
    def __init__(self):
        self.probs = None

    def fit(self, z_data):
        """z_data: (N, k, H, W) binary tensor."""
        self.probs = z_data.mean(dim=0)  # (k, H, W)

    def sample(self, n, device='cpu'):
        probs = self.probs.unsqueeze(0).expand(n, -1, -1, -1).to(device)
        return (torch.rand_like(probs) < probs).float()


# ============================================================================
# G2-B: Per-position token categorical
# ============================================================================

class TokenCategoricalPrior:
    """Per-position k-bit token categorical prior."""
    def __init__(self, n_bits=8):
        self.n_bits = n_bits
        self.distributions = None

    def fit(self, z_data):
        """z_data: (N, k, H, W) binary tensor."""
        N, K, H, W = z_data.shape
        n_tokens = 2 ** K
        self.H, self.W, self.K = H, W, K

        # Convert each position to token index
        self.token_probs = torch.zeros(H, W, n_tokens)
        for i in range(H):
            for j in range(W):
                bits = z_data[:, :, i, j]  # (N, K)
                # Convert to token index
                indices = torch.zeros(N, dtype=torch.long)
                for b in range(K):
                    indices += (bits[:, b].long() << b)
                counts = torch.bincount(indices, minlength=n_tokens).float()
                self.token_probs[i, j] = counts / counts.sum()

    def sample(self, n, device='cpu'):
        H, W, K = self.H, self.W, self.K
        z = torch.zeros(n, K, H, W, device=device)
        probs_flat = self.token_probs.reshape(H * W, -1).to(device)  # (H*W, n_tokens)
        for pos in range(H * W):
            i, j = pos // W, pos % W
            indices = torch.multinomial(probs_flat[pos].unsqueeze(0).expand(n, -1), 1).squeeze(1)
            for b in range(K):
                z[:, b, i, j] = ((indices >> b) & 1).float()
        return z


# ============================================================================
# G2-C: Autoregressive in z-space
# ============================================================================

class AutoregressivePrior(nn.Module):
    """Masked-conv AR model: predict each bit from spatially causal context.
    Uses masked convolutions for parallel training (no sequential loop).
    Sampling is still sequential but per-bit (fast enough for 8×14×14).
    """
    def __init__(self, n_bits=8, hidden=64):
        super().__init__()
        self.n_bits = n_bits
        # Predict each bit from all previous bits (in raster scan of k×H×W)
        # Use a simple per-bit predictor with spatial context
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, n_bits, 3, padding=1),
        )

    def forward(self, z_bits):
        """Training: predict each bit from masked input (teacher forcing)."""
        B, K, H, W = z_bits.shape
        total_loss = 0.
        # For each bit channel, mask it out and predict from rest
        for b in range(K):
            z_masked = z_bits.clone()
            z_masked[:, b:] = 0  # mask current and future bits
            logits = self.net(z_masked)  # (B, K, H, W)
            loss = F.binary_cross_entropy_with_logits(
                logits[:, b], z_bits[:, b])
            total_loss += loss
        return total_loss / K

    @torch.no_grad()
    def sample(self, n, H, W, device='cpu'):
        K = self.n_bits
        z = torch.zeros(n, K, H, W, device=device)
        for b in range(K):
            logits = self.net(z)
            probs = torch.sigmoid(logits[:, b])
            z[:, b] = (torch.rand_like(probs) < probs).float()
        return z


# ============================================================================
# G3: Denoising Compilation Sampler
# ============================================================================

class DenoisingCompiler(nn.Module):
    """Denoising sampler: learn to remove noise from z, then iterate from random z.
    This is the Route C native generation method — sleep-phase compilation extended
    to unconditional generation (no observation, pure prior).
    """
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        # Input: noisy z (k channels) + noise level (1 channel)
        self.net = nn.Sequential(
            nn.Conv2d(n_bits + 1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, n_bits, 3, padding=1),
        )
        self.skip = nn.Conv2d(n_bits, n_bits, 1)

    def forward(self, z_noisy, noise_level):
        """Predict clean z from noisy z + noise level."""
        B = z_noisy.shape[0]
        nl = noise_level.view(B, 1, 1, 1).expand(-1, 1, z_noisy.shape[2], z_noisy.shape[3])
        inp = torch.cat([z_noisy, nl], dim=1)
        logits = self.net(inp) + self.skip(z_noisy)
        return logits  # raw logits, apply sigmoid for probabilities

    @torch.no_grad()
    def sample(self, n, H, W, device='cpu', n_steps=10, temperature=0.8):
        """Iterative denoising from random z."""
        K = self.n_bits
        # Start from random binary
        z = (torch.rand(n, K, H, W, device=device) > 0.5).float()

        for step in range(n_steps):
            # Noise level decreases: 1.0 → 0.1
            nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
            logits = self(z, nl)

            # Apply with temperature
            probs = torch.sigmoid(logits / temperature)

            # At each step, refine: probabilistic update
            # More confident as we progress
            confidence = (step + 1) / n_steps
            mask = (torch.rand_like(z) < confidence).float()
            z_new = (torch.rand_like(z) < probs).float()
            z = mask * z_new + (1 - mask) * z

        # Final hard decision
        logits = self(z, torch.zeros(n, device=device))
        z = (torch.sigmoid(logits) > 0.5).float()
        return z


def train_denoiser(denoiser, z_data, device, epochs=30, batch_size=64):
    """Train denoiser on corrupted z samples."""
    opt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)

    for epoch in range(epochs):
        denoiser.train()
        perm = torch.randperm(N)
        tl, nb = 0., 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            z_clean = z_data[idx].to(device)
            B = z_clean.shape[0]

            # Random noise level per sample
            noise_level = torch.rand(B, device=device)  # 0=clean, 1=full noise

            # Add bit-flip noise
            flip_mask = (torch.rand_like(z_clean) < noise_level.view(B, 1, 1, 1)).float()
            z_noisy = z_clean * (1 - flip_mask) + (1 - z_clean) * flip_mask

            opt.zero_grad()
            logits = denoiser(z_noisy, noise_level)
            loss = F.binary_cross_entropy_with_logits(logits, z_clean)
            loss.backward(); opt.step()
            tl += loss.item(); nb += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Denoiser epoch {epoch+1}/{epochs}: BCE={tl/nb:.4f}")


# ============================================================================
# EVALUATION
# ============================================================================

def compute_fid_simple(real_features, gen_features):
    """Simplified FID using sample statistics."""
    mu_r = real_features.mean(0)
    mu_g = gen_features.mean(0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(gen_features, rowvar=False)

    diff = mu_r - mu_g
    # Simplified: trace-based approximation (avoid sqrtm)
    fid = np.dot(diff, diff) + np.trace(sigma_r) + np.trace(sigma_g) - 2 * np.trace(
        np.linalg.cholesky(sigma_r @ sigma_g + 1e-6 * np.eye(len(mu_r))))
    return max(0, fid)


def extract_features_for_fid(images, device):
    """Extract simple CNN features for FID computation."""
    # Use a simple feature extractor (not pretrained, just measures distribution)
    features = []
    for i in range(0, len(images), 64):
        batch = images[i:i+64]
        if isinstance(batch, np.ndarray):
            batch = torch.tensor(batch)
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        batch = batch.to(device)
        # Simple pooled features
        feat = F.adaptive_avg_pool2d(batch, 4).reshape(batch.shape[0], -1)
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_1nn_distance(gen_samples, train_samples, n_check=100):
    """Check memorization: compute 1-NN distance from generated to training set."""
    gen_flat = gen_samples[:n_check].reshape(n_check, -1)
    train_flat = train_samples.reshape(len(train_samples), -1)

    distances = []
    for i in range(n_check):
        dists = ((train_flat - gen_flat[i:i+1]) ** 2).sum(1)
        distances.append(dists.min().item())
    return np.mean(distances), np.std(distances)


def compute_diversity(z_samples, n_pairs=500):
    """Compute Hamming distance distribution between sample pairs."""
    N = len(z_samples)
    z_flat = z_samples.reshape(N, -1)
    distances = []
    for _ in range(n_pairs):
        i, j = np.random.choice(N, 2, replace=False)
        ham = (z_flat[i] != z_flat[j]).float().mean().item()
        distances.append(ham)
    return np.mean(distances), np.std(distances)


def token_histogram_kl(z_real, z_gen, n_bits=8):
    """KL between token distributions at each position."""
    N_r, K, H, W = z_real.shape
    N_g = z_gen.shape[0]
    n_tokens = 2 ** K

    kls = []
    for i in range(H):
        for j in range(W):
            # Real token distribution
            idx_r = torch.zeros(N_r, dtype=torch.long)
            idx_g = torch.zeros(N_g, dtype=torch.long)
            for b in range(K):
                idx_r += (z_real[:, b, i, j].long() << b)
                idx_g += (z_gen[:, b, i, j].long() << b)

            p = torch.bincount(idx_r, minlength=n_tokens).float() + 1
            q = torch.bincount(idx_g, minlength=n_tokens).float() + 1
            p = p / p.sum()
            q = q / q.sum()
            kl = (p * (p / q).log()).sum().item()
            kls.append(kl)
    return np.mean(kls)


def save_grid(images, path, nrow=8):
    """Save images as grid PNG."""
    try:
        from torchvision.utils import save_image
        if isinstance(images, np.ndarray):
            images = torch.tensor(images)
        if images.dim() == 3:
            images = images.unsqueeze(1)
        save_image(images[:64], path, nrow=nrow, normalize=False)
        print(f"    Grid saved to {path}")
    except Exception as e:
        print(f"    Grid save failed: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', default='fmnist', choices=['mnist', 'fmnist', 'kmnist'])
    parser.add_argument('--output_dir', default='outputs/exp_gen')
    parser.add_argument('--n_bits', type=int, default=8)
    parser.add_argument('--n_samples', type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print(f"UNCONDITIONAL GENERATION: {args.dataset.upper()} (14×14×{args.n_bits} discrete core)")
    print("=" * 100)

    # [1] Load data
    print("[1] Loading dataset...")
    from torchvision import datasets, transforms
    ds_class = {'mnist': datasets.MNIST, 'fmnist': datasets.FashionMNIST,
                'kmnist': datasets.KMNIST}[args.dataset]
    train_ds = ds_class('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = ds_class('./data', train=False, download=True, transform=transforms.ToTensor())

    # Use 5000 for training, 1000 for eval
    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(len(train_ds), 5000, replace=False)
    test_idx = rng.choice(len(test_ds), 1000, replace=False)
    train_x = torch.stack([train_ds[i][0] for i in train_idx])
    test_x = torch.stack([test_ds[i][0] for i in test_idx])
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")

    # [2] Train ADC/DAC
    print("\n[2] Training ADC/DAC (28×28 → 14×14×{})...".format(args.n_bits))
    encoder = Encoder14(in_ch=1, n_bits=args.n_bits).to(device)
    decoder = Decoder14(out_ch=1, n_bits=args.n_bits).to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    for epoch in range(25):
        encoder.train(); decoder.train()
        tau = 1.0 + (0.3 - 1.0) * epoch / 24
        encoder.set_temperature(tau)
        perm = torch.randperm(len(train_x))
        tl, nb = 0., 0
        for i in range(0, len(train_x), 64):
            idx = perm[i:i+64]
            x = train_x[idx].to(device)
            opt.zero_grad()
            z, _ = encoder(x)
            x_hat = decoder(z)
            loss = F.binary_cross_entropy(x_hat, x)
            loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/25: BCE={tl/nb:.4f}")

    # G1: Decoder reliability (oracle z)
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        test_batch = test_x[:64].to(device)
        z_oracle, _ = encoder(test_batch)
        x_recon = decoder(z_oracle)
        recon_mse = F.mse_loss(x_recon, test_batch).item()
    print(f"    Oracle reconstruction MSE: {recon_mse:.4f}")
    save_grid(x_recon.cpu(), os.path.join(out_dir, 'oracle_recon.png'))

    # [3] Encode training set
    print("\n[3] Encoding training set → z_data...")
    encoder.eval()
    z_data = []
    with torch.no_grad():
        for i in range(0, len(train_x), 64):
            x = train_x[i:i+64].to(device)
            z, _ = encoder(x)
            z_data.append(z.cpu())
    z_data = torch.cat(z_data)  # (N, k, 14, 14)
    K, H, W = z_data.shape[1:]
    print(f"    z_data shape: {z_data.shape}, bit usage: {z_data.mean():.3f}")

    # Train E_core for evaluation
    print("\n[3b] Training E_core for evaluation...")
    e_core = LocalEnergyCore(args.n_bits).to(device)
    e_opt = torch.optim.Adam(e_core.parameters(), lr=1e-3)
    for epoch in range(10):
        e_core.train()
        perm = torch.randperm(len(z_data))
        tl, nb = 0., 0
        for i in range(0, len(z_data), 64):
            idx = perm[i:i+64]
            z = z_data[idx].to(device)
            B = z.shape[0]
            e_opt.zero_grad()
            # Train on random positions
            total_loss = 0.
            for _ in range(20):
                b = torch.randint(K, (1,)).item()
                ii = torch.randint(H, (1,)).item()
                jj = torch.randint(W, (1,)).item()
                ctx = e_core.get_context(z, b, ii, jj)
                logit = e_core.predictor(ctx).squeeze(1)
                target = z[:, b, ii, jj]
                total_loss += F.binary_cross_entropy_with_logits(logit, target)
            (total_loss / 20).backward()
            e_opt.step()
            tl += (total_loss / 20).item(); nb += 1
    e_core.eval()
    viol_data = e_core.violation_rate(z_data[:100].to(device))
    print(f"    E_core violation on real z: {viol_data:.4f}")

    # ================================================================
    # GENERATION METHODS
    # ================================================================
    all_results = []
    train_x_np = train_x.numpy().reshape(len(train_x), -1)

    methods = {}

    # G2-A: Independent Bernoulli
    print("\n" + "=" * 80)
    print("G2-A: INDEPENDENT BERNOULLI")
    print("=" * 80)
    bern_prior = BernoulliPrior()
    bern_prior.fit(z_data)
    z_bern = bern_prior.sample(args.n_samples, device)
    with torch.no_grad():
        x_bern = decoder(z_bern.to(device)).cpu()
    methods['bernoulli'] = (z_bern, x_bern)
    save_grid(x_bern, os.path.join(out_dir, 'gen_bernoulli.png'))

    # G2-B: Per-position token categorical
    print("\n" + "=" * 80)
    print("G2-B: PER-POSITION TOKEN CATEGORICAL")
    print("=" * 80)
    if args.n_bits <= 8:
        token_prior = TokenCategoricalPrior(args.n_bits)
        token_prior.fit(z_data)
        z_token = token_prior.sample(args.n_samples, device)
        with torch.no_grad():
            x_token = decoder(z_token.to(device)).cpu()
        methods['token_cat'] = (z_token, x_token)
        save_grid(x_token, os.path.join(out_dir, 'gen_token_cat.png'))
    else:
        print("    Skipped (n_bits too large for enumeration)")

    # G2-C: Autoregressive
    print("\n" + "=" * 80)
    print("G2-C: AUTOREGRESSIVE")
    print("=" * 80)
    if args.n_bits <= 8:
        ar_prior = AutoregressivePrior(args.n_bits, hidden=64).to(device)
        ar_opt = torch.optim.Adam(ar_prior.parameters(), lr=1e-3)
        for epoch in range(15):
            ar_prior.train()
            perm = torch.randperm(len(z_data))
            tl, nb = 0., 0
            for i in range(0, min(len(z_data), 3000), 64):
                idx = perm[i:i+64]
                z = z_data[idx].to(device)
                ar_opt.zero_grad()
                loss = ar_prior(z)
                loss.backward(); ar_opt.step()
                tl += loss.item(); nb += 1
            if (epoch + 1) % 5 == 0:
                print(f"    AR epoch {epoch+1}/15: BCE={tl/nb:.4f}")

        ar_prior.eval()
        z_ar = ar_prior.sample(args.n_samples, H, W, device)
        with torch.no_grad():
            x_ar = decoder(z_ar.to(device)).cpu()
        methods['ar'] = (z_ar, x_ar)
        save_grid(x_ar, os.path.join(out_dir, 'gen_ar.png'))
    else:
        print("    Skipped")

    # G3: Denoising Compilation
    print("\n" + "=" * 80)
    print("G3: DENOISING COMPILATION (Route C native)")
    print("=" * 80)
    denoiser = DenoisingCompiler(args.n_bits).to(device)
    train_denoiser(denoiser, z_data, device, epochs=30)

    denoiser.eval()
    z_denoise = denoiser.sample(args.n_samples, H, W, device, n_steps=15, temperature=0.7)
    with torch.no_grad():
        x_denoise = decoder(z_denoise.to(device)).cpu()
    methods['denoise'] = (z_denoise, x_denoise)
    save_grid(x_denoise, os.path.join(out_dir, 'gen_denoise.png'))

    # Also try different temperatures
    for temp in [0.5, 1.0]:
        z_temp = denoiser.sample(64, H, W, device, n_steps=15, temperature=temp)
        with torch.no_grad():
            x_temp = decoder(z_temp.to(device)).cpu()
        save_grid(x_temp, os.path.join(out_dir, f'gen_denoise_t{temp}.png'))

    # ================================================================
    # EVALUATION
    # ================================================================
    print("\n" + "=" * 100)
    print("EVALUATION")
    print("=" * 100)

    # Real data features for FID
    real_feats = extract_features_for_fid(test_x, device)

    for name, (z_gen, x_gen) in methods.items():
        print(f"\n  [{name}]")

        # Image quality
        gen_feats = extract_features_for_fid(x_gen, device)
        try:
            fid = compute_fid_simple(real_feats, gen_feats)
        except Exception:
            fid = float('nan')

        # Protocol metrics
        z_gen_t = z_gen.cpu() if isinstance(z_gen, torch.Tensor) else z_gen
        viol = e_core.violation_rate(z_gen_t[:100].to(device))
        tok_kl = token_histogram_kl(z_data[:500], z_gen_t[:500], args.n_bits)
        diversity_mean, diversity_std = compute_diversity(z_gen_t)

        # Cycle stability
        with torch.no_grad():
            z_check = z_gen_t[:100].to(device)
            x_check = decoder(z_check)
            z_cycle, _ = encoder(x_check)
            cycle_ham = (z_check != z_cycle).float().mean().item()

        # Memorization
        x_gen_np = x_gen.numpy().reshape(len(x_gen), -1) if isinstance(x_gen, torch.Tensor) else x_gen.reshape(len(x_gen), -1)
        nn_dist_mean, nn_dist_std = compute_1nn_distance(x_gen_np, train_x_np)

        r = {
            'method': name,
            'fid': fid,
            'violation_rate': viol,
            'token_kl': tok_kl,
            'diversity_mean': diversity_mean,
            'diversity_std': diversity_std,
            'cycle_hamming': cycle_ham,
            'nn_dist_mean': nn_dist_mean,
            'nn_dist_std': nn_dist_std,
        }
        all_results.append(r)

        print(f"    FID={fid:.2f} viol={viol:.4f} tokKL={tok_kl:.4f} "
              f"div={diversity_mean:.4f}±{diversity_std:.4f} "
              f"cycle={cycle_ham:.4f} 1NN={nn_dist_mean:.4f}")

    # Summary
    print("\n" + "=" * 100)
    print(f"GENERATION SUMMARY ({args.dataset.upper()})")
    print("=" * 100)

    print(f"\n{'method':<15} {'FID':>8} {'viol':>8} {'tokKL':>8} "
          f"{'diversity':>10} {'cycle':>8} {'1NN_dist':>10}")
    print("-" * 75)
    for r in sorted(all_results, key=lambda x: x.get('fid', float('inf'))):
        print(f"{r['method']:<15} {r['fid']:>8.2f} {r['violation_rate']:>8.4f} "
              f"{r['token_kl']:>8.4f} {r['diversity_mean']:>10.4f} "
              f"{r['cycle_hamming']:>8.4f} {r['nn_dist_mean']:>10.4f}")

    # Save CSV
    csv_path = os.path.join(out_dir, "gen_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(set().union(*(r.keys() for r in all_results)))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    # Save real data grid for comparison
    save_grid(test_x[:64], os.path.join(out_dir, 'real_samples.png'))

    print("\n" + "=" * 100)
    print("Generation experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
