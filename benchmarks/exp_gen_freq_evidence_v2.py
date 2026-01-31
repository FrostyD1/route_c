#!/usr/bin/env python3
"""
Route A1v2: Freq-Aware Denoiser Training + Patch-Level Guidance
===============================================================
Lessons from A1v1:
  - Whole-image BEP guidance improved connectedness (+28%) but was too coarse
  - Conservative vs aggressive weights gave identical results (binary accept)
  - Need: (1) freq-aware denoiser TRAINING, (2) patch-level guidance

Key changes:
  1. Denoiser training loss = pixel_BCE + λ_freq * freq_band_BCE
     (trains denoiser to preserve frequency structure during denoising)
  2. Patch-level DCT: 4×4 overlapping patches for local freq coherence
  3. Multi-scale sampling: low-freq corrections first, then high-freq refinement

Experiment matrix (6 runs):
  1. baseline:     pixel-only denoiser, no freq guidance
  2. freq_train_01: denoiser trained with λ_freq=0.1
  3. freq_train_03: denoiser trained with λ_freq=0.3
  4. freq_train_05: denoiser trained with λ_freq=0.5
  5. freq_train_03_ms: λ_freq=0.3 + multi-scale sampling schedule
  6. freq_train_05_ms: λ_freq=0.5 + multi-scale sampling schedule

Usage:
    python3 -u benchmarks/exp_gen_freq_evidence_v2.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, csv, argparse
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# DCT UTILITIES
# ============================================================================

def dct2d(x):
    """2D DCT (Type-II, orthonormal). x: (B, C, H, W)."""
    B, C, H, W = x.shape
    def dct_matrix(N):
        n = torch.arange(N, dtype=x.dtype, device=x.device)
        k = n.unsqueeze(1)
        D = torch.cos(np.pi * (2 * n + 1) * k / (2 * N))
        D[0] *= 1.0 / np.sqrt(N)
        D[1:] *= np.sqrt(2.0 / N)
        return D
    DH = dct_matrix(H)
    DW = dct_matrix(W)
    return torch.einsum('hH,bcHW,wW->bchw', DH, x, DW)


def get_freq_masks(H, W, device='cpu'):
    """Low/mid/high frequency masks."""
    fy = torch.arange(H, device=device).float()
    fx = torch.arange(W, device=device).float()
    freq_grid = fy.unsqueeze(1) + fx.unsqueeze(0)
    max_freq = H + W - 2
    t1 = max_freq / 3.0
    t2 = 2 * max_freq / 3.0
    low = (freq_grid <= t1).float()
    mid = ((freq_grid > t1) & (freq_grid <= t2)).float()
    high = (freq_grid > t2).float()
    return low, mid, high


def band_energy_profile(images, device='cpu'):
    """BEP: [low_frac, mid_frac, high_frac]."""
    if isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float32)
    if images.dim() == 3:
        images = images.unsqueeze(1)
    images = images.to(device)
    B, C, H, W = images.shape
    low_m, mid_m, high_m = get_freq_masks(H, W, device)
    dct_c = dct2d(images)
    energy = (dct_c ** 2).mean(dim=(0, 1))
    total = energy.sum() + 1e-12
    return np.array([(energy * m).sum().item() / total.item() for m in [low_m, mid_m, high_m]])


def bep_distance(images_gen, images_real, device='cpu'):
    bep_gen = band_energy_profile(images_gen, device)
    bep_real = band_energy_profile(images_real, device)
    return float(np.sqrt(((bep_gen - bep_real) ** 2).sum()))


def freq_band_loss(x_pred, x_target):
    """Frequency-band aware reconstruction loss.
    Computes BCE in DCT domain per frequency band with band-specific weighting.
    Higher weight on low frequencies to encourage global coherence.
    """
    # DCT of both
    dct_pred = dct2d(x_pred)
    dct_target = dct2d(x_target)

    B, C, H, W = x_pred.shape
    low_m, mid_m, high_m = get_freq_masks(H, W, x_pred.device)

    # MSE per band (DCT coefficients are real-valued, use MSE)
    loss_low = ((dct_pred - dct_target) ** 2 * low_m).sum() / (low_m.sum() * B * C + 1e-8)
    loss_mid = ((dct_pred - dct_target) ** 2 * mid_m).sum() / (mid_m.sum() * B * C + 1e-8)
    loss_high = ((dct_pred - dct_target) ** 2 * high_m).sum() / (high_m.sum() * B * C + 1e-8)

    # Weight: prioritize low-freq coherence
    return 3.0 * loss_low + 1.0 * loss_mid + 0.3 * loss_high


# ============================================================================
# CONNECTEDNESS PROXY
# ============================================================================

def connectedness_proxy(images, threshold=0.5):
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if images.ndim == 4:
        images = images[:, 0]
    scores = []
    for img in images[:100]:
        binary = (img > threshold).astype(np.int32)
        total_fg = binary.sum()
        if total_fg < 5:
            scores.append(0.0)
            continue
        H, W = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        max_comp = 0
        for i in range(H):
            for j in range(W):
                if binary[i, j] == 1 and not visited[i, j]:
                    stack = [(i, j)]
                    visited[i, j] = True
                    sz = 0
                    while stack:
                        ci, cj = stack.pop()
                        sz += 1
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < H and 0 <= nj < W and binary[ni, nj] == 1 and not visited[ni, nj]:
                                visited[ni, nj] = True
                                stack.append((ni, nj))
                    max_comp = max(max_comp, sz)
        scores.append(max_comp / total_fg)
    return float(np.mean(scores))


# ============================================================================
# ADC/DAC
# ============================================================================

class GumbelSigmoid(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    def forward(self, logits):
        if self.training:
            u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            noisy = (logits - torch.log(-torch.log(u))) / self.temperature
        else:
            noisy = logits / self.temperature
        soft = torch.sigmoid(noisy)
        hard = (soft > 0.5).float()
        return hard - soft.detach() + soft
    def set_temperature(self, tau): self.temperature = tau


class Encoder14(nn.Module):
    def __init__(self, in_ch=1, n_bits=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(),
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
    def __init__(self, out_ch=1, n_bits=8):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, out_ch, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z):
        return self.deconv(z)


# ============================================================================
# E_CORE
# ============================================================================

class LocalEnergyCore(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        context_size = 9 * n_bits - 1
        self.predictor = nn.Sequential(
            nn.Linear(context_size, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def get_context(self, z, bit_idx, i, j):
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
        B, K, H, W = z.shape
        violations = []
        for _ in range(min(50, H * W * K)):
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
# FREQ-AWARE DENOISER
# ============================================================================

class FreqDenoisingCompiler(nn.Module):
    """Denoiser with optional freq-aware training loss."""
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.net = nn.Sequential(
            nn.Conv2d(n_bits + 1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, n_bits, 3, padding=1),
        )
        self.skip = nn.Conv2d(n_bits, n_bits, 1)

    def forward(self, z_noisy, noise_level):
        B = z_noisy.shape[0]
        nl = noise_level.view(B, 1, 1, 1).expand(-1, 1, z_noisy.shape[2], z_noisy.shape[3])
        inp = torch.cat([z_noisy, nl], dim=1)
        return self.net(inp) + self.skip(z_noisy)

    @torch.no_grad()
    def sample(self, n, H, W, device='cpu', n_steps=15, temperature=0.7):
        """Standard denoising sampling."""
        K = self.n_bits
        z = (torch.rand(n, K, H, W, device=device) > 0.5).float()
        for step in range(n_steps):
            nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
            logits = self(z, nl)
            probs = torch.sigmoid(logits / temperature)
            confidence = (step + 1) / n_steps
            mask = (torch.rand_like(z) < confidence).float()
            z_new = (torch.rand_like(z) < probs).float()
            z = mask * z_new + (1 - mask) * z
        logits = self(z, torch.zeros(n, device=device))
        return (torch.sigmoid(logits) > 0.5).float()

    @torch.no_grad()
    def sample_multiscale(self, n, H, W, decoder, device='cpu',
                          n_steps=15, temperature=0.7):
        """Multi-scale sampling: low-freq corrections first, high-freq later.
        Phase 1 (steps 0-7): aggressive updates, favoring global structure
        Phase 2 (steps 8-14): conservative updates, refining details
        """
        K = self.n_bits
        z = (torch.rand(n, K, H, W, device=device) > 0.5).float()

        for step in range(n_steps):
            nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
            logits = self(z, nl)

            # Phase-dependent temperature: lower temp early (more deterministic global)
            # higher temp later (allow variation in details)
            progress = step / n_steps
            if progress < 0.5:
                # Phase 1: low temperature for global coherence
                step_temp = temperature * 0.6
                confidence = 0.3 + 0.4 * (progress / 0.5)  # 0.3 → 0.7
            else:
                # Phase 2: normal temperature for detail refinement
                step_temp = temperature * 1.2
                confidence = 0.7 + 0.3 * ((progress - 0.5) / 0.5)  # 0.7 → 1.0

            probs = torch.sigmoid(logits / step_temp)
            mask = (torch.rand_like(z) < confidence).float()
            z_new = (torch.rand_like(z) < probs).float()
            z = mask * z_new + (1 - mask) * z

        logits = self(z, torch.zeros(n, device=device))
        return (torch.sigmoid(logits) > 0.5).float()


def train_freq_denoiser(denoiser, z_data, decoder, device,
                         lam_freq=0.0, epochs=30, batch_size=64):
    """Train denoiser with optional freq-band loss.

    Loss = BCE(z_logits, z_clean) + lam_freq * freq_band_loss(decoded_pred, decoded_clean)

    The freq loss operates in pixel space via the decoder, so it teaches the
    denoiser to produce z values whose decoded images have correct frequency structure.
    """
    opt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)

    for epoch in range(epochs):
        denoiser.train()
        perm = torch.randperm(N)
        tl, fl, nb = 0., 0., 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            z_clean = z_data[idx].to(device)
            B = z_clean.shape[0]

            noise_level = torch.rand(B, device=device)
            flip_mask = (torch.rand_like(z_clean) < noise_level.view(B, 1, 1, 1)).float()
            z_noisy = z_clean * (1 - flip_mask) + (1 - z_clean) * flip_mask

            opt.zero_grad()
            logits = denoiser(z_noisy, noise_level)
            loss_bce = F.binary_cross_entropy_with_logits(logits, z_clean)

            loss_freq = torch.tensor(0.0, device=device)
            if lam_freq > 0:
                # Decode predicted z (using STE through hard threshold)
                z_pred_soft = torch.sigmoid(logits)
                z_pred_hard = (z_pred_soft > 0.5).float()
                z_pred = z_pred_hard - z_pred_soft.detach() + z_pred_soft  # STE

                with torch.no_grad():
                    x_clean = decoder(z_clean)
                x_pred = decoder(z_pred)
                loss_freq = freq_band_loss(x_pred, x_clean)

            loss = loss_bce + lam_freq * loss_freq
            loss.backward()
            opt.step()
            tl += loss_bce.item()
            fl += loss_freq.item()
            nb += 1

        if (epoch + 1) % 10 == 0:
            msg = f"    epoch {epoch+1}/{epochs}: BCE={tl/nb:.4f}"
            if lam_freq > 0:
                msg += f" freq={fl/nb:.4f}"
            print(msg)


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def compute_diversity(z_samples, n_pairs=500):
    N = len(z_samples)
    z_flat = z_samples.reshape(N, -1)
    dists = []
    for _ in range(n_pairs):
        i, j = np.random.choice(N, 2, replace=False)
        dists.append((z_flat[i] != z_flat[j]).float().mean().item())
    return np.mean(dists), np.std(dists)


def compute_1nn_distance(gen_samples, train_samples, n_check=100):
    gen_flat = gen_samples[:n_check].reshape(n_check, -1)
    train_flat = train_samples.reshape(len(train_samples), -1)
    dists = []
    for i in range(n_check):
        d = ((train_flat - gen_flat[i:i + 1]) ** 2).sum(1)
        dists.append(d.min().item())
    return np.mean(dists), np.std(dists)


def token_histogram_kl(z_real, z_gen, n_bits=8):
    N_r, K, H, W = z_real.shape
    N_g = z_gen.shape[0]
    n_tokens = 2 ** K
    kls = []
    for i in range(H):
        for j in range(W):
            idx_r = torch.zeros(N_r, dtype=torch.long)
            idx_g = torch.zeros(N_g, dtype=torch.long)
            for b in range(K):
                idx_r += (z_real[:, b, i, j].long() << b)
                idx_g += (z_gen[:, b, i, j].long() << b)
            p = torch.bincount(idx_r, minlength=n_tokens).float() + 1
            q = torch.bincount(idx_g, minlength=n_tokens).float() + 1
            p = p / p.sum(); q = q / q.sum()
            kls.append((p * (p / q).log()).sum().item())
    return np.mean(kls)


def save_grid(images, path, nrow=8):
    try:
        from torchvision.utils import save_image
        if isinstance(images, np.ndarray):
            images = torch.tensor(images)
        if images.dim() == 3:
            images = images.unsqueeze(1)
        save_image(images[:64], path, nrow=nrow, normalize=False)
        print(f"    Grid saved: {path}")
    except Exception as e:
        print(f"    Grid save failed: {e}")


# ============================================================================
# CONFIGS
# ============================================================================

CONFIGS = OrderedDict([
    ("baseline",         {"lam_freq": 0.0, "multiscale": False}),
    ("freq_train_01",    {"lam_freq": 0.1, "multiscale": False}),
    ("freq_train_03",    {"lam_freq": 0.3, "multiscale": False}),
    ("freq_train_05",    {"lam_freq": 0.5, "multiscale": False}),
    ("freq_train_03_ms", {"lam_freq": 0.3, "multiscale": True}),
    ("freq_train_05_ms", {"lam_freq": 0.5, "multiscale": True}),
])


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', default='fmnist', choices=['mnist', 'fmnist', 'kmnist'])
    parser.add_argument('--output_dir', default='outputs/exp_gen_freq_v2')
    parser.add_argument('--n_bits', type=int, default=8)
    parser.add_argument('--n_samples', type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print(f"ROUTE A1v2: FREQ-AWARE DENOISER — {args.dataset.upper()} — 6-run matrix")
    print("=" * 100)

    # [1] Load data
    print("\n[1] Loading dataset...")
    from torchvision import datasets, transforms
    ds_class = {'mnist': datasets.MNIST, 'fmnist': datasets.FashionMNIST,
                'kmnist': datasets.KMNIST}[args.dataset]
    train_ds = ds_class('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = ds_class('./data', train=False, download=True, transform=transforms.ToTensor())

    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(len(train_ds), 5000, replace=False)
    test_idx = rng.choice(len(test_ds), 1000, replace=False)
    train_x = torch.stack([train_ds[i][0] for i in train_idx])
    test_x = torch.stack([test_ds[i][0] for i in test_idx])

    # [2] Train shared ADC/DAC
    print(f"\n[2] Training ADC/DAC (28→14×14×{args.n_bits})...")
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
            idx = perm[i:i + 64]
            x = train_x[idx].to(device)
            opt.zero_grad()
            z, _ = encoder(x)
            x_hat = decoder(z)
            loss = F.binary_cross_entropy(x_hat, x)
            loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/25: BCE={tl/nb:.4f}")

    encoder.eval(); decoder.eval()

    # [3] Encode training set
    print("\n[3] Encoding training set...")
    z_data = []
    with torch.no_grad():
        for i in range(0, len(train_x), 64):
            x = train_x[i:i + 64].to(device)
            z, _ = encoder(x)
            z_data.append(z.cpu())
    z_data = torch.cat(z_data)
    K, H, W = z_data.shape[1:]
    print(f"    z_data: {z_data.shape}, bit usage: {z_data.mean():.3f}")

    # [4] Train E_core (shared)
    print("\n[4] Training E_core...")
    e_core = LocalEnergyCore(args.n_bits).to(device)
    e_opt = torch.optim.Adam(e_core.parameters(), lr=1e-3)
    for epoch in range(10):
        e_core.train()
        perm = torch.randperm(len(z_data))
        for i in range(0, len(z_data), 64):
            idx = perm[i:i + 64]
            z = z_data[idx].to(device)
            e_opt.zero_grad()
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
    e_core.eval()

    # [5] Reference metrics
    print("\n[5] Reference structural metrics...")
    real_bep = band_energy_profile(train_x[:500], device)
    real_conn = connectedness_proxy(test_x[:100].numpy())
    print(f"    Real BEP: [{real_bep[0]:.4f}, {real_bep[1]:.4f}, {real_bep[2]:.4f}]")
    print(f"    Real connectedness: {real_conn:.4f}")

    # [6] Run configs
    print("\n" + "=" * 100)
    print("RUNNING 6-CONFIG EXPERIMENT MATRIX")
    print("=" * 100)

    all_results = []
    train_x_np = train_x.numpy().reshape(len(train_x), -1)

    for config_name, cfg in CONFIGS.items():
        print(f"\n{'=' * 80}")
        print(f"CONFIG: {config_name} | lam_freq={cfg['lam_freq']} multiscale={cfg['multiscale']}")
        print("=" * 80)

        torch.manual_seed(args.seed + hash(config_name) % 10000)

        # Train fresh denoiser for each config
        denoiser = FreqDenoisingCompiler(args.n_bits).to(device)
        print(f"  Training denoiser (lam_freq={cfg['lam_freq']})...")
        train_freq_denoiser(denoiser, z_data, decoder, device,
                            lam_freq=cfg['lam_freq'], epochs=30)
        denoiser.eval()

        # Sample
        print(f"  Sampling {args.n_samples} images...")
        torch.manual_seed(args.seed)  # same seed for sampling
        if cfg['multiscale']:
            z_gen = denoiser.sample_multiscale(
                args.n_samples, H, W, decoder, device, n_steps=15, temperature=0.7)
        else:
            z_gen = denoiser.sample(
                args.n_samples, H, W, device, n_steps=15, temperature=0.7)

        with torch.no_grad():
            x_gen = decoder(z_gen.to(device)).cpu()

        save_grid(x_gen, os.path.join(out_dir, f'gen_{config_name}.png'))

        # Evaluate
        z_gen_cpu = z_gen.cpu()
        viol = e_core.violation_rate(z_gen_cpu[:100].to(device))
        tok_kl = token_histogram_kl(z_data[:500], z_gen_cpu[:500], args.n_bits)
        div_mean, div_std = compute_diversity(z_gen_cpu)

        with torch.no_grad():
            z_check = z_gen_cpu[:100].to(device)
            x_check = decoder(z_check)
            z_cyc, _ = encoder(x_check)
            cycle_ham = (z_check != z_cyc).float().mean().item()

        x_gen_np = x_gen.numpy().reshape(len(x_gen), -1)
        nn_mean, nn_std = compute_1nn_distance(x_gen_np, train_x_np)

        gen_bep = band_energy_profile(x_gen[:100], device)
        bep_dist = float(np.sqrt(((gen_bep - real_bep) ** 2).sum()))
        conn = connectedness_proxy(x_gen[:100])

        r = {
            'config': config_name,
            'lam_freq': cfg['lam_freq'],
            'multiscale': cfg['multiscale'],
            'violation': viol,
            'token_kl': tok_kl,
            'diversity': div_mean,
            'cycle_hamming': cycle_ham,
            'nn_dist': nn_mean,
            'bep_distance': bep_dist,
            'bep_low': gen_bep[0],
            'bep_mid': gen_bep[1],
            'bep_high': gen_bep[2],
            'connectedness': conn,
        }
        all_results.append(r)

        print(f"    viol={viol:.4f} tokKL={tok_kl:.4f} div={div_mean:.4f} "
              f"cycle={cycle_ham:.4f} 1NN={nn_mean:.2f}")
        print(f"    BEP_d={bep_dist:.4f} BEP=[{gen_bep[0]:.3f},{gen_bep[1]:.3f},{gen_bep[2]:.3f}] "
              f"conn={conn:.4f}")

    # Summary
    print("\n" + "=" * 100)
    print(f"FREQ-AWARE DENOISER SUMMARY ({args.dataset.upper()})")
    print(f"Real BEP: [{real_bep[0]:.3f},{real_bep[1]:.3f},{real_bep[2]:.3f}]  "
          f"Real conn: {real_conn:.4f}")
    print("=" * 100)

    header = (f"{'config':<20} {'viol':>7} {'tokKL':>7} {'div':>7} "
              f"{'cycle':>7} {'1NN':>7} {'BEP_d':>7} {'conn':>7}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['config']:<20} {r['violation']:>7.4f} {r['token_kl']:>7.4f} "
              f"{r['diversity']:>7.4f} {r['cycle_hamming']:>7.4f} {r['nn_dist']:>7.2f} "
              f"{r['bep_distance']:>7.4f} {r['connectedness']:>7.4f}")

    # Gate check
    print("\n" + "-" * 60)
    print("GATE CHECK (vs baseline):")
    bl = all_results[0]
    for r in all_results[1:]:
        viol_d = (r['violation'] - bl['violation']) / (bl['violation'] + 1e-8) * 100
        cyc_d = r['cycle_hamming'] - bl['cycle_hamming']
        div_d = r['diversity'] - bl['diversity']
        conn_d = r['connectedness'] - bl['connectedness']
        bep_d = r['bep_distance'] - bl['bep_distance']
        g1 = "PASS" if viol_d < 20 else "FAIL"
        g2 = "PASS" if cyc_d <= 0.01 else "FAIL"
        print(f"  {r['config']:<20} viol Δ={viol_d:+.1f}% [{g1}]  "
              f"cycle Δ={cyc_d:+.4f} [{g2}]  div Δ={div_d:+.4f}  "
              f"conn Δ={conn_d:+.4f}  BEP Δ={bep_d:+.4f}")

    # CSV
    csv_path = os.path.join(out_dir, "freq_v2_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(all_results[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    print(f"\nCSV: {csv_path}")

    save_grid(test_x[:64], os.path.join(out_dir, 'real_samples.png'))

    print("\n" + "=" * 100)
    print("Route A1v2 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
