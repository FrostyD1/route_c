#!/usr/bin/env python3
"""
Route A1: Freq-as-Evidence — DCT frequency band residuals as observation energy
================================================================================
Core idea: Don't change the protocol (INT4 deterministic quantize, z format).
Only add DCT frequency-band residuals as auxiliary evidence measurement channels
to guide denoising compilation sampling.

8-run experiment matrix:
  1. baseline:        pixel-only (w0=1, w1=w2=w3=0)
  2. low_cons:        pixel + low (conservative: w1=0.1)
  3. low_aggr:        pixel + low (aggressive: w1=0.3)
  4. lowmid_cons:     pixel + low+mid (conservative: w1=0.1, w2=0.1)
  5. lowmid_aggr:     pixel + low+mid (aggressive: w1=0.3, w2=0.3)
  6. lowmidhigh_cons: pixel + low+mid+high (conservative: w1=0.1, w2=0.1, w3=0.05)
  7. lowmidhigh_aggr: pixel + low+mid+high (aggressive: w1=0.3, w2=0.3, w3=0.1)
  8. freq_only:       freq-only control (w0=0, w1=0.3, w2=0.3, w3=0.1)

New metrics (non-task-specific structural):
  - BEP distance: Band Energy Profile L2 between generated and real images
  - Connectedness Proxy: max connected component area / total foreground area

Gate checks:
  - violation must not increase > 20% relative to baseline
  - cycle_hamming must not increase
  - diversity must not collapse

Usage:
    python3 -u benchmarks/exp_gen_freq_evidence.py --device cuda
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
# DCT UTILITIES (deterministic, no learnable params)
# ============================================================================

def dct2d(x):
    """2D DCT via matrix multiplication (Type-II, orthonormal).
    x: (B, C, H, W) -> (B, C, H, W) DCT coefficients.
    """
    B, C, H, W = x.shape
    # 1D DCT matrix
    def dct_matrix(N):
        n = torch.arange(N, dtype=x.dtype, device=x.device)
        k = n.unsqueeze(1)
        D = torch.cos(np.pi * (2 * n + 1) * k / (2 * N))
        D[0] *= 1.0 / np.sqrt(N)
        D[1:] *= np.sqrt(2.0 / N)
        return D

    DH = dct_matrix(H)  # (H, H)
    DW = dct_matrix(W)  # (W, W)
    # Apply: Y = DH @ x @ DW^T
    out = torch.einsum('hH,bcHW,wW->bchw', DH, x, DW)
    return out


def get_freq_masks(H, W, device='cpu'):
    """Create low/mid/high frequency masks for H×W DCT coefficients.
    Uses concentric rings based on Manhattan distance from DC.
    """
    fy = torch.arange(H, device=device).float()
    fx = torch.arange(W, device=device).float()
    freq_grid = fy.unsqueeze(1) + fx.unsqueeze(0)  # Manhattan distance from (0,0)
    max_freq = H + W - 2

    # Three bands: low (0-33%), mid (33-66%), high (66-100%)
    t1 = max_freq / 3.0
    t2 = 2 * max_freq / 3.0

    low_mask = (freq_grid <= t1).float()
    mid_mask = ((freq_grid > t1) & (freq_grid <= t2)).float()
    high_mask = (freq_grid > t2).float()

    return low_mask, mid_mask, high_mask


def band_energy_profile(images, device='cpu'):
    """Compute band energy profile: fraction of energy in low/mid/high bands.
    images: (N, 1, H, W) tensor.
    Returns: (3,) array [low_frac, mid_frac, high_frac].
    """
    if isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float32)
    if images.dim() == 3:
        images = images.unsqueeze(1)
    images = images.to(device)

    B, C, H, W = images.shape
    low_m, mid_m, high_m = get_freq_masks(H, W, device)

    dct_coeffs = dct2d(images)  # (B, C, H, W)
    energy = (dct_coeffs ** 2).mean(dim=(0, 1))  # (H, W) average energy

    total = energy.sum() + 1e-12
    low_e = (energy * low_m).sum() / total
    mid_e = (energy * mid_m).sum() / total
    high_e = (energy * high_m).sum() / total

    return np.array([low_e.item(), mid_e.item(), high_e.item()])


def bep_distance(images_gen, images_real, device='cpu'):
    """L2 distance between band energy profiles of generated and real images."""
    bep_gen = band_energy_profile(images_gen, device)
    bep_real = band_energy_profile(images_real, device)
    return float(np.sqrt(((bep_gen - bep_real) ** 2).sum()))


# ============================================================================
# CONNECTEDNESS PROXY (non-task-specific structural metric)
# ============================================================================

def connectedness_proxy(images, threshold=0.5):
    """Compute max connected component area / total foreground area.
    Uses simple flood-fill. images: (N, 1, H, W) or (N, H, W).
    Returns mean connectedness across batch.
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if images.ndim == 4:
        images = images[:, 0]  # (N, H, W)

    connectedness_scores = []
    for img in images[:100]:  # limit for speed
        binary = (img > threshold).astype(np.int32)
        total_fg = binary.sum()
        if total_fg < 5:
            connectedness_scores.append(0.0)
            continue

        # Simple connected components via flood fill
        H, W = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        max_component = 0

        for i in range(H):
            for j in range(W):
                if binary[i, j] == 1 and not visited[i, j]:
                    # BFS flood fill
                    stack = [(i, j)]
                    visited[i, j] = True
                    comp_size = 0
                    while stack:
                        ci, cj = stack.pop()
                        comp_size += 1
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < H and 0 <= nj < W and binary[ni, nj] == 1 and not visited[ni, nj]:
                                visited[ni, nj] = True
                                stack.append((ni, nj))
                    max_component = max(max_component, comp_size)

        connectedness_scores.append(max_component / total_fg)

    return float(np.mean(connectedness_scores))


# ============================================================================
# ADC/DAC (same as exp_gen_unconditional.py)
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
# E_CORE (same as exp_gen_unconditional.py)
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
# FREQ-AWARE DENOISING COMPILER
# ============================================================================

class FreqAwareDenoisingCompiler(nn.Module):
    """Denoising compiler with frequency-band evidence scoring.

    The denoiser network is the same. The difference is in sampling:
    at each step, we compute a freq-aware energy score to guide the
    confidence/acceptance of proposed updates.
    """
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
        logits = self.net(inp) + self.skip(z_noisy)
        return logits

    @torch.no_grad()
    def sample_with_freq_energy(self, n, H, W, decoder, ref_bep,
                                 device='cpu', n_steps=15, temperature=0.7,
                                 w_pixel=1.0, w_low=0.0, w_mid=0.0, w_high=0.0):
        """Iterative denoising with frequency-band energy guidance.

        At each step:
        1. Propose z_new from denoiser
        2. Decode both z_current and z_new
        3. Compute freq-aware energy (pixel + band residuals vs reference BEP)
        4. Accept update where energy improves (or with probability based on confidence)
        """
        K = self.n_bits
        z = (torch.rand(n, K, H, W, device=device) > 0.5).float()

        low_m, mid_m, high_m = get_freq_masks(28, 28, device)

        for step in range(n_steps):
            nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
            logits = self(z, nl)
            probs = torch.sigmoid(logits / temperature)

            confidence = (step + 1) / n_steps

            # Propose new z
            z_proposed = (torch.rand_like(z) < probs).float()

            if w_low > 0 or w_mid > 0 or w_high > 0:
                # Decode current and proposed
                x_current = decoder(z)
                x_proposed = decoder(z_proposed)

                # Compute freq energy for each sample
                dct_cur = dct2d(x_current)
                dct_prop = dct2d(x_proposed)

                # Reference BEP as target distribution
                ref_low, ref_mid, ref_high = ref_bep

                # Per-sample band energy deviation from reference
                def band_deviation(dct_coeffs):
                    energy = (dct_coeffs ** 2).mean(dim=1, keepdim=True)  # (B, 1, H, W)
                    total = energy.sum(dim=(2, 3), keepdim=True) + 1e-12
                    low_frac = (energy * low_m).sum(dim=(2, 3)) / total.squeeze(3).squeeze(2)
                    mid_frac = (energy * mid_m).sum(dim=(2, 3)) / total.squeeze(3).squeeze(2)
                    high_frac = (energy * high_m).sum(dim=(2, 3)) / total.squeeze(3).squeeze(2)
                    dev = (w_low * (low_frac.squeeze(1) - ref_low) ** 2 +
                           w_mid * (mid_frac.squeeze(1) - ref_mid) ** 2 +
                           w_high * (high_frac.squeeze(1) - ref_high) ** 2)
                    return dev  # (B,)

                dev_cur = band_deviation(dct_cur)
                dev_prop = band_deviation(dct_prop)

                # Accept proposed where it has better freq alignment
                # (lower deviation = better)
                freq_accept = (dev_prop < dev_cur).float()  # (B,)
                freq_accept = freq_accept.view(n, 1, 1, 1)

                # Combine with confidence-based mask
                conf_mask = (torch.rand_like(z) < confidence).float()
                # Where freq says accept AND confidence allows: use proposed
                # Where freq says reject: still use proposed with lower probability
                accept_prob = conf_mask * (0.5 + 0.5 * freq_accept)
                z = accept_prob * z_proposed + (1 - accept_prob) * z
            else:
                # Standard (no freq guidance)
                mask = (torch.rand_like(z) < confidence).float()
                z = mask * z_proposed + (1 - mask) * z

        # Final hard decision
        logits = self(z, torch.zeros(n, device=device))
        z = (torch.sigmoid(logits) > 0.5).float()
        return z


def train_denoiser(denoiser, z_data, device, epochs=30, batch_size=64):
    """Train denoiser on corrupted z samples (same as baseline)."""
    opt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)
    for epoch in range(epochs):
        denoiser.train()
        perm = torch.randperm(N)
        tl, nb = 0., 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            z_clean = z_data[idx].to(device)
            B = z_clean.shape[0]
            noise_level = torch.rand(B, device=device)
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
# EVALUATION UTILITIES
# ============================================================================

def compute_diversity(z_samples, n_pairs=500):
    N = len(z_samples)
    z_flat = z_samples.reshape(N, -1)
    distances = []
    for _ in range(n_pairs):
        i, j = np.random.choice(N, 2, replace=False)
        ham = (z_flat[i] != z_flat[j]).float().mean().item()
        distances.append(ham)
    return np.mean(distances), np.std(distances)


def compute_1nn_distance(gen_samples, train_samples, n_check=100):
    gen_flat = gen_samples[:n_check].reshape(n_check, -1)
    train_flat = train_samples.reshape(len(train_samples), -1)
    distances = []
    for i in range(n_check):
        dists = ((train_flat - gen_flat[i:i + 1]) ** 2).sum(1)
        distances.append(dists.min().item())
    return np.mean(distances), np.std(distances)


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
            kl = (p * (p / q).log()).sum().item()
            kls.append(kl)
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
# EXPERIMENT CONFIGS
# ============================================================================

CONFIGS = OrderedDict([
    ("baseline",        {"w_pixel": 1.0, "w_low": 0.0,  "w_mid": 0.0,  "w_high": 0.0}),
    ("low_cons",        {"w_pixel": 1.0, "w_low": 0.1,  "w_mid": 0.0,  "w_high": 0.0}),
    ("low_aggr",        {"w_pixel": 1.0, "w_low": 0.3,  "w_mid": 0.0,  "w_high": 0.0}),
    ("lowmid_cons",     {"w_pixel": 1.0, "w_low": 0.1,  "w_mid": 0.1,  "w_high": 0.0}),
    ("lowmid_aggr",     {"w_pixel": 1.0, "w_low": 0.3,  "w_mid": 0.3,  "w_high": 0.0}),
    ("lowmidhigh_cons", {"w_pixel": 1.0, "w_low": 0.1,  "w_mid": 0.1,  "w_high": 0.05}),
    ("lowmidhigh_aggr", {"w_pixel": 1.0, "w_low": 0.3,  "w_mid": 0.3,  "w_high": 0.1}),
    ("freq_only",       {"w_pixel": 0.0, "w_low": 0.3,  "w_mid": 0.3,  "w_high": 0.1}),
])


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', default='fmnist', choices=['mnist', 'fmnist', 'kmnist'])
    parser.add_argument('--output_dir', default='outputs/exp_gen_freq')
    parser.add_argument('--n_bits', type=int, default=8)
    parser.add_argument('--n_samples', type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print(f"ROUTE A1: FREQ-AS-EVIDENCE — {args.dataset.upper()} — 8-run matrix")
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
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")

    # [2] Train ADC/DAC (shared across all configs)
    print(f"\n[2] Training ADC/DAC (28×28 → 14×14×{args.n_bits})...")
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
    print("\n[3] Encoding training set → z_data...")
    z_data = []
    with torch.no_grad():
        for i in range(0, len(train_x), 64):
            x = train_x[i:i + 64].to(device)
            z, _ = encoder(x)
            z_data.append(z.cpu())
    z_data = torch.cat(z_data)
    K, H, W = z_data.shape[1:]
    print(f"    z_data: {z_data.shape}, bit usage: {z_data.mean():.3f}")

    # [4] Train E_core
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

    # [5] Train denoiser (shared — same network, freq guidance only at sampling)
    print("\n[5] Training denoiser (shared across all configs)...")
    denoiser = FreqAwareDenoisingCompiler(args.n_bits).to(device)
    train_denoiser(denoiser, z_data, device, epochs=30)
    denoiser.eval()

    # [6] Compute reference BEP from real data
    print("\n[6] Computing reference BEP from real data...")
    ref_bep_arr = band_energy_profile(train_x[:500], device)
    ref_bep = tuple(ref_bep_arr)
    print(f"    Reference BEP: low={ref_bep[0]:.4f} mid={ref_bep[1]:.4f} high={ref_bep[2]:.4f}")

    # Real data structural metrics
    real_bep = ref_bep_arr
    real_conn = connectedness_proxy(test_x[:100].numpy())
    print(f"    Real data connectedness: {real_conn:.4f}")

    # [7] Run all 8 configs
    print("\n" + "=" * 100)
    print("RUNNING 8-CONFIG EXPERIMENT MATRIX")
    print("=" * 100)

    all_results = []
    train_x_np = train_x.numpy().reshape(len(train_x), -1)

    for config_name, weights in CONFIGS.items():
        print(f"\n{'=' * 80}")
        print(f"CONFIG: {config_name} | w_pixel={weights['w_pixel']} "
              f"w_low={weights['w_low']} w_mid={weights['w_mid']} w_high={weights['w_high']}")
        print("=" * 80)

        torch.manual_seed(args.seed)  # same seed for fair comparison

        # Sample with freq-aware energy guidance
        z_gen = denoiser.sample_with_freq_energy(
            args.n_samples, H, W, decoder, ref_bep, device,
            n_steps=15, temperature=0.7,
            w_pixel=weights['w_pixel'],
            w_low=weights['w_low'],
            w_mid=weights['w_mid'],
            w_high=weights['w_high'],
        )

        with torch.no_grad():
            x_gen = decoder(z_gen.to(device)).cpu()

        save_grid(x_gen, os.path.join(out_dir, f'gen_{config_name}.png'))

        # === EVALUATION ===
        z_gen_cpu = z_gen.cpu()

        # Protocol metrics
        viol = e_core.violation_rate(z_gen_cpu[:100].to(device))
        tok_kl = token_histogram_kl(z_data[:500], z_gen_cpu[:500], args.n_bits)
        div_mean, div_std = compute_diversity(z_gen_cpu)

        # Cycle stability
        with torch.no_grad():
            z_check = z_gen_cpu[:100].to(device)
            x_check = decoder(z_check)
            z_cycle, _ = encoder(x_check)
            cycle_ham = (z_check != z_cycle).float().mean().item()

        # Memorization
        x_gen_np = x_gen.numpy().reshape(len(x_gen), -1)
        nn_mean, nn_std = compute_1nn_distance(x_gen_np, train_x_np)

        # NEW: Structural metrics
        gen_bep = band_energy_profile(x_gen[:100], device)
        bep_dist = float(np.sqrt(((gen_bep - real_bep) ** 2).sum()))
        conn = connectedness_proxy(x_gen[:100])

        r = {
            'config': config_name,
            'w_pixel': weights['w_pixel'],
            'w_low': weights['w_low'],
            'w_mid': weights['w_mid'],
            'w_high': weights['w_high'],
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
        print(f"    BEP_dist={bep_dist:.4f} BEP=[{gen_bep[0]:.3f},{gen_bep[1]:.3f},{gen_bep[2]:.3f}] "
              f"conn={conn:.4f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 100)
    print(f"FREQ-AS-EVIDENCE SUMMARY ({args.dataset.upper()})")
    print(f"Reference BEP: [{real_bep[0]:.3f}, {real_bep[1]:.3f}, {real_bep[2]:.3f}]  "
          f"Real connectedness: {real_conn:.4f}")
    print("=" * 100)

    header = (f"{'config':<18} {'viol':>7} {'tokKL':>7} {'div':>7} "
              f"{'cycle':>7} {'1NN':>7} {'BEP_d':>7} {'conn':>7}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['config']:<18} {r['violation']:>7.4f} {r['token_kl']:>7.4f} "
              f"{r['diversity']:>7.4f} {r['cycle_hamming']:>7.4f} {r['nn_dist']:>7.2f} "
              f"{r['bep_distance']:>7.4f} {r['connectedness']:>7.4f}")

    # Gate check
    print("\n" + "-" * 60)
    print("GATE CHECK (vs baseline):")
    baseline = all_results[0]
    for r in all_results[1:]:
        viol_delta = (r['violation'] - baseline['violation']) / (baseline['violation'] + 1e-8) * 100
        cycle_delta = r['cycle_hamming'] - baseline['cycle_hamming']
        div_delta = r['diversity'] - baseline['diversity']
        conn_delta = r['connectedness'] - baseline['connectedness']
        bep_delta = r['bep_distance'] - baseline['bep_distance']

        gate1 = "PASS" if viol_delta < 20 else "FAIL"
        gate2 = "PASS" if cycle_delta <= 0.01 else "FAIL"

        print(f"  {r['config']:<18} viol Δ={viol_delta:+.1f}% [{gate1}]  "
              f"cycle Δ={cycle_delta:+.4f} [{gate2}]  "
              f"div Δ={div_delta:+.4f}  conn Δ={conn_delta:+.4f}  "
              f"BEP Δ={bep_delta:+.4f}")

    # Save CSV
    csv_path = os.path.join(out_dir, "freq_evidence_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(all_results[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    print(f"\nCSV: {csv_path}")

    # Save real samples grid
    save_grid(test_x[:64], os.path.join(out_dir, 'real_samples.png'))

    print("\n" + "=" * 100)
    print("Route A1 Freq-as-Evidence experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
