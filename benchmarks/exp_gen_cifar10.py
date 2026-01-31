#!/usr/bin/env python3
"""
CIFAR-10 Unconditional Generation via Discrete Core
====================================================
Extends generation to 32×32×3 RGB images.

Architecture:
  Encoder: 32×32×3 → 16×16×8 binary (stride-2 conv)
  Decoder: 16×16×8 → 32×32×3 (transposed conv)

Methods:
  1. Bernoulli prior
  2. Autoregressive (masked-conv)
  3. Denoising Compilation (baseline)
  4. Denoising Compilation + freq training (λ=0.3)
  5. Denoising Compilation + freq training (λ=0.3) + multiscale sampling

Token categorical skipped (2^8=256 tokens × 16×16=256 positions — manageable
but less interesting than denoising for CIFAR-10).

4GB GPU constraint: 3000 train, 500 test, batch_size=32.

Usage:
    python3 -u benchmarks/exp_gen_cifar10.py --device cuda
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
# DCT + BEP + Connectedness (reused from freq_evidence_v2)
# ============================================================================

def dct2d(x):
    B, C, H, W = x.shape
    def dct_matrix(N):
        n = torch.arange(N, dtype=x.dtype, device=x.device)
        k = n.unsqueeze(1)
        D = torch.cos(np.pi * (2 * n + 1) * k / (2 * N))
        D[0] *= 1.0 / np.sqrt(N)
        D[1:] *= np.sqrt(2.0 / N)
        return D
    DH = dct_matrix(H); DW = dct_matrix(W)
    return torch.einsum('hH,bcHW,wW->bchw', DH, x, DW)


def get_freq_masks(H, W, device='cpu'):
    fy = torch.arange(H, device=device).float()
    fx = torch.arange(W, device=device).float()
    freq_grid = fy.unsqueeze(1) + fx.unsqueeze(0)
    max_freq = H + W - 2
    t1, t2 = max_freq / 3.0, 2 * max_freq / 3.0
    return ((freq_grid <= t1).float(),
            ((freq_grid > t1) & (freq_grid <= t2)).float(),
            (freq_grid > t2).float())


def band_energy_profile(images, device='cpu'):
    if isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float32)
    if images.dim() == 3:
        images = images.unsqueeze(1)
    images = images.to(device)
    B, C, H, W = images.shape
    masks = get_freq_masks(H, W, device)
    dct_c = dct2d(images)
    energy = (dct_c ** 2).mean(dim=(0, 1))
    total = energy.sum() + 1e-12
    return np.array([(energy * m).sum().item() / total.item() for m in masks])


def freq_band_loss(x_pred, x_target):
    dct_p = dct2d(x_pred); dct_t = dct2d(x_target)
    B, C, H, W = x_pred.shape
    low_m, mid_m, high_m = get_freq_masks(H, W, x_pred.device)
    loss_low = ((dct_p - dct_t)**2 * low_m).sum() / (low_m.sum() * B * C + 1e-8)
    loss_mid = ((dct_p - dct_t)**2 * mid_m).sum() / (mid_m.sum() * B * C + 1e-8)
    loss_high = ((dct_p - dct_t)**2 * high_m).sum() / (high_m.sum() * B * C + 1e-8)
    return 3.0 * loss_low + 1.0 * loss_mid + 0.3 * loss_high


def connectedness_proxy(images, threshold=0.3):
    """For RGB: convert to grayscale first, use lower threshold."""
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if images.ndim == 4 and images.shape[1] == 3:
        # RGB → grayscale
        images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
    elif images.ndim == 4:
        images = images[:, 0]
    scores = []
    for img in images[:100]:
        binary = (img > threshold).astype(np.int32)
        total_fg = binary.sum()
        if total_fg < 5:
            scores.append(0.0); continue
        H, W = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        max_comp = 0
        for i in range(H):
            for j in range(W):
                if binary[i, j] == 1 and not visited[i, j]:
                    stack = [(i, j)]; visited[i, j] = True; sz = 0
                    while stack:
                        ci, cj = stack.pop(); sz += 1
                        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ni, nj = ci+di, cj+dj
                            if 0<=ni<H and 0<=nj<W and binary[ni,nj]==1 and not visited[ni,nj]:
                                visited[ni,nj] = True; stack.append((ni,nj))
                    max_comp = max(max_comp, sz)
        scores.append(max_comp / total_fg)
    return float(np.mean(scores))


# ============================================================================
# ADC/DAC for CIFAR-10: 32×32×3 → 16×16×8 binary
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


class CifarEncoder(nn.Module):
    """32×32×3 → 16×16×8 binary with residual blocks."""
    def __init__(self, n_bits=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),  # 16×16
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.res1 = self._res_block(64, 64)
        self.res2 = self._res_block(64, 64)
        self.head = nn.Conv2d(64, n_bits, 3, padding=1)
        self.quantizer = GumbelSigmoid()

    def _res_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        h = self.stem(x)
        h = F.relu(h + self.res1(h))
        h = F.relu(h + self.res2(h))
        logits = self.head(h)
        return self.quantizer(logits), logits

    def set_temperature(self, tau):
        self.quantizer.set_temperature(tau)


class CifarDecoder(nn.Module):
    """16×16×8 → 32×32×3 with residual blocks."""
    def __init__(self, n_bits=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ConvTranspose2d(n_bits, 64, 4, stride=2, padding=1),  # 32×32
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.res1 = self._res_block(64, 64)
        self.res2 = self._res_block(64, 64)
        self.head = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def _res_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, z):
        h = self.stem(z)
        h = F.relu(h + self.res1(h))
        h = F.relu(h + self.res2(h))
        return self.head(h)


# ============================================================================
# E_CORE for 16×16 z
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
                ni, nj = (i+di) % H, (j+dj) % W
                for b in range(K):
                    if di==0 and dj==0 and b==bit_idx: continue
                    contexts.append(z[:, b, ni, nj])
        return torch.stack(contexts, dim=1)

    def violation_rate(self, z):
        B, K, H, W = z.shape
        violations = []
        for _ in range(min(50, H*W*K)):
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
# GENERATION METHODS
# ============================================================================

class BernoulliPrior:
    def __init__(self): self.probs = None
    def fit(self, z_data): self.probs = z_data.mean(dim=0)
    def sample(self, n, device='cpu'):
        probs = self.probs.unsqueeze(0).expand(n, -1, -1, -1).to(device)
        return (torch.rand_like(probs) < probs).float()


class AutoregressivePrior(nn.Module):
    def __init__(self, n_bits=8, hidden=64):
        super().__init__()
        self.n_bits = n_bits
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, n_bits, 3, padding=1),
        )
    def forward(self, z_bits):
        B, K, H, W = z_bits.shape
        total_loss = 0.
        for b in range(K):
            z_masked = z_bits.clone()
            z_masked[:, b:] = 0
            logits = self.net(z_masked)
            loss = F.binary_cross_entropy_with_logits(logits[:, b], z_bits[:, b])
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


class DenoisingCompiler(nn.Module):
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
    def sample_multiscale(self, n, H, W, device='cpu', n_steps=15, temperature=0.7):
        K = self.n_bits
        z = (torch.rand(n, K, H, W, device=device) > 0.5).float()
        for step in range(n_steps):
            nl = torch.tensor([1.0 - step / n_steps], device=device).expand(n)
            logits = self(z, nl)
            progress = step / n_steps
            if progress < 0.5:
                step_temp = temperature * 0.6
                confidence = 0.3 + 0.4 * (progress / 0.5)
            else:
                step_temp = temperature * 1.2
                confidence = 0.7 + 0.3 * ((progress - 0.5) / 0.5)
            probs = torch.sigmoid(logits / step_temp)
            mask = (torch.rand_like(z) < confidence).float()
            z_new = (torch.rand_like(z) < probs).float()
            z = mask * z_new + (1 - mask) * z
        logits = self(z, torch.zeros(n, device=device))
        return (torch.sigmoid(logits) > 0.5).float()


def train_denoiser(denoiser, z_data, decoder, device,
                    lam_freq=0.0, epochs=30, batch_size=32):
    opt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)
    for epoch in range(epochs):
        denoiser.train()
        perm = torch.randperm(N)
        tl, fl, nb = 0., 0., 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            z_clean = z_data[idx].to(device)
            B = z_clean.shape[0]
            noise_level = torch.rand(B, device=device)
            flip_mask = (torch.rand_like(z_clean) < noise_level.view(B,1,1,1)).float()
            z_noisy = z_clean * (1 - flip_mask) + (1 - z_clean) * flip_mask
            opt.zero_grad()
            logits = denoiser(z_noisy, noise_level)
            loss_bce = F.binary_cross_entropy_with_logits(logits, z_clean)
            loss_freq = torch.tensor(0.0, device=device)
            if lam_freq > 0:
                z_pred_soft = torch.sigmoid(logits)
                z_pred_hard = (z_pred_soft > 0.5).float()
                z_pred = z_pred_hard - z_pred_soft.detach() + z_pred_soft
                with torch.no_grad():
                    x_clean = decoder(z_clean)
                x_pred = decoder(z_pred)
                loss_freq = freq_band_loss(x_pred, x_clean)
            loss = loss_bce + lam_freq * loss_freq
            loss.backward(); opt.step()
            tl += loss_bce.item(); fl += loss_freq.item(); nb += 1
        if (epoch+1) % 10 == 0:
            msg = f"    epoch {epoch+1}/{epochs}: BCE={tl/nb:.4f}"
            if lam_freq > 0: msg += f" freq={fl/nb:.4f}"
            print(msg)


# ============================================================================
# EVALUATION
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
        d = ((train_flat - gen_flat[i:i+1])**2).sum(1)
        dists.append(d.min().item())
    return np.mean(dists), np.std(dists)


def token_histogram_kl(z_real, z_gen, n_bits=8):
    N_r, K, H, W = z_real.shape
    N_g = z_gen.shape[0]
    n_tokens = 2 ** K
    kls = []
    # Sample positions to avoid O(H*W) with 16×16
    positions = [(i, j) for i in range(H) for j in range(W)]
    np.random.shuffle(positions)
    for i, j in positions[:50]:  # sample 50 positions
        idx_r = torch.zeros(N_r, dtype=torch.long)
        idx_g = torch.zeros(N_g, dtype=torch.long)
        for b in range(K):
            idx_r += (z_real[:, b, i, j].long() << b)
            idx_g += (z_gen[:, b, i, j].long() << b)
        p = torch.bincount(idx_r, minlength=n_tokens).float() + 1
        q = torch.bincount(idx_g, minlength=n_tokens).float() + 1
        p = p / p.sum(); q = q / q.sum()
        kls.append((p * (p/q).log()).sum().item())
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
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_gen/cifar10')
    parser.add_argument('--n_bits', type=int, default=8)
    parser.add_argument('--n_samples', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print(f"CIFAR-10 UNCONDITIONAL GENERATION (32×32×3 → 16×16×{args.n_bits} discrete core)")
    print("=" * 100)

    # [1] Load CIFAR-10
    print("\n[1] Loading CIFAR-10...")
    from torchvision import datasets, transforms
    train_ds = datasets.CIFAR10('./data', train=True, download=True,
                                 transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10('./data', train=False, download=True,
                                transform=transforms.ToTensor())

    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(len(train_ds), 3000, replace=False)
    test_idx = rng.choice(len(test_ds), 500, replace=False)
    train_x = torch.stack([train_ds[i][0] for i in train_idx])
    test_x = torch.stack([test_ds[i][0] for i in test_idx])
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")

    # [2] Train ADC/DAC
    print(f"\n[2] Training ADC/DAC (32×32×3 → 16×16×{args.n_bits})...")
    encoder = CifarEncoder(n_bits=args.n_bits).to(device)
    decoder = CifarDecoder(n_bits=args.n_bits).to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    for epoch in range(40):
        encoder.train(); decoder.train()
        tau = 1.0 + (0.3 - 1.0) * epoch / 39
        encoder.set_temperature(tau)
        perm = torch.randperm(len(train_x))
        tl, nb = 0., 0
        for i in range(0, len(train_x), 32):
            idx = perm[i:i+32]
            x = train_x[idx].to(device)
            opt.zero_grad()
            z, _ = encoder(x)
            x_hat = decoder(z)
            loss = F.mse_loss(x_hat, x) + 0.5 * F.binary_cross_entropy(x_hat, x)
            loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        if (epoch+1) % 10 == 0:
            print(f"    epoch {epoch+1}/40: loss={tl/nb:.4f}")

    encoder.eval(); decoder.eval()

    # Oracle recon
    with torch.no_grad():
        test_batch = test_x[:64].to(device)
        z_oracle, _ = encoder(test_batch)
        x_recon = decoder(z_oracle)
        recon_mse = F.mse_loss(x_recon, test_batch).item()
    print(f"    Oracle MSE: {recon_mse:.4f}")
    save_grid(x_recon.cpu(), os.path.join(args.output_dir, 'oracle_recon.png'))
    save_grid(test_x[:64], os.path.join(args.output_dir, 'real_samples.png'))

    # [3] Encode training set
    print("\n[3] Encoding training set → z_data...")
    z_data = []
    with torch.no_grad():
        for i in range(0, len(train_x), 32):
            x = train_x[i:i+32].to(device)
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
            idx = perm[i:i+64]
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
            (total_loss / 20).backward(); e_opt.step()
    e_core.eval()

    # Reference metrics
    print("\n[5] Reference structural metrics...")
    real_bep = band_energy_profile(train_x[:200], device)
    real_conn = connectedness_proxy(test_x[:100].numpy())
    print(f"    Real BEP: [{real_bep[0]:.4f}, {real_bep[1]:.4f}, {real_bep[2]:.4f}]")
    print(f"    Real connectedness: {real_conn:.4f}")

    # ================================================================
    # GENERATION METHODS
    # ================================================================
    all_results = []
    train_x_np = train_x.numpy().reshape(len(train_x), -1)

    # --- Bernoulli ---
    print("\n" + "=" * 80)
    print("BERNOULLI PRIOR")
    print("=" * 80)
    bern = BernoulliPrior()
    bern.fit(z_data)
    z_bern = bern.sample(args.n_samples, device)
    with torch.no_grad():
        x_bern = decoder(z_bern.to(device)).cpu()
    save_grid(x_bern, os.path.join(args.output_dir, 'gen_bernoulli.png'))

    # --- AR ---
    print("\n" + "=" * 80)
    print("AUTOREGRESSIVE")
    print("=" * 80)
    ar = AutoregressivePrior(args.n_bits, hidden=64).to(device)
    ar_opt = torch.optim.Adam(ar.parameters(), lr=1e-3)
    for epoch in range(15):
        ar.train()
        perm = torch.randperm(len(z_data))
        tl, nb = 0., 0
        for i in range(0, min(len(z_data), 2000), 32):
            idx = perm[i:i+32]
            z = z_data[idx].to(device)
            ar_opt.zero_grad()
            loss = ar(z)
            loss.backward(); ar_opt.step()
            tl += loss.item(); nb += 1
        if (epoch+1) % 5 == 0:
            print(f"    AR epoch {epoch+1}/15: BCE={tl/nb:.4f}")
    ar.eval()
    z_ar = ar.sample(args.n_samples, H, W, device)
    with torch.no_grad():
        x_ar = decoder(z_ar.to(device)).cpu()
    save_grid(x_ar, os.path.join(args.output_dir, 'gen_ar.png'))

    # --- Denoising baseline ---
    print("\n" + "=" * 80)
    print("DENOISING COMPILATION (baseline)")
    print("=" * 80)
    den_base = DenoisingCompiler(args.n_bits).to(device)
    train_denoiser(den_base, z_data, decoder, device, lam_freq=0.0, epochs=30, batch_size=32)
    den_base.eval()
    z_den_base = den_base.sample(args.n_samples, H, W, device, n_steps=15, temperature=0.7)
    with torch.no_grad():
        x_den_base = decoder(z_den_base.to(device)).cpu()
    save_grid(x_den_base, os.path.join(args.output_dir, 'gen_denoise_base.png'))

    # --- Denoising + freq λ=0.3 ---
    print("\n" + "=" * 80)
    print("DENOISING + FREQ (λ=0.3)")
    print("=" * 80)
    den_freq = DenoisingCompiler(args.n_bits).to(device)
    train_denoiser(den_freq, z_data, decoder, device, lam_freq=0.3, epochs=30, batch_size=32)
    den_freq.eval()
    z_den_freq = den_freq.sample(args.n_samples, H, W, device, n_steps=15, temperature=0.7)
    with torch.no_grad():
        x_den_freq = decoder(z_den_freq.to(device)).cpu()
    save_grid(x_den_freq, os.path.join(args.output_dir, 'gen_denoise_freq.png'))

    # --- Denoising + freq λ=0.3 + multiscale ---
    print("\n" + "=" * 80)
    print("DENOISING + FREQ (λ=0.3) + MULTISCALE")
    print("=" * 80)
    den_freq_ms = DenoisingCompiler(args.n_bits).to(device)
    train_denoiser(den_freq_ms, z_data, decoder, device, lam_freq=0.3, epochs=30, batch_size=32)
    den_freq_ms.eval()
    z_den_ms = den_freq_ms.sample_multiscale(args.n_samples, H, W, device,
                                               n_steps=15, temperature=0.7)
    with torch.no_grad():
        x_den_ms = decoder(z_den_ms.to(device)).cpu()
    save_grid(x_den_ms, os.path.join(args.output_dir, 'gen_denoise_freq_ms.png'))

    # ================================================================
    # EVALUATION
    # ================================================================
    print("\n" + "=" * 100)
    print("EVALUATION")
    print("=" * 100)

    methods = {
        'bernoulli': (z_bern, x_bern),
        'ar': (z_ar, x_ar),
        'denoise_base': (z_den_base, x_den_base),
        'denoise_freq': (z_den_freq, x_den_freq),
        'denoise_freq_ms': (z_den_ms, x_den_ms),
    }

    for name, (z_gen, x_gen) in methods.items():
        print(f"\n  [{name}]")
        z_gen_cpu = z_gen.cpu()

        viol = e_core.violation_rate(z_gen_cpu[:100].to(device))
        tok_kl = token_histogram_kl(z_data[:500], z_gen_cpu[:min(500, len(z_gen_cpu))], args.n_bits)
        div_mean, div_std = compute_diversity(z_gen_cpu)

        with torch.no_grad():
            z_check = z_gen_cpu[:100].to(device)
            x_check = decoder(z_check)
            z_cyc, _ = encoder(x_check)
            cycle_ham = (z_check != z_cyc).float().mean().item()

        x_gen_np = x_gen.numpy().reshape(len(x_gen), -1)
        nn_mean, nn_std = compute_1nn_distance(x_gen_np, train_x_np)

        gen_bep = band_energy_profile(x_gen[:100], device)
        bep_dist = float(np.sqrt(((gen_bep - real_bep)**2).sum()))
        conn = connectedness_proxy(x_gen[:100])

        r = {
            'method': name,
            'violation': viol,
            'token_kl': tok_kl,
            'diversity': div_mean,
            'cycle_hamming': cycle_ham,
            'nn_dist': nn_mean,
            'bep_distance': bep_dist,
            'connectedness': conn,
        }
        all_results.append(r)

        print(f"    viol={viol:.4f} tokKL={tok_kl:.4f} div={div_mean:.4f} "
              f"cycle={cycle_ham:.4f} 1NN={nn_mean:.2f}")
        print(f"    BEP_d={bep_dist:.4f} conn={conn:.4f}")

    # Summary
    print("\n" + "=" * 100)
    print("CIFAR-10 GENERATION SUMMARY")
    print(f"Real BEP: [{real_bep[0]:.3f},{real_bep[1]:.3f},{real_bep[2]:.3f}]  "
          f"Real conn: {real_conn:.4f}")
    print("=" * 100)

    header = (f"{'method':<20} {'viol':>7} {'tokKL':>7} {'div':>7} "
              f"{'cycle':>7} {'1NN':>7} {'BEP_d':>7} {'conn':>7}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['method']:<20} {r['violation']:>7.4f} {r['token_kl']:>7.4f} "
              f"{r['diversity']:>7.4f} {r['cycle_hamming']:>7.4f} {r['nn_dist']:>7.2f} "
              f"{r['bep_distance']:>7.4f} {r['connectedness']:>7.4f}")

    # CSV
    csv_path = os.path.join(args.output_dir, "cifar10_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(all_results[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV: {csv_path}")

    print("\n" + "=" * 100)
    print("CIFAR-10 generation experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
