#!/usr/bin/env python3
"""
CIFAR-10 Classification — C2: Staged ResBlock Encoder + VICReg Self-Supervised
===============================================================================
Core hypothesis: 45% accuracy is limited by encoder architecture (flat conv, no
stage hierarchy, insufficient receptive field for semantic abstraction).

Architecture:
  Stage1: 32×32, C=64, ResBlock×2 (texture)
  Stage2: 16×16, C=128, ResBlock×2 (structure)
  Stage3: 8×8, C=256, ResBlock×2 (semantic)

  z_tex: from Stage2 output (16×16×K_tex)
  z_sem: from Stage3 output (8×8×K_sem)

Self-supervised signal:
  VICReg on z_sem (variance + invariance + covariance)
  Better than SimCLR for small batch sizes (no negative pair dependency)

Evaluation: mixed probe (clean + repaired), report clean/repair/gap.

Usage:
    python3 -u benchmarks/exp_cifar10_staged_encoder.py --device cuda
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


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class GumbelSigmoid(nn.Module):
    def __init__(self, tau=1.0):
        super().__init__(); self.tau = tau
    def forward(self, logits):
        if self.training:
            u = torch.rand_like(logits).clamp(1e-8, 1-1e-8)
            noisy = (logits - torch.log(-torch.log(u))) / self.tau
        else:
            noisy = logits / self.tau
        soft = torch.sigmoid(noisy); hard = (soft > 0.5).float()
        return hard - soft.detach() + soft
    def set_tau(self, tau): self.tau = tau


class ResBlock(nn.Module):
    """Standard pre-activation ResBlock."""
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
    def forward(self, x): return x + self.net(x)


class DownBlock(nn.Module):
    """Downsample 2× + change channels."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x): return F.relu(self.bn(self.conv(x)))


class UpBlock(nn.Module):
    """Upsample 2× + change channels."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x): return F.relu(self.bn(self.up(x)))


# ============================================================================
# STAGED ENCODER (Stage1→Stage2→Stage3, z_tex + z_sem)
# ============================================================================

class StagedEncoder(nn.Module):
    """
    3-stage ResBlock encoder:
      Stage1: 32×32, C=64, ResBlock×2
      Stage2: 16×16, C=128, ResBlock×2
      Stage3: 8×8, C=256, ResBlock×2

    Output:
      z_tex: 16×16 × k_tex (from Stage2)
      z_sem: 8×8 × k_sem (from Stage3)
    """
    def __init__(self, k_tex=8, k_sem=16):
        super().__init__()
        self.k_tex, self.k_sem = k_tex, k_sem

        # Input projection
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
        )

        # Stage 1: 32×32, C=48
        self.stage1 = nn.Sequential(ResBlock(48))

        # Downsample 32→16
        self.down1 = DownBlock(48, 96)

        # Stage 2: 16×16, C=96
        self.stage2 = nn.Sequential(ResBlock(96))

        # z_tex head: 16×16 → k_tex channels
        self.tex_head = nn.Conv2d(96, k_tex, 3, padding=1)
        self.q_tex = GumbelSigmoid()

        # Downsample 16→8
        self.down2 = DownBlock(96, 192)

        # Stage 3: 8×8, C=192
        self.stage3 = nn.Sequential(ResBlock(192))

        # z_sem head: 8×8 → k_sem channels
        self.sem_head = nn.Conv2d(192, k_sem, 3, padding=1)
        self.q_sem = GumbelSigmoid()

    def forward(self, x):
        h = self.stem(x)          # 32×32×64
        h1 = self.stage1(h)       # 32×32×64
        h2 = self.stage2(self.down1(h1))  # 16×16×128
        h3 = self.stage3(self.down2(h2))  # 8×8×256

        z_tex = self.q_tex(self.tex_head(h2))  # 16×16×k_tex
        z_sem = self.q_sem(self.sem_head(h3))  # 8×8×k_sem
        return z_tex, z_sem

    def forward_features(self, x):
        """Return continuous features before quantization (for VICReg)."""
        h = self.stem(x)
        h1 = self.stage1(h)
        h2 = self.stage2(self.down1(h1))
        h3 = self.stage3(self.down2(h2))
        return h3  # 8×8×192 continuous features

    def set_temperature(self, tau):
        self.q_tex.set_tau(tau)
        self.q_sem.set_tau(tau)


# ============================================================================
# STAGED DECODER
# ============================================================================

class StagedDecoder(nn.Module):
    """Reconstruct 32×32×3 from z_tex (16×16) + z_sem (8×8)."""
    def __init__(self, k_tex=8, k_sem=16):
        super().__init__()
        # Upsample z_sem: 8→16
        self.sem_up = nn.Sequential(
            UpBlock(k_sem, 64),  # 8→16
        )
        # Merge z_tex + upsampled z_sem at 16×16
        self.merge16 = nn.Sequential(
            nn.Conv2d(k_tex + 64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128),
        )
        # Upsample 16→32
        self.up_final = UpBlock(128, 64)
        # Output
        self.out = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z_tex, z_sem):
        sem16 = self.sem_up(z_sem)      # 8→16
        h = torch.cat([z_tex, sem16], dim=1)  # 16×16×(k_tex+64)
        h = self.merge16(h)              # 16×16×128
        h = self.up_final(h)             # 32×32×64
        return self.out(h)               # 32×32×3


# ============================================================================
# FLAT BASELINE (same param budget, no staging)
# ============================================================================

class FlatEncoder(nn.Module):
    def __init__(self, k=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, k, 3, padding=1),
        )
        self.q = GumbelSigmoid()
    def forward(self, x):
        return self.q(self.net(x)), None
    def set_temperature(self, tau): self.q.set_tau(tau)


class FlatDecoder(nn.Module):
    def __init__(self, k=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(k, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z_tex, z_sem=None): return self.net(z_tex)


# ============================================================================
# DENOISER (repairs z_tex only)
# ============================================================================

class Denoiser(nn.Module):
    def __init__(self, k, z_h=16):
        super().__init__()
        hid = min(128, max(64, k * 4))
        self.net = nn.Sequential(
            nn.Conv2d(k+1, hid, 3, padding=1), nn.ReLU(),
            ResBlock(hid),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, k, 3, padding=1))
        self.skip = nn.Conv2d(k, k, 1)

    def forward(self, z_noisy, noise_level):
        B = z_noisy.shape[0]
        nl = noise_level.view(B,1,1,1).expand(-1,1,z_noisy.shape[2],z_noisy.shape[3])
        return self.net(torch.cat([z_noisy, nl], dim=1)) + self.skip(z_noisy)

    @torch.no_grad()
    def repair(self, z_tex, mask, n_steps=5, temperature=0.5):
        B, K, H, W = z_tex.shape
        z_rep = z_tex.clone()
        for step in range(n_steps):
            nl = torch.tensor([1.0-step/n_steps], device=z_tex.device).expand(B)
            logits = self(z_rep, nl)
            probs = torch.sigmoid(logits / temperature)
            z_new = (torch.rand_like(z_rep) < probs).float()
            z_rep = mask * z_tex + (1-mask) * z_new
        logits = self(z_rep, torch.zeros(B, device=z_tex.device))
        z_final = (torch.sigmoid(logits) > 0.5).float()
        return mask * z_tex + (1-mask) * z_final


# ============================================================================
# VICReg LOSS (variance + invariance + covariance, no negatives needed)
# ============================================================================

def vicreg_loss(z1, z2, lam=25.0, mu=25.0, nu=1.0):
    """
    VICReg: variance-invariance-covariance regularization.
    z1, z2: B × D embeddings from two augmented views.
    """
    B, D = z1.shape

    # Invariance: MSE between paired embeddings
    inv_loss = F.mse_loss(z1, z2)

    # Variance: force std > 1 per dimension
    z1_std = z1.std(dim=0)
    z2_std = z2.std(dim=0)
    var_loss = (F.relu(1.0 - z1_std).mean() + F.relu(1.0 - z2_std).mean()) / 2

    # Covariance: off-diagonal elements of cov matrix → 0
    z1c = z1 - z1.mean(0)
    z2c = z2 - z2.mean(0)
    cov1 = (z1c.T @ z1c) / (B - 1)
    cov2 = (z2c.T @ z2c) / (B - 1)
    # Zero diagonal, sum off-diagonal squared
    off1 = cov1.pow(2).sum() - cov1.diag().pow(2).sum()
    off2 = cov2.pow(2).sum() - cov2.diag().pow(2).sum()
    cov_loss = (off1 + off2) / (2 * D)

    return lam * inv_loss + mu * var_loss + nu * cov_loss


# ============================================================================
# AUGMENTATION
# ============================================================================

def augment_batch(x):
    B, C, H, W = x.shape
    # Random horizontal flip (per-image)
    flip_mask = (torch.rand(B, 1, 1, 1, device=x.device) > 0.5).float()
    x_flip = x.flip(-1)
    x = x * (1 - flip_mask) + x_flip * flip_mask
    # Random crop (pad 4, then crop)
    x = F.pad(x, [4,4,4,4], mode='reflect')
    i = torch.randint(0, 8, (1,)).item()
    j = torch.randint(0, 8, (1,)).item()
    x = x[:, :, i:i+H, j:j+W]
    # Brightness jitter
    brightness = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=x.device)
    x = (x * brightness).clamp(0, 1)
    return x


# ============================================================================
# PROBES
# ============================================================================

class SemProbe(nn.Module):
    def __init__(self, k_sem, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(k_sem, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
            nn.Linear(64*4, n_classes),
        )
    def forward(self, z): return self.net(z)


class TexProbe(nn.Module):
    def __init__(self, k_tex, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(k_tex, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(32*16, n_classes),
        )
    def forward(self, z): return self.net(z)


class DualProbe(nn.Module):
    def __init__(self, k_tex, k_sem, n_classes=10):
        super().__init__()
        self.sem_arm = nn.Sequential(
            nn.Conv2d(k_sem, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )
        self.tex_arm = nn.Sequential(
            nn.Conv2d(k_tex, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )
        self.head = nn.Linear(64*4 + 32*4, n_classes)

    def forward(self, z_tex, z_sem):
        return self.head(torch.cat([self.sem_arm(z_sem), self.tex_arm(z_tex)], 1))


class FlatConvProbe(nn.Module):
    def __init__(self, k, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(k, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(32*16, n_classes),
        )
    def forward(self, z): return self.net(z)


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def train_adc(encoder, decoder, train_x, device, epochs=30, bs=128):
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    for epoch in tqdm(range(epochs), desc="ADC"):
        encoder.train(); decoder.train()
        encoder.set_temperature(1.0 + (0.3-1.0)*epoch/(max(epochs-1,1)))
        perm = torch.randperm(len(train_x)); tl, nb = 0., 0
        for i in range(0, len(train_x), bs):
            x = train_x[perm[i:i+bs]].to(device)
            opt.zero_grad()
            z_tex, z_sem = encoder(x)
            xh = decoder(z_tex, z_sem)
            loss = F.mse_loss(xh, x) + 0.5 * F.binary_cross_entropy(xh, x)
            loss.backward(); opt.step(); tl += loss.item(); nb += 1
    encoder.eval(); decoder.eval()
    return tl / nb


def train_vicreg(encoder, train_x, device, epochs=20, bs=128, proj_dim=128):
    """VICReg self-supervised training on continuous features before z_sem quantization."""
    # Projector for VICReg (operates on pooled Stage3 features)
    projector = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(192, 192), nn.BatchNorm1d(192), nn.ReLU(),
        nn.Linear(192, proj_dim),
    ).to(device)

    # Fine-tune: sem_head + stage3 + projector
    params = list(projector.parameters())
    if hasattr(encoder, 'stage3'):
        params += list(encoder.stage3.parameters())
        params += list(encoder.sem_head.parameters())
        params += list(encoder.q_sem.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)

    for epoch in tqdm(range(epochs), desc="VICReg"):
        encoder.train(); projector.train()
        perm = torch.randperm(len(train_x)); tl, nb = 0., 0
        for i in range(0, len(train_x), bs):
            x = train_x[perm[i:i+bs]].to(device)
            x1 = augment_batch(x)
            x2 = augment_batch(x)
            opt.zero_grad()
            feat1 = encoder.forward_features(x1)  # 8×8×256
            feat2 = encoder.forward_features(x2)
            e1 = projector(feat1)
            e2 = projector(feat2)
            loss = vicreg_loss(e1, e2)
            loss.backward(); opt.step(); tl += loss.item(); nb += 1
    print(f"    VICReg done: loss={tl/nb:.4f}")
    encoder.eval()
    del projector
    torch.cuda.empty_cache()


def train_denoiser(denoiser, z_data, device, epochs=25, bs=128):
    opt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)
    for epoch in tqdm(range(epochs), desc="Denoiser"):
        denoiser.train(); perm = torch.randperm(N); tl, nb = 0., 0
        for i in range(0, N, bs):
            z = z_data[perm[i:i+bs]].to(device); B_ = z.shape[0]
            nl = torch.rand(B_, device=device)
            flip = (torch.rand_like(z) < nl.view(B_,1,1,1)).float()
            z_noisy = z*(1-flip) + (1-z)*flip
            opt.zero_grad()
            logits = denoiser(z_noisy, nl)
            loss = F.binary_cross_entropy_with_logits(logits, z)
            loss.backward(); opt.step(); tl += loss.item(); nb += 1
    print(f"    Denoiser done: loss={tl/nb:.4f}")
    denoiser.eval()


def encode_hier(encoder, data, device, bs=128):
    tex_l, sem_l = [], []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            zt, zs = encoder(data[i:i+bs].to(device))
            tex_l.append(zt.cpu()); sem_l.append(zs.cpu())
    return torch.cat(tex_l), torch.cat(sem_l)


def encode_flat(encoder, data, device, bs=128):
    zs = []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            z, _ = encoder(data[i:i+bs].to(device))
            zs.append(z.cpu())
    return torch.cat(zs)


def make_center_mask(z):
    B, K, H, W = z.shape
    mask = torch.ones(B, K, H, W)
    h4, w4 = H//4, W//4
    mask[:, :, h4:3*h4, w4:3*w4] = 0
    return mask


def apply_repair(denoiser, z_data, device, bs=64):
    reps = []
    with torch.no_grad():
        for i in range(0, len(z_data), bs):
            z = z_data[i:i+bs]
            mask = make_center_mask(z).to(device)
            z_rep = denoiser.repair(z.to(device) * mask, mask)
            reps.append(z_rep.cpu())
    return torch.cat(reps)


def train_probe(probe, z_data, labels, device, epochs=50, bs=256):
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for ep in tqdm(range(epochs), desc="Probe", leave=False):
        probe.train(); perm = torch.randperm(len(z_data))
        for i in range(0, len(z_data), bs):
            idx = perm[i:i+bs]
            z = z_data[idx].to(device); y = labels[idx].to(device)
            opt.zero_grad(); F.cross_entropy(probe(z), y).backward(); opt.step()
    probe.eval()


def train_dual_probe(probe, zt, zs, labels, device, epochs=50, bs=256):
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for ep in tqdm(range(epochs), desc="Probe", leave=False):
        probe.train(); perm = torch.randperm(len(zt))
        for i in range(0, len(zt), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            F.cross_entropy(probe(zt[idx].to(device), zs[idx].to(device)),
                          labels[idx].to(device)).backward()
            opt.step()
    probe.eval()


def eval_probe(probe, z, labels, device, bs=256):
    probe.eval(); nc, nt = 0, 0
    with torch.no_grad():
        for i in range(0, len(z), bs):
            pred = probe(z[i:i+bs].to(device)).argmax(1)
            nc += (pred == labels[i:i+bs].to(device)).sum().item()
            nt += len(pred)
    return nc / nt


def eval_dual_probe(probe, zt, zs, labels, device, bs=256):
    probe.eval(); nc, nt = 0, 0
    with torch.no_grad():
        for i in range(0, len(zt), bs):
            pred = probe(zt[i:i+bs].to(device), zs[i:i+bs].to(device)).argmax(1)
            nc += (pred == labels[i:i+bs].to(device)).sum().item()
            nt += len(pred)
    return nc / nt


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("CIFAR-10 — C2: STAGED RESBLOCK ENCODER + VICREG SELF-SUPERVISED")
    print("=" * 100)

    from torchvision import datasets, transforms
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(args.seed)
    train_x = torch.stack([train_ds[i][0] for i in rng.choice(len(train_ds), args.n_train, replace=False)])
    train_y = torch.tensor([train_ds[i][1] for i in rng.choice(len(train_ds), args.n_train, replace=False)])
    # Fix: use same indices for x and y
    rng2 = np.random.default_rng(args.seed)
    train_idx = rng2.choice(len(train_ds), args.n_train, replace=False)
    test_idx = rng2.choice(len(test_ds), args.n_test, replace=False)
    train_x = torch.stack([train_ds[i][0] for i in train_idx])
    train_y = torch.tensor([train_ds[i][1] for i in train_idx])
    test_x = torch.stack([test_ds[i][0] for i in test_idx])
    test_y = torch.tensor([test_ds[i][1] for i in test_idx])
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")

    results = {}
    k_tex, k_sem = 8, 16

    # ========================================================================
    # CONFIG A: FLAT RESBLOCK BASELINE (32×32×8, same ResBlock budget)
    # ========================================================================
    print(f"\n{'='*100}")
    print("CONFIG A: FLAT RESBLOCK BASELINE (32×32×8)")
    print("="*100)

    torch.manual_seed(args.seed)
    enc_flat = FlatEncoder(k=8).to(device)
    dec_flat = FlatDecoder(k=8).to(device)
    den_flat = Denoiser(k=8, z_h=32).to(device)

    enc_p = sum(p.numel() for p in enc_flat.parameters())
    print(f"    Encoder params: {enc_p:,}")

    print("  Training ADC...")
    train_adc(enc_flat, dec_flat, train_x, device, epochs=25)

    z_train_flat = encode_flat(enc_flat, train_x, device)
    z_test_flat = encode_flat(enc_flat, test_x, device)
    print(f"    z: {z_train_flat.shape}")

    print("  Training denoiser...")
    train_denoiser(den_flat, z_train_flat, device, epochs=20)

    z_train_rep_flat = apply_repair(den_flat, z_train_flat, device)
    z_test_rep_flat = apply_repair(den_flat, z_test_flat, device)

    # Mixed probe
    z_mixed_flat = torch.cat([z_train_flat, z_train_rep_flat])
    y_mixed = torch.cat([train_y, train_y])

    probe_flat = FlatConvProbe(k=8).to(device)
    train_probe(probe_flat, z_mixed_flat, y_mixed, device)
    acc_c = eval_probe(probe_flat, z_test_flat, test_y, device)
    acc_r = eval_probe(probe_flat, z_test_rep_flat, test_y, device)
    print(f"    FLAT_RESBLOCK: clean={acc_c:.3f}  repair={acc_r:.3f}  gap={abs(acc_c-acc_r):.3f}")
    results['flat_resblock'] = {'clean': acc_c, 'repair': acc_r, 'gap': abs(acc_c-acc_r)}
    del enc_flat, dec_flat, den_flat, probe_flat; torch.cuda.empty_cache()

    # ========================================================================
    # CONFIG B: STAGED ENCODER (no VICReg)
    # ========================================================================
    print(f"\n{'='*100}")
    print(f"CONFIG B: STAGED ENCODER (z_tex 16×16×{k_tex} + z_sem 8×8×{k_sem}, no VICReg)")
    print("="*100)

    torch.manual_seed(args.seed)
    enc_staged = StagedEncoder(k_tex, k_sem).to(device)
    dec_staged = StagedDecoder(k_tex, k_sem).to(device)
    den_staged = Denoiser(k=k_tex, z_h=16).to(device)

    enc_p = sum(p.numel() for p in enc_staged.parameters())
    print(f"    Encoder params: {enc_p:,}")
    print(f"    z_tex: 16×16×{k_tex} ({16*16*k_tex} bits)")
    print(f"    z_sem: 8×8×{k_sem} ({8*8*k_sem} bits)")

    print("  Training ADC...")
    train_adc(enc_staged, dec_staged, train_x, device, epochs=25)

    ztex_train, zsem_train = encode_hier(enc_staged, train_x, device)
    ztex_test, zsem_test = encode_hier(enc_staged, test_x, device)
    print(f"    z_tex: {ztex_train.shape}, z_sem: {zsem_train.shape}")

    print("  Training denoiser (z_tex only)...")
    train_denoiser(den_staged, ztex_train, device, epochs=20)

    ztex_train_rep = apply_repair(den_staged, ztex_train, device)
    ztex_test_rep = apply_repair(den_staged, ztex_test, device)

    # Mixed
    ztex_mixed = torch.cat([ztex_train, ztex_train_rep])
    zsem_mixed = torch.cat([zsem_train, zsem_train])
    y_mixed = torch.cat([train_y, train_y])

    # Probes
    for probe_name, make_probe, is_dual, z_clean, z_rep in [
        ("sem_only", lambda: SemProbe(k_sem).to(device), False, zsem_train, zsem_test),
        ("tex_only", lambda: TexProbe(k_tex).to(device), False, ztex_train, ztex_test),
    ]:
        p = make_probe()
        z_mix = torch.cat([z_clean, z_clean])  # sem never changes
        if probe_name == "tex_only":
            z_mix = torch.cat([ztex_train, ztex_train_rep])
        train_probe(p, z_mix, y_mixed, device)
        ac = eval_probe(p, zsem_test if 'sem' in probe_name else ztex_test, test_y, device)
        ar = eval_probe(p, zsem_test if 'sem' in probe_name else ztex_test_rep, test_y, device)
        if 'sem' in probe_name:
            ar = ac  # z_sem is identical after repair
        print(f"    staged_{probe_name}: clean={ac:.3f}  repair={ar:.3f}  gap={abs(ac-ar):.3f}")
        results[f'staged_{probe_name}'] = {'clean': ac, 'repair': ar, 'gap': abs(ac-ar)}
        del p; torch.cuda.empty_cache()

    # Dual probe
    dp = DualProbe(k_tex, k_sem).to(device)
    train_dual_probe(dp, ztex_mixed, zsem_mixed, y_mixed, device)
    ac = eval_dual_probe(dp, ztex_test, zsem_test, test_y, device)
    ar = eval_dual_probe(dp, ztex_test_rep, zsem_test, test_y, device)
    print(f"    staged_dual: clean={ac:.3f}  repair={ar:.3f}  gap={abs(ac-ar):.3f}")
    results['staged_dual'] = {'clean': ac, 'repair': ar, 'gap': abs(ac-ar)}
    del dp; torch.cuda.empty_cache()

    # ========================================================================
    # CONFIG C: STAGED + VICReg (self-supervised semantic carrier)
    # ========================================================================
    print(f"\n{'='*100}")
    print("CONFIG C: STAGED + VICReg (self-supervised z_sem, no labels)")
    print("="*100)

    print("  VICReg training on z_sem features...")
    train_vicreg(enc_staged, train_x, device, epochs=20, bs=128)

    # Re-encode
    ztex_train_v, zsem_train_v = encode_hier(enc_staged, train_x, device)
    ztex_test_v, zsem_test_v = encode_hier(enc_staged, test_x, device)

    # Re-repair z_tex (reuse denoiser — z_tex distribution may have shifted)
    ztex_train_rep_v = apply_repair(den_staged, ztex_train_v, device)
    ztex_test_rep_v = apply_repair(den_staged, ztex_test_v, device)

    ztex_mixed_v = torch.cat([ztex_train_v, ztex_train_rep_v])
    zsem_mixed_v = torch.cat([zsem_train_v, zsem_train_v])
    y_mixed = torch.cat([train_y, train_y])

    # Probes after VICReg
    p_sem = SemProbe(k_sem).to(device)
    train_probe(p_sem, torch.cat([zsem_train_v, zsem_train_v]), y_mixed, device)
    ac = eval_probe(p_sem, zsem_test_v, test_y, device)
    print(f"    vicreg_sem: clean={ac:.3f}  repair={ac:.3f}  gap=0.000")
    results['vicreg_sem'] = {'clean': ac, 'repair': ac, 'gap': 0.0}
    del p_sem; torch.cuda.empty_cache()

    dp_v = DualProbe(k_tex, k_sem).to(device)
    train_dual_probe(dp_v, ztex_mixed_v, zsem_mixed_v, y_mixed, device)
    ac = eval_dual_probe(dp_v, ztex_test_v, zsem_test_v, test_y, device)
    ar = eval_dual_probe(dp_v, ztex_test_rep_v, zsem_test_v, test_y, device)
    print(f"    vicreg_dual: clean={ac:.3f}  repair={ar:.3f}  gap={abs(ac-ar):.3f}")
    results['vicreg_dual'] = {'clean': ac, 'repair': ar, 'gap': abs(ac-ar)}
    del dp_v, den_staged, enc_staged, dec_staged; torch.cuda.empty_cache()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*100}")
    print("FINAL SUMMARY — STAGED ENCODER EXPERIMENT")
    print("="*100)

    print(f"\n  {'config':<25} {'clean':>8} {'repair':>8} {'gap':>8}")
    print(f"  {'-'*49}")
    for name, r in results.items():
        print(f"  {name:<25} {r['clean']:>8.3f} {r['repair']:>8.3f} {r['gap']:>8.3f}")

    # Diagnose
    flat_clean = results['flat_resblock']['clean']
    best_staged = max(r['clean'] for k, r in results.items() if 'staged' in k or 'vicreg' in k)
    print(f"\n  Flat ResBlock baseline: {flat_clean:.3f}")
    print(f"  Best staged config: {best_staged:.3f}")
    print(f"  Staging gain: {best_staged - flat_clean:+.3f}")

    if 'vicreg_sem' in results and 'staged_sem_only' in results:
        vic_gain = results['vicreg_sem']['clean'] - results['staged_sem_only']['clean']
        print(f"  VICReg gain on z_sem: {vic_gain:+.3f}")

    if 'vicreg_dual' in results and 'staged_dual' in results:
        vic_gain = results['vicreg_dual']['clean'] - results['staged_dual']['clean']
        print(f"  VICReg gain on dual: {vic_gain:+.3f}")

    print(f"\n{'='*100}")
    print("EXPERIMENT COMPLETE")
    print("="*100)

    os.makedirs('outputs/exp_cifar10_staged_encoder', exist_ok=True)
    with open('outputs/exp_cifar10_staged_encoder/results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
