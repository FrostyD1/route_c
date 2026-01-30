#!/usr/bin/env python3
"""
Global Discrete Attention (GDA) Experiment
============================================
InpaintNet_v2: local CNN + Global Discrete Attention via XOR/popcount.

GDA layer:
  1. bitpack: z (B,k,H,W) → code (B,N) uint8, code = Σ z_b << b
  2. Hamming distance: dist(i,j) = popcount(code_i XOR code_j) via 256-LUT
  3. Attention: A = softmax(-dist/T), value = Embedding(code) → ctx = A @ V
  4. Fusion: concat(local_feat, ctx) → Conv → bit logits

Benchmark: center/stripes × clean/noise
Compare: InpaintNet_v1(mask-only) vs InpaintNet_v2(+GDA, mask-only)

Usage:
    python3 -u benchmarks/exp_gda.py --device cuda --seed 42
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import time
import csv
import argparse
from typing import Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))


# ============================================================================
# MODEL (same as Phase 1 — BCE, no L_cls)
# ============================================================================

class GumbelSigmoid(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits):
        if self.training:
            u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            gumbel = -torch.log(-torch.log(u))
            noisy = (logits + gumbel) / self.temperature
        else:
            noisy = logits / self.temperature
        soft = torch.sigmoid(noisy)
        hard = (soft > 0.5).float()
        return hard - soft.detach() + soft

    def set_temperature(self, tau):
        self.temperature = tau


class Encoder(nn.Module):
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(z)


class LocalPredictor(nn.Module):
    def __init__(self, n_bits, hidden_dim=32):
        super().__init__()
        self.n_bits = n_bits
        self.net = nn.Sequential(
            nn.Linear(9 * n_bits, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_bits),
        )

    def forward(self, z):
        B, k, H, W = z.shape
        z_pad = F.pad(z, (1, 1, 1, 1), mode='constant', value=0)
        windows = F.unfold(z_pad, kernel_size=3)
        windows = windows.reshape(B, k, 9, H * W)
        windows[:, :, 4, :] = 0
        windows = windows.reshape(B, k * 9, H * W)
        windows = windows.permute(0, 2, 1)
        logits = self.net(windows)
        return logits.permute(0, 2, 1).reshape(B, k, H, W)


class Classifier(nn.Module):
    def __init__(self, n_bits, latent_size=7, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(n_bits * latent_size * latent_size, n_classes)

    def forward(self, z):
        return self.fc(z.reshape(z.shape[0], -1))


class RouteCModel(nn.Module):
    def __init__(self, n_bits=8, hidden_dim=64, energy_hidden=32,
                 latent_size=7, tau=1.0):
        super().__init__()
        self.n_bits = n_bits
        self.latent_size = latent_size
        self.encoder = Encoder(n_bits, hidden_dim)
        self.quantizer = GumbelSigmoid(tau)
        self.decoder = Decoder(n_bits, hidden_dim)
        self.local_pred = LocalPredictor(n_bits, energy_hidden)
        self.classifier = Classifier(n_bits, latent_size)

    def encode(self, x):
        return self.quantizer(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        cls_logits = self.classifier(z)
        core_logits = self.local_pred(z)
        return z, x_hat, cls_logits, core_logits

    def set_temperature(self, tau):
        self.quantizer.set_temperature(tau)


# ============================================================================
# GLOBAL DISCRETE ATTENTION (GDA)
# ============================================================================

class GlobalDiscreteAttention(nn.Module):
    """
    Content-addressed global attention via XOR/popcount on binary codes.

    For each position, computes Hamming distance to all other positions,
    then applies softmax attention weighted by distance.

    This is the Hopfield isomorphism made explicit:
    - Hamming distance = energy between binary patterns
    - Softmax attention = Boltzmann distribution over patterns
    - Value aggregation = associative memory retrieval
    """

    def __init__(self, k: int = 8, d_v: int = 32, temperature: float = 1.0):
        super().__init__()
        self.k = k
        self.d_v = d_v
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Popcount LUT: pop_lut[i] = number of 1-bits in byte i
        pop_lut = torch.tensor([bin(i).count("1") for i in range(256)],
                               dtype=torch.float32)
        self.register_buffer('pop_lut', pop_lut)

        # Value embedding: code (uint8 in [0,255]) → R^{d_v}
        self.value_embed = nn.Embedding(256, d_v)

    def bitpack(self, z: torch.Tensor) -> torch.Tensor:
        """
        Pack k binary channels into uint8 codes.

        Args:
            z: (B, k, H, W) float — thresholded to {0,1}

        Returns:
            codes: (B, N) long, where N = H*W, values in [0, 255]
        """
        B, k, H, W = z.shape
        bits = (z > 0.5).long()  # (B, k, H, W)
        # code = Σ bits[:,b,:,:] << b
        shifts = (2 ** torch.arange(k, device=z.device)).view(1, k, 1, 1)
        codes = (bits * shifts).sum(dim=1)  # (B, H, W)
        return codes.reshape(B, H * W)  # (B, N)

    def hamming_matrix(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Hamming distance matrix via XOR + LUT popcount.

        Args:
            codes: (B, N) long

        Returns:
            dist: (B, N, N) float — Hamming distances
        """
        B, N = codes.shape
        # XOR all pairs: (B, N, 1) XOR (B, 1, N) → (B, N, N)
        xor = codes.unsqueeze(2) ^ codes.unsqueeze(1)  # (B, N, N)
        # Popcount via LUT
        dist = self.pop_lut[xor]  # (B, N, N) float
        return dist

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Global discrete attention.

        Args:
            z: (B, k, H, W) — binary latent grid (may be soft during training)

        Returns:
            ctx: (B, d_v, H, W) — global context features
        """
        B, k, H, W = z.shape
        N = H * W

        # 1. Bitpack
        codes = self.bitpack(z)  # (B, N)

        # 2. Hamming distance matrix
        dist = self.hamming_matrix(codes)  # (B, N, N)

        # 3. Attention weights: A = softmax(-dist / T)
        # Temperature is learnable — controls sharpness of retrieval
        scores = -dist / self.temperature.clamp(min=0.1)  # (B, N, N)
        attn = F.softmax(scores, dim=-1)  # (B, N, N), rows sum to 1

        # 4. Values from code embedding
        values = self.value_embed(codes.clamp(0, 255))  # (B, N, d_v)

        # 5. Context: weighted aggregation
        ctx = torch.bmm(attn, values)  # (B, N, d_v)

        # Reshape to spatial
        ctx = ctx.permute(0, 2, 1).reshape(B, self.d_v, H, W)  # (B, d_v, H, W)
        return ctx


# ============================================================================
# INPAINTING NETWORKS
# ============================================================================

class InpaintNetV1(nn.Module):
    """Original: 3-layer residual CNN, local only."""
    def __init__(self, k=8, hidden=64, n_layers=3):
        super().__init__()
        self.k = k
        layers = []
        in_ch = k + 1
        for i in range(n_layers):
            out_ch = hidden if i < n_layers - 1 else k
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1,
                                    padding_mode='circular'))
            if i < n_layers - 1:
                layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.skip = nn.Conv2d(k + 1, k, 1)

    def forward(self, z_masked, mask):
        x = torch.cat([z_masked, mask], dim=1)
        return self.net(x) + self.skip(x)


class InpaintNetV2(nn.Module):
    """
    InpaintNet + Global Discrete Attention (GDA).

    Architecture:
      1. Local branch: Conv(k+1 → hidden) — 2 layers, captures 3×3 context
      2. Global branch: GDA(z) → (B, d_v, H, W) — captures full-grid Hamming relations
      3. Fusion: Conv(hidden + d_v → k) — combines local + global → bit logits
      4. Skip: Conv(k+1 → k, 1×1) — residual connection

    The GDA layer operates on the MASKED input z (with zeros at masked positions),
    so its Hamming distances reflect "what we know" not "what we guess".
    """

    def __init__(self, k=8, hidden=64, d_v=32, temperature=1.0):
        super().__init__()
        self.k = k

        # Local branch: 2 conv layers
        self.local_branch = nn.Sequential(
            nn.Conv2d(k + 1, hidden, 3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
        )

        # Global branch: GDA
        self.gda = GlobalDiscreteAttention(k=k, d_v=d_v, temperature=temperature)

        # Fusion: local + global → output logits
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden + d_v, hidden, 3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, k, 3, padding=1, padding_mode='circular'),
        )

        # Skip connection
        self.skip = nn.Conv2d(k + 1, k, 1)

    def forward(self, z_masked, mask):
        """
        Args:
            z_masked: (B, k, H, W) — observed bits (masked positions zeroed)
            mask: (B, 1, H, W) — 1 where masked, 0 where observed

        Returns:
            logits: (B, k, H, W) — per-bit prediction logits
        """
        x = torch.cat([z_masked, mask], dim=1)  # (B, k+1, H, W)

        # Local features
        f_local = self.local_branch(x)  # (B, hidden, H, W)

        # Global features via GDA on the masked z
        f_global = self.gda(z_masked)  # (B, d_v, H, W)

        # Fuse
        combined = torch.cat([f_local, f_global], dim=1)  # (B, hidden+d_v, H, W)
        out = self.fusion(combined) + self.skip(x)  # residual

        return out


# ============================================================================
# DATA + MODEL TRAINING
# ============================================================================

def load_data(train_n=2000, test_n=1000, seed=42):
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(train_ds), train_n, replace=False)
    test_idx = rng.choice(len(test_ds), test_n, replace=False)
    train_x = torch.stack([train_ds[i][0] for i in train_idx])
    train_y = torch.tensor([train_ds[i][1] for i in train_idx])
    test_x = torch.stack([test_ds[i][0] for i in test_idx])
    test_y = torch.tensor([test_ds[i][1] for i in test_idx])
    return train_x, train_y, test_x, test_y


def train_model_bce(train_x, train_y, device, epochs=5, lr=1e-3, batch_size=64,
                    tau_start=1.0, tau_end=0.2):
    """Train Route C model: BCE E_obs, NO L_cls."""
    model = RouteCModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(train_x, train_y),
                        batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        tau = tau_start + (tau_end - tau_start) * epoch / max(1, epochs - 1)
        model.set_temperature(tau)
        epoch_loss = 0.0
        n_b = 0
        for x, y in loader:
            x = x.to(device)
            optimizer.zero_grad()
            z, x_hat, _, core_logits = model(x)
            loss_recon = F.binary_cross_entropy(x_hat.clamp(1e-6, 1-1e-6), x)
            mask = torch.rand_like(z) < 0.15
            loss_core = F.binary_cross_entropy_with_logits(
                core_logits[mask], z.detach()[mask]
            ) if mask.any() else torch.tensor(0.0, device=device)
            loss = loss_recon + 0.5 * loss_core
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        print(f"    Epoch {epoch+1}/{epochs}: loss={epoch_loss/max(n_b,1):.4f}")

    # Train classifier probe separately
    print("    Training classifier probe (frozen)...")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    cls_opt = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    for ep in range(3):
        model.classifier.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                z = model.encode(x)
            logits = model.classifier(z)
            F.cross_entropy(logits, y).backward()
            cls_opt.step()
            cls_opt.zero_grad()
    for p in model.parameters():
        p.requires_grad = True
    return model


# ============================================================================
# INPAINTING TRAINING (mask-only, NO L_cls/L_obs/L_core)
# ============================================================================

def train_inpaint(model, inpaint_net, train_x, device,
                  epochs=20, batch_size=64, lr=1e-3,
                  mask_ratio_min=0.1, mask_ratio_max=0.7):
    """
    Pure L_mask training (sleep-phase compilation).
    No L_cls, no L_obs, no L_core — isolates the GDA contribution.
    """
    optimizer = torch.optim.Adam(inpaint_net.parameters(), lr=lr)
    model.eval()
    N = len(train_x)

    for epoch in range(epochs):
        inpaint_net.train()
        perm = torch.randperm(N)
        total_loss = 0.0
        n_b = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            x = train_x[idx].to(device)

            with torch.no_grad():
                z = model.encode(x)
                z_hard = (z > 0.5).float()

            B, k, H, W = z_hard.shape
            ratio = torch.rand(B, 1, 1, 1, device=device)
            ratio = mask_ratio_min + ratio * (mask_ratio_max - mask_ratio_min)
            mask = (torch.rand(B, 1, H, W, device=device) < ratio).float()
            mask_exp = mask.expand(-1, k, -1, -1)

            z_masked = z_hard * (1 - mask_exp)
            logits = inpaint_net(z_masked, mask)

            # L_mask only
            loss = F.binary_cross_entropy_with_logits(
                logits[mask_exp.bool()], z_hard[mask_exp.bool()]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_b += 1

        if (epoch + 1) % 5 == 0:
            print(f"      epoch {epoch+1}/{epochs}: loss={total_loss/max(n_b,1):.4f}")

    return inpaint_net


# ============================================================================
# MASKS + EVALUATION
# ============================================================================

def make_center_mask(H=28, W=28, occ_h=14, occ_w=14):
    mask = np.ones((H, W), dtype=np.float32)
    y, x = (H - occ_h) // 2, (W - occ_w) // 2
    mask[y:y+occ_h, x:x+occ_w] = 0
    return mask


def make_stripe_mask(H=28, W=28, stripe_width=2, gap=6):
    mask = np.ones((H, W), dtype=np.float32)
    for y in range(0, H, gap):
        mask[y:min(y + stripe_width, H), :] = 0
    return mask


def pixel_to_bit_mask(pixel_mask, n_bits=8, latent_size=7):
    patch_size = 28 // latent_size
    bit_mask = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    for i in range(latent_size):
        for j in range(latent_size):
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            if pixel_mask[y0:y1, x0:x1].mean() < 1.0 - 1e-6:
                bit_mask[:, i, j] = True
    return bit_mask


def apply_noise(image, noise_type, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    if noise_type == 'noise':
        noisy = image + rng.normal(0, 0.1, image.shape).astype(np.float32)
        return np.clip(noisy, 0, 1)
    return image.copy()


def amortized_inpaint(inpaint_net, z_init, bit_mask, device):
    inpaint_net.eval()
    k, H, W = z_init.shape
    z = z_init.clone().to(device)
    bm = torch.from_numpy(bit_mask).float().to(device)
    mask = bm.max(dim=0, keepdim=True)[0].unsqueeze(0)
    bm_exp = bm.unsqueeze(0)
    z_masked = z.unsqueeze(0) * (1 - bm_exp)

    with torch.no_grad():
        logits = inpaint_net(z_masked, mask)
        preds = (torch.sigmoid(logits) > 0.5).float()

    bm_bool = torch.from_numpy(bit_mask).to(device)
    z_result = z.clone()
    z_result[bm_bool] = preds[0][bm_bool]
    return z_result


def evaluate_method(model, inpaint_net, method_name, test_x, test_y,
                    mask_type, noise_type, device, n_samples=100, seed=42):
    model.eval()
    inpaint_net.eval()

    pixel_mask = make_center_mask() if mask_type == 'center' else make_stripe_mask()
    bit_mask = pixel_to_bit_mask(pixel_mask)
    occ = 1 - pixel_mask

    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    correct_before, correct_after = [], []
    bce_before_l, bce_after_l = [], []
    mse_before_l, mse_after_l = [], []
    runtimes = []

    for idx in eval_idx:
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()
        x_noisy = apply_noise(x_clean, noise_type, rng)
        x_occ = x_noisy * pixel_mask

        with torch.no_grad():
            x_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_init = model.encode(x_t)[0]
            pred_b = model.classifier(z_init.unsqueeze(0)).argmax(1).item()
            o_hat_b = model.decode(z_init.unsqueeze(0))[0, 0]

        t0 = time.time()
        z_final = amortized_inpaint(inpaint_net, z_init, bit_mask, device)
        rt = (time.time() - t0) * 1000

        with torch.no_grad():
            pred_a = model.classifier(z_final.unsqueeze(0)).argmax(1).item()
            o_hat_a = model.decode(z_final.unsqueeze(0))[0, 0]

        x_t_full = torch.from_numpy(x_clean).to(device)
        occ_t = torch.from_numpy(occ).to(device)
        occ_sum = occ_t.sum().clamp(min=1.0).item()

        def occ_mse(o_hat):
            d = (o_hat - x_t_full) * occ_t
            return (d * d).sum().item() / occ_sum

        def occ_bce(o_hat):
            lam = o_hat.clamp(1e-6, 1-1e-6)
            b = -(x_t_full * torch.log(lam) + (1 - x_t_full) * torch.log(1 - lam))
            return (b * occ_t).sum().item() / occ_sum

        correct_before.append(int(pred_b == label))
        correct_after.append(int(pred_a == label))
        mse_before_l.append(occ_mse(o_hat_b))
        mse_after_l.append(occ_mse(o_hat_a))
        bce_before_l.append(occ_bce(o_hat_b))
        bce_after_l.append(occ_bce(o_hat_a))
        runtimes.append(rt)

    cb = np.array(correct_before)
    ca = np.array(correct_after)
    n = len(eval_idx)

    return {
        'method': method_name,
        'mask_type': mask_type,
        'noise_type': noise_type,
        'acc_before': cb.mean(),
        'acc_after': ca.mean(),
        'delta_acc': (ca.sum() - cb.sum()) / n,
        'mse_before': np.mean(mse_before_l),
        'mse_after': np.mean(mse_after_l),
        'bce_before': np.mean(bce_before_l),
        'bce_after': np.mean(bce_after_l),
        'runtime_ms': np.mean(runtimes),
        'bit_mask_ratio': float(bit_mask[0].mean()),
        'n_samples': n,
    }


# ============================================================================
# MAIN
# ============================================================================

def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    parser = argparse.ArgumentParser(description='GDA Experiment')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--train_samples', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--inpaint_epochs', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='outputs/exp_gda')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 100)
    print("GDA EXPERIMENT: InpaintNet_v1 (local) vs InpaintNet_v2 (+GDA)")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Training: mask-only (L_mask), NO L_cls/L_obs/L_core")
    print()

    # Data
    print("[1] Loading data...")
    train_x, train_y, test_x, test_y = load_data(args.train_samples, 1000, args.seed)

    # Model
    print("\n[2] Training Route C model (BCE, no L_cls)...")
    model = train_model_bce(train_x, train_y, device, epochs=args.epochs)
    model.eval()

    with torch.no_grad():
        z_test = model.encode(test_x[:500].to(device))
        preds = model.classifier(z_test).argmax(1).cpu()
        clean_acc = (preds == test_y[:500]).float().mean().item()
    print(f"    Clean accuracy (probe): {clean_acc:.1%}")

    # Train both InpaintNets
    v1 = InpaintNetV1(k=8).to(device)
    v2 = InpaintNetV2(k=8, d_v=32, temperature=1.0).to(device)

    print(f"\n    V1 params: {count_params(v1):,}")
    print(f"    V2 params: {count_params(v2):,}")
    print(f"    GDA params: {count_params(v2.gda):,}")

    print(f"\n[3a] Training InpaintNet_v1 (local only, mask-only)...")
    v1 = train_inpaint(model, v1, train_x, device, epochs=args.inpaint_epochs)

    print(f"\n[3b] Training InpaintNet_v2 (+GDA, mask-only)...")
    v2 = train_inpaint(model, v2, train_x, device, epochs=args.inpaint_epochs)

    # Evaluate
    configs = [
        ('center', 'clean'),
        ('center', 'noise'),
        ('stripes', 'clean'),
        ('stripes', 'noise'),
    ]

    print(f"\n[4] Evaluating...")
    all_results = []

    for mask_type, noise_type in configs:
        print(f"\n  --- {mask_type} + {noise_type} ---")

        r1 = evaluate_method(model, v1, 'v1_local', test_x, test_y,
                             mask_type, noise_type, device,
                             n_samples=args.eval_samples, seed=args.seed)
        print(f"    v1_local:  Δacc={r1['delta_acc']:+.1%}, "
              f"bce={r1['bce_before']:.2f}→{r1['bce_after']:.2f}, "
              f"t={r1['runtime_ms']:.1f}ms")

        r2 = evaluate_method(model, v2, 'v2_gda', test_x, test_y,
                             mask_type, noise_type, device,
                             n_samples=args.eval_samples, seed=args.seed)
        print(f"    v2_gda:    Δacc={r2['delta_acc']:+.1%}, "
              f"bce={r2['bce_before']:.2f}→{r2['bce_after']:.2f}, "
              f"t={r2['runtime_ms']:.1f}ms")

        delta = r2['delta_acc'] - r1['delta_acc']
        print(f"    GDA gain:  {delta:+.1%}")

        all_results.extend([r1, r2])

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY: InpaintNet_v1 (local) vs InpaintNet_v2 (+GDA)")
    print("=" * 120)

    header = (f"{'method':<12} {'mask':<10} {'noise':<8} "
              f"{'acc_bef':>8} {'acc_aft':>8} {'Δacc':>7} "
              f"{'bce_bef':>8} {'bce_aft':>8} "
              f"{'mse_bef':>8} {'mse_aft':>8} "
              f"{'ms':>7} {'bit_r':>6}")
    print(header)
    print("-" * 120)

    for r in all_results:
        print(f"{r['method']:<12} {r['mask_type']:<10} {r['noise_type']:<8} "
              f"{r['acc_before']:>8.1%} {r['acc_after']:>8.1%} {r['delta_acc']:>+7.1%} "
              f"{r['bce_before']:>8.2f} {r['bce_after']:>8.2f} "
              f"{r['mse_before']:>8.4f} {r['mse_after']:>8.4f} "
              f"{r['runtime_ms']:>7.1f} {r['bit_mask_ratio']:>6.2f}")

    # Per-config comparison
    print("\n" + "=" * 80)
    print("GDA GAIN (v2 - v1)")
    print("=" * 80)

    for mask_type, noise_type in configs:
        r1 = next(r for r in all_results
                  if r['method'] == 'v1_local'
                  and r['mask_type'] == mask_type
                  and r['noise_type'] == noise_type)
        r2 = next(r for r in all_results
                  if r['method'] == 'v2_gda'
                  and r['mask_type'] == mask_type
                  and r['noise_type'] == noise_type)

        gain_acc = r2['delta_acc'] - r1['delta_acc']
        gain_bce = r2['bce_after'] - r1['bce_after']
        speed_ratio = r2['runtime_ms'] / max(r1['runtime_ms'], 0.01)
        status = "BETTER" if gain_acc > 0 else ("SAME" if gain_acc == 0 else "WORSE")

        print(f"  {mask_type}+{noise_type}: "
              f"Δacc gain={gain_acc:+.1%}, "
              f"Δbce gain={gain_bce:+.3f}, "
              f"speed ratio={speed_ratio:.1f}× → {status}")

    # Key question: does stripes improve?
    print("\n" + "=" * 80)
    print("KEY QUESTION: Does GDA help on stripes (the hard OOD case)?")
    print("=" * 80)
    for noise_type in ['clean', 'noise']:
        r1 = next(r for r in all_results
                  if r['method'] == 'v1_local'
                  and r['mask_type'] == 'stripes'
                  and r['noise_type'] == noise_type)
        r2 = next(r for r in all_results
                  if r['method'] == 'v2_gda'
                  and r['mask_type'] == 'stripes'
                  and r['noise_type'] == noise_type)
        gain = r2['delta_acc'] - r1['delta_acc']
        verdict = "YES — GDA helps" if gain > 0.01 else ("MARGINAL" if gain > -0.01 else "NO")
        print(f"  stripes+{noise_type}: v1 Δacc={r1['delta_acc']:+.1%}, "
              f"v2 Δacc={r2['delta_acc']:+.1%}, gain={gain:+.1%} → {verdict}")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "gda_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    # Model info
    print(f"\nModel sizes:")
    print(f"  Route C:      {count_params(model):,}")
    print(f"  InpaintNet v1: {count_params(v1):,}")
    print(f"  InpaintNet v2: {count_params(v2):,} (+{count_params(v2)-count_params(v1):,})")
    print(f"    GDA layer:   {count_params(v2.gda):,}")

    print("\n" + "=" * 100)
    print("GDA experiment complete.")
    print("=" * 100)

    return all_results


if __name__ == "__main__":
    main()
