#!/usr/bin/env python3
"""
E_obs Geometry Ablation: MSE vs BCE vs Continuous Bernoulli
============================================================
Paradigm 4 validation: "The observation energy E_obs defines the metric tensor
on the decoder manifold. Changing E_obs changes the topology of the energy
landscape over z, not just gradient magnitudes."

Hypothesis: sigmoid decoder + MSE = geometric mismatch → corr(Δmse,Δacc) < 0.
BCE/CB should align the observation geometry with the decoder's distributional
assumptions, yielding corr(Δmetric,Δacc) ≥ 0.

Three levels where E_obs acts:
1. Model training (loss_recon) — shapes z-space
2. InpaintNet training (L_obs) — teaches inpainter about observation consistency
3. Evaluation energy — the metric used to measure improvement

Usage:
    python3 -u benchmarks/exp_eobs.py --device cuda --seed 42
    python3 -u benchmarks/exp_eobs.py --device cpu --eval_samples 50
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
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))


# ============================================================================
# CONTINUOUS BERNOULLI UTILITIES
# ============================================================================

def cont_bern_log_norm(lam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    log C(λ) where C(λ) = 2·atanh(1-2λ) / (1-2λ).

    Numerically stable implementation:
    - Near λ=0.5: Taylor expand → log C ≈ log 2
    - Away from 0.5: direct formula
    - Clamp λ to [eps, 1-eps] to avoid log(0)
    """
    lam = lam.clamp(eps, 1.0 - eps)

    # Near λ=0.5 the formula is numerically unstable; use Taylor approximation
    # C(λ) = 2 + 2/3·(2λ-1)² + O((2λ-1)⁴)  →  log C ≈ log 2 + (2λ-1)²/3
    near_half = (lam - 0.5).abs() < 0.02
    safe_lam = torch.where(near_half, torch.tensor(0.3, device=lam.device), lam)

    # Standard formula: log C(λ) = log(2·atanh(1-2λ)) - log(1-2λ)
    # But 1-2λ can be negative, so use absolute value + care:
    # C(λ) = 2·atanh(1-2λ)/(1-2λ)
    # When λ < 0.5: 1-2λ > 0, atanh > 0 → C > 0 ✓
    # When λ > 0.5: 1-2λ < 0, atanh < 0 → C > 0 ✓ (both negative → positive)
    one_minus_2lam = 1.0 - 2.0 * safe_lam
    # atanh(x) = 0.5 * log((1+x)/(1-x)), defined for |x|<1
    log_C = torch.log(2.0 * torch.atanh(one_minus_2lam).abs() + 1e-10) - torch.log(one_minus_2lam.abs() + 1e-10)

    # Near-half approximation
    taylor_log_C = torch.log(torch.tensor(2.0, device=lam.device)) + (2 * lam - 1).pow(2) / 3.0

    return torch.where(near_half, taylor_log_C, log_C)


def loss_cb(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Continuous Bernoulli negative log-likelihood (mean over batch).

    -log p(x|λ) = -x·log(λ) - (1-x)·log(1-λ) - log C(λ)
                 = BCE(x, λ) - log C(λ)

    Note the MINUS: CB adds log C(λ) to the likelihood, so the NLL subtracts it.
    Since log C(λ) ≥ log 2 > 0, CB loss < BCE loss always.
    """
    eps = 1e-6
    lam = x_hat.clamp(eps, 1.0 - eps)
    bce = F.binary_cross_entropy(lam, x, reduction='none')
    log_C = cont_bern_log_norm(lam)
    # NLL = BCE - log C (per element), then mean
    return (bce - log_C).mean()


def loss_bce(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Standard BCE loss (Bernoulli NLL). Decoder output is already sigmoid."""
    return F.binary_cross_entropy(x_hat.clamp(1e-6, 1-1e-6), x)


def loss_mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Standard MSE loss (Gaussian NLL, up to constants)."""
    return F.mse_loss(x_hat, x)


RECON_LOSSES = {
    'mse': loss_mse,
    'bce': loss_bce,
    'cb': loss_cb,
}


# ============================================================================
# MODEL (same architecture as run_suite.py)
# ============================================================================

class GumbelSigmoid(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits):
        if self.training:
            u = torch.rand_like(logits).clamp(1e-8, 1-1e-8)
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
            nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(z)


class LocalPredictor(nn.Module):
    def __init__(self, n_bits, hidden_dim=32):
        super().__init__()
        self.n_bits = n_bits
        self.net = nn.Sequential(
            nn.Linear(9 * n_bits, hidden_dim),
            nn.ReLU(),
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
# INPAINTING NETWORK (simplified from inpainting/__init__.py)
# ============================================================================

class InpaintNet(nn.Module):
    def __init__(self, k: int = 8, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        self.k = k
        layers = []
        in_ch = k + 1
        for i in range(n_layers):
            out_ch = hidden if i < n_layers - 1 else k
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='circular'))
            if i < n_layers - 1:
                layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.skip = nn.Conv2d(k + 1, k, 1)

    def forward(self, z_masked, mask):
        x = torch.cat([z_masked, mask], dim=1)
        return self.net(x) + self.skip(x)


# ============================================================================
# DATA
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


# ============================================================================
# TRAINING
# ============================================================================

def train_model(eobs_type: str, train_x, train_y, device,
                epochs=5, lr=1e-3, batch_size=64,
                tau_start=1.0, tau_end=0.2,
                alpha_recon=1.0, beta_core=0.5, gamma_cls=1.0):
    """Train Route C model with specified E_obs type."""
    recon_loss_fn = RECON_LOSSES[eobs_type]

    model = RouteCModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(train_x, train_y),
                        batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        tau = tau_start + (tau_end - tau_start) * epoch / max(1, epochs - 1)
        model.set_temperature(tau)

        epoch_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            z, x_hat, cls_logits, core_logits = model(x)

            loss_recon = recon_loss_fn(x_hat, x)

            mask = torch.rand_like(z) < 0.15
            loss_core = F.binary_cross_entropy_with_logits(
                core_logits[mask], z.detach()[mask]
            ) if mask.any() else torch.tensor(0.0, device=device)
            loss_cls = F.cross_entropy(cls_logits, y)

            loss = alpha_recon * loss_recon + beta_core * loss_core + gamma_cls * loss_cls
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        print(f"    [{eobs_type}] Epoch {epoch+1}/{epochs}: loss={avg:.4f}")

    return model


def train_inpaint(model, eobs_type, train_x, train_y, device,
                  epochs=20, batch_size=64, lr=1e-3,
                  mask_ratio_min=0.1, mask_ratio_max=0.7,
                  alpha_core=0.1, alpha_obs=0.1):
    """Train InpaintNet with specified E_obs type for L_obs."""
    recon_loss_fn = RECON_LOSSES[eobs_type]

    inpaint_net = InpaintNet(k=model.n_bits).to(device)
    optimizer = torch.optim.Adam(inpaint_net.parameters(), lr=lr)

    model.eval()
    N = len(train_x)

    for epoch in range(epochs):
        inpaint_net.train()
        perm = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            x = train_x[idx].to(device)

            with torch.no_grad():
                z = model.encode(x)
                z_hard = (z > 0.5).float()

            B, k, H, W = z_hard.shape

            # Random per-position mask
            ratio = torch.rand(B, 1, 1, 1, device=device)
            ratio = mask_ratio_min + ratio * (mask_ratio_max - mask_ratio_min)
            mask = (torch.rand(B, 1, H, W, device=device) < ratio).float()
            mask_exp = mask.expand(-1, k, -1, -1)

            z_masked = z_hard * (1 - mask_exp)
            logits = inpaint_net(z_masked, mask)

            # L_mask: BCE on masked bits
            loss_mask = F.binary_cross_entropy_with_logits(
                logits[mask_exp.bool()], z_hard[mask_exp.bool()]
            )
            loss = loss_mask

            # L_core: local predictor consistency
            if alpha_core > 0:
                z_soft = z_hard * (1 - mask_exp) + torch.sigmoid(logits) * mask_exp
                core_logits = model.local_pred(z_soft)
                loss_core = F.binary_cross_entropy_with_logits(
                    core_logits[mask_exp.bool()], z_hard[mask_exp.bool()]
                )
                loss = loss + alpha_core * loss_core

            # L_obs: reconstruction consistency — uses E_obs-matched loss
            if alpha_obs > 0:
                z_soft = z_hard * (1 - mask_exp) + torch.sigmoid(logits) * mask_exp
                x_hat = model.decode(z_soft)
                loss_obs = recon_loss_fn(x_hat, x)
                loss = loss + alpha_obs * loss_obs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            avg = total_loss / max(n_batches, 1)
            print(f"    [{eobs_type}] InpaintNet epoch {epoch+1}/{epochs}: loss={avg:.4f}")

    return inpaint_net


# ============================================================================
# MASK UTILITIES
# ============================================================================

def make_center_mask(H=28, W=28, occ_h=14, occ_w=14):
    mask = np.ones((H, W), dtype=np.float32)
    y, x = (H - occ_h) // 2, (W - occ_w) // 2
    mask[y:y+occ_h, x:x+occ_w] = 0
    return mask


def pixel_to_bit_mask(pixel_mask, n_bits=8, latent_size=7):
    """'any' policy: masked if any pixel in patch is occluded."""
    patch_size = 28 // latent_size
    bit_mask = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    for i in range(latent_size):
        for j in range(latent_size):
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            if pixel_mask[y0:y1, x0:x1].mean() < 1.0 - 1e-6:
                bit_mask[:, i, j] = True
    return bit_mask


def amortized_inpaint(inpaint_net, z_init, bit_mask, device):
    """Single-pass amortized inpainting."""
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


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, inpaint_net, eobs_type, test_x, test_y,
             device, n_samples=100, seed=42):
    """
    Evaluate amortized inpainting on center mask.

    Returns per-sample metrics for correlation analysis.
    """
    model.eval()
    inpaint_net.eval()

    pixel_mask = make_center_mask()
    bit_mask = pixel_to_bit_mask(pixel_mask)
    occ_pixels = (1 - pixel_mask)

    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    correct_before = []
    correct_after = []
    mse_before = []
    mse_after = []
    bce_before = []
    bce_after = []
    cb_before = []
    cb_after = []
    runtimes = []

    for idx in eval_idx:
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()
        x_occ = x_clean * pixel_mask

        with torch.no_grad():
            x_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_init = model.encode(x_t)[0]

            # Before
            pred_b = model.classifier(z_init.unsqueeze(0)).argmax(1).item()
            o_hat_b = model.decode(z_init.unsqueeze(0))[0, 0]

        # Inpaint
        t0 = time.time()
        z_final = amortized_inpaint(inpaint_net, z_init, bit_mask, device)
        rt = (time.time() - t0) * 1000

        with torch.no_grad():
            pred_a = model.classifier(z_final.unsqueeze(0)).argmax(1).item()
            o_hat_a = model.decode(z_final.unsqueeze(0))[0, 0]

        # Compute all three metrics on occluded region
        x_clean_t = torch.from_numpy(x_clean).to(device)
        occ_t = torch.from_numpy(occ_pixels).to(device)
        occ_sum = occ_t.sum().clamp(min=1.0)

        def occluded_mse(o_hat):
            d = (o_hat - x_clean_t) * occ_t
            return (d * d).sum().item() / occ_sum.item()

        def occluded_bce(o_hat):
            lam = o_hat.clamp(1e-6, 1 - 1e-6)
            bce_elem = -(x_clean_t * torch.log(lam) + (1 - x_clean_t) * torch.log(1 - lam))
            return (bce_elem * occ_t).sum().item() / occ_sum.item()

        def occluded_cb(o_hat):
            lam = o_hat.clamp(1e-6, 1 - 1e-6)
            bce_elem = -(x_clean_t * torch.log(lam) + (1 - x_clean_t) * torch.log(1 - lam))
            log_C = cont_bern_log_norm(lam)
            return ((bce_elem - log_C) * occ_t).sum().item() / occ_sum.item()

        correct_before.append(int(pred_b == label))
        correct_after.append(int(pred_a == label))
        mse_before.append(occluded_mse(o_hat_b))
        mse_after.append(occluded_mse(o_hat_a))
        bce_before.append(occluded_bce(o_hat_b))
        bce_after.append(occluded_bce(o_hat_a))
        cb_before.append(occluded_cb(o_hat_b))
        cb_after.append(occluded_cb(o_hat_a))
        runtimes.append(rt)

    # Compute correlations
    correct_before = np.array(correct_before)
    correct_after = np.array(correct_after)
    delta_acc = correct_after - correct_before

    def safe_corr(delta_metric, delta_acc):
        if np.std(delta_metric) > 1e-12 and np.std(delta_acc) > 1e-12:
            return float(np.corrcoef(delta_metric, delta_acc)[0, 1])
        return 0.0

    delta_mse = np.array(mse_after) - np.array(mse_before)
    delta_bce = np.array(bce_after) - np.array(bce_before)
    delta_cb = np.array(cb_after) - np.array(cb_before)

    n = len(eval_idx)
    return {
        'eobs_type': eobs_type,
        'acc_before': correct_before.mean(),
        'acc_after': correct_after.mean(),
        'delta_acc': (correct_after.sum() - correct_before.sum()) / n,
        'mse_before': np.mean(mse_before),
        'mse_after': np.mean(mse_after),
        'bce_before': np.mean(bce_before),
        'bce_after': np.mean(bce_after),
        'cb_before': np.mean(cb_before),
        'cb_after': np.mean(cb_after),
        'corr_dmse_dacc': safe_corr(delta_mse, delta_acc),
        'corr_dbce_dacc': safe_corr(delta_bce, delta_acc),
        'corr_dcb_dacc': safe_corr(delta_cb, delta_acc),
        'runtime_ms': np.mean(runtimes),
        'n_samples': n,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='E_obs Geometry Ablation')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=200)
    parser.add_argument('--train_samples', type=int, default=2000)
    parser.add_argument('--test_samples', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--inpaint_epochs', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='outputs/exp_eobs')
    parser.add_argument('--eobs_types', type=str, default='mse,bce,cb',
                        help='Comma-separated E_obs types to test')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    eobs_types = args.eobs_types.split(',')

    print("=" * 100)
    print("E_obs GEOMETRY ABLATION — Paradigm 4 Validation")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"E_obs types: {eobs_types}")
    print(f"Eval samples: {args.eval_samples}")
    print(f"Train samples: {args.train_samples}")
    print()

    # Load data once
    print("[1] Loading data...")
    train_x, train_y, test_x, test_y = load_data(
        args.train_samples, args.test_samples, args.seed
    )
    print(f"    Train: {len(train_x)}, Test: {len(test_x)}")

    # ── Quick sanity check: CB log norm ──
    print("\n[1.5] Continuous Bernoulli sanity check...")
    test_lam = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    test_logC = cont_bern_log_norm(test_lam)
    print(f"    λ = {test_lam.tolist()}")
    print(f"    log C(λ) = {test_logC.tolist()}")
    print(f"    C(λ) = {test_logC.exp().tolist()}")
    # C(0.5) should be 2.0
    assert abs(test_logC[2].item() - np.log(2.0)) < 0.01, \
        f"C(0.5) should be 2.0, got {test_logC[2].exp().item()}"
    # C(λ) ≥ 2 for all λ
    assert (test_logC >= np.log(2.0) - 0.01).all(), \
        f"C(λ) should be ≥ 2 for all λ, got {test_logC.exp().tolist()}"
    print("    ✓ CB log norm verified")

    all_results = []

    for eobs_type in eobs_types:
        print(f"\n{'='*80}")
        print(f"[2] Training Route C model with E_obs = {eobs_type.upper()}")
        print(f"{'='*80}")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        model = train_model(
            eobs_type, train_x, train_y, device,
            epochs=args.epochs,
        )

        # Baseline accuracy (no occlusion)
        model.eval()
        with torch.no_grad():
            test_batch = test_x[:500].to(device)
            z_test = model.encode(test_batch)
            preds = model.classifier(z_test).argmax(1).cpu()
            baseline_acc = (preds == test_y[:500]).float().mean().item()
        print(f"    Baseline clean accuracy: {baseline_acc:.1%}")

        # Reconstruction quality
        with torch.no_grad():
            sample = test_x[:100].to(device)
            z_s = model.encode(sample)
            x_hat_s = model.decode(z_s)
            recon_mse = F.mse_loss(x_hat_s, sample).item()
            recon_bce = F.binary_cross_entropy(
                x_hat_s.clamp(1e-6, 1-1e-6), sample).item()
        print(f"    Recon MSE: {recon_mse:.4f}, Recon BCE: {recon_bce:.4f}")

        print(f"\n[3] Training InpaintNet with E_obs = {eobs_type.upper()}")
        inpaint_net = train_inpaint(
            model, eobs_type, train_x, train_y, device,
            epochs=args.inpaint_epochs,
        )

        print(f"\n[4] Evaluating {eobs_type.upper()}...")
        res = evaluate(
            model, inpaint_net, eobs_type, test_x, test_y,
            device, n_samples=args.eval_samples, seed=args.seed,
        )
        all_results.append(res)

        print(f"    acc: {res['acc_before']:.1%} → {res['acc_after']:.1%} "
              f"(Δ={res['delta_acc']:+.1%})")
        print(f"    corr(Δmse,Δacc) = {res['corr_dmse_dacc']:+.3f}")
        print(f"    corr(Δbce,Δacc) = {res['corr_dbce_dacc']:+.3f}")
        print(f"    corr(Δcb, Δacc) = {res['corr_dcb_dacc']:+.3f}")
        print(f"    runtime: {res['runtime_ms']:.1f} ms/sample")

    # ── Summary table ──
    print("\n" + "=" * 120)
    print("SUMMARY: E_obs Geometry Ablation (center mask, clean)")
    print("=" * 120)

    header = (f"{'E_obs':<6} {'acc_bef':>8} {'acc_aft':>8} {'Δacc':>7} "
              f"{'mse_bef':>8} {'mse_aft':>8} "
              f"{'bce_bef':>8} {'bce_aft':>8} "
              f"{'cb_bef':>8} {'cb_aft':>8} "
              f"{'corr_mse':>9} {'corr_bce':>9} {'corr_cb':>9} "
              f"{'ms':>6}")
    print(header)
    print("-" * 130)

    for r in all_results:
        print(f"{r['eobs_type']:<6} "
              f"{r['acc_before']:>8.1%} {r['acc_after']:>8.1%} {r['delta_acc']:>+7.1%} "
              f"{r['mse_before']:>8.4f} {r['mse_after']:>8.4f} "
              f"{r['bce_before']:>8.4f} {r['bce_after']:>8.4f} "
              f"{r['cb_before']:>8.4f} {r['cb_after']:>8.4f} "
              f"{r['corr_dmse_dacc']:>+9.3f} {r['corr_dbce_dacc']:>+9.3f} {r['corr_dcb_dacc']:>+9.3f} "
              f"{r['runtime_ms']:>6.1f}")

    # ── Key hypothesis tests ──
    print("\n" + "=" * 120)
    print("HYPOTHESIS TESTS")
    print("=" * 120)

    if len(all_results) >= 2:
        mse_res = next((r for r in all_results if r['eobs_type'] == 'mse'), None)
        bce_res = next((r for r in all_results if r['eobs_type'] == 'bce'), None)
        cb_res = next((r for r in all_results if r['eobs_type'] == 'cb'), None)

        print("\nH1: sigmoid decoder + MSE = geometric mismatch → corr(Δmse,Δacc) < 0")
        if mse_res:
            sign = "CONFIRMED" if mse_res['corr_dmse_dacc'] < 0 else "REJECTED"
            print(f"    MSE: corr(Δmse,Δacc) = {mse_res['corr_dmse_dacc']:+.3f} → {sign}")

        print("\nH2: BCE/CB should have better corr(Δmetric,Δacc) than MSE")
        if mse_res and bce_res:
            # Use native metric for each: MSE model uses corr_dmse, BCE uses corr_dbce
            mse_native = mse_res['corr_dmse_dacc']
            bce_native = bce_res['corr_dbce_dacc']
            sign = "CONFIRMED" if bce_native > mse_native else "REJECTED"
            print(f"    MSE native corr = {mse_native:+.3f}, BCE native corr = {bce_native:+.3f} → {sign}")
        if mse_res and cb_res:
            mse_native = mse_res['corr_dmse_dacc']
            cb_native = cb_res['corr_dcb_dacc']
            sign = "CONFIRMED" if cb_native > mse_native else "REJECTED"
            print(f"    MSE native corr = {mse_native:+.3f}, CB native corr = {cb_native:+.3f} → {sign}")

        print("\nH3: CB should have better Δacc than BCE (normalization constant matters)")
        if bce_res and cb_res:
            sign = "CONFIRMED" if cb_res['delta_acc'] > bce_res['delta_acc'] else "NOT CONFIRMED"
            print(f"    BCE Δacc = {bce_res['delta_acc']:+.1%}, CB Δacc = {cb_res['delta_acc']:+.1%} → {sign}")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "eobs_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    print("\n" + "=" * 100)
    print("E_obs ablation complete.")
    print("=" * 100)

    return all_results


if __name__ == "__main__":
    main()
