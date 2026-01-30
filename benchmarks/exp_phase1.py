#!/usr/bin/env python3
"""
Phase 1: MCMC Block Proposal + Paradigm Contract Validation
=============================================================
Goals:
1. Implement MCMC_block (token/2×2 block update with informed proposals)
   — upgrade from blind bit-flip to local-conditional sampling
2. Remove L_cls entirely from training (amortized_maskonly as main line)
3. Fix E_obs to BCE (geometrically matched to sigmoid decoder)
4. Run 4 hard configs: {center, stripes} × {clean, noise}
5. Compare: MCMC_bit(old), MCMC_block(new), amortized_maskonly, iterative(1..4)

Key diagnostic outputs per config:
- Δacc_probe (classification as readout, NOT objective)
- Δrecon (BCE on occluded region)
- runtime_ms
- accept_rate (MCMC only)
- energy_curve (MCMC only)
- E_core_violation_rate (fraction of positions where local pred disagrees)

Usage:
    python3 -u benchmarks/exp_phase1.py --device cuda --seed 42
    python3 -u benchmarks/exp_phase1.py --device cpu --eval_samples 50
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
import json
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))


# ============================================================================
# E_obs: BCE (geometrically matched to sigmoid decoder)
# ============================================================================

def loss_bce(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """BCE loss — correct likelihood for sigmoid decoder."""
    return F.binary_cross_entropy(x_hat.clamp(1e-6, 1 - 1e-6), x)


# ============================================================================
# MODEL (identical architecture, BCE training, NO L_cls)
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
    """E_core: predicts z_i from 3×3 neighborhood (center masked out)."""
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
        windows[:, :, 4, :] = 0  # mask center
        windows = windows.reshape(B, k * 9, H * W)
        windows = windows.permute(0, 2, 1)
        logits = self.net(windows)
        return logits.permute(0, 2, 1).reshape(B, k, H, W)

    def predict_position(self, z, i, j):
        """Get logits for a single position (i,j) given its neighborhood.

        Args:
            z: (k, H, W) single latent grid
            i, j: position indices

        Returns:
            logits: (k,) per-bit prediction logits
        """
        k, H, W = z.shape
        # Extract 3×3 neighborhood (zero-padded)
        window = torch.zeros(k, 9, device=z.device)
        idx = 0
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    window[:, idx] = z[:, ni, nj]
                idx += 1
        window[:, 4] = 0  # mask center
        flat = window.reshape(-1)  # (k*9,)
        logits = self.net(flat.unsqueeze(0))  # (1, k)
        return logits[0]  # (k,)


class Classifier(nn.Module):
    """Probe ONLY — never enters training loss."""
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
# INPAINTING NETWORK
# ============================================================================

class InpaintNet(nn.Module):
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


# ============================================================================
# ENERGY FUNCTIONS (BCE-based, not MSE)
# ============================================================================

class BCEEnergy:
    """
    E_total = λ_core · E_core + λ_obs · E_obs(BCE)

    E_core = -Σ log p_θ(z_i | neigh)   (local pseudo-likelihood)
    E_obs  = BCE(decode(z), x_obs)      (on visible pixels only)

    BCE is the geometrically correct metric for sigmoid decoder.
    """
    def __init__(self, model, lambda_core=1.0, lambda_obs=1.0, device=None):
        self.model = model
        self.lambda_core = lambda_core
        self.lambda_obs = lambda_obs
        self.device = device or torch.device('cpu')
        self._o_obs_t = None
        self._mask_t = None
        self._mask_sum = 1.0

    def set_observation(self, o_obs, pixel_mask):
        self._o_obs_t = torch.from_numpy(o_obs.astype(np.float32)).to(self.device)
        self._mask_t = torch.from_numpy(pixel_mask.astype(np.float32)).to(self.device)
        self._mask_sum = self._mask_t.sum().clamp(min=1.0)

    def energy(self, z):
        """Total energy. z: (k,H,W) or (1,k,H,W)."""
        with torch.inference_mode():
            z_t = z.unsqueeze(0) if z.dim() == 3 else z

            e_core = 0.0
            if self.lambda_core > 0:
                logits = self.model.local_pred(z_t)
                e_core = F.binary_cross_entropy_with_logits(
                    logits, z_t, reduction='sum'
                )

            e_obs = 0.0
            if self.lambda_obs > 0 and self._o_obs_t is not None:
                o_hat = self.model.decode(z_t)[0, 0]
                lam = o_hat.clamp(1e-6, 1 - 1e-6)
                # BCE on visible pixels only
                bce = -(self._o_obs_t * torch.log(lam)
                        + (1 - self._o_obs_t) * torch.log(1 - lam))
                e_obs = (bce * self._mask_t).sum() / self._mask_sum

            total = self.lambda_core * e_core + self.lambda_obs * e_obs
            return total.item() if isinstance(total, torch.Tensor) else total

    def energy_decomposed(self, z):
        """Return (e_core, e_obs) separately for diagnostics."""
        with torch.inference_mode():
            z_t = z.unsqueeze(0) if z.dim() == 3 else z

            logits = self.model.local_pred(z_t)
            e_core = F.binary_cross_entropy_with_logits(
                logits, z_t, reduction='sum'
            ).item()

            e_obs = 0.0
            if self._o_obs_t is not None:
                o_hat = self.model.decode(z_t)[0, 0]
                lam = o_hat.clamp(1e-6, 1 - 1e-6)
                bce = -(self._o_obs_t * torch.log(lam)
                        + (1 - self._o_obs_t) * torch.log(1 - lam))
                e_obs = (bce * self._mask_t).sum().item() / self._mask_sum.item()

            return e_core, e_obs


# ============================================================================
# MCMC SOLVERS
# ============================================================================

class MCMCBitSolver:
    """Original bit-flip MCMC (baseline, expected to be bad)."""
    def __init__(self, energy_fn, block_size=(2, 2), device=None):
        self.energy = energy_fn
        self.block_size = block_size
        self.device = device or torch.device('cpu')

    def run(self, z_init, o_obs, pixel_mask, bit_mask, n_sweeps=30):
        z = z_init.clone().to(self.device)
        k, H, W = z.shape
        bh, bw = self.block_size

        self.energy.set_observation(o_obs, pixel_mask)
        bit_mask_t = torch.from_numpy(bit_mask).to(self.device)
        z[bit_mask_t] = torch.randint(0, 2, (bit_mask_t.sum().item(),),
                                       device=self.device, dtype=z.dtype)

        E_curr = self.energy.energy(z)
        energies = [E_curr]
        accepts = []

        for sweep in range(n_sweeps):
            n_proposed = 0
            n_accepted = 0
            for bi in range(0, H, bh):
                for bj in range(0, W, bw):
                    i_end, j_end = min(bi + bh, H), min(bj + bw, W)
                    block_bm = bit_mask_t[:, bi:i_end, bj:j_end]
                    if not block_bm.any():
                        continue
                    n_proposed += 1
                    # Flip masked bits
                    z[:, bi:i_end, bj:j_end][block_bm] = \
                        1 - z[:, bi:i_end, bj:j_end][block_bm]
                    E_prop = self.energy.energy(z)
                    dE = E_prop - E_curr
                    if dE < 0 or torch.rand(1).item() < np.exp(-min(dE, 20)):
                        E_curr = E_prop
                        n_accepted += 1
                    else:
                        z[:, bi:i_end, bj:j_end][block_bm] = \
                            1 - z[:, bi:i_end, bj:j_end][block_bm]

            energies.append(E_curr)
            accepts.append(n_accepted / max(n_proposed, 1))

        return z, {
            'energies': energies,
            'accept_rates': accepts,
            'mean_accept': np.mean(accepts) if accepts else 0.0,
        }


class MCMCBlockSolver:
    """
    Block Gibbs with INFORMED proposals from local predictor.

    Update unit: one latent position (all k bits) or 2×2 block of positions.
    Proposal: sample from p_θ(z_i | neighborhood) — the local predictor.

    This addresses the scale mismatch:
    - Token-level update (k=8 bits at once) → E_obs can "see" the change
    - Informed proposal from E_core → proposals are already locally consistent
    - Accept/reject on full energy → balances E_core + E_obs
    """
    def __init__(self, energy_fn, model, block_size=(1, 1), device=None):
        self.energy = energy_fn
        self.model = model
        self.block_size = block_size  # (1,1)=single token, (2,2)=2×2 block
        self.device = device or torch.device('cpu')

    def _sample_token_from_local_pred(self, z, i, j):
        """Sample k bits at position (i,j) from local predictor p_θ(z_i|neigh).

        Returns:
            new_bits: (k,) tensor of {0, 1}
        """
        with torch.inference_mode():
            logits = self.model.local_pred.predict_position(z, i, j)
            probs = torch.sigmoid(logits)
            new_bits = (torch.rand_like(probs) < probs).float()
        return new_bits

    def run(self, z_init, o_obs, pixel_mask, bit_mask, n_sweeps=30):
        z = z_init.clone().to(self.device)
        k, H, W = z.shape
        bh, bw = self.block_size

        self.energy.set_observation(o_obs, pixel_mask)
        bit_mask_t = torch.from_numpy(bit_mask).to(self.device)

        # Initialize masked positions from local predictor (informed init)
        pos_mask = bit_mask_t[0]  # (H, W) — same for all bits
        masked_positions = [(i, j) for i in range(H) for j in range(W) if pos_mask[i, j]]

        for i, j in masked_positions:
            z[:, i, j] = self._sample_token_from_local_pred(z, i, j)

        E_curr = self.energy.energy(z)
        energies = [E_curr]
        accepts = []

        for sweep in range(n_sweeps):
            n_proposed = 0
            n_accepted = 0

            # Shuffle update order
            np.random.shuffle(masked_positions)

            for i, j in masked_positions:
                if not bit_mask_t[0, i, j]:
                    continue

                old_bits = z[:, i, j].clone()
                n_proposed += 1

                # Informed proposal from local predictor
                new_bits = self._sample_token_from_local_pred(z, i, j)

                # Apply proposal
                z[:, i, j] = new_bits
                E_prop = self.energy.energy(z)
                dE = E_prop - E_curr

                # Metropolis-Hastings acceptance
                # Note: proposal is symmetric (both sample from same local_pred
                # given current neighbors), so MH ratio simplifies to exp(-dE)
                if dE < 0 or torch.rand(1).item() < np.exp(-min(dE, 20)):
                    E_curr = E_prop
                    n_accepted += 1
                else:
                    z[:, i, j] = old_bits  # reject

            energies.append(E_curr)
            accepts.append(n_accepted / max(n_proposed, 1))

        return z, {
            'energies': energies,
            'accept_rates': accepts,
            'mean_accept': np.mean(accepts) if accepts else 0.0,
        }


# ============================================================================
# AMORTIZED INFERENCE
# ============================================================================

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


def iterative_inpaint(inpaint_net, z_init, bit_mask, n_steps, device):
    """MaskGIT-style iterative with cosine schedule."""
    inpaint_net.eval()
    k, H, W = z_init.shape
    z = z_init.clone().to(device)
    current_mask = torch.from_numpy(bit_mask).bool().to(device)
    total_masked = current_mask.sum().item()

    if total_masked == 0:
        return z

    with torch.no_grad():
        for step in range(n_steps):
            progress = (step + 1) / n_steps
            fraction_to_unmask = 1.0 - np.cos(progress * np.pi / 2)
            n_to_unmask = int(fraction_to_unmask * total_masked)
            n_to_unmask = min(n_to_unmask, current_mask.sum().item())

            if current_mask.sum().item() == 0:
                break

            mask_spatial = current_mask.float().max(dim=0, keepdim=True)[0].unsqueeze(0)
            z_masked = z.unsqueeze(0) * (1 - current_mask.float().unsqueeze(0))
            logits = inpaint_net(z_masked, mask_spatial)
            probs = torch.sigmoid(logits[0])
            predictions = (probs > 0.5).float()
            confidence = (probs - 0.5).abs() * 2

            confidence_masked = confidence.clone()
            confidence_masked[~current_mask] = -1.0

            flat_conf = confidence_masked.flatten()
            flat_mask = current_mask.flatten()

            if n_to_unmask > 0 and flat_mask.sum() > 0:
                sorted_indices = flat_conf.argsort(descending=True)
                count = 0
                for idx in sorted_indices:
                    if count >= n_to_unmask:
                        break
                    if flat_mask[idx]:
                        flat_mask[idx] = False
                        z.view(-1)[idx] = predictions.view(-1)[idx]
                        count += 1
                current_mask = flat_mask.reshape(k, H, W)

        # Final pass
        if current_mask.sum().item() > 0:
            mask_spatial = current_mask.float().max(dim=0, keepdim=True)[0].unsqueeze(0)
            z_masked = z.unsqueeze(0) * (1 - current_mask.float().unsqueeze(0))
            logits = inpaint_net(z_masked, mask_spatial)
            predictions = (torch.sigmoid(logits[0]) > 0.5).float()
            z[current_mask] = predictions[current_mask]

    return z


# ============================================================================
# MASKS
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
    """'any' policy."""
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
    if noise_type == 'clean':
        return image.copy()
    elif noise_type == 'noise':
        noisy = image + rng.normal(0, 0.1, image.shape).astype(np.float32)
        return np.clip(noisy, 0, 1)
    return image.copy()


# ============================================================================
# DATA + TRAINING
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
                    tau_start=1.0, tau_end=0.2, alpha_recon=1.0, beta_core=0.5):
    """
    Train Route C model with BCE E_obs, NO L_cls.

    Loss = α_recon · BCE(x̂, x) + β_core · E_core
    Classification is a PROBE — never in the loss.
    """
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
            x = x.to(device)
            optimizer.zero_grad()
            z, x_hat, cls_logits, core_logits = model(x)

            # E_obs: BCE (geometrically matched)
            loss_recon = loss_bce(x_hat, x)

            # E_core: local pseudo-likelihood
            mask = torch.rand_like(z) < 0.15
            loss_core = F.binary_cross_entropy_with_logits(
                core_logits[mask], z.detach()[mask]
            ) if mask.any() else torch.tensor(0.0, device=device)

            # NO L_cls — classifier trains separately as probe
            loss = alpha_recon * loss_recon + beta_core * loss_core
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"    Epoch {epoch+1}/{epochs}: loss={epoch_loss/max(n_batches,1):.4f}")

    # Train classifier probe separately (frozen world model)
    print("    Training classifier probe (frozen encoder)...")
    model.eval()
    # Freeze everything except classifier
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    cls_opt = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    for epoch in range(3):
        model.classifier.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                z = model.encode(x)
            logits = model.classifier(z)
            loss = F.cross_entropy(logits, y)
            cls_opt.zero_grad()
            loss.backward()
            cls_opt.step()

    # Unfreeze
    for p in model.parameters():
        p.requires_grad = True

    return model


def train_inpaint_maskonly(model, train_x, device, epochs=20,
                           batch_size=64, lr=1e-3,
                           mask_ratio_min=0.1, mask_ratio_max=0.7,
                           alpha_core=0.1, alpha_obs=0.1):
    """
    Train InpaintNet: L_mask + L_core + L_obs(BCE).
    NO L_cls. This is pure sleep-phase compilation.
    """
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
            ratio = torch.rand(B, 1, 1, 1, device=device)
            ratio = mask_ratio_min + ratio * (mask_ratio_max - mask_ratio_min)
            mask = (torch.rand(B, 1, H, W, device=device) < ratio).float()
            mask_exp = mask.expand(-1, k, -1, -1)

            z_masked = z_hard * (1 - mask_exp)
            logits = inpaint_net(z_masked, mask)

            # L_mask
            loss = F.binary_cross_entropy_with_logits(
                logits[mask_exp.bool()], z_hard[mask_exp.bool()]
            )

            # L_core
            if alpha_core > 0:
                z_soft = z_hard * (1 - mask_exp) + torch.sigmoid(logits) * mask_exp
                core_logits = model.local_pred(z_soft)
                loss_core = F.binary_cross_entropy_with_logits(
                    core_logits[mask_exp.bool()], z_hard[mask_exp.bool()]
                )
                loss = loss + alpha_core * loss_core

            # L_obs (BCE)
            if alpha_obs > 0:
                z_soft = z_hard * (1 - mask_exp) + torch.sigmoid(logits) * mask_exp
                x_hat = model.decode(z_soft)
                loss_obs = loss_bce(x_hat, x)
                loss = loss + alpha_obs * loss_obs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"    InpaintNet epoch {epoch+1}/{epochs}: "
                  f"loss={total_loss/max(n_batches,1):.4f}")

    return inpaint_net


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def compute_core_violation_rate(model, z, device):
    """Fraction of positions where local predictor disagrees with actual bits.

    This measures how well z satisfies the local consistency rules (E_core).
    Lower = more consistent.
    """
    with torch.no_grad():
        z_t = z.unsqueeze(0).to(device)
        logits = model.local_pred(z_t)
        preds = (logits > 0).float()
        z_hard = (z_t > 0.5).float()
        disagreement = (preds != z_hard).float().mean().item()
    return disagreement


def compute_occluded_bce(o_hat_t, x_clean_t, occ_t, occ_sum):
    """BCE on occluded pixels."""
    lam = o_hat_t.clamp(1e-6, 1 - 1e-6)
    bce = -(x_clean_t * torch.log(lam) + (1 - x_clean_t) * torch.log(1 - lam))
    return (bce * occ_t).sum().item() / occ_sum


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_config(model, inpaint_net, test_x, test_y,
                    mask_type, noise_type, methods, device,
                    n_samples=100, n_sweeps=30, seed=42):
    """
    Evaluate all methods on one (mask_type, noise_type) config.

    Returns list of result dicts.
    """
    model.eval()
    inpaint_net.eval()

    # Generate mask
    if mask_type == 'center':
        pixel_mask = make_center_mask()
    elif mask_type == 'stripes':
        pixel_mask = make_stripe_mask()
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

    bit_mask = pixel_to_bit_mask(pixel_mask, n_bits=8)
    occ_pixels = 1 - pixel_mask
    bit_mask_ratio = bit_mask[0].mean()

    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    # Build energy + solvers
    energy_fn = BCEEnergy(model, lambda_core=1.0, lambda_obs=1.0, device=device)
    mcmc_bit = MCMCBitSolver(energy_fn, block_size=(2, 2), device=device)
    mcmc_block = MCMCBlockSolver(energy_fn, model, block_size=(1, 1), device=device)

    all_results = []

    for method in methods:
        print(f"      {method}...", end='', flush=True)
        correct_before = []
        correct_after = []
        bce_before_list = []
        bce_after_list = []
        runtimes = []
        accept_rates = []
        energy_curves = []
        core_violations_before = []
        core_violations_after = []

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

            # Core violation before
            core_viol_b = compute_core_violation_rate(model, z_init, device)
            core_violations_before.append(core_viol_b)

            # Inference
            t0 = time.time()
            mcmc_stats = None

            if method == 'mcmc_bit':
                z_final, mcmc_stats = mcmc_bit.run(
                    z_init, x_occ, pixel_mask, bit_mask, n_sweeps=n_sweeps
                )
            elif method == 'mcmc_block':
                z_final, mcmc_stats = mcmc_block.run(
                    z_init, x_occ, pixel_mask, bit_mask, n_sweeps=n_sweeps
                )
            elif method == 'amortized':
                z_final = amortized_inpaint(inpaint_net, z_init, bit_mask, device)
            elif method.startswith('iterative_'):
                n_steps = int(method.split('_')[1])
                z_final = iterative_inpaint(
                    inpaint_net, z_init, bit_mask, n_steps, device
                )
            else:
                z_final = z_init

            rt = (time.time() - t0) * 1000

            with torch.no_grad():
                pred_a = model.classifier(z_final.unsqueeze(0)).argmax(1).item()
                o_hat_a = model.decode(z_final.unsqueeze(0))[0, 0]

            # Core violation after
            core_viol_a = compute_core_violation_rate(model, z_final, device)
            core_violations_after.append(core_viol_a)

            # Metrics
            x_clean_t = torch.from_numpy(x_clean).to(device)
            occ_t = torch.from_numpy(occ_pixels).to(device)
            occ_sum = occ_t.sum().clamp(min=1.0).item()

            bce_b = compute_occluded_bce(o_hat_b, x_clean_t, occ_t, occ_sum)
            bce_a = compute_occluded_bce(o_hat_a, x_clean_t, occ_t, occ_sum)

            correct_before.append(int(pred_b == label))
            correct_after.append(int(pred_a == label))
            bce_before_list.append(bce_b)
            bce_after_list.append(bce_a)
            runtimes.append(rt)

            if mcmc_stats:
                accept_rates.append(mcmc_stats['mean_accept'])
                energy_curves.append(mcmc_stats['energies'])

        # Aggregate
        cb = np.array(correct_before)
        ca = np.array(correct_after)
        delta_acc = ca - cb
        delta_bce = np.array(bce_after_list) - np.array(bce_before_list)

        corr_val = 0.0
        if np.std(delta_bce) > 1e-12 and np.std(delta_acc) > 1e-12:
            corr_val = float(np.corrcoef(delta_bce, delta_acc)[0, 1])

        n = len(eval_idx)
        result = {
            'method': method,
            'mask_type': mask_type,
            'noise_type': noise_type,
            'acc_before': cb.mean(),
            'acc_after': ca.mean(),
            'delta_acc': (ca.sum() - cb.sum()) / n,
            'bce_before': np.mean(bce_before_list),
            'bce_after': np.mean(bce_after_list),
            'delta_bce': np.mean(delta_bce),
            'corr_dbce_dacc': corr_val,
            'runtime_ms': np.mean(runtimes),
            'bit_mask_ratio': float(bit_mask_ratio),
            'core_viol_before': np.mean(core_violations_before),
            'core_viol_after': np.mean(core_violations_after),
            'n_samples': n,
        }

        if accept_rates:
            result['accept_rate'] = np.mean(accept_rates)
            # Energy curve: first, middle, last
            if energy_curves:
                avg_curve = np.mean(
                    [c for c in energy_curves if len(c) > 0], axis=0
                ).tolist()
                result['energy_start'] = avg_curve[0] if avg_curve else 0
                result['energy_end'] = avg_curve[-1] if avg_curve else 0
                result['energy_drop'] = (avg_curve[0] - avg_curve[-1]) if avg_curve else 0
        else:
            result['accept_rate'] = None
            result['energy_start'] = None
            result['energy_end'] = None
            result['energy_drop'] = None

        all_results.append(result)
        print(f" Δacc={result['delta_acc']:+.1%}, "
              f"bce={result['bce_before']:.2f}→{result['bce_after']:.2f}, "
              f"t={result['runtime_ms']:.1f}ms"
              + (f", acc_rate={result['accept_rate']:.2f}" if result['accept_rate'] is not None else ""))

    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 1: MCMC Block + Contract')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--train_samples', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--inpaint_epochs', type=int, default=20)
    parser.add_argument('--n_sweeps', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='outputs/exp_phase1')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 1: MCMC Block Proposal + Paradigm Contract Validation")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"E_obs: BCE (fixed — geometrically matched to sigmoid decoder)")
    print(f"L_cls: REMOVED from training (classifier is probe only)")
    print(f"Eval samples: {args.eval_samples}")
    print(f"MCMC sweeps: {args.n_sweeps}")
    print()

    # Load data
    print("[1] Loading data...")
    train_x, train_y, test_x, test_y = load_data(
        args.train_samples, 1000, args.seed
    )
    print(f"    Train: {len(train_x)}, Test: {len(test_x)}")

    # Train model (BCE, no L_cls)
    print("\n[2] Training Route C model (BCE E_obs, NO L_cls)...")
    model = train_model_bce(train_x, train_y, device, epochs=args.epochs)
    model.eval()

    # Baseline accuracy
    with torch.no_grad():
        z_test = model.encode(test_x[:500].to(device))
        preds = model.classifier(z_test).argmax(1).cpu()
        clean_acc = (preds == test_y[:500]).float().mean().item()
    print(f"    Clean accuracy (probe): {clean_acc:.1%}")

    # Train InpaintNet (mask + core + obs, NO cls)
    print("\n[3] Training InpaintNet (L_mask + L_core + L_obs(BCE), NO L_cls)...")
    inpaint_net = train_inpaint_maskonly(
        model, train_x, device, epochs=args.inpaint_epochs
    )

    # Phase 1 experiment matrix
    configs = [
        ('center', 'clean'),
        ('center', 'noise'),
        ('stripes', 'clean'),
        ('stripes', 'noise'),
    ]
    methods = ['mcmc_bit', 'mcmc_block', 'amortized',
               'iterative_1', 'iterative_2', 'iterative_4']

    print(f"\n[4] Running Phase 1 evaluation...")
    print(f"    Configs: {len(configs)}")
    print(f"    Methods: {methods}")
    print(f"    Total runs: {len(configs) * len(methods)}")

    all_results = []
    for mask_type, noise_type in configs:
        print(f"\n    --- {mask_type} + {noise_type} ---")
        results = evaluate_config(
            model, inpaint_net, test_x, test_y,
            mask_type, noise_type, methods, device,
            n_samples=args.eval_samples, n_sweeps=args.n_sweeps,
            seed=args.seed,
        )
        all_results.extend(results)

    # ── Summary tables ──
    print("\n" + "=" * 140)
    print("PHASE 1 SUMMARY")
    print("=" * 140)

    header = (f"{'method':<16} {'mask':<10} {'noise':<8} "
              f"{'acc_bef':>8} {'acc_aft':>8} {'Δacc':>7} "
              f"{'bce_bef':>8} {'bce_aft':>8} {'Δbce':>8} "
              f"{'corr':>7} {'viol_b':>7} {'viol_a':>7} "
              f"{'acc_r':>6} {'E_drop':>7} {'ms':>8}")
    print(header)
    print("-" * 140)

    for r in all_results:
        acc_r = f"{r['accept_rate']:.2f}" if r['accept_rate'] is not None else "  n/a"
        e_drop = f"{r['energy_drop']:.1f}" if r['energy_drop'] is not None else "   n/a"
        print(f"{r['method']:<16} {r['mask_type']:<10} {r['noise_type']:<8} "
              f"{r['acc_before']:>8.1%} {r['acc_after']:>8.1%} {r['delta_acc']:>+7.1%} "
              f"{r['bce_before']:>8.2f} {r['bce_after']:>8.2f} {r['delta_bce']:>+8.3f} "
              f"{r['corr_dbce_dacc']:>+7.3f} "
              f"{r['core_viol_before']:>7.3f} {r['core_viol_after']:>7.3f} "
              f"{acc_r:>6} {e_drop:>7} {r['runtime_ms']:>8.1f}")

    # ── MCMC comparison ──
    print("\n" + "=" * 100)
    print("MCMC COMPARISON: bit-flip vs block (informed proposal)")
    print("=" * 100)

    for mask_type, noise_type in configs:
        mcmc_bit_res = next((r for r in all_results
                             if r['method'] == 'mcmc_bit'
                             and r['mask_type'] == mask_type
                             and r['noise_type'] == noise_type), None)
        mcmc_block_res = next((r for r in all_results
                               if r['method'] == 'mcmc_block'
                               and r['mask_type'] == mask_type
                               and r['noise_type'] == noise_type), None)
        if mcmc_bit_res and mcmc_block_res:
            print(f"\n  {mask_type} + {noise_type}:")
            print(f"    MCMC_bit:   Δacc={mcmc_bit_res['delta_acc']:+.1%}, "
                  f"accept={mcmc_bit_res['accept_rate']:.2f}, "
                  f"E_drop={mcmc_bit_res['energy_drop']:.1f}, "
                  f"core_viol={mcmc_bit_res['core_viol_after']:.3f}")
            print(f"    MCMC_block: Δacc={mcmc_block_res['delta_acc']:+.1%}, "
                  f"accept={mcmc_block_res['accept_rate']:.2f}, "
                  f"E_drop={mcmc_block_res['energy_drop']:.1f}, "
                  f"core_viol={mcmc_block_res['core_viol_after']:.3f}")
            improved = mcmc_block_res['delta_acc'] > mcmc_bit_res['delta_acc']
            print(f"    Block better? {'YES' if improved else 'NO'} "
                  f"(Δacc gap: {mcmc_block_res['delta_acc'] - mcmc_bit_res['delta_acc']:+.1%})")

    # ── Speed comparison ──
    print("\n" + "=" * 100)
    print("SPEED COMPARISON (center + clean)")
    print("=" * 100)
    for method in methods:
        r = next((r for r in all_results
                  if r['method'] == method
                  and r['mask_type'] == 'center'
                  and r['noise_type'] == 'clean'), None)
        if r:
            print(f"  {method:<16}: {r['runtime_ms']:>8.1f} ms/sample")

    # ── Key diagnostic: does MCMC_block "not harm"? ──
    print("\n" + "=" * 100)
    print("KEY DIAGNOSTIC: Is MCMC_block 'at least not harmful'?")
    print("=" * 100)
    for mask_type, noise_type in configs:
        r = next((r for r in all_results
                  if r['method'] == 'mcmc_block'
                  and r['mask_type'] == mask_type
                  and r['noise_type'] == noise_type), None)
        if r:
            status = "STABLE" if r['delta_acc'] >= -0.02 else "HARMFUL"
            print(f"  {mask_type}+{noise_type}: Δacc={r['delta_acc']:+.1%} → {status}")

    # Save
    csv_path = os.path.join(args.output_dir, "phase1_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    # Save paradigm contract
    contract = {
        'representation': {
            'encode_output': '(k=8, H=7, W=7) binary via Gumbel-Sigmoid STE',
            'bit_mask_policy': 'any',
            'k': 8, 'H': 7, 'W': 7,
        },
        'observation_protocol': {
            'E_obs': 'BCE (geometrically matched to sigmoid decoder)',
            'future': 'Continuous Bernoulli, token-likelihood (placeholder)',
        },
        'inference_operators': {
            'amortized': 'InpaintNet single pass (L_mask + L_core + L_obs, NO L_cls)',
            'iterative': 'MaskGIT-style cosine schedule, steps=1..4',
            'mcmc_bit': 'bit-flip block Gibbs (2×2 spatial), DIAGNOSTIC ONLY',
            'mcmc_block': 'token-level Gibbs with local-pred proposal, DIAGNOSTIC/TEACHER',
        },
        'probe': {
            'classifier': 'Linear(392, 10), trained separately, frozen during world model training',
            'note': 'Classification accuracy is a PROBE, not an objective',
        },
    }
    contract_path = os.path.join(args.output_dir, "paradigm_contract.json")
    with open(contract_path, 'w') as f:
        json.dump(contract, f, indent=2)
    print(f"Contract saved to {contract_path}")

    print("\n" + "=" * 100)
    print("Phase 1 complete.")
    print("=" * 100)

    return all_results


if __name__ == "__main__":
    main()
