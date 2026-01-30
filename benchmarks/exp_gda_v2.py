#!/usr/bin/env python3
"""
Evidence-gated Iterative GDA + Mask Mixture Training
======================================================
Three targeted fixes to make GDA work on stripes:

Fix 1: Evidence gating — only high-confidence observed tokens serve as K/V
       in GDA. Masked/low-conf tokens don't pollute global mixing.

Fix 2: Learned mask token — masked positions get a learnable embedding
       instead of code=0, preventing Hamming distance collapse.

Fix 3: Mask mixture training — random:center:stripes:multi_hole = 0.4:0.2:0.2:0.2
       with randomized stripe parameters (width, gap, phase, direction).

Architecture: Iterative GDA (4 steps)
  Each step t:
    1. Compute confidence from current predictions
    2. Build memory bank = observed ∪ high-conf predicted tokens
    3. GDA: query=all positions, K/V=memory bank only (evidence-gated)
    4. Local CNN + GDA context → updated predictions
    5. MaskGIT unmask schedule: reveal top-confidence positions

Usage:
    python3 -u benchmarks/exp_gda_v2.py --device cuda --seed 42
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))


# ============================================================================
# MODEL (reuse — BCE, no L_cls)
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
    def forward(self, x): return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z): return self.deconv(z)


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
        windows = windows.reshape(B, k * 9, H * W).permute(0, 2, 1)
        logits = self.net(windows)
        return logits.permute(0, 2, 1).reshape(B, k, H, W)


class Classifier(nn.Module):
    def __init__(self, n_bits, latent_size=7, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(n_bits * latent_size * latent_size, n_classes)
    def forward(self, z): return self.fc(z.reshape(z.shape[0], -1))


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

    def encode(self, x): return self.quantizer(self.encoder(x))
    def decode(self, z): return self.decoder(z)
    def set_temperature(self, tau): self.quantizer.set_temperature(tau)

    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z), self.classifier(z), self.local_pred(z)


# ============================================================================
# EVIDENCE-GATED GDA
# ============================================================================

class EvidenceGatedGDA(nn.Module):
    """
    Global Discrete Attention with evidence gating.

    Key difference from v1: masked/low-confidence positions are EXCLUDED
    from the K/V memory bank. Only observed + high-confidence tokens
    contribute to global context, preventing noise propagation.

    Also uses a learned mask embedding instead of code=0 for masked positions,
    preventing Hamming distance collapse.
    """

    def __init__(self, k: int = 8, d_v: int = 32, temperature: float = 1.0):
        super().__init__()
        self.k = k
        self.d_v = d_v
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Popcount LUT
        pop_lut = torch.tensor([bin(i).count("1") for i in range(256)],
                               dtype=torch.float32)
        self.register_buffer('pop_lut', pop_lut)

        # Value embedding: code → R^{d_v}
        self.value_embed = nn.Embedding(256, d_v)

        # Learned mask token embedding (Fix 2)
        # When a position is masked, use this instead of code=0
        self.mask_value = nn.Parameter(torch.randn(d_v) * 0.1)

    def bitpack(self, z: torch.Tensor) -> torch.Tensor:
        B, k, H, W = z.shape
        bits = (z > 0.5).long()
        shifts = (2 ** torch.arange(k, device=z.device)).view(1, k, 1, 1)
        codes = (bits * shifts).sum(dim=1)  # (B, H, W)
        return codes.reshape(B, H * W)  # (B, N)

    def forward(self, z: torch.Tensor, evidence_mask: torch.Tensor) -> torch.Tensor:
        """
        Evidence-gated global attention.

        Args:
            z: (B, k, H, W) — current binary latent (may include predictions)
            evidence_mask: (B, N) bool — True where position has evidence
                           (observed or high-confidence prediction)

        Returns:
            ctx: (B, d_v, H, W) — global context, gated by evidence
        """
        B, k, H, W = z.shape
        N = H * W

        # Bitpack
        codes = self.bitpack(z)  # (B, N)

        # Hamming distance matrix
        xor = codes.unsqueeze(2) ^ codes.unsqueeze(1)  # (B, N, N)
        dist = self.pop_lut[xor]  # (B, N, N)

        # Attention scores
        scores = -dist / self.temperature.clamp(min=0.1)  # (B, N, N)

        # Evidence gating (Fix 1): mask out non-evidence K positions
        # evidence_mask: (B, N) → (B, 1, N) for broadcasting
        # Where evidence_mask is False, set score to -inf so softmax gives 0
        gate = evidence_mask.unsqueeze(1).expand(-1, N, -1)  # (B, N, N)
        scores = scores.masked_fill(~gate, float('-inf'))

        # Handle rows where ALL K positions are masked (no evidence at all)
        # In this case softmax would be nan; replace with uniform
        all_masked = ~evidence_mask.unsqueeze(1).expand(-1, N, -1).any(dim=-1)
        scores[all_masked] = 0.0  # uniform fallback

        attn = F.softmax(scores, dim=-1)  # (B, N, N)

        # Values: evidence positions get code embedding, masked get learned token
        values = self.value_embed(codes.clamp(0, 255))  # (B, N, d_v)
        # Replace non-evidence values with mask_value (Fix 2)
        no_evidence = ~evidence_mask  # (B, N)
        # For each non-evidence position, replace its value embedding with learned mask_value
        mask_val_expanded = self.mask_value.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        values = torch.where(no_evidence.unsqueeze(-1), mask_val_expanded, values)

        # Context
        ctx = torch.bmm(attn, values)  # (B, N, d_v)
        return ctx.permute(0, 2, 1).reshape(B, self.d_v, H, W)


# ============================================================================
# INPAINTING NET V3: Iterative Evidence-gated GDA
# ============================================================================

class InpaintNetV1(nn.Module):
    """Baseline: local CNN only."""
    def __init__(self, k=8, hidden=64):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Conv2d(k + 1, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(hidden, k, 3, padding=1, padding_mode='circular'),
        )
        self.skip = nn.Conv2d(k + 1, k, 1)

    def forward(self, z_masked, mask):
        x = torch.cat([z_masked, mask], dim=1)
        return self.net(x) + self.skip(x)


class InpaintNetV3(nn.Module):
    """
    InpaintNet with Evidence-gated GDA.

    Architecture:
      Local branch: Conv(k+1 → hidden) × 2
      Global branch: EvidenceGatedGDA
      Fusion: Conv(hidden + d_v → hidden → k) + skip
    """

    def __init__(self, k=8, hidden=64, d_v=32, temperature=1.0):
        super().__init__()
        self.k = k

        self.local_branch = nn.Sequential(
            nn.Conv2d(k + 1, hidden, 3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
        )

        self.gda = EvidenceGatedGDA(k=k, d_v=d_v, temperature=temperature)

        self.fusion = nn.Sequential(
            nn.Conv2d(hidden + d_v, hidden, 3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, k, 3, padding=1, padding_mode='circular'),
        )

        self.skip = nn.Conv2d(k + 1, k, 1)

    def forward(self, z_masked, mask, evidence_mask=None):
        """
        Args:
            z_masked: (B, k, H, W)
            mask: (B, 1, H, W) — 1=masked, 0=observed
            evidence_mask: (B, N) bool — True=has evidence. If None, uses ~mask.

        Returns:
            logits: (B, k, H, W)
        """
        B, k, H, W = z_masked.shape

        if evidence_mask is None:
            # Default: observed positions are evidence
            evidence_mask = (mask[:, 0] < 0.5).reshape(B, H * W)  # (B, N)

        x = torch.cat([z_masked, mask], dim=1)
        f_local = self.local_branch(x)
        f_global = self.gda(z_masked, evidence_mask)
        combined = torch.cat([f_local, f_global], dim=1)
        return self.fusion(combined) + self.skip(x)


# ============================================================================
# ITERATIVE INFERENCE WITH EVIDENCE GATING
# ============================================================================

def iterative_inpaint_gated(
    inpaint_net,  # InpaintNetV3
    z_init: torch.Tensor,
    bit_mask: np.ndarray,
    n_steps: int = 4,
    device: torch.device = None,
    conf_threshold: float = 0.7,
) -> torch.Tensor:
    """
    MaskGIT-style iterative with evidence-gated GDA.

    Each step:
      1. Build evidence_mask = observed ∪ {predicted with conf > threshold}
      2. Forward pass with evidence gating
      3. Unmask top-confidence positions (cosine schedule)

    Args:
        conf_threshold: minimum |σ(logit)-0.5|*2 to count as evidence
    """
    if device is None:
        device = next(inpaint_net.parameters()).device

    inpaint_net.eval()
    k, H, W = z_init.shape
    N = H * W
    z = z_init.clone().to(device)

    current_mask = torch.from_numpy(bit_mask).bool().to(device)  # (k, H, W)
    total_masked = current_mask.sum().item()
    # Per-position mask (all bits share)
    pos_mask = current_mask[0]  # (H, W)
    observed = ~pos_mask  # (H, W) — True where observed

    if total_masked == 0:
        return z

    with torch.no_grad():
        for step in range(n_steps):
            # Cosine unmask schedule
            progress = (step + 1) / n_steps
            fraction_to_unmask = 1.0 - np.cos(progress * np.pi / 2)
            n_to_unmask = int(fraction_to_unmask * total_masked // k)
            n_to_unmask = min(n_to_unmask, pos_mask.sum().item())

            if current_mask.sum().item() == 0:
                break

            # Build evidence mask: observed + high-confidence predicted
            # Start with observed positions
            evidence = observed.clone().reshape(N)  # (N,)

            if step > 0:
                # Add high-confidence predictions from previous step
                # confidence = how far sigmoid(logit) is from 0.5
                conf_per_bit = (prev_probs - 0.5).abs() * 2  # (k, H, W)
                conf_per_pos = conf_per_bit.mean(dim=0)  # (H, W) — avg across bits
                high_conf = conf_per_pos.reshape(N) > conf_threshold
                evidence = evidence | high_conf

            evidence_batch = evidence.unsqueeze(0)  # (1, N)

            # Prepare input
            mask_spatial = current_mask.float().max(dim=0, keepdim=True)[0].unsqueeze(0)
            z_masked = z.unsqueeze(0) * (1 - current_mask.float().unsqueeze(0))

            # Forward with evidence gating
            logits = inpaint_net(z_masked, mask_spatial, evidence_batch)
            probs = torch.sigmoid(logits[0])  # (k, H, W)
            prev_probs = probs  # save for next step's confidence

            predictions = (probs > 0.5).float()
            confidence = (probs - 0.5).abs() * 2  # (k, H, W)

            # Per-position confidence (average over bits)
            pos_confidence = confidence.mean(dim=0)  # (H, W)

            # Only consider currently masked positions
            pos_conf_masked = pos_confidence.clone()
            pos_conf_masked[~pos_mask] = -1.0

            # Unmask top-confidence positions
            flat_conf = pos_conf_masked.flatten()
            flat_pos_mask = pos_mask.flatten()

            if n_to_unmask > 0 and flat_pos_mask.sum() > 0:
                sorted_indices = flat_conf.argsort(descending=True)
                count = 0
                for idx in sorted_indices:
                    if count >= n_to_unmask:
                        break
                    if flat_pos_mask[idx]:
                        flat_pos_mask[idx] = False
                        # Write prediction for all k bits at this position
                        h_idx = idx.item() // W
                        w_idx = idx.item() % W
                        z[:, h_idx, w_idx] = predictions[:, h_idx, w_idx]
                        count += 1

                pos_mask = flat_pos_mask.reshape(H, W)
                current_mask = pos_mask.unsqueeze(0).expand(k, -1, -1)

        # Final pass
        if current_mask.sum().item() > 0:
            evidence = (~pos_mask).reshape(N).unsqueeze(0)
            mask_spatial = current_mask.float().max(dim=0, keepdim=True)[0].unsqueeze(0)
            z_masked = z.unsqueeze(0) * (1 - current_mask.float().unsqueeze(0))
            logits = inpaint_net(z_masked, mask_spatial, evidence)
            predictions = (torch.sigmoid(logits[0]) > 0.5).float()
            z[current_mask] = predictions[current_mask]

    return z


# Single-pass amortized (for V1 baseline)
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


# V1 iterative (no gating, for comparison)
def iterative_inpaint_v1(inpaint_net, z_init, bit_mask, n_steps, device):
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
            fraction = 1.0 - np.cos(progress * np.pi / 2)
            n_to_unmask = int(fraction * total_masked)
            n_to_unmask = min(n_to_unmask, current_mask.sum().item())
            if current_mask.sum().item() == 0:
                break
            mask_spatial = current_mask.float().max(dim=0, keepdim=True)[0].unsqueeze(0)
            z_masked = z.unsqueeze(0) * (1 - current_mask.float().unsqueeze(0))
            logits = inpaint_net(z_masked, mask_spatial)
            probs = torch.sigmoid(logits[0])
            predictions = (probs > 0.5).float()
            confidence = (probs - 0.5).abs() * 2
            conf_masked = confidence.clone()
            conf_masked[~current_mask] = -1.0
            flat_conf = conf_masked.flatten()
            flat_mask = current_mask.flatten()
            if n_to_unmask > 0 and flat_mask.sum() > 0:
                sorted_idx = flat_conf.argsort(descending=True)
                count = 0
                for idx in sorted_idx:
                    if count >= n_to_unmask:
                        break
                    if flat_mask[idx]:
                        flat_mask[idx] = False
                        z.view(-1)[idx] = predictions.view(-1)[idx]
                        count += 1
                current_mask = flat_mask.reshape(k, H, W)
        if current_mask.sum().item() > 0:
            mask_spatial = current_mask.float().max(dim=0, keepdim=True)[0].unsqueeze(0)
            z_masked = z.unsqueeze(0) * (1 - current_mask.float().unsqueeze(0))
            logits = inpaint_net(z_masked, mask_spatial)
            preds = (torch.sigmoid(logits[0]) > 0.5).float()
            z[current_mask] = preds[current_mask]
    return z


# ============================================================================
# MASK GENERATORS (Fix 3: Mask mixture)
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


def make_random_stripe_mask(H=28, W=28, rng=None):
    """Randomized stripes: random width, gap, phase, direction."""
    if rng is None:
        rng = np.random.default_rng()
    width = rng.integers(1, 4)  # 1-3 pixels
    gap = rng.integers(4, 10)   # 4-9 pixels
    phase = rng.integers(0, gap)
    horizontal = rng.random() < 0.5

    mask = np.ones((H, W), dtype=np.float32)
    if horizontal:
        for y in range(phase, H, gap):
            mask[y:min(y + width, H), :] = 0
    else:
        for x in range(phase, W, gap):
            mask[:, x:min(x + width, W)] = 0
    return mask


def make_random_block_mask(H=28, W=28, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    mask = np.ones((H, W), dtype=np.float32)
    occ_h = rng.integers(8, 18)
    occ_w = rng.integers(8, 18)
    y = rng.integers(0, max(1, H - occ_h + 1))
    x = rng.integers(0, max(1, W - occ_w + 1))
    mask[y:y+occ_h, x:x+occ_w] = 0
    return mask


def make_multi_hole_mask(H=28, W=28, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    mask = np.ones((H, W), dtype=np.float32)
    n_holes = rng.integers(3, 8)
    for _ in range(n_holes):
        hs = rng.integers(2, 6)
        y = rng.integers(0, max(1, H - hs + 1))
        x = rng.integers(0, max(1, W - hs + 1))
        mask[y:y+hs, x:x+hs] = 0
    return mask


def sample_training_mask(H=28, W=28, rng=None):
    """
    Mix of mask types for training: random_block:center:stripes:multi_hole = 0.4:0.2:0.2:0.2
    """
    if rng is None:
        rng = np.random.default_rng()
    p = rng.random()
    if p < 0.4:
        return make_random_block_mask(H, W, rng)
    elif p < 0.6:
        return make_center_mask(H, W)
    elif p < 0.8:
        return make_random_stripe_mask(H, W, rng)
    else:
        return make_multi_hole_mask(H, W, rng)


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
        return np.clip(image + rng.normal(0, 0.1, image.shape).astype(np.float32), 0, 1)
    return image.copy()


# ============================================================================
# DATA + MODEL
# ============================================================================

def load_data(train_n=2000, test_n=1000, seed=42):
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(train_ds), train_n, replace=False)
    si = rng.choice(len(test_ds), test_n, replace=False)
    return (torch.stack([train_ds[i][0] for i in ti]),
            torch.tensor([train_ds[i][1] for i in ti]),
            torch.stack([test_ds[i][0] for i in si]),
            torch.tensor([test_ds[i][1] for i in si]))


def train_model_bce(train_x, train_y, device, epochs=5, lr=1e-3, batch_size=64):
    model = RouteCModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        tau = 1.0 + (0.2 - 1.0) * epoch / max(1, epochs - 1)
        model.set_temperature(tau)
        el, nb = 0.0, 0
        for x, y in loader:
            x = x.to(device)
            optimizer.zero_grad()
            z, x_hat, _, core_logits = model(x)
            loss_r = F.binary_cross_entropy(x_hat.clamp(1e-6, 1-1e-6), x)
            m = torch.rand_like(z) < 0.15
            loss_c = F.binary_cross_entropy_with_logits(
                core_logits[m], z.detach()[m]) if m.any() else torch.tensor(0., device=device)
            loss = loss_r + 0.5 * loss_c
            loss.backward(); optimizer.step()
            el += loss.item(); nb += 1
        print(f"    Epoch {epoch+1}/{epochs}: loss={el/max(nb,1):.4f}")

    # Probe classifier
    for p in model.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True
    co = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    for _ in range(3):
        model.classifier.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): z = model.encode(x)
            F.cross_entropy(model.classifier(z), y).backward()
            co.step(); co.zero_grad()
    for p in model.parameters(): p.requires_grad = True
    return model


# ============================================================================
# TRAINING: MASK MIXTURE (Fix 3)
# ============================================================================

def train_inpaint_mask_mixture(model, inpaint_net, train_x, device,
                               epochs=20, batch_size=64, lr=1e-3):
    """
    Train InpaintNet with mask MIXTURE distribution.
    L_mask only (no L_cls/L_obs/L_core) — isolates architecture contribution.

    Mask distribution: random_block 40%, center 20%, stripes 20%, multi_hole 20%
    Stripes are randomized (width, gap, phase, direction).
    """
    optimizer = torch.optim.Adam(inpaint_net.parameters(), lr=lr)
    model.eval()
    N = len(train_x)
    rng = np.random.default_rng(42)

    for epoch in range(epochs):
        inpaint_net.train()
        perm = torch.randperm(N)
        total_loss = 0.0
        nb = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            x = train_x[idx].to(device)

            with torch.no_grad():
                z = model.encode(x)
                z_hard = (z > 0.5).float()

            B, k, H, W = z_hard.shape

            # Per-sample mask from mixture distribution
            # Convert pixel masks to latent-space masks
            masks = []
            for b in range(B):
                pm = sample_training_mask(28, 28, rng)
                bm = pixel_to_bit_mask(pm, k, H)
                masks.append(torch.from_numpy(bm).float())

            bit_masks = torch.stack(masks).to(device)  # (B, k, H, W)
            pos_masks = bit_masks[:, 0:1, :, :]  # (B, 1, H, W)

            z_masked = z_hard * (1 - bit_masks)

            # Forward — handle V3 with evidence_mask
            if hasattr(inpaint_net, 'gda'):
                # V3: pass evidence_mask (observed positions)
                evidence = (pos_masks[:, 0] < 0.5).reshape(B, H * W)
                logits = inpaint_net(z_masked, pos_masks, evidence)
            else:
                logits = inpaint_net(z_masked, pos_masks)

            loss = F.binary_cross_entropy_with_logits(
                logits[bit_masks.bool()], z_hard[bit_masks.bool()]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            nb += 1

        if (epoch + 1) % 5 == 0:
            print(f"      epoch {epoch+1}/{epochs}: loss={total_loss/max(nb,1):.4f}")

    return inpaint_net


def train_inpaint_random_only(model, inpaint_net, train_x, device,
                              epochs=20, batch_size=64, lr=1e-3):
    """Original: random ratio mask only (for comparison)."""
    optimizer = torch.optim.Adam(inpaint_net.parameters(), lr=lr)
    model.eval()
    N = len(train_x)

    for epoch in range(epochs):
        inpaint_net.train()
        perm = torch.randperm(N)
        tl, nb = 0.0, 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            x = train_x[idx].to(device)
            with torch.no_grad():
                z = model.encode(x)
                z_hard = (z > 0.5).float()
            B, k, H, W = z_hard.shape
            ratio = 0.1 + torch.rand(B, 1, 1, 1, device=device) * 0.6
            mask = (torch.rand(B, 1, H, W, device=device) < ratio).float()
            mask_exp = mask.expand(-1, k, -1, -1)
            z_masked = z_hard * (1 - mask_exp)

            if hasattr(inpaint_net, 'gda'):
                evidence = (mask[:, 0] < 0.5).reshape(B, H * W)
                logits = inpaint_net(z_masked, mask, evidence)
            else:
                logits = inpaint_net(z_masked, mask)

            loss = F.binary_cross_entropy_with_logits(
                logits[mask_exp.bool()], z_hard[mask_exp.bool()])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tl += loss.item(); nb += 1

        if (epoch + 1) % 5 == 0:
            print(f"      epoch {epoch+1}/{epochs}: loss={tl/max(nb,1):.4f}")

    return inpaint_net


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, inpaint_net, method_name, test_x, test_y,
             mask_type, noise_type, device, n_samples=100, seed=42,
             use_gated_iter=False, n_iter_steps=4):
    model.eval()
    inpaint_net.eval()

    pm = make_center_mask() if mask_type == 'center' else make_stripe_mask()
    bm = pixel_to_bit_mask(pm)
    occ = 1 - pm

    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    cb_list, ca_list, bce_b, bce_a, mse_b, mse_a, rts = [], [], [], [], [], [], []

    for idx in eval_idx:
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()
        x_noisy = apply_noise(x_clean, noise_type, rng)
        x_occ = x_noisy * pm

        with torch.no_grad():
            x_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_init = model.encode(x_t)[0]
            pred_b = model.classifier(z_init.unsqueeze(0)).argmax(1).item()
            o_hat_b = model.decode(z_init.unsqueeze(0))[0, 0]

        t0 = time.time()
        if use_gated_iter:
            z_final = iterative_inpaint_gated(
                inpaint_net, z_init, bm, n_steps=n_iter_steps, device=device)
        elif 'iterative' in method_name:
            z_final = iterative_inpaint_v1(
                inpaint_net, z_init, bm, n_steps=n_iter_steps, device=device)
        else:
            z_final = amortized_inpaint(inpaint_net, z_init, bm, device)
        rt = (time.time() - t0) * 1000

        with torch.no_grad():
            pred_a = model.classifier(z_final.unsqueeze(0)).argmax(1).item()
            o_hat_a = model.decode(z_final.unsqueeze(0))[0, 0]

        xt = torch.from_numpy(x_clean).to(device)
        ot = torch.from_numpy(occ).to(device)
        os_ = ot.sum().clamp(min=1.0).item()

        def om(oh):
            d = (oh - xt) * ot; return (d*d).sum().item() / os_
        def ob(oh):
            l = oh.clamp(1e-6,1-1e-6)
            return (-(xt*torch.log(l)+(1-xt)*torch.log(1-l))*ot).sum().item()/os_

        cb_list.append(int(pred_b == label))
        ca_list.append(int(pred_a == label))
        mse_b.append(om(o_hat_b)); mse_a.append(om(o_hat_a))
        bce_b.append(ob(o_hat_b)); bce_a.append(ob(o_hat_a))
        rts.append(rt)

    cb, ca = np.array(cb_list), np.array(ca_list)
    n = len(eval_idx)
    return {
        'method': method_name,
        'mask_type': mask_type,
        'noise_type': noise_type,
        'acc_before': cb.mean(),
        'acc_after': ca.mean(),
        'delta_acc': (ca.sum() - cb.sum()) / n,
        'bce_before': np.mean(bce_b),
        'bce_after': np.mean(bce_a),
        'mse_before': np.mean(mse_b),
        'mse_after': np.mean(mse_a),
        'runtime_ms': np.mean(rts),
        'bit_mask_ratio': float(bm[0].mean()),
        'n_samples': n,
    }


# ============================================================================
# MAIN
# ============================================================================

def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--train_samples', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--inpaint_epochs', type=int, default=20)
    parser.add_argument('--output_dir', default='outputs/exp_gda_v2')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 100)
    print("EVIDENCE-GATED GDA + MASK MIXTURE TRAINING")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Fixes: evidence gating + learned mask token + mask mixture training")
    print()

    print("[1] Loading data...")
    train_x, train_y, test_x, test_y = load_data(args.train_samples, 1000, args.seed)

    print("\n[2] Training Route C model (BCE, no L_cls)...")
    model = train_model_bce(train_x, train_y, device, epochs=args.epochs)
    model.eval()
    with torch.no_grad():
        z_t = model.encode(test_x[:500].to(device))
        clean_acc = (model.classifier(z_t).argmax(1).cpu() == test_y[:500]).float().mean().item()
    print(f"    Clean accuracy: {clean_acc:.1%}")

    # Create all 4 variants
    configs = {
        'v1_rand':   ('V1 + random mask',       InpaintNetV1(k=8).to(device), 'random'),
        'v1_mix':    ('V1 + mask mixture',       InpaintNetV1(k=8).to(device), 'mixture'),
        'v3_rand':   ('V3(gated) + random mask', InpaintNetV3(k=8).to(device), 'random'),
        'v3_mix':    ('V3(gated) + mask mixture', InpaintNetV3(k=8).to(device), 'mixture'),
    }

    print(f"\n    Model sizes:")
    for name, (desc, net, _) in configs.items():
        print(f"      {name}: {count_params(net):,} params")

    for name, (desc, net, mask_train) in configs.items():
        print(f"\n[3] Training {desc}...")
        # Reset seed for fair comparison
        torch.manual_seed(args.seed)
        if mask_train == 'mixture':
            net = train_inpaint_mask_mixture(
                model, net, train_x, device, epochs=args.inpaint_epochs)
        else:
            net = train_inpaint_random_only(
                model, net, train_x, device, epochs=args.inpaint_epochs)
        configs[name] = (desc, net, mask_train)

    # Evaluate all variants
    eval_configs = [
        ('center', 'clean'),
        ('center', 'noise'),
        ('stripes', 'clean'),
        ('stripes', 'noise'),
    ]

    # Methods to evaluate per InpaintNet:
    # - amortized (single pass)
    # - iterative_4 (4-step MaskGIT)
    # For V3: also iterative_4_gated (with evidence gating in each step)

    print(f"\n[4] Evaluating...")
    all_results = []

    for mt, nt in eval_configs:
        print(f"\n  === {mt} + {nt} ===")
        for name, (desc, net, mask_train) in configs.items():
            is_v3 = name.startswith('v3')

            # Amortized
            r = evaluate(model, net, f"{name}_amort", test_x, test_y,
                         mt, nt, device, args.eval_samples, args.seed)
            all_results.append(r)
            print(f"    {name}_amort:    Δacc={r['delta_acc']:+.1%}, "
                  f"bce={r['bce_after']:.2f}, t={r['runtime_ms']:.1f}ms")

            # Iterative 4 (standard, no gating even for V3)
            r = evaluate(model, net, f"{name}_iter4", test_x, test_y,
                         mt, nt, device, args.eval_samples, args.seed,
                         use_gated_iter=False, n_iter_steps=4)
            all_results.append(r)
            print(f"    {name}_iter4:    Δacc={r['delta_acc']:+.1%}, "
                  f"bce={r['bce_after']:.2f}, t={r['runtime_ms']:.1f}ms")

            # V3 only: gated iterative
            if is_v3:
                r = evaluate(model, net, f"{name}_giter4", test_x, test_y,
                             mt, nt, device, args.eval_samples, args.seed,
                             use_gated_iter=True, n_iter_steps=4)
                all_results.append(r)
                print(f"    {name}_giter4:   Δacc={r['delta_acc']:+.1%}, "
                      f"bce={r['bce_after']:.2f}, t={r['runtime_ms']:.1f}ms")

    # Summary
    print("\n" + "=" * 130)
    print("FULL SUMMARY")
    print("=" * 130)
    header = (f"{'method':<20} {'mask':<10} {'noise':<8} "
              f"{'acc_bef':>7} {'acc_aft':>7} {'Δacc':>7} "
              f"{'bce_aft':>8} {'mse_aft':>8} {'ms':>7}")
    print(header)
    print("-" * 130)
    for r in all_results:
        print(f"{r['method']:<20} {r['mask_type']:<10} {r['noise_type']:<8} "
              f"{r['acc_before']:>7.1%} {r['acc_after']:>7.1%} {r['delta_acc']:>+7.1%} "
              f"{r['bce_after']:>8.2f} {r['mse_after']:>8.4f} {r['runtime_ms']:>7.1f}")

    # Key comparison: stripes
    print("\n" + "=" * 80)
    print("KEY: STRIPES PERFORMANCE (the hard case)")
    print("=" * 80)
    for nt in ['clean', 'noise']:
        print(f"\n  stripes + {nt}:")
        stripe_results = [(r['method'], r['delta_acc'], r['bce_after'])
                          for r in all_results
                          if r['mask_type'] == 'stripes' and r['noise_type'] == nt]
        stripe_results.sort(key=lambda x: -x[1])
        for m, da, bce in stripe_results:
            status = "+" if da > 0 else ("~" if da > -0.05 else "-")
            print(f"    [{status}] {m:<20}: Δacc={da:+.1%}, bce={bce:.2f}")

    # Center comparison
    print("\n" + "=" * 80)
    print("CENTER PERFORMANCE (should not regress)")
    print("=" * 80)
    for nt in ['clean', 'noise']:
        print(f"\n  center + {nt}:")
        center_results = [(r['method'], r['delta_acc'], r['bce_after'])
                          for r in all_results
                          if r['mask_type'] == 'center' and r['noise_type'] == nt]
        center_results.sort(key=lambda x: -x[1])
        for m, da, bce in center_results:
            print(f"    {m:<20}: Δacc={da:+.1%}, bce={bce:.2f}")

    # Save
    csv_path = os.path.join(args.output_dir, "gda_v2_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    print("\n" + "=" * 100)
    print("Experiment complete.")
    print("=" * 100)

    return all_results


if __name__ == "__main__":
    main()
