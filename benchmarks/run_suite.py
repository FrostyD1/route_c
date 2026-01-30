#!/usr/bin/env python3
"""
Route C Benchmark Suite
========================
Evaluates baseline, D-RoPE, learned D-RoPE, and amortized inpainting across
multiple mask types, noise conditions, and OOD settings.

Usage:
    python3 -u route_c/benchmarks/run_suite.py --model all --device cuda --seed 42
    python3 -u route_c/benchmarks/run_suite.py --model baseline --device cpu --seed 42

Output: CSV table + console summary
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
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional

# Setup path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw): return x


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class SuiteConfig:
    # Data (reduced for fast trend-checking; scale up for final results)
    train_samples: int = 2000
    test_samples: int = 1000
    eval_samples: int = 100   # Per configuration
    batch_size: int = 64

    # Model architecture (must match exp09)
    n_bits: int = 8
    latent_size: int = 7
    hidden_dim: int = 64
    energy_hidden: int = 32

    # Training
    epochs: int = 5
    lr: float = 1e-3
    tau_start: float = 1.0
    tau_end: float = 0.2
    alpha_recon: float = 1.0
    beta_core: float = 0.5
    gamma_cls: float = 1.0

    # Inference
    n_sweeps: int = 30
    block_size: Tuple[int, int] = (2, 2)
    lambda_core: float = 1.0
    lambda_obs: float = 1.0
    lambda_rope: float = 0.5

    # Inpainting
    inpaint_epochs: int = 20
    inpaint_hidden: int = 64
    inpaint_lr: float = 1e-3

    # Learned routing
    learned_gate_epochs: int = 10
    learned_gate_lr: float = 1e-3

    # Bit mask policy: 'any' (conservative) | 'majority' | 'soft' (future)
    bitmask_policy: str = 'any'

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs/benchmark"


# ============================================================================
# IMPORT MODELS (re-use exp09 model architecture)
# ============================================================================

# We replicate the model classes here to avoid import issues
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
    def __init__(self, cfg):
        super().__init__()
        self.n_bits = cfg.n_bits
        self.encoder = Encoder(cfg.n_bits, cfg.hidden_dim)
        self.quantizer = GumbelSigmoid(cfg.tau_start)
        self.decoder = Decoder(cfg.n_bits, cfg.hidden_dim)
        self.local_pred = LocalPredictor(cfg.n_bits, cfg.energy_hidden)
        self.classifier = Classifier(cfg.n_bits, cfg.latent_size)

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
# MASK GENERATORS
# ============================================================================

def make_center_mask(H: int, W: int, occ_h: int = 14, occ_w: int = 14) -> np.ndarray:
    """Center occlusion mask. Returns pixel mask (1=visible, 0=occluded)."""
    mask = np.ones((H, W), dtype=np.float32)
    y, x = (H - occ_h) // 2, (W - occ_w) // 2
    mask[y:y+occ_h, x:x+occ_w] = 0
    return mask


def make_random_mask(H: int, W: int, occ_h: int = 14, occ_w: int = 14, rng=None) -> np.ndarray:
    """Random position block occlusion."""
    if rng is None:
        rng = np.random.default_rng(42)
    mask = np.ones((H, W), dtype=np.float32)
    y = rng.integers(0, max(1, H - occ_h + 1))
    x = rng.integers(0, max(1, W - occ_w + 1))
    mask[y:y+occ_h, x:x+occ_w] = 0
    return mask


def make_multi_hole_mask(H: int, W: int, n_holes: int = 5, hole_size: int = 4, rng=None) -> np.ndarray:
    """Multiple small holes."""
    if rng is None:
        rng = np.random.default_rng(42)
    mask = np.ones((H, W), dtype=np.float32)
    for _ in range(n_holes):
        y = rng.integers(0, max(1, H - hole_size + 1))
        x = rng.integers(0, max(1, W - hole_size + 1))
        mask[y:y+hole_size, x:x+hole_size] = 0
    return mask


def make_stripe_mask(H: int, W: int, stripe_width: int = 2, gap: int = 6) -> np.ndarray:
    """Horizontal stripe occlusion."""
    mask = np.ones((H, W), dtype=np.float32)
    for y in range(0, H, gap):
        mask[y:min(y+stripe_width, H), :] = 0
    return mask


MASK_GENERATORS = {
    'center': make_center_mask,
    'random': make_random_mask,
    'multi_hole': make_multi_hole_mask,
    'stripes': make_stripe_mask,
}


# ============================================================================
# NOISE GENERATORS
# ============================================================================

def apply_noise(image: np.ndarray, noise_type: str, rng=None) -> np.ndarray:
    """Apply noise/corruption to image."""
    if rng is None:
        rng = np.random.default_rng(42)

    if noise_type == 'clean':
        return image.copy()
    elif noise_type == 'blur':
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=1.0).astype(np.float32)
    elif noise_type == 'noise':
        noisy = image + rng.normal(0, 0.1, image.shape).astype(np.float32)
        return np.clip(noisy, 0, 1)
    elif noise_type == 'bias':
        return np.clip(image + 0.2, 0, 1).astype(np.float32)
    elif noise_type == 'dropout':
        drop_mask = rng.random(image.shape) > 0.2
        return (image * drop_mask).astype(np.float32)
    else:
        return image.copy()


NOISE_TYPES = ['clean', 'noise', 'bias', 'dropout']  # blur needs scipy


# ============================================================================
# BIT MASK UTILITY
# ============================================================================

def pixel_to_bit_mask(pixel_mask: np.ndarray, n_bits: int, latent_size: int = 7,
                      policy: str = 'any') -> np.ndarray:
    """Convert pixel mask to bit mask for latent grid.

    Args:
        policy: How to decide if a latent position is masked.
            'any'      — masked if ANY pixel in the patch is occluded (conservative).
                         Threshold: patch_visible_ratio < 1.0 - ε.
            'majority' — masked if MORE THAN HALF of patch pixels are occluded.
                         Threshold: patch_visible_ratio < 0.5.
            'soft'     — (future) return continuous reliability weight per position.

    The v2.0 bug used an implicit 'majority' policy with strict '<0.5', which
    caused stripes (stripe_width=2, gap=6) to produce an all-zero bit_mask —
    every 4×4 patch had exactly 50% occlusion.  v2.1 defaults to 'any'.
    """
    if policy == 'soft':
        raise NotImplementedError("Soft bitmask policy not yet implemented; use 'any' or 'majority'.")

    patch_size = 28 // latent_size
    bit_mask = np.zeros((n_bits, latent_size, latent_size), dtype=bool)

    if policy == 'any':
        threshold = 1.0 - 1e-6
    elif policy == 'majority':
        threshold = 0.5
    else:
        raise ValueError(f"Unknown bitmask policy: {policy!r}. Use 'any' or 'majority'.")

    for i in range(latent_size):
        for j in range(latent_size):
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            patch_visible_ratio = pixel_mask[y0:y1, x0:x1].mean()
            if patch_visible_ratio < threshold:
                bit_mask[:, i, j] = True
    return bit_mask


def compute_mask_ratio(pixel_mask: np.ndarray) -> float:
    """Fraction of pixels that are occluded (mask=0)."""
    return float(1.0 - pixel_mask.mean())


def compute_bit_mask_ratio(bit_mask: np.ndarray) -> float:
    """Fraction of latent positions that are masked."""
    # bit_mask shape: (n_bits, H, W) — all bits at a position share the mask
    return float(bit_mask[0].mean())


def compute_occluded_mse(o_hat, o_orig, mask):
    occ = (1 - mask)
    if occ.sum() == 0:
        return 0.0
    return ((o_hat - o_orig)**2 * occ).sum() / occ.sum()


# ============================================================================
# GOLDEN MASK TESTS — prevent regressions in mask generation / bit conversion
# ============================================================================

# Expected pixel_mask_ratio per mask type (approximate ranges).
# stripes: 5 stripes × 2 rows × 28 cols = 280/784 ≈ 0.357
# center:  14×14 / 28×28 = 0.25
# multi_hole: 5 holes × 4×4 / 784 ≈ 0.102 (varies with overlap)
# random: 14×14 / 784 = 0.25 (varies with position)
EXPECTED_MASK_RATIOS = {
    'center':     (0.20, 0.30),   # 14×14 center = 25%
    'random':     (0.15, 0.40),   # 14×14 block, random pos
    'multi_hole': (0.05, 0.20),   # 5 small holes, possible overlap
    'stripes':    (0.30, 0.40),   # ~35.7% theoretical
}

# Minimum expected bit_mask_ratio per mask type under 'any' policy.
# Under 'any' policy, even partial patch occlusion counts → these are lower bounds.
EXPECTED_BIT_MASK_RATIO_MIN = {
    'center':     0.10,
    'random':     0.05,
    'multi_hole': 0.02,
    'stripes':    0.10,  # was 0.0 under the old bug!
}


def run_golden_mask_tests(n_bits: int = 8, latent_size: int = 7,
                          policy: str = 'any', verbose: bool = True) -> bool:
    """Verify mask generation and bit conversion for all mask types.

    Returns True if all tests pass.  Raises AssertionError on failure.
    """
    all_ok = True
    rng = np.random.default_rng(12345)  # fixed seed for reproducibility

    for mask_type in ['center', 'random', 'multi_hole', 'stripes']:
        if mask_type == 'center':
            pm = make_center_mask(28, 28)
        elif mask_type == 'random':
            pm = make_random_mask(28, 28, rng=rng)
        elif mask_type == 'multi_hole':
            pm = make_multi_hole_mask(28, 28, rng=rng)
        elif mask_type == 'stripes':
            pm = make_stripe_mask(28, 28)

        bm = pixel_to_bit_mask(pm, n_bits, latent_size, policy=policy)

        px_ratio = compute_mask_ratio(pm)
        bit_ratio = compute_bit_mask_ratio(bm)

        lo, hi = EXPECTED_MASK_RATIOS[mask_type]
        min_bit = EXPECTED_BIT_MASK_RATIO_MIN[mask_type]

        ok = True
        msgs = []

        if not (lo <= px_ratio <= hi):
            ok = False
            msgs.append(f"pixel_mask_ratio={px_ratio:.3f} outside [{lo}, {hi}]")
        if bit_ratio < min_bit:
            ok = False
            msgs.append(f"bit_mask_ratio={bit_ratio:.3f} < min {min_bit}")
        if px_ratio <= 0:
            ok = False
            msgs.append("pixel_mask_ratio=0 (no occlusion)")
        if bit_ratio <= 0:
            ok = False
            msgs.append("bit_mask_ratio=0 (no latent masking — likely threshold bug)")

        status = "PASS" if ok else "FAIL"
        if verbose:
            print(f"  [{status}] {mask_type:<12} policy={policy}: "
                  f"px_ratio={px_ratio:.3f}, bit_ratio={bit_ratio:.3f}"
                  + (f"  !! {'; '.join(msgs)}" if msgs else ""))
        if not ok:
            all_ok = False

    if not all_ok:
        raise AssertionError(
            "Golden mask tests FAILED — mask generation or bit conversion is broken. "
            "See output above for details."
        )
    return True


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(cfg: SuiteConfig):
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)

    rng = np.random.default_rng(cfg.seed)
    train_idx = rng.choice(len(train_ds), cfg.train_samples, replace=False)
    test_idx = rng.choice(len(test_ds), cfg.test_samples, replace=False)

    train_x = torch.stack([train_ds[i][0] for i in train_idx])
    train_y = torch.tensor([train_ds[i][1] for i in train_idx])
    test_x = torch.stack([test_ds[i][0] for i in test_idx])
    test_y = torch.tensor([test_ds[i][1] for i in test_idx])

    return train_x, train_y, test_x, test_y


# ============================================================================
# MODEL TRAINING / LOADING
# ============================================================================

def train_or_load_model(cfg: SuiteConfig):
    """Train or load cached Route C model."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    model_path = os.path.join(cfg.output_dir, "routec_model.pt")

    # Also check exp09 model
    exp09_path = os.path.join(PROJECT_ROOT, "experiments", "outputs", "exp09_model.pt")

    model = RouteCModel(cfg).to(cfg.device)

    if os.path.exists(model_path):
        print(f"  Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=cfg.device, weights_only=True))
        return model
    elif os.path.exists(exp09_path):
        print(f"  Loading model from {exp09_path}")
        model.load_state_dict(torch.load(exp09_path, map_location=cfg.device, weights_only=True))
        torch.save(model.state_dict(), model_path)
        return model

    print("  Training Route C model...")
    train_x, train_y, _, _ = load_data(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.epochs):
        model.train()
        tau = cfg.tau_start + (cfg.tau_end - cfg.tau_start) * epoch / max(1, cfg.epochs - 1)
        model.set_temperature(tau)

        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            z, x_hat, cls_logits, core_logits = model(x)
            loss_recon = F.mse_loss(x_hat, x)
            mask = torch.rand_like(z) < 0.15
            loss_core = F.binary_cross_entropy_with_logits(
                core_logits[mask], z.detach()[mask]
            ) if mask.any() else 0.0
            loss_cls = F.cross_entropy(cls_logits, y)
            loss = cfg.alpha_recon * loss_recon + cfg.beta_core * loss_core + cfg.gamma_cls * loss_cls
            loss.backward()
            optimizer.step()

        print(f"    Epoch {epoch+1}/{cfg.epochs}")

    torch.save(model.state_dict(), model_path)
    return model


def estimate_sigma_sq(model, test_x, device, n_samples=500):
    model.eval()
    residuals = []
    with torch.no_grad():
        for i in range(min(n_samples, len(test_x))):
            x = test_x[i:i+1].to(device)
            z = model.encode(x)
            x_hat = model.decode(z)
            residuals.append(((x_hat - x) ** 2).mean().item())
    return np.mean(residuals)


# ============================================================================
# ENERGY + SOLVER (baseline & D-RoPE)
# ============================================================================

class BaselineEnergy:
    def __init__(self, model, sigma_sq, lambda_core=1.0, lambda_obs=1.0, device=None):
        self.model = model
        self.sigma_sq = sigma_sq
        self.lambda_core = lambda_core
        self.lambda_obs = lambda_obs
        self.device = device or torch.device('cpu')
        self._o_obs_t = None
        self._mask_t = None
        self._mask_sum = 1.0

    def set_observation(self, o_obs, mask):
        self._o_obs_t = torch.from_numpy(o_obs.astype(np.float32)).to(self.device)
        self._mask_t = torch.from_numpy(mask.astype(np.float32)).to(self.device)
        self._mask_sum = self._mask_t.sum().clamp(min=1.0)

    def energy(self, z):
        with torch.inference_mode():
            z_t = z.unsqueeze(0) if z.dim() == 3 else z
            e_core = 0.0
            if self.lambda_core > 0:
                logits = self.model.local_pred(z_t)
                e_core = F.binary_cross_entropy_with_logits(logits, z_t, reduction='sum')
            e_obs = 0.0
            if self.lambda_obs > 0 and self._o_obs_t is not None:
                o_hat = self.model.decode(z_t)[0, 0]
                diff = (o_hat - self._o_obs_t) * self._mask_t
                mse = (diff * diff).sum() / self._mask_sum
                e_obs = mse / (2 * self.sigma_sq)
            total = self.lambda_core * e_core + self.lambda_obs * e_obs
            return total.item() if isinstance(total, torch.Tensor) else total


class MCMCSolver:
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
        z[bit_mask_t] = torch.randint(0, 2, (bit_mask_t.sum().item(),), device=self.device, dtype=z.dtype)

        E_curr = self.energy.energy(z)

        for sweep in range(n_sweeps):
            for bi in range(0, H, bh):
                for bj in range(0, W, bw):
                    i_end, j_end = min(bi + bh, H), min(bj + bw, W)
                    block_bit_mask = bit_mask_t[:, bi:i_end, bj:j_end]
                    if not block_bit_mask.any():
                        continue
                    z[:, bi:i_end, bj:j_end][block_bit_mask] = 1 - z[:, bi:i_end, bj:j_end][block_bit_mask]
                    E_prop = self.energy.energy(z)
                    dE = E_prop - E_curr
                    if dE < 0 or torch.rand(1).item() < np.exp(-min(dE, 20)):
                        E_curr = E_prop
                    else:
                        z[:, bi:i_end, bj:j_end][block_bit_mask] = 1 - z[:, bi:i_end, bj:j_end][block_bit_mask]

        return z


# ============================================================================
# D-RoPE ENERGY (from core)
# ============================================================================

def _build_drope_energy(model, sigma_sq, cfg, device):
    """Build D-RoPE combined energy."""
    from core.drope import DRoPEEnergy, CandidateConfig, CombinedEnergyWithRope
    rope_config = CandidateConfig(density='sparse')
    drope = DRoPEEnergy(
        H=cfg.latent_size, W=cfg.latent_size, k=cfg.n_bits,
        config=rope_config, device=device
    )
    return CombinedEnergyWithRope(
        model, drope, sigma_sq,
        lambda_core=cfg.lambda_core, lambda_obs=cfg.lambda_obs,
        lambda_rope=cfg.lambda_rope, device=device
    )


# ============================================================================
# LEARNED ROUTING ENERGY
# ============================================================================

def train_learned_gate(model, train_x, cfg, device):
    """Train the learned Hamming gate on mask-prediction loss."""
    from learned_routing import LearnedDRoPEEnergy

    learned_drope = LearnedDRoPEEnergy(
        H=cfg.latent_size, W=cfg.latent_size, k=cfg.n_bits,
        use_lsh=False, temperature=1.0, device=device,
    ).to(device)

    optimizer = torch.optim.Adam(learned_drope.parameters(), lr=cfg.learned_gate_lr)

    model.eval()
    N = len(train_x)

    for epoch in range(cfg.learned_gate_epochs):
        perm = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, N, cfg.batch_size):
            idx = perm[i:i+cfg.batch_size]
            x = train_x[idx].to(device)

            with torch.no_grad():
                z = model.encode(x)
                z_hard = (z > 0.5).float()

            # The energy should be lower for correct z than corrupted z
            B = z_hard.shape[0]
            z_corrupted = z_hard.clone()
            flip_mask = torch.rand_like(z_corrupted) < 0.15
            z_corrupted[flip_mask] = 1 - z_corrupted[flip_mask]

            e_correct = learned_drope.compute_energy(z_hard)
            e_corrupted = learned_drope.compute_energy(z_corrupted)

            # Contrastive: correct should have lower energy
            margin = 1.0
            loss = F.relu(e_correct - e_corrupted + margin).mean()

            # Gate weight sparsity regularization
            loss = loss + 0.01 * learned_drope.gate.w.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"    Learned gate epoch {epoch+1}/{cfg.learned_gate_epochs}: loss={total_loss/max(n_batches,1):.4f}")

    return learned_drope


def _build_learned_energy(model, learned_drope, sigma_sq, cfg, device):
    """Build learned D-RoPE combined energy."""
    from learned_routing import CombinedEnergyLearned
    return CombinedEnergyLearned(
        model, learned_drope, sigma_sq,
        lambda_core=cfg.lambda_core, lambda_obs=cfg.lambda_obs,
        lambda_rope=cfg.lambda_rope, device=device
    )


# ============================================================================
# INPAINTING MODEL
# ============================================================================

def train_inpaint_model(model, train_x, cfg, device, train_y=None,
                        use_cls=True, use_energy=True):
    """Train amortized inpainting network.

    Args:
        use_cls: If True (default), include L_cls in training loss.
        use_energy: If True (default), include L_core + L_obs.
    """
    from inpainting import InpaintTrainer, InpaintConfig

    icfg = InpaintConfig(
        epochs=cfg.inpaint_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.inpaint_lr,
        hidden=cfg.inpaint_hidden,
        alpha_core=0.1 if use_energy else 0.0,
        alpha_obs=0.1 if use_energy else 0.0,
        gamma_cls=0.5 if use_cls else 0.0,
    )
    trainer = InpaintTrainer(model, icfg, device)
    trainer.train(train_x, train_y=train_y, verbose=True)
    return trainer.inpaint_net


# ============================================================================
# SINGLE SAMPLE EVALUATION
# ============================================================================

def evaluate_sample(
    model, x_clean, label, pixel_mask, bit_mask, method, device, cfg,
    energy_fn=None, solver=None, inpaint_net=None, inpaint_net_maskonly=None,
):
    """Evaluate a single sample with a given method. Returns dict of metrics."""
    # Occlude image
    x_occ = x_clean * pixel_mask

    # Encode occluded
    with torch.no_grad():
        x_occ_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
        z_init = model.encode(x_occ_t)[0]

        logits = model.classifier(z_init.unsqueeze(0))
        pred_before = logits.argmax(dim=1).item()

        o_hat_before = model.decode(z_init.unsqueeze(0))[0, 0].cpu().numpy()
        mse_before = compute_occluded_mse(o_hat_before, x_clean, pixel_mask)

    # Inference
    t0 = time.time()

    if method in ('baseline', 'drope', 'learned_drope'):
        z_final = solver.run(z_init, x_occ, pixel_mask, bit_mask, n_sweeps=cfg.n_sweeps)
    elif method == 'amortized':
        from inpainting import amortized_inpaint
        z_final = amortized_inpaint(inpaint_net, z_init, bit_mask, device)
    elif method == 'amortized_maskonly':
        from inpainting import amortized_inpaint
        z_final = amortized_inpaint(inpaint_net_maskonly, z_init, bit_mask, device)
    elif method == 'iterative':
        from inpainting import iterative_inpaint
        z_final = iterative_inpaint(inpaint_net, z_init, bit_mask, n_steps=4, device=device)
    else:
        z_final = z_init  # no-op

    runtime_ms = (time.time() - t0) * 1000

    # After inference
    with torch.no_grad():
        logits = model.classifier(z_final.unsqueeze(0))
        pred_after = logits.argmax(dim=1).item()

        o_hat_after = model.decode(z_final.unsqueeze(0))[0, 0].cpu().numpy()
        mse_after = compute_occluded_mse(o_hat_after, x_clean, pixel_mask)

    return {
        'correct_before': int(pred_before == label),
        'correct_after': int(pred_after == label),
        'mse_before': mse_before,
        'mse_after': mse_after,
        'runtime_ms': runtime_ms,
    }


# ============================================================================
# FULL EVALUATION
# ============================================================================

def run_evaluation(
    model, test_x, test_y, method, mask_type, noise_type, cfg, device,
    energy_fn=None, solver=None, inpaint_net=None, inpaint_net_maskonly=None,
):
    """Run evaluation for a specific (method, mask_type, noise_type) configuration."""
    model.eval()
    rng = np.random.default_rng(cfg.seed + 100)
    eval_idx = rng.choice(len(test_x), min(cfg.eval_samples, len(test_x)), replace=False)

    results = {
        'acc_before': 0, 'acc_after': 0,
        'mse_before': [], 'mse_after': [],
        'runtime_ms': [],
        'correct_before': [], 'correct_after': [],  # per-sample for correlation
    }
    pixel_mask_ratio = None
    bit_mask_ratio = None

    for idx in eval_idx:
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()

        # Apply noise
        x_noisy = apply_noise(x_clean, noise_type, rng)

        # Generate mask
        if mask_type == 'center':
            pixel_mask = make_center_mask(28, 28)
        elif mask_type == 'random':
            pixel_mask = make_random_mask(28, 28, rng=rng)
        elif mask_type == 'multi_hole':
            pixel_mask = make_multi_hole_mask(28, 28, rng=rng)
        elif mask_type == 'stripes':
            pixel_mask = make_stripe_mask(28, 28)
        else:
            pixel_mask = make_center_mask(28, 28)

        bit_mask = pixel_to_bit_mask(pixel_mask, cfg.n_bits, cfg.latent_size,
                                     policy=cfg.bitmask_policy)

        # ── Sanity checks (computed once per config) ──
        if pixel_mask_ratio is None:
            pixel_mask_ratio = compute_mask_ratio(pixel_mask)
            bit_mask_ratio = compute_bit_mask_ratio(bit_mask)
            assert pixel_mask_ratio > 0, (
                f"SANITY FAIL: pixel_mask_ratio=0 for mask_type={mask_type}. "
                f"Mask is all-visible — occlusion not applied."
            )
            assert bit_mask_ratio > 0, (
                f"SANITY FAIL: bit_mask_ratio=0 for mask_type={mask_type}. "
                f"No latent positions are masked — inference will be a no-op. "
                f"Check pixel_to_bit_mask threshold."
            )

        res = evaluate_sample(
            model, x_noisy, label, pixel_mask, bit_mask,
            method, device, cfg,
            energy_fn=energy_fn, solver=solver, inpaint_net=inpaint_net,
            inpaint_net_maskonly=inpaint_net_maskonly,
        )

        results['acc_before'] += res['correct_before']
        results['acc_after'] += res['correct_after']
        results['correct_before'].append(res['correct_before'])
        results['correct_after'].append(res['correct_after'])
        results['mse_before'].append(res['mse_before'])
        results['mse_after'].append(res['mse_after'])
        results['runtime_ms'].append(res['runtime_ms'])

    n = len(eval_idx)

    # Per-sample Δmse and Δacc for correlation analysis
    mse_before_arr = np.array(results['mse_before'])
    mse_after_arr = np.array(results['mse_after'])
    correct_before_arr = np.array(results['correct_before'])
    correct_after_arr = np.array(results['correct_after'])
    delta_mse_per_sample = mse_after_arr - mse_before_arr
    delta_acc_per_sample = correct_after_arr - correct_before_arr

    # corr(Δmse, Δacc) — quantifies objective misalignment
    if np.std(delta_mse_per_sample) > 1e-12 and np.std(delta_acc_per_sample) > 1e-12:
        corr_mse_acc = float(np.corrcoef(delta_mse_per_sample, delta_acc_per_sample)[0, 1])
    else:
        corr_mse_acc = 0.0

    return {
        'method': method,
        'mask_type': mask_type,
        'noise_type': noise_type,
        'acc_before': results['acc_before'] / n,
        'acc_after': results['acc_after'] / n,
        'delta_acc': (results['acc_after'] - results['acc_before']) / n,
        'mse_before': np.mean(mse_before_arr),
        'mse_after': np.mean(mse_after_arr),
        'delta_mse': np.mean(delta_mse_per_sample),
        'runtime_ms': np.mean(results['runtime_ms']),
        'n_samples': n,
        # ── Hard metrics (reviewer feedback) ──
        'pixel_mask_ratio': pixel_mask_ratio,
        'bit_mask_ratio': bit_mask_ratio,
        'corr_dmse_dacc': corr_mse_acc,
        'mse_before_abs': float(np.mean(mse_before_arr)),
        'mse_after_abs': float(np.mean(mse_after_arr)),
        'runtime_p10': float(np.percentile(results['runtime_ms'], 10)),
        'runtime_p50': float(np.percentile(results['runtime_ms'], 50)),
        'runtime_p90': float(np.percentile(results['runtime_ms'], 90)),
        'bitmask_policy': cfg.bitmask_policy,
    }


# ============================================================================
# MAIN
# ============================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description='Route C Benchmark Suite')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'baseline', 'drope', 'learned_drope',
                                 'amortized', 'amortized_maskonly', 'iterative'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=200)
    parser.add_argument('--mask_types', type=str, default='center,random,multi_hole,stripes')
    parser.add_argument('--noise_types', type=str, default='clean,noise,bias,dropout')
    parser.add_argument('--output_dir', type=str, default='outputs/benchmark')
    parser.add_argument('--bitmask_policy', type=str, default='any',
                        choices=['any', 'majority'],
                        help="How to convert pixel mask to latent bit mask. "
                             "'any'=conservative (any occlusion marks position), "
                             "'majority'=only if >50%% occluded.")
    args = parser.parse_args()

    cfg = SuiteConfig(
        seed=args.seed,
        device=args.device,
        eval_samples=args.eval_samples,
        output_dir=args.output_dir,
        bitmask_policy=args.bitmask_policy,
    )

    mask_types = args.mask_types.split(',')
    noise_types = args.noise_types.split(',')

    if args.model == 'all':
        methods = ['baseline', 'drope', 'learned_drope',
                   'amortized', 'amortized_maskonly', 'iterative']
    else:
        methods = [args.model]

    device = torch.device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 100)
    print("ROUTE C BENCHMARK SUITE")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Methods: {methods}")
    print(f"Mask types: {mask_types}")
    print(f"Noise types: {noise_types}")
    print(f"Bitmask policy: {cfg.bitmask_policy}")
    print(f"Eval samples per config: {cfg.eval_samples}")

    # ── Golden mask tests (prevent regressions) ──
    print("\n[0] Running golden mask tests...")
    run_golden_mask_tests(n_bits=cfg.n_bits, latent_size=cfg.latent_size,
                          policy=cfg.bitmask_policy, verbose=True)
    print("    All golden mask tests PASSED.")

    # Load data
    print("\n[1] Loading data...")
    train_x, train_y, test_x, test_y = load_data(cfg)
    print(f"    Train: {len(train_x)}, Test: {len(test_x)}")

    # Load/train base model
    print("\n[2] Preparing Route C model...")
    model = train_or_load_model(cfg)
    model = model.to(device)
    model.eval()

    sigma_sq = estimate_sigma_sq(model, test_x, device)
    print(f"    σ² = {sigma_sq:.4f}")
    print(f"    Parameters: {count_params(model):,}")

    # Prepare method-specific components
    solvers = {}
    energy_fns = {}
    inpaint_net = None
    learned_drope = None

    if 'baseline' in methods:
        energy_fns['baseline'] = BaselineEnergy(model, sigma_sq, cfg.lambda_core, cfg.lambda_obs, device)
        solvers['baseline'] = MCMCSolver(energy_fns['baseline'], cfg.block_size, device)

    if 'drope' in methods:
        print("\n[3a] Setting up D-RoPE energy...")
        energy_fns['drope'] = _build_drope_energy(model, sigma_sq, cfg, device)
        from core.drope import BlockGibbsWithRope
        solvers['drope'] = MCMCSolver(energy_fns['drope'], cfg.block_size, device)

    if 'learned_drope' in methods:
        print("\n[3b] Training learned gate...")
        learned_drope = train_learned_gate(model, train_x, cfg, device)
        energy_fns['learned_drope'] = _build_learned_energy(
            model, learned_drope, sigma_sq, cfg, device
        )
        solvers['learned_drope'] = MCMCSolver(energy_fns['learned_drope'], cfg.block_size, device)

    inpaint_net_maskonly = None

    if 'amortized' in methods or 'iterative' in methods:
        print("\n[3c] Training inpainting network (L_mask + L_cls + L_core + L_obs)...")
        inpaint_net = train_inpaint_model(model, train_x, cfg, device,
                                          train_y=train_y, use_cls=True, use_energy=True)
        print(f"    InpaintNet (full) parameters: {count_params(inpaint_net):,}")

    if 'amortized_maskonly' in methods:
        print("\n[3d] Training inpainting network (L_mask ONLY — ablation baseline)...")
        inpaint_net_maskonly = train_inpaint_model(model, train_x, cfg, device,
                                                    train_y=None, use_cls=False, use_energy=False)
        print(f"    InpaintNet (mask-only) parameters: {count_params(inpaint_net_maskonly):,}")

    # Run evaluations
    print("\n[4] Running evaluations...")
    all_results = []

    total_configs = len(methods) * len(mask_types) * len(noise_types)
    config_idx = 0

    for method in methods:
        for mask_type in mask_types:
            for noise_type in noise_types:
                config_idx += 1
                print(f"  [{config_idx}/{total_configs}] {method} | {mask_type} | {noise_type}")

                res = run_evaluation(
                    model, test_x, test_y,
                    method=method,
                    mask_type=mask_type,
                    noise_type=noise_type,
                    cfg=cfg,
                    device=device,
                    energy_fn=energy_fns.get(method),
                    solver=solvers.get(method),
                    inpaint_net=inpaint_net,
                    inpaint_net_maskonly=inpaint_net_maskonly,
                )
                all_results.append(res)

                print(f"    acc: {res['acc_before']:.1%} → {res['acc_after']:.1%} "
                      f"(Δ={res['delta_acc']:+.1%}) | "
                      f"mse: {res['mse_before']:.4f} → {res['mse_after']:.4f} | "
                      f"time: {res['runtime_ms']:.1f}ms | "
                      f"mask_ratio: px={res['pixel_mask_ratio']:.2f} bit={res['bit_mask_ratio']:.2f} | "
                      f"corr(Δmse,Δacc)={res['corr_dmse_dacc']:+.3f}")

    # Save CSV with extended columns (reviewer hard metrics)
    csv_path = os.path.join(cfg.output_dir, "results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'method', 'mask_type', 'noise_type',
            'acc_before', 'acc_after', 'delta_acc',
            'mse_before', 'mse_after', 'delta_mse',
            'runtime_ms', 'n_samples',
            'pixel_mask_ratio', 'bit_mask_ratio', 'bitmask_policy',
            'corr_dmse_dacc',
            'mse_before_abs', 'mse_after_abs',
            'runtime_p10', 'runtime_p50', 'runtime_p90',
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    # Hardware / environment info header
    import platform
    hw_device = str(device)
    hw_info = {
        'device': hw_device,
        'platform': platform.platform(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'batch_size': 1,  # evaluation is per-sample
    }
    if hw_device.startswith('cuda') and torch.cuda.is_available():
        hw_info['gpu'] = torch.cuda.get_device_name(0)
        hw_info['gpu_mem_gb'] = f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}"
    hw_path = os.path.join(cfg.output_dir, "hardware_info.txt")
    with open(hw_path, 'w') as f:
        for k, v in hw_info.items():
            f.write(f"{k}: {v}\n")
    print(f"Hardware info saved to {hw_path}")

    # Pretty print summary tables
    print("\n" + "=" * 120)
    print("SUMMARY TABLE: All Methods × Mask Types (clean noise only)")
    print("=" * 120)

    header = (f"{'method':<18} {'mask':<12} {'noise':<10} {'acc_bef':>8} {'acc_aft':>8} "
              f"{'Δacc':>7} {'mse_bef':>8} {'mse_aft':>8} {'Δmse':>8} "
              f"{'ms':>7} {'px_mask':>7} {'bit_mask':>8} {'corr':>6}")
    print(header)
    print("-" * 140)

    for r in all_results:
        print(f"{r['method']:<18} {r['mask_type']:<12} {r['noise_type']:<10} "
              f"{r['acc_before']:>8.1%} {r['acc_after']:>8.1%} {r['delta_acc']:>+7.1%} "
              f"{r['mse_before']:>8.4f} {r['mse_after']:>8.4f} {r['delta_mse']:>+8.4f} "
              f"{r['runtime_ms']:>7.1f} {r['pixel_mask_ratio']:>7.2f} {r['bit_mask_ratio']:>8.2f} "
              f"{r['corr_dmse_dacc']:>+6.2f}")

    # OOD analysis
    print("\n" + "=" * 120)
    print("OOD ANALYSIS: Train mask (center) vs Test mask (random, multi_hole, stripes)")
    print("=" * 120)

    clean_results = [r for r in all_results if r['noise_type'] == 'clean']
    for method in methods:
        method_results = [r for r in clean_results if r['method'] == method]
        if method_results:
            center_res = [r for r in method_results if r['mask_type'] == 'center']
            other_res = [r for r in method_results if r['mask_type'] != 'center']
            if center_res and other_res:
                center_acc = center_res[0]['delta_acc']
                for r in other_res:
                    gap = r['delta_acc'] - center_acc
                    print(f"  {method:<15} center→{r['mask_type']:<12}: "
                          f"Δacc_center={center_acc:+.1%}, Δacc_{r['mask_type']}={r['delta_acc']:+.1%}, "
                          f"OOD gap={gap:+.1%}")

    # Speed comparison
    print("\n" + "=" * 120)
    print("SPEED COMPARISON (center mask, clean noise)")
    print("=" * 120)

    for method in methods:
        speed_results = [r for r in all_results
                        if r['method'] == method and r['mask_type'] == 'center' and r['noise_type'] == 'clean']
        if speed_results:
            r = speed_results[0]
            print(f"  {method:<15}: {r['runtime_ms']:.1f} ms/sample")

    # Parameter counts
    print("\n" + "=" * 120)
    print("MODEL SIZES")
    print("=" * 120)
    print(f"  Route C base model: {count_params(model):,} params")
    if learned_drope is not None:
        print(f"  Learned gate:       {count_params(learned_drope):,} params")
    if inpaint_net is not None:
        print(f"  InpaintNet (full):  {count_params(inpaint_net):,} params")
    if inpaint_net_maskonly is not None:
        print(f"  InpaintNet (mask):  {count_params(inpaint_net_maskonly):,} params")

    # ── Mask ratio summary (sanity check visibility) ──
    print("\n" + "=" * 120)
    print("MASK RATIO SUMMARY (sanity check)")
    print("=" * 120)
    seen_masks = set()
    for r in all_results:
        mt = r['mask_type']
        if mt not in seen_masks:
            seen_masks.add(mt)
            print(f"  {mt:<12}: pixel_mask_ratio={r['pixel_mask_ratio']:.3f}, "
                  f"bit_mask_ratio={r['bit_mask_ratio']:.3f}")

    # ── Iterative steps curve (acc_after vs n_steps = 1,2,3,4) ──
    if inpaint_net is not None:
        print("\n" + "=" * 120)
        print("ITERATIVE STEPS CURVE: acc_after vs n_steps (center mask, clean noise)")
        print("=" * 120)

        from inpainting import iterative_inpaint
        steps_csv_rows = []
        for n_steps in [1, 2, 3, 4]:
            rng_steps = np.random.default_rng(cfg.seed + 200)
            eval_idx = rng_steps.choice(len(test_x), min(cfg.eval_samples, len(test_x)), replace=False)
            correct = 0
            for idx_s in eval_idx:
                x_c = test_x[idx_s].numpy()[0]
                lbl = test_y[idx_s].item()
                pm = make_center_mask(28, 28)
                bm = pixel_to_bit_mask(pm, cfg.n_bits, cfg.latent_size,
                                       policy=cfg.bitmask_policy)
                x_occ = x_c * pm
                with torch.no_grad():
                    x_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
                    z_i = model.encode(x_t)[0]
                    z_f = iterative_inpaint(inpaint_net, z_i, bm, n_steps=n_steps, device=device)
                    pred = model.classifier(z_f.unsqueeze(0)).argmax(1).item()
                    correct += int(pred == lbl)
            acc = correct / len(eval_idx)
            print(f"  n_steps={n_steps}: acc_after={acc:.1%}")
            steps_csv_rows.append({'n_steps': n_steps, 'acc_after': acc})

        steps_csv_path = os.path.join(cfg.output_dir, "iterative_steps_curve.csv")
        with open(steps_csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['n_steps', 'acc_after'])
            w.writeheader()
            for row in steps_csv_rows:
                w.writerow(row)
        print(f"  Steps curve saved to {steps_csv_path}")

    print("\n" + "=" * 120)
    print(f"All results saved to {csv_path}")
    print("=" * 120)

    return all_results


if __name__ == "__main__":
    main()
