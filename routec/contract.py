"""
Route C: Paradigm Contract (Phase 0)
=====================================
Frozen interfaces for the discrete-core inference paradigm.
Any new operator (FGO, GDA, etc.) must implement these interfaces.
Any new dataset only changes ADC/DAC parameters, not the contract.

Usage:
    from routec.contract import RouteCConfig, build_model, build_inpaint_net
    from routec.contract import RepairPolicy, EvidenceRepairPolicy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Callable
from enum import Enum


# ============================================================================
# CONFIG: What changes per dataset
# ============================================================================

@dataclass
class RouteCConfig:
    """Paradigm configuration. Only these parameters change per dataset."""
    # Representation
    n_bits: int = 8           # k: bits per spatial position
    latent_size: int = 7      # H = W of latent grid
    # ADC/DAC
    in_channels: int = 1      # observation channels (1=grayscale, 3=RGB)
    out_channels: int = 1
    hidden_dim: int = 64      # encoder/decoder width
    # E_core
    energy_hidden: int = 32   # local predictor MLP width
    # E_obs
    obs_likelihood: str = 'bce'  # 'bce' (sigmoid), 'mse' (linear), 'cb'
    # Training
    mask_mixture: Dict[str, float] = field(default_factory=lambda: {
        'random_block': 0.25, 'center': 0.20,
        'random_stripe': 0.20, 'multi_hole': 0.20,
        'random_sparse': 0.15,
    })
    # Image size (for mask generation)
    img_h: int = 28
    img_w: int = 28


# ============================================================================
# ENERGY CONTRACT: E_core, E_obs
# ============================================================================

class EnergyTerm(nn.Module):
    """Abstract energy term. All energy components implement this."""
    def energy(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute energy. Returns (B,) tensor."""
        raise NotImplementedError

    def violation_rate(self, z: torch.Tensor, **kwargs) -> float:
        """Fraction of positions violating this energy term."""
        raise NotImplementedError


class LocalEnergyCore(EnergyTerm):
    """
    E_core: pseudo-likelihood MRF on z.
    Task-agnostic local consistency. 3×3 neighborhood.
    """
    def __init__(self, n_bits=8, hidden_dim=32):
        super().__init__()
        self.n_bits = n_bits
        self.predictor = nn.Sequential(
            nn.Linear(9 * n_bits, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_bits),
        )

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """Predict all bits at all positions. Returns logits (B, k, H, W)."""
        B, k, H, W = z.shape
        z_pad = F.pad(z, (1,1,1,1), mode='constant', value=0)
        windows = F.unfold(z_pad, kernel_size=3)
        windows = windows.reshape(B, k, 9, H*W)
        windows[:, :, 4, :] = 0  # mask center
        windows = windows.reshape(B, k*9, H*W).permute(0,2,1)
        logits = self.predictor(windows)
        return logits.permute(0,2,1).reshape(B, k, H, W)

    def energy(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.predict(z)
        loss = F.binary_cross_entropy_with_logits(logits, z, reduction='none')
        return loss.sum(dim=(1, 2, 3))

    def violation_rate(self, z: torch.Tensor, **kwargs) -> float:
        with torch.no_grad():
            logits = self.predict(z)
            preds = (torch.sigmoid(logits) > 0.5).float()
            return (preds != z).float().mean().item()


class ObservationEnergy(EnergyTerm):
    """
    E_obs: -log p(o | decode(z)).
    Must match decoder output distribution.
    """
    def __init__(self, decoder: nn.Module, likelihood: str = 'bce'):
        super().__init__()
        self.decoder = decoder
        self.likelihood = likelihood

    def energy(self, z: torch.Tensor, observation: torch.Tensor = None,
               pixel_mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Args:
            z: (B, k, H, W) binary latent
            observation: (B, C, H_img, W_img) original image
            pixel_mask: (B, 1, H_img, W_img) 1=observed, 0=occluded
        """
        o_hat = self.decoder(z)
        if self.likelihood == 'bce':
            o_hat = o_hat.clamp(1e-6, 1-1e-6)
            nll = -(observation * torch.log(o_hat) +
                    (1-observation) * torch.log(1-o_hat))
        elif self.likelihood == 'mse':
            nll = (observation - o_hat) ** 2
        else:
            raise ValueError(f"Unknown likelihood: {self.likelihood}")

        if pixel_mask is not None:
            nll = nll * pixel_mask

        return nll.sum(dim=(1, 2, 3))

    def per_patch_residual(self, z: torch.Tensor, observation: torch.Tensor,
                           pixel_mask: torch.Tensor, latent_size: int) -> torch.Tensor:
        """
        Compute E_obs residual per latent patch.
        Returns (B, latent_size, latent_size) residual map.
        """
        B = z.shape[0]
        o_hat = self.decoder(z)
        H_img, W_img = observation.shape[2], observation.shape[3]
        patch_h, patch_w = H_img // latent_size, W_img // latent_size

        residuals = torch.zeros(B, latent_size, latent_size, device=z.device)

        for i in range(latent_size):
            for j in range(latent_size):
                y0, y1 = i * patch_h, (i+1) * patch_h
                x0, x1 = j * patch_w, (j+1) * patch_w
                p_obs = observation[:, :, y0:y1, x0:x1]
                p_hat = o_hat[:, :, y0:y1, x0:x1].clamp(1e-6, 1-1e-6)
                p_mask = pixel_mask[:, :, y0:y1, x0:x1]

                obs_count = p_mask.sum(dim=(1,2,3))
                if self.likelihood == 'bce':
                    bce = -(p_obs * torch.log(p_hat) +
                            (1-p_obs) * torch.log(1-p_hat))
                else:
                    bce = (p_obs - p_hat) ** 2

                masked_bce = (bce * p_mask).sum(dim=(1,2,3))
                residuals[:, i, j] = torch.where(
                    obs_count > 0,
                    masked_bce / obs_count.clamp(min=1),
                    torch.tensor(float('inf'), device=z.device)
                )

        return residuals


# ============================================================================
# REPAIR POLICY CONTRACT (Phase 3)
# ============================================================================

class RepairPolicy:
    """Abstract repair policy. Decides which positions to repair."""

    def compute_repair_mask(self, z_init: torch.Tensor, z_repaired: torch.Tensor,
                            bit_mask: np.ndarray, **kwargs) -> np.ndarray:
        """
        Given initial z, repaired z, and the occlusion bit_mask,
        return a boolean mask of which positions to actually repair.

        Args:
            z_init: (k, H, W) initial encoded z
            z_repaired: (k, H, W) fully repaired z (from amortized inpaint)
            bit_mask: (k, H, W) bool — True where occluded
            **kwargs: additional info (observation, pixel_mask, model, etc.)

        Returns:
            repair_mask: (k, H, W) bool — True where to apply repair
        """
        raise NotImplementedError


class RepairAll(RepairPolicy):
    """Replace all masked positions (baseline 'any' policy)."""
    def compute_repair_mask(self, z_init, z_repaired, bit_mask, **kwargs):
        return bit_mask.copy()


class EvidenceRepairPolicy(RepairPolicy):
    """
    E_obs residual-driven repair (Phase 3 module).
    Only repair positions where observation evidence suggests mismatch.

    Fully occluded patches (inf residual) → always repair.
    Partially observed patches → repair only if residual > threshold.
    """
    def __init__(self, obs_energy: ObservationEnergy, latent_size: int,
                 threshold: float = 1.0):
        self.obs_energy = obs_energy
        self.latent_size = latent_size
        self.threshold = threshold

    def compute_repair_mask(self, z_init, z_repaired, bit_mask,
                            observation=None, pixel_mask=None, **kwargs):
        """
        Args:
            observation: (H_img, W_img) numpy array
            pixel_mask: (H_img, W_img) numpy array, 1=observed
        """
        device = z_init.device
        obs_t = torch.from_numpy(observation[None, None].astype(np.float32)).to(device)
        pmask_t = torch.from_numpy(pixel_mask[None, None].astype(np.float32)).to(device)

        with torch.no_grad():
            residuals = self.obs_energy.per_patch_residual(
                z_init.unsqueeze(0), obs_t, pmask_t, self.latent_size
            )[0]  # (ls, ls)

        repair_mask = bit_mask.copy()
        k = bit_mask.shape[0]
        for i in range(self.latent_size):
            for j in range(self.latent_size):
                if bit_mask[0, i, j]:
                    r = residuals[i, j].item()
                    if not (np.isinf(r) or r > self.threshold):
                        repair_mask[:, i, j] = False

        return repair_mask


class ConstraintInterface:
    """
    Phase 6: Constraint writing interface.
    Users declare constraints; system encodes them into energy function.

    Example:
        constraints = ConstraintInterface()
        constraints.add_symmetry('D4', weight=0.1)  # rotation equivariance
        constraints.add_scale_invariance(scales=[0.8, 1.0, 1.2])
        E_total = E_core + constraints.compile(z)
    """
    def __init__(self):
        self.constraints = []

    def add_symmetry(self, group: str = 'D4', weight: float = 0.1):
        """Add group equivariance constraint to E_core."""
        self.constraints.append({
            'type': 'symmetry', 'group': group, 'weight': weight
        })

    def add_consistency(self, transform_fn: Callable, weight: float = 0.1):
        """Add custom consistency constraint."""
        self.constraints.append({
            'type': 'consistency', 'fn': transform_fn, 'weight': weight
        })

    def compile(self, e_core: LocalEnergyCore, z: torch.Tensor) -> torch.Tensor:
        """Compile constraints into additional energy terms."""
        extra_energy = torch.tensor(0.0, device=z.device)
        for c in self.constraints:
            if c['type'] == 'symmetry' and c['group'] == 'D4':
                # Average E_core over 4 rotations
                e0 = e_core.energy(z)
                e90 = e_core.energy(torch.rot90(z, 1, [2, 3]))
                e180 = e_core.energy(torch.rot90(z, 2, [2, 3]))
                e270 = e_core.energy(torch.rot90(z, 3, [2, 3]))
                # Consistency = variance across rotations
                e_stack = torch.stack([e0, e90, e180, e270])
                extra_energy = extra_energy + c['weight'] * e_stack.var(dim=0).mean()
        return extra_energy
