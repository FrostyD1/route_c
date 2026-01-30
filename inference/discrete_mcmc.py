"""
Route C: Discrete MCMC Inference
================================

Inference in z-space using block Gibbs / Metropolis-Hastings.

z* = argmin_z [E_core(z) + E_obs(z, o) + E_task(z, y)]

Where:
- E_core(z) = -Σ log pθ(z_i | neigh)     (local structure)
- E_obs(z,o) = (1/(2σ²)) ||M ⊙ (D(z)-o)||²  (analog coupling)
- E_task(z,y) = -log pω(y|z)               (task objective)

All inference is DISCRETE: we flip bits or blocks, never optimize in continuous space.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Configuration for discrete inference."""
    n_steps: int = 30
    block_size: Tuple[int, int] = (2, 2)  # Spatial block for updates
    temperature: float = 1.0
    sigma_sq: float = 0.1  # Observation noise variance
    lambda_core: float = 1.0
    lambda_obs: float = 1.0
    lambda_task: float = 0.0  # Set >0 if using task energy
    seed: int = 42


class DiscreteInference:
    """
    Discrete MCMC inference for Route C.
    
    Supports:
    - Single bit flips (Gibbs)
    - Block updates (flip multiple bits at once)
    - Metropolis-Hastings acceptance
    """
    
    def __init__(
        self,
        decode_fn: Callable[[np.ndarray], np.ndarray],
        energy_core_fn: Optional[Callable[[np.ndarray], float]] = None,
        classifier_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        config: InferenceConfig = None,
    ):
        """
        Args:
            decode_fn: z (k,H,W) → o_hat (28,28)
            energy_core_fn: z → scalar energy (optional)
            classifier_fn: z → class logits (optional)
            config: inference configuration
        """
        self.decode = decode_fn
        self.energy_core = energy_core_fn
        self.classifier = classifier_fn
        self.config = config or InferenceConfig()
    
    def energy_observation(
        self,
        z: np.ndarray,
        o_obs: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """
        E_obs = (1/(2σ²)) ||M ⊙ (D(z) - o)||²
        
        Args:
            z: (k, H, W) binary latent
            o_obs: (28, 28) observed image (with occlusion)
            mask: (28, 28) bool, True where observed
        
        Returns:
            energy: scalar
        """
        o_hat = self.decode(z)
        diff = (o_hat - o_obs) * mask
        mse = (diff ** 2).sum() / max(1, mask.sum())
        return mse / (2 * self.config.sigma_sq)
    
    def energy_task(
        self,
        z: np.ndarray,
        target_class: int,
    ) -> float:
        """
        E_task = -log P(y | z)
        
        Args:
            z: (k, H, W) binary latent
            target_class: target class index
        
        Returns:
            energy: scalar (higher = worse)
        """
        if self.classifier is None:
            return 0.0
        
        logits = self.classifier(z)  # (n_classes,)
        log_probs = logits - np.log(np.exp(logits).sum() + 1e-10)
        return -log_probs[target_class]
    
    def total_energy(
        self,
        z: np.ndarray,
        o_obs: np.ndarray,
        mask: np.ndarray,
        target_class: Optional[int] = None,
    ) -> float:
        """
        E_total = λ_core * E_core + λ_obs * E_obs + λ_task * E_task
        """
        cfg = self.config
        E = 0.0
        
        if cfg.lambda_core > 0 and self.energy_core is not None:
            E += cfg.lambda_core * self.energy_core(z)
        
        if cfg.lambda_obs > 0:
            E += cfg.lambda_obs * self.energy_observation(z, o_obs, mask)
        
        if cfg.lambda_task > 0 and target_class is not None:
            E += cfg.lambda_task * self.energy_task(z, target_class)
        
        return E
    
    def gibbs_step_single_bit(
        self,
        z: np.ndarray,
        o_obs: np.ndarray,
        mask: np.ndarray,
        bit_mask: np.ndarray,
        target_class: Optional[int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Single Gibbs sweep: visit each masked bit and sample.
        
        Args:
            z: current state (k, H, W)
            o_obs: observed image
            mask: pixel observation mask
            bit_mask: (k, H, W) bool, True where bit is unknown
            target_class: optional target for E_task
            rng: random generator
        
        Returns:
            z_new: updated state
        """
        k, H, W = z.shape
        z = z.copy()
        
        # Get positions to update
        positions = np.argwhere(bit_mask)
        rng.shuffle(positions)
        
        for pos in positions:
            b, i, j = pos
            
            # Compute energy for z_b=0 and z_b=1
            z[b, i, j] = 0
            E0 = self.total_energy(z, o_obs, mask, target_class)
            
            z[b, i, j] = 1
            E1 = self.total_energy(z, o_obs, mask, target_class)
            
            # Sample from Boltzmann
            dE = (E1 - E0) / self.config.temperature
            p1 = 1 / (1 + np.exp(dE))  # P(z=1)
            
            z[b, i, j] = 1 if rng.random() < p1 else 0
        
        return z
    
    def gibbs_step_block(
        self,
        z: np.ndarray,
        o_obs: np.ndarray,
        mask: np.ndarray,
        bit_mask: np.ndarray,
        target_class: Optional[int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Block Gibbs: update spatial blocks of bits together.
        
        For a 2×2 block, enumerate all 2^(k*4) configurations
        and sample from Boltzmann distribution.
        
        If too many configs, use MH proposal instead.
        """
        k, H, W = z.shape
        bh, bw = self.config.block_size
        z = z.copy()
        
        # Iterate over blocks
        for bi in range(0, H, bh):
            for bj in range(0, W, bw):
                # Get block bounds
                i_end = min(bi + bh, H)
                j_end = min(bj + bw, W)
                
                # Check if any bit in block is masked
                block_mask = bit_mask[:, bi:i_end, bj:j_end]
                if not block_mask.any():
                    continue
                
                # For small blocks, enumerate all configurations
                n_bits_in_block = block_mask.sum()
                
                if n_bits_in_block <= 8:  # Enumerate up to 256 configs
                    z = self._enumerate_block_gibbs(
                        z, o_obs, mask, bi, i_end, bj, j_end,
                        block_mask, target_class, rng
                    )
                else:
                    # Too many configs, use MH with random flip proposal
                    z = self._mh_block_update(
                        z, o_obs, mask, bi, i_end, bj, j_end,
                        block_mask, target_class, rng
                    )
        
        return z
    
    def _enumerate_block_gibbs(
        self,
        z: np.ndarray,
        o_obs: np.ndarray,
        mask: np.ndarray,
        bi: int, i_end: int,
        bj: int, j_end: int,
        block_mask: np.ndarray,
        target_class: Optional[int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Enumerate all configurations for a small block."""
        z = z.copy()
        
        # Get masked positions in block
        masked_pos = np.argwhere(block_mask)
        n = len(masked_pos)
        
        if n == 0:
            return z
        
        # Enumerate all 2^n configurations
        energies = []
        for config in range(2 ** n):
            # Set block bits according to config
            for idx, (b, i, j) in enumerate(masked_pos):
                bit_val = (config >> idx) & 1
                z[b, bi + i, bj + j] = bit_val
            
            E = self.total_energy(z, o_obs, mask, target_class)
            energies.append(E)
        
        # Sample from Boltzmann
        energies = np.array(energies)
        energies = (energies - energies.min()) / self.config.temperature
        probs = np.exp(-energies)
        probs = probs / probs.sum()
        
        chosen = rng.choice(2 ** n, p=probs)
        
        # Set final configuration
        for idx, (b, i, j) in enumerate(masked_pos):
            bit_val = (chosen >> idx) & 1
            z[b, bi + i, bj + j] = bit_val
        
        return z
    
    def _mh_block_update(
        self,
        z: np.ndarray,
        o_obs: np.ndarray,
        mask: np.ndarray,
        bi: int, i_end: int,
        bj: int, j_end: int,
        block_mask: np.ndarray,
        target_class: Optional[int],
        rng: np.random.Generator,
        n_proposals: int = 5,
    ) -> np.ndarray:
        """MH update for large blocks: propose random flips."""
        z = z.copy()
        E_current = self.total_energy(z, o_obs, mask, target_class)
        
        masked_pos = np.argwhere(block_mask)
        
        for _ in range(n_proposals):
            # Propose: flip a random subset of masked bits
            n_flip = rng.integers(1, min(4, len(masked_pos)) + 1)
            flip_idx = rng.choice(len(masked_pos), size=n_flip, replace=False)
            
            z_prop = z.copy()
            for idx in flip_idx:
                b, i, j = masked_pos[idx]
                z_prop[b, bi + i, bj + j] = 1 - z_prop[b, bi + i, bj + j]
            
            E_prop = self.total_energy(z_prop, o_obs, mask, target_class)
            
            # MH acceptance
            dE = (E_prop - E_current) / self.config.temperature
            if dE < 0 or rng.random() < np.exp(-dE):
                z = z_prop
                E_current = E_prop
        
        return z
    
    def run(
        self,
        z_init: np.ndarray,
        o_obs: np.ndarray,
        pixel_mask: np.ndarray,
        bit_mask: np.ndarray,
        target_class: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run discrete inference.
        
        Args:
            z_init: initial latent state (k, H, W)
            o_obs: observed image (28, 28)
            pixel_mask: (28, 28) bool, True where pixel is observed
            bit_mask: (k, H, W) bool, True where bit is unknown
            target_class: optional target for task energy
            verbose: print progress
        
        Returns:
            z_final: optimized latent state
            stats: dictionary with energy trace, etc.
        """
        rng = np.random.default_rng(self.config.seed)
        z = z_init.copy()
        
        # Initialize masked bits randomly
        z[bit_mask] = rng.integers(0, 2, size=bit_mask.sum())
        
        stats = {'energies': [], 'delta_E': []}
        
        for step in range(self.config.n_steps):
            E_before = self.total_energy(z, o_obs, pixel_mask, target_class)
            
            # Block Gibbs step
            z = self.gibbs_step_block(
                z, o_obs, pixel_mask, bit_mask, target_class, rng
            )
            
            E_after = self.total_energy(z, o_obs, pixel_mask, target_class)
            
            stats['energies'].append(E_after)
            stats['delta_E'].append(E_before - E_after)
            
            if verbose and step % 10 == 0:
                print(f"  Step {step}: E = {E_after:.3f}, ΔE = {E_before - E_after:.3f}")
        
        return z, stats


def create_bit_mask_from_pixel_mask(
    pixel_mask: np.ndarray,
    n_bits: int,
    latent_size: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Create bit-level mask from pixel observation mask.
    
    A latent position (i,j) is masked if <threshold of its
    corresponding pixel patch is observed.
    
    Args:
        pixel_mask: (28, 28) bool, True = observed
        n_bits: number of bits per position
        latent_size: H=W of latent grid
        threshold: fraction threshold
    
    Returns:
        bit_mask: (k, H, W) bool, True = unknown bit
    """
    H = W = latent_size
    patch_h = 28 // H
    patch_w = 28 // W
    
    bit_mask = np.zeros((n_bits, H, W), dtype=bool)
    
    for i in range(H):
        for j in range(W):
            y0, y1 = i * patch_h, (i + 1) * patch_h
            x0, x1 = j * patch_w, (j + 1) * patch_w
            
            patch_obs = pixel_mask[y0:y1, x0:x1].mean()
            
            if patch_obs < threshold:
                # Most of patch is unobserved → all bits unknown
                bit_mask[:, i, j] = True
    
    return bit_mask


def estimate_sigma_squared(
    model_decode: Callable,
    model_encode: Callable,
    images: np.ndarray,
    n_samples: int = 500,
) -> float:
    """
    Estimate observation noise variance σ² from reconstruction residuals.
    
    σ² = E[(D(E(o)) - o)²]
    
    This is used to set the analog coupling weight without tuning.
    """
    residuals = []
    
    for i in range(min(n_samples, len(images))):
        o = images[i]
        z = model_encode(o)
        o_hat = model_decode(z)
        
        res = (o_hat - o) ** 2
        residuals.append(res.mean())
    
    return np.mean(residuals)
