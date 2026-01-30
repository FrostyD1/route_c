"""
Route C: D-RoPE (Discrete Rotary Position Embedding)
=====================================================

D-RoPE enables long-range routing in discrete z-space using:
1. Permutation alignment Π_Δ (2D cyclic shift)
2. XOR match (alignment residual)
3. Threshold gating (content addressing)

Key difference from Transformer RoPE:
- NO sin/cos embeddings
- NO floating-point attention scores
- Pure discrete operations: shift, XOR, popcount, threshold compare

This is COMPILE-FRIENDLY: all operations map to wires/permutation, XOR gates,
popcount circuits, and threshold comparators.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CandidateConfig:
    """Configuration for candidate set construction"""
    mode: str = 'anchors+radii'  # 'anchors+radii', 'grid', 'random'
    density: str = 'sparse'  # 'sparse', 'medium', 'dense'
    n_anchors: int = 8  # Number of anchor positions
    radii: List[int] = None  # Multi-scale radii
    
    def __post_init__(self):
        if self.radii is None:
            if self.density == 'sparse':
                self.radii = [1, 2]  # ~8 candidates/site
            elif self.density == 'medium':
                self.radii = [1, 2, 3]  # ~16 candidates/site
            else:  # dense
                self.radii = [1, 2, 3, 4]  # ~32 candidates/site


# ============================================================================
# CORE OPERATIONS (Compile-Friendly)
# ============================================================================

def shift2d(z: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """
    2D cyclic shift of latent grid.
    
    Args:
        z: (B, k, H, W) or (k, H, W) binary tensor
        dx: shift in x (column) direction
        dy: shift in y (row) direction
    
    Returns:
        z_shifted: same shape, cyclically shifted
    
    Hardware: This is just wire routing (no gates needed)!
    """
    if z.dim() == 3:
        z = z.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    # torch.roll implements cyclic shift
    z_shifted = torch.roll(z, shifts=(dy, dx), dims=(2, 3))
    
    if squeeze:
        z_shifted = z_shifted.squeeze(0)
    
    return z_shifted


def xor_match(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    XOR between two binary tensors (alignment residual).
    
    Args:
        z1, z2: binary tensors of same shape
    
    Returns:
        residual: binary tensor where 1 indicates mismatch
    
    Hardware: XOR gates
    """
    return (z1 != z2).float()


def hamming_distance(z1: torch.Tensor, z2: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Hamming distance (popcount of XOR).
    
    Args:
        z1, z2: (B, k, H, W) binary tensors
        dim: dimension to sum over (1 for k-bits)
    
    Returns:
        distance: (B, H, W) integer tensor
    
    Hardware: XOR gates + popcount circuit
    """
    xor = xor_match(z1, z2)
    return xor.sum(dim=dim)


def threshold_gate(distance: torch.Tensor, threshold: int) -> torch.Tensor:
    """
    Threshold gating for content addressing.
    
    Args:
        distance: Hamming distances
        threshold: gate opens if distance < threshold
    
    Returns:
        gate: binary tensor (1 = connected)
    
    Hardware: Comparator (magnitude compare)
    """
    return (distance < threshold).float()


# ============================================================================
# CANDIDATE SET CONSTRUCTION
# ============================================================================

def build_candidate_offsets(H: int, W: int, config: CandidateConfig) -> List[Tuple[int, int]]:
    """
    Build list of (dy, dx) offsets for candidate positions.
    
    For each site j, candidates are positions i = j + offset (mod H, W).
    
    Returns:
        offsets: list of (dy, dx) tuples
    """
    offsets = []
    
    if config.mode == 'anchors+radii':
        # Anchor positions at corners and edges
        anchors = [
            (0, 0),  # self (for baseline)
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 4-neighbors
        ]
        
        # Add multi-scale radii
        for r in config.radii:
            # Add positions at distance r in cardinal + diagonal directions
            anchors.extend([
                (-r, 0), (r, 0), (0, -r), (0, r),  # Cardinal
                (-r, -r), (-r, r), (r, -r), (r, r),  # Diagonal
            ])
        
        # Remove duplicates and self
        seen = set()
        for dy, dx in anchors:
            if (dy, dx) != (0, 0) and (dy, dx) not in seen:
                offsets.append((dy, dx))
                seen.add((dy, dx))
    
    elif config.mode == 'grid':
        # Regular grid sampling
        step = max(1, H // 4) if config.density == 'sparse' else max(1, H // 6)
        for dy in range(-H//2, H//2 + 1, step):
            for dx in range(-W//2, W//2 + 1, step):
                if (dy, dx) != (0, 0):
                    offsets.append((dy, dx))
    
    return offsets


def build_candidate_sets(
    H: int,
    W: int,
    config: CandidateConfig,
    device: torch.device = None,
) -> Dict:
    """
    Build candidate set structure for all positions.
    
    Returns:
        dict with:
            'offsets': list of (dy, dx)
            'n_candidates': number of candidates per site
    """
    offsets = build_candidate_offsets(H, W, config)
    
    return {
        'offsets': offsets,
        'n_candidates': len(offsets),
        'H': H,
        'W': W,
    }


# ============================================================================
# E_ROPE ENERGY COMPUTATION
# ============================================================================

class DRoPEEnergy:
    """
    D-RoPE long-range energy term.
    
    E_rope(z) = Σ_j min_{i∈C(j), Gate=1} d(i→j)
    
    Where:
    - C(j) is candidate set for position j
    - d(i→j) = Hamming(align(z_i, Δ), z_j)
    - Gate = 1 if d < threshold
    """
    
    def __init__(
        self,
        H: int = 7,
        W: int = 7,
        k: int = 8,
        threshold: int = None,
        config: CandidateConfig = None,
        device: torch.device = None,
    ):
        self.H = H
        self.W = W
        self.k = k
        self.device = device or torch.device('cpu')
        
        # Threshold: default to 0.25 * k
        self.threshold = threshold if threshold is not None else max(1, round(0.25 * k))
        
        # Build candidate sets
        self.config = config or CandidateConfig()
        self.candidates = build_candidate_sets(H, W, self.config, device)
        self.offsets = self.candidates['offsets']
        self.n_candidates = self.candidates['n_candidates']
    
    def compute_all_distances(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute Hamming distances from all candidates to all positions.
        
        Args:
            z: (B, k, H, W) or (k, H, W) binary tensor
        
        Returns:
            distances: (B, n_candidates, H, W) or (n_candidates, H, W)
        """
        squeeze = z.dim() == 3
        if squeeze:
            z = z.unsqueeze(0)
        
        B, k, H, W = z.shape
        distances = []
        
        for dy, dx in self.offsets:
            # Shift z to align candidate i with position j
            # If j = i + offset, then to compare z_i with z_j,
            # we shift z by -offset to bring z_i to position j
            z_aligned = shift2d(z, -dx, -dy)
            
            # Hamming distance at each position
            d = hamming_distance(z, z_aligned, dim=1)  # (B, H, W)
            distances.append(d)
        
        distances = torch.stack(distances, dim=1)  # (B, n_candidates, H, W)
        
        if squeeze:
            distances = distances.squeeze(0)
        
        return distances
    
    def compute_energy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute E_rope(z) = Σ_j min_{i∈C(j), Gate=1} d(i→j)
        
        Args:
            z: (B, k, H, W) or (k, H, W)
        
        Returns:
            energy: (B,) or scalar
        """
        squeeze = z.dim() == 3
        if squeeze:
            z = z.unsqueeze(0)
        
        # Get all distances
        distances = self.compute_all_distances(z)  # (B, n_candidates, H, W)
        
        # Apply threshold gate
        gate = threshold_gate(distances, self.threshold)  # (B, n_candidates, H, W)
        
        # Masked min: set ungated distances to large value
        large_val = self.k + 1
        masked_distances = distances + (1 - gate) * large_val
        
        # Min over candidates
        min_distances, _ = masked_distances.min(dim=1)  # (B, H, W)
        
        # Clamp: if all candidates are gated out, use penalty
        min_distances = torch.clamp(min_distances, max=self.k)
        
        # Sum over spatial positions
        energy = min_distances.sum(dim=(1, 2))  # (B,)
        
        if squeeze:
            energy = energy.squeeze(0)
        
        return energy
    
    def compute_delta_energy(
        self,
        z: torch.Tensor,
        z_new: torch.Tensor,
        changed_positions: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Incremental ΔE_rope computation for efficiency.
        
        Only recomputes energy for positions affected by the change.
        
        Args:
            z: old state (k, H, W)
            z_new: new state (k, H, W)
            changed_positions: list of (i, j) that changed
        
        Returns:
            delta_E: scalar
        """
        # For simplicity, we compute full energy difference
        # A truly incremental version would track which positions are affected
        # by each changed position through the candidate relationships
        
        E_old = self.compute_energy(z)
        E_new = self.compute_energy(z_new)
        
        return E_new - E_old
    
    def compute_energy_per_position(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute per-position E_rope contributions (for visualization).
        
        Returns:
            energy_map: (H, W) tensor
        """
        squeeze = z.dim() == 3
        if squeeze:
            z = z.unsqueeze(0)
        
        distances = self.compute_all_distances(z)
        gate = threshold_gate(distances, self.threshold)
        
        large_val = self.k + 1
        masked_distances = distances + (1 - gate) * large_val
        min_distances, _ = masked_distances.min(dim=1)
        min_distances = torch.clamp(min_distances, max=self.k)
        
        if squeeze:
            min_distances = min_distances.squeeze(0)
        
        return min_distances


# ============================================================================
# COMBINED ENERGY FOR INFERENCE
# ============================================================================

class CombinedEnergyWithRope:
    """
    Combined energy for Route C inference with D-RoPE.
    ALL TORCH, no numpy in energy computation.
    """
    
    def __init__(
        self,
        model,  # RouteCModel
        drope: DRoPEEnergy,
        sigma_sq: float,
        lambda_core: float = 1.0,
        lambda_obs: float = 1.0,
        lambda_rope: float = 0.5,
        device: torch.device = None,
    ):
        self.model = model
        self.drope = drope
        self.sigma_sq = sigma_sq
        self.lambda_core = lambda_core
        self.lambda_obs = lambda_obs
        self.lambda_rope = lambda_rope
        self.device = device or torch.device('cpu')
        # Cache for observation
        self._o_obs_t = None
        self._mask_t = None
        self._mask_sum = 1.0
    
    def set_observation(self, o_obs: np.ndarray, mask: np.ndarray):
        """Cache observation tensors on GPU - call once per sample"""
        self._o_obs_t = torch.from_numpy(o_obs.astype(np.float32)).to(self.device)
        self._mask_t = torch.from_numpy(mask.astype(np.float32)).to(self.device)
        self._mask_sum = self._mask_t.sum().clamp(min=1.0)
    
    def energy(self, z: torch.Tensor) -> float:
        """Compute total energy - ALL ON GPU, returns scalar"""
        with torch.inference_mode():
            z_t = z.unsqueeze(0) if z.dim() == 3 else z
            
            # E_core
            if self.lambda_core > 0:
                logits = self.model.local_pred(z_t)
                e_core = F.binary_cross_entropy_with_logits(logits, z_t, reduction='sum')
            else:
                e_core = 0.0
            
            # E_obs - ALL TORCH
            if self.lambda_obs > 0 and self._o_obs_t is not None:
                o_hat = self.model.decode(z_t)[0, 0]
                diff = (o_hat - self._o_obs_t) * self._mask_t
                mse = (diff * diff).sum() / self._mask_sum
                e_obs = mse / (2 * self.sigma_sq)
            else:
                e_obs = 0.0
            
            # E_rope
            if self.lambda_rope > 0:
                e_rope = self.drope.compute_energy(z if z.dim() == 3 else z[0])
            else:
                e_rope = 0.0
            
            total = self.lambda_core * e_core + self.lambda_obs * e_obs + self.lambda_rope * e_rope
            return total.item() if isinstance(total, torch.Tensor) else total


# ============================================================================
# SOLVER WITH D-ROPE
# ============================================================================

class BlockGibbsWithRope:
    """
    Block Gibbs sampler with D-RoPE - FAST version.
    Whole-block proposal, in-place flip, no numpy in hot path.
    """
    
    def __init__(
        self,
        combined_energy: CombinedEnergyWithRope,
        block_size: Tuple[int, int] = (2, 2),
        device: torch.device = None,
    ):
        self.energy = combined_energy
        self.block_size = block_size
        self.device = device or torch.device('cpu')
    
    def run(
        self,
        z_init: torch.Tensor,
        o_obs: np.ndarray,
        pixel_mask: np.ndarray,
        bit_mask: np.ndarray,
        n_sweeps: int = 30,
        track_metrics: bool = False,
        classifier_fn = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Run block Gibbs with whole-block proposals."""
        z = z_init.clone().to(self.device)
        k, H, W = z.shape
        bh, bw = self.block_size
        
        # Cache observation on GPU once
        self.energy.set_observation(o_obs, pixel_mask)
        
        # Convert bit_mask to torch
        bit_mask_t = torch.from_numpy(bit_mask).to(self.device)
        
        # Initialize masked bits randomly
        z[bit_mask_t] = torch.randint(0, 2, (bit_mask_t.sum().item(),), device=self.device, dtype=z.dtype)
        
        # Current energy
        E_curr = self.energy.energy(z)
        
        metrics = {'energy': [E_curr]}
        
        for sweep in range(n_sweeps):
            for bi in range(0, H, bh):
                for bj in range(0, W, bw):
                    i_end = min(bi + bh, H)
                    j_end = min(bj + bw, W)
                    
                    # Get block mask
                    block_bit_mask = bit_mask_t[:, bi:i_end, bj:j_end]
                    if not block_bit_mask.any():
                        continue
                    
                    # IN-PLACE FLIP whole block (only masked bits)
                    z[:, bi:i_end, bj:j_end][block_bit_mask] = 1 - z[:, bi:i_end, bj:j_end][block_bit_mask]
                    
                    # Compute proposal energy
                    E_prop = self.energy.energy(z)
                    
                    # MH accept/reject
                    dE = E_prop - E_curr
                    if dE < 0 or torch.rand(1).item() < np.exp(-dE):
                        E_curr = E_prop  # Accept
                    else:
                        # Reject: flip back
                        z[:, bi:i_end, bj:j_end][block_bit_mask] = 1 - z[:, bi:i_end, bj:j_end][block_bit_mask]
            
            if track_metrics:
                metrics['energy'].append(E_curr)
        
        return z, metrics


# ============================================================================
# UTILITY: Compute mixing time
# ============================================================================

def compute_mixing_time(
    metrics: Dict,
    target_fraction: float = 0.95,
) -> Dict:
    """
    Compute mixing time metrics.
    
    Args:
        metrics: dict from solver with 'energy' list
        target_fraction: fraction of final improvement to reach
    
    Returns:
        dict with mixing time statistics
    """
    energies = np.array(metrics['energy'])
    
    if len(energies) < 2:
        return {'sweeps_95': len(energies), 'converged': False}
    
    E_init = energies[0]
    E_final = energies[-1]
    improvement = E_init - E_final
    
    if improvement <= 0:
        return {'sweeps_95': len(energies), 'converged': False}
    
    target = E_init - target_fraction * improvement
    
    # Find first sweep where energy drops below target
    for i, E in enumerate(energies):
        if E <= target:
            return {'sweeps_95': i, 'converged': True}
    
    return {'sweeps_95': len(energies), 'converged': False}
