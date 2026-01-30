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
    
    E(z) = λ_core * E_core(z) + λ_obs * E_obs(z;o,M) + λ_rope * E_rope(z)
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
    
    def energy_core(self, z: torch.Tensor) -> float:
        """E_core = -Σ log p(z_i | neigh)"""
        with torch.no_grad():
            z_t = z.unsqueeze(0) if z.dim() == 3 else z
            z_t = z_t.to(self.device)
            logits = self.model.local_pred(z_t)
            loss = F.binary_cross_entropy_with_logits(logits, z_t, reduction='sum')
            return loss.item()
    
    def energy_obs(self, z: torch.Tensor, o_obs: np.ndarray, mask: np.ndarray) -> float:
        """E_obs = (1/2σ²) ||M ⊙ (D(z) - o)||²"""
        with torch.no_grad():
            z_t = z.unsqueeze(0) if z.dim() == 3 else z
            z_t = z_t.to(self.device)
            o_hat = self.model.decode(z_t)[0, 0].cpu().numpy()
            
            diff = (o_hat - o_obs) * mask
            mse = (diff ** 2).sum() / max(1, mask.sum())
            return mse / (2 * self.sigma_sq)
    
    def energy_rope(self, z: torch.Tensor) -> float:
        """E_rope from D-RoPE module"""
        z_t = z if isinstance(z, torch.Tensor) else torch.from_numpy(z).float()
        z_t = z_t.to(self.device)
        return self.drope.compute_energy(z_t).item()
    
    def total_energy(
        self,
        z: torch.Tensor,
        o_obs: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total energy with breakdown.
        
        Returns:
            total: scalar
            breakdown: dict of component energies
        """
        e_core = self.energy_core(z) if self.lambda_core > 0 else 0.0
        e_obs = self.energy_obs(z, o_obs, mask) if self.lambda_obs > 0 else 0.0
        e_rope = self.energy_rope(z) if self.lambda_rope > 0 else 0.0
        
        total = (self.lambda_core * e_core +
                 self.lambda_obs * e_obs +
                 self.lambda_rope * e_rope)
        
        return total, {
            'core': e_core,
            'obs': e_obs,
            'rope': e_rope,
            'total': total,
        }


# ============================================================================
# SOLVER WITH D-ROPE
# ============================================================================

class BlockGibbsWithRope:
    """
    Block Gibbs sampler with D-RoPE energy integration.
    
    Tracks per-sweep metrics for mixing time analysis.
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
        track_metrics: bool = True,
        classifier_fn = None,  # For tracking classification
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Run block Gibbs inference with D-RoPE.
        
        Args:
            z_init: initial latent (k, H, W)
            o_obs: observed image (28, 28)
            pixel_mask: (28, 28) float, 1=observed
            bit_mask: (k, H, W) bool, True=unknown
            n_sweeps: number of Gibbs sweeps
            track_metrics: whether to track per-sweep metrics
            classifier_fn: optional function z -> logits for tracking
        
        Returns:
            z_final: optimized latent
            metrics: dict with per-sweep tracking
        """
        rng = np.random.default_rng(42)
        z = z_init.clone().to(self.device)
        k, H, W = z.shape
        bh, bw = self.block_size
        
        # Initialize masked bits randomly
        bit_mask_t = torch.from_numpy(bit_mask).to(self.device)
        z[bit_mask_t] = torch.randint(0, 2, (bit_mask_t.sum().item(),), 
                                       device=self.device, dtype=z.dtype)
        
        # Tracking
        metrics = {
            'energy': [],
            'energy_breakdown': [],
            'mse_occluded': [],
            'pred_label': [],
            'pred_confidence': [],
        }
        
        # Initial metrics
        if track_metrics:
            E, breakdown = self.energy.total_energy(z, o_obs, pixel_mask)
            metrics['energy'].append(E)
            metrics['energy_breakdown'].append(breakdown)
            
            if classifier_fn is not None:
                with torch.no_grad():
                    logits = classifier_fn(z.unsqueeze(0))
                    probs = F.softmax(logits, dim=1)
                    conf, pred = probs.max(dim=1)
                    metrics['pred_label'].append(pred.item())
                    metrics['pred_confidence'].append(conf.item())
        
        # Run sweeps
        for sweep in range(n_sweeps):
            # Iterate over blocks
            for bi in range(0, H, bh):
                for bj in range(0, W, bw):
                    i_end = min(bi + bh, H)
                    j_end = min(bj + bw, W)
                    
                    # Get masked bits in this block
                    block_mask = bit_mask[:, bi:i_end, bj:j_end]
                    masked_pos = np.argwhere(block_mask)
                    
                    if len(masked_pos) == 0:
                        continue
                    
                    n_bits_block = len(masked_pos)
                    
                    if n_bits_block <= 6:  # Enumerate
                        energies = []
                        for config in range(2 ** n_bits_block):
                            for idx, (b, i, j) in enumerate(masked_pos):
                                z[b, bi + i, bj + j] = (config >> idx) & 1
                            E, _ = self.energy.total_energy(z, o_obs, pixel_mask)
                            energies.append(E)
                        
                        energies = np.array(energies)
                        energies = energies - energies.min()
                        probs = np.exp(-energies)
                        probs = probs / (probs.sum() + 1e-10)
                        
                        chosen = rng.choice(2 ** n_bits_block, p=probs)
                        for idx, (b, i, j) in enumerate(masked_pos):
                            z[b, bi + i, bj + j] = (chosen >> idx) & 1
                    
                    else:  # MH
                        E_curr, _ = self.energy.total_energy(z, o_obs, pixel_mask)
                        for _ in range(3):
                            n_flip = rng.integers(1, min(4, n_bits_block) + 1)
                            flip_idx = rng.choice(n_bits_block, size=n_flip, replace=False)
                            
                            z_prop = z.clone()
                            for idx in flip_idx:
                                b, i, j = masked_pos[idx]
                                z_prop[b, bi + i, bj + j] = 1 - z_prop[b, bi + i, bj + j]
                            
                            E_prop, _ = self.energy.total_energy(z_prop, o_obs, pixel_mask)
                            
                            if E_prop < E_curr or rng.random() < np.exp(-(E_prop - E_curr)):
                                z = z_prop
                                E_curr = E_prop
            
            # Track metrics after each sweep
            if track_metrics:
                E, breakdown = self.energy.total_energy(z, o_obs, pixel_mask)
                metrics['energy'].append(E)
                metrics['energy_breakdown'].append(breakdown)
                
                if classifier_fn is not None:
                    with torch.no_grad():
                        logits = classifier_fn(z.unsqueeze(0))
                        probs = F.softmax(logits, dim=1)
                        conf, pred = probs.max(dim=1)
                        metrics['pred_label'].append(pred.item())
                        metrics['pred_confidence'].append(conf.item())
        
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
