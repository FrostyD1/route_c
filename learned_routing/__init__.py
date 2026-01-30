"""
Route C: Learned Discrete Routing
==================================
Upgrades fixed XOR/Hamming gate to learnable weighted Hamming with LSH candidate generation.

Key components:
1. LearnedHammingGate: w^T(z_i XOR z_j) < τ  with learnable w, τ
2. LSHCandidateGenerator: bit-sampling LSH for Hamming-space ANN
3. LearnedDRoPEEnergy: drop-in replacement for DRoPEEnergy with learned gate

References:
- Routing Transformer (Roy et al., 2021): learned k-means routing for sparse attention
- Reformer (Kitaev et al., 2020): LSH attention, bit-sampling is canonical LSH for Hamming
- XNOR-Net (Rastegari et al., 2016): w^T(z_i XOR z_j) = learned XNOR-popcount
- BinaryConnect (Courbariaux et al., 2015): STE for binary parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# ============================================================================
# LSH CANDIDATE GENERATOR (Bit-Sampling for Hamming Space)
# ============================================================================

class LSHCandidateGenerator:
    """
    Bit-sampling LSH for candidate generation in binary latent space.

    For z ∈ {0,1}^k, the canonical LSH is bit-sampling:
        hash_l(z) = z[bits_l]  where bits_l is a random subset of b bit indices

    Collision probability: Pr[hash(z) = hash(z')] = (1 - d_H(z,z')/k)^b

    Multi-round (L rounds) reduces miss probability exponentially:
        Pr[miss in ALL rounds] = (1 - (1 - d_H/k)^b)^L

    Reference: Indyk & Motwani (1998), Section 3; O'Donnell et al. (2014).
    """

    def __init__(
        self,
        k: int,
        b: int = 4,
        n_rounds: int = 4,
        seed: int = 42,
    ):
        """
        Args:
            k: number of bits per token
            b: bits per hash (controls bucket granularity, 2^b buckets)
            n_rounds: number of independent hash rounds (controls recall)
            seed: random seed for reproducibility
        """
        self.k = k
        self.b = b
        self.n_rounds = n_rounds

        rng = np.random.default_rng(seed)
        # Pre-sample bit indices for each round
        self.bit_indices = []
        for _ in range(n_rounds):
            bits = rng.choice(k, size=b, replace=False)
            self.bit_indices.append(bits)

    def hash_tokens(self, z: torch.Tensor, round_idx: int) -> torch.Tensor:
        """
        Hash binary tokens by extracting selected bits.

        Args:
            z: (..., k) binary tensor
            round_idx: which hash round to use

        Returns:
            hash_vals: (...,) integer hash values in [0, 2^b)
        """
        bits = self.bit_indices[round_idx]
        selected = z[..., bits]  # (..., b)
        # Convert binary vector to integer
        powers = (2 ** torch.arange(self.b, device=z.device, dtype=z.dtype)).flip(0)
        return (selected * powers).sum(dim=-1).long()

    def build_candidate_sets(
        self,
        z_grid: torch.Tensor,
        max_candidates: int = 32,
    ) -> torch.Tensor:
        """
        Build candidate sets for all positions in a token grid.

        Args:
            z_grid: (k, H, W) binary tensor — the latent grid
            max_candidates: maximum candidates per position

        Returns:
            candidates: (H*W, max_candidates, 2) — indices (i,j) of candidates for each position
            n_valid: (H*W,) — number of valid candidates per position
        """
        k, H, W = z_grid.shape
        N = H * W

        # Reshape to (N, k) for hashing
        z_flat = z_grid.permute(1, 2, 0).reshape(N, k)  # (N, k)

        # Collect candidates across all rounds
        candidate_sets = [set() for _ in range(N)]

        for r in range(self.n_rounds):
            hashes = self.hash_tokens(z_flat, r)  # (N,)

            # Group by hash value
            bucket_map = {}
            for idx in range(N):
                h = hashes[idx].item()
                if h not in bucket_map:
                    bucket_map[h] = []
                bucket_map[h].append(idx)

            # Add same-bucket tokens as candidates
            for idx in range(N):
                h = hashes[idx].item()
                for other in bucket_map[h]:
                    if other != idx:
                        candidate_sets[idx].add(other)

        # Convert to padded tensor
        candidates = torch.full((N, max_candidates), -1, dtype=torch.long, device=z_grid.device)
        n_valid = torch.zeros(N, dtype=torch.long, device=z_grid.device)

        for idx in range(N):
            cands = sorted(candidate_sets[idx])[:max_candidates]
            n_valid[idx] = len(cands)
            for c_idx, c in enumerate(cands):
                candidates[idx, c_idx] = c

        return candidates, n_valid


# ============================================================================
# LEARNED HAMMING GATE
# ============================================================================

class LearnedHammingGate(nn.Module):
    """
    Learned weighted Hamming gate:
        gate(i→j) = 1[ w^T (z_i XOR z_j) < τ ]

    where w ∈ R_+^k is a learnable per-bit importance vector
    and τ is a learnable threshold.

    This generalizes the fixed gate: popcount(z_i XOR z_j) < τ
    (which is w = 1_k, uniform weights).

    Gradient flow:
    - w is continuous → standard backprop
    - z is binary → STE (already handled by encoder)
    - Gate is discrete → use soft relaxation during training:
        soft_gate = σ((τ - w^T(z_i XOR z_j)) / temperature)

    Reference: XNOR-Net (Rastegari et al., 2016) — w^T(z_i XOR z_j)
    is exactly a learned XNOR-popcount with per-bit weights.
    """

    def __init__(self, k: int, init_threshold: float = None, temperature: float = 1.0):
        super().__init__()
        self.k = k

        # Per-bit importance weights (initialized uniform, constrained positive via softplus)
        self.w_logit = nn.Parameter(torch.zeros(k))

        # Learnable threshold
        if init_threshold is None:
            init_threshold = 0.25 * k
        self.tau_logit = nn.Parameter(torch.tensor(float(init_threshold)))

        self.temperature = temperature

    @property
    def w(self):
        """Positive weights via softplus."""
        return F.softplus(self.w_logit)

    @property
    def tau(self):
        """Threshold (unconstrained)."""
        return self.tau_logit

    def weighted_hamming(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted Hamming distance: w^T |z1 - z2|

        Args:
            z1, z2: (..., k) binary tensors

        Returns:
            distance: (...) weighted distance
        """
        xor = (z1 != z2).float()  # (..., k)
        return (xor * self.w).sum(dim=-1)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        hard: bool = False,
    ) -> torch.Tensor:
        """
        Compute gate values.

        Args:
            z1, z2: (..., k) binary tensors
            hard: if True, return hard 0/1 gate; if False, return soft sigmoid gate

        Returns:
            gate: (...) gate values in [0, 1]
        """
        dist = self.weighted_hamming(z1, z2)

        if hard:
            return (dist < self.tau).float()
        else:
            # Soft gate for training (differentiable)
            return torch.sigmoid((self.tau - dist) / self.temperature)


# ============================================================================
# LEARNED D-ROPE ENERGY
# ============================================================================

class LearnedDRoPEEnergy(nn.Module):
    """
    D-RoPE with learned weighted Hamming gate + optional LSH candidates.

    E_rope(z; w, τ) = Σ_j min_{i∈C(j), gate_w(i,j)>0} [w^T(z_i XOR z_j)]

    Upgrades from fixed DRoPEEnergy:
    1. Weighted Hamming distance (learnable w)
    2. Learnable threshold τ
    3. LSH-based candidate generation (optional, replaces fixed spatial offsets)
    """

    def __init__(
        self,
        H: int = 7,
        W: int = 7,
        k: int = 8,
        use_lsh: bool = False,
        lsh_b: int = 4,
        lsh_rounds: int = 4,
        max_candidates: int = 16,
        temperature: float = 1.0,
        device: torch.device = None,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.k = k
        self.use_lsh = use_lsh
        self.max_candidates = max_candidates
        self.device = device or torch.device('cpu')

        # Learned gate
        self.gate = LearnedHammingGate(k, temperature=temperature)

        # LSH generator (if used)
        if use_lsh:
            self.lsh = LSHCandidateGenerator(k, b=lsh_b, n_rounds=lsh_rounds)
        else:
            self.lsh = None

        # Fallback: fixed spatial offsets (same as original DRoPE)
        self._fixed_offsets = self._build_fixed_offsets()

    def _build_fixed_offsets(self) -> List[Tuple[int, int]]:
        """Build fixed spatial candidate offsets (fallback when not using LSH)."""
        offsets = []
        seen = set()
        for r in [1, 2]:
            for dy, dx in [(-r,0),(r,0),(0,-r),(0,r),(-r,-r),(-r,r),(r,-r),(r,r)]:
                if (dy, dx) not in seen:
                    offsets.append((dy, dx))
                    seen.add((dy, dx))
        # Also add 4-neighbors
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            if (dy, dx) not in seen:
                offsets.append((dy, dx))
                seen.add((dy, dx))
        return offsets

    def compute_energy(self, z: torch.Tensor, hard_gate: bool = False) -> torch.Tensor:
        """
        Compute E_rope with learned gate.

        Uses DIFFERENTIABLE weighted-sum form (not min):
            E_rope(z; w) = Σ_j Σ_{i∈C(j)} g_{ij} · w^T(z_i ⊕ z_j)
        This is fully differentiable w.r.t. w, τ, and (via STE) z.

        Args:
            z: (k, H, W) or (B, k, H, W) binary tensor
            hard_gate: use hard (non-differentiable) gate

        Returns:
            energy: scalar or (B,)
        """
        squeeze = z.dim() == 3
        if squeeze:
            z = z.unsqueeze(0)

        B, k, H, W = z.shape
        N = H * W

        # Reshape to (B, N, k) for pairwise computation
        z_flat = z.permute(0, 2, 3, 1).reshape(B, N, k)  # (B, N, k)

        if self.use_lsh and self.lsh is not None:
            # Use LSH candidates (per-sample, using first sample for structure)
            candidates, n_valid = self.lsh.build_candidate_sets(
                z[0], max_candidates=self.max_candidates
            )
            return self._energy_with_candidates(z_flat, candidates, n_valid, hard_gate, B, N)
        else:
            # Use fixed spatial offsets
            return self._energy_with_offsets(z, z_flat, hard_gate, B, N, k, H, W, squeeze)

    def _energy_with_offsets(self, z, z_flat, hard_gate, B, N, k, H, W, squeeze):
        """Compute energy using fixed spatial offsets.

        Uses differentiable WEIGHTED SUM form:
            E = Σ_j Σ_{i∈C(j)} g_{ij} · d_{ij}
        where g_{ij} = soft gate, d_{ij} = weighted Hamming distance.
        This is fully differentiable (no hard min).
        """
        all_distances = []
        all_gates = []

        for dy, dx in self._fixed_offsets:
            z_shifted = torch.roll(z, shifts=(-dy, -dx), dims=(2, 3))
            z_shifted_flat = z_shifted.permute(0, 2, 3, 1).reshape(B, N, k)

            dist = self.gate.weighted_hamming(z_flat, z_shifted_flat)  # (B, N)
            gate_val = self.gate(z_flat, z_shifted_flat, hard=hard_gate)  # (B, N)

            all_distances.append(dist)
            all_gates.append(gate_val)

        distances = torch.stack(all_distances, dim=2)  # (B, N, n_cand)
        gates = torch.stack(all_gates, dim=2)  # (B, N, n_cand)

        # Differentiable weighted sum: E = Σ g_{ij} · d_{ij}
        weighted = gates * distances  # (B, N, n_cand)
        energy_per_pos = weighted.sum(dim=2)  # (B, N)
        energy = energy_per_pos.sum(dim=1)  # (B,)

        if squeeze:
            energy = energy.squeeze(0)
        return energy

    def _energy_with_candidates(self, z_flat, candidates, n_valid, hard_gate, B, N):
        """Compute energy using LSH candidates. Uses differentiable weighted sum."""
        max_c = candidates.shape[1]

        valid_mask = (candidates >= 0).float().to(z_flat.device)  # (N, max_c)
        cand_clamped = candidates.clamp(min=0)  # (N, max_c)

        all_energies = []
        for b_idx in range(B):
            z_b = z_flat[b_idx]  # (N, k)

            z_query = z_b.unsqueeze(1).expand(-1, max_c, -1)  # (N, max_c, k)
            z_cand = z_b[cand_clamped]  # (N, max_c, k)

            dist = self.gate.weighted_hamming(z_query, z_cand)  # (N, max_c)
            gate_val = self.gate(z_query, z_cand, hard=hard_gate)  # (N, max_c)

            # Weighted sum (differentiable), zeroing invalid candidates
            weighted = gate_val * dist * valid_mask  # (N, max_c)
            all_energies.append(weighted.sum())

        return torch.stack(all_energies)


# ============================================================================
# COMBINED ENERGY WITH LEARNED ROUTING
# ============================================================================

class CombinedEnergyLearned:
    """
    Combined energy for Route C inference with learned D-RoPE.
    Drop-in replacement for CombinedEnergyWithRope.
    """

    def __init__(
        self,
        model,
        learned_drope: LearnedDRoPEEnergy,
        sigma_sq: float,
        lambda_core: float = 1.0,
        lambda_obs: float = 1.0,
        lambda_rope: float = 0.5,
        device: torch.device = None,
    ):
        self.model = model
        self.drope = learned_drope
        self.sigma_sq = sigma_sq
        self.lambda_core = lambda_core
        self.lambda_obs = lambda_obs
        self.lambda_rope = lambda_rope
        self.device = device or torch.device('cpu')
        self._o_obs_t = None
        self._mask_t = None
        self._mask_sum = 1.0

    def set_observation(self, o_obs: np.ndarray, mask: np.ndarray):
        self._o_obs_t = torch.from_numpy(o_obs.astype(np.float32)).to(self.device)
        self._mask_t = torch.from_numpy(mask.astype(np.float32)).to(self.device)
        self._mask_sum = self._mask_t.sum().clamp(min=1.0)

    def energy(self, z: torch.Tensor) -> float:
        with torch.inference_mode():
            z_t = z.unsqueeze(0) if z.dim() == 3 else z

            # E_core
            if self.lambda_core > 0:
                logits = self.model.local_pred(z_t)
                e_core = F.binary_cross_entropy_with_logits(logits, z_t, reduction='sum')
            else:
                e_core = 0.0

            # E_obs
            if self.lambda_obs > 0 and self._o_obs_t is not None:
                o_hat = self.model.decode(z_t)[0, 0]
                diff = (o_hat - self._o_obs_t) * self._mask_t
                mse = (diff * diff).sum() / self._mask_sum
                e_obs = mse / (2 * self.sigma_sq)
            else:
                e_obs = 0.0

            # E_rope (learned)
            if self.lambda_rope > 0:
                e_rope = self.drope.compute_energy(
                    z if z.dim() == 3 else z[0], hard_gate=True
                )
            else:
                e_rope = 0.0

            total = self.lambda_core * e_core + self.lambda_obs * e_obs + self.lambda_rope * e_rope
            return total.item() if isinstance(total, torch.Tensor) else total
