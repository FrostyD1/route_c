"""
Route C Energy Models
=====================
Modular energy functions for inference-time dynamics.

Energy Model Interface:
    energy(z, y=None, obs=None) -> float
    local_delta(z, i, j, new_token, y=None, obs=None) -> float

Available Energies:
    E1: TokenConditionalEnergy - based on -log P(z | neighborhood)
    E2: ClassifierEnergy - based on classification log-odds
    Combined: λ1*E1 + λ2*E2 for joint token+classification energy
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from abc import ABC, abstractmethod


class EnergyModel(ABC):
    """
    Abstract base class for energy functions in discrete inference.
    
    Energy is the quantity to minimize during inference. Lower energy = better state.
    
    Convention:
        E(z) = -log P(z | ...) + penalties + priors
    """
    
    @abstractmethod
    def energy(self, z: np.ndarray, y: Optional[int] = None, obs: Optional[np.ndarray] = None) -> float:
        """
        Compute total energy of configuration z.
        
        Args:
            z: token grid (H, W)
            y: optional class label (for classification tasks)
            obs: optional observation (original/partial image)
        
        Returns:
            energy: scalar energy value (lower is better)
        """
        pass
    
    def local_delta(
        self,
        z: np.ndarray,
        i: int,
        j: int,
        new_token: int,
        y: Optional[int] = None,
        obs: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute change in energy from flipping token at (i, j).
        
        Default implementation: compute full energy before/after.
        Subclasses should override for efficient local computation.
        
        Args:
            z: current token grid
            i, j: position to change
            new_token: proposed new token value
            y: optional class label
            obs: optional observation
        
        Returns:
            delta_E: E(z_new) - E(z_old)
        """
        old_token = z[i, j]
        old_energy = self.energy(z, y, obs)
        
        z[i, j] = new_token
        new_energy = self.energy(z, y, obs)
        z[i, j] = old_token  # Restore
        
        return new_energy - old_energy


class TokenConditionalEnergy(EnergyModel):
    """
    Energy based on token conditional model: E = -sum log P(z_ij | context).
    
    This is the negative log-likelihood under the count-based n-gram model.
    Lower energy = higher likelihood = more consistent with learned local structure.
    """
    
    def __init__(self, model, mask: Optional[np.ndarray] = None):
        """
        Args:
            model: TokenConditionalModel instance
            mask: optional (H, W) boolean mask, True where to compute energy
                  If None, compute over all positions
        """
        self.model = model
        self.mask = mask
    
    def energy(self, z: np.ndarray, y: Optional[int] = None, obs: Optional[np.ndarray] = None) -> float:
        """Compute -log P(z | model) = sum of -log P(z_ij | context)."""
        H, W = z.shape
        total = 0.0
        
        for i in range(H):
            for j in range(W):
                if self.mask is not None and not self.mask[i, j]:
                    continue
                
                context = self.model._extract_context(z, i, j)
                probs = self.model.predict_proba(context)
                total += -np.log(probs[z[i, j]] + 1e-10)
        
        return total
    
    def local_delta(
        self,
        z: np.ndarray,
        i: int,
        j: int,
        new_token: int,
        y: Optional[int] = None,
        obs: Optional[np.ndarray] = None
    ) -> float:
        """
        Efficient local energy change computation.
        
        Only considers:
        1. The token's own probability given its context
        2. How neighbors' probabilities change (their context includes this position)
        """
        H, W = z.shape
        old_token = z[i, j]
        
        if self.mask is not None and not self.mask[i, j]:
            return 0.0  # Position not in energy computation
        
        # Energy change at position (i, j) itself
        context = self.model._extract_context(z, i, j)
        old_probs = self.model.predict_proba(context)
        delta = -np.log(old_probs[new_token] + 1e-10) + np.log(old_probs[old_token] + 1e-10)
        
        # Energy change at neighbors (their context changes)
        z[i, j] = new_token
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                
                ni, nj = i + di, j + dj
                if self.model.boundary == "torus":
                    ni, nj = ni % H, nj % W
                elif not (0 <= ni < H and 0 <= nj < W):
                    continue
                
                if self.mask is not None and not self.mask[ni, nj]:
                    continue
                
                neighbor_token = z[ni, nj]
                
                # New context (with new token at i,j)
                context_new = self.model._extract_context(z, ni, nj)
                probs_new = self.model.predict_proba(context_new)
                
                # Old context
                z[i, j] = old_token
                context_old = self.model._extract_context(z, ni, nj)
                probs_old = self.model.predict_proba(context_old)
                z[i, j] = new_token
                
                delta += -np.log(probs_new[neighbor_token] + 1e-10) + np.log(probs_old[neighbor_token] + 1e-10)
        
        z[i, j] = old_token  # Restore
        return delta


class ClassifierEnergy(EnergyModel):
    """
    Energy based on classification score: E = -log P(y | z).
    
    When used with inference, this drives tokens toward configurations
    that support the given class label. Useful for:
    - Occlusion recovery (fill tokens to maximize P(y | z))
    - Adversarial robustness
    - Class-conditional generation
    """
    
    def __init__(
        self,
        classifier,
        vocab_size: int,
        spatial_bins: Tuple[int, int] = (2, 2),
        target_class: Optional[int] = None,
    ):
        """
        Args:
            classifier: LogOddsClassifier instance
            vocab_size: number of token types (K)
            spatial_bins: bins for feature extraction
            target_class: if set, energy is -log P(this class | z)
                         if None, uses the most likely class
        """
        self.classifier = classifier
        self.vocab_size = vocab_size
        self.spatial_bins = spatial_bins
        self.target_class = target_class
    
    def _extract_features(self, z: np.ndarray) -> np.ndarray:
        """Extract spatial histogram features from token grid."""
        # Handle both package and direct imports
        try:
            from route_c.learning import token_grid_to_spatial_histogram
        except ImportError:
            from ..learning import token_grid_to_spatial_histogram
        return token_grid_to_spatial_histogram(z, self.vocab_size, self.spatial_bins)
    
    def energy(self, z: np.ndarray, y: Optional[int] = None, obs: Optional[np.ndarray] = None) -> float:
        """
        Compute classification energy.
        
        If y is provided, returns -log P(y | z).
        If target_class is set, uses that.
        Otherwise, uses argmax class (which gives 0 energy if perfectly confident).
        """
        features = self._extract_features(z)
        probs = self.classifier.predict_proba(features)
        
        if y is not None:
            target = y
        elif self.target_class is not None:
            target = self.target_class
        else:
            target = np.argmax(probs)
        
        return -np.log(probs[target] + 1e-10)
    
    def local_delta(
        self,
        z: np.ndarray,
        i: int,
        j: int,
        new_token: int,
        y: Optional[int] = None,
        obs: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute classification energy change.
        
        Note: This is not as local as TokenConditionalEnergy because 
        the classifier looks at global/spatial histogram features.
        However, single token changes have small effect on histograms.
        """
        old_token = z[i, j]
        
        old_energy = self.energy(z, y, obs)
        z[i, j] = new_token
        new_energy = self.energy(z, y, obs)
        z[i, j] = old_token
        
        return new_energy - old_energy


class CombinedEnergy(EnergyModel):
    """
    Linear combination of multiple energy models: E = sum(λ_i * E_i).
    
    Useful for balancing multiple objectives:
    - Token coherence (local structure)
    - Classification confidence
    - Prior terms (sparsity, smoothness)
    """
    
    def __init__(self, energies: List[Tuple[float, EnergyModel]]):
        """
        Args:
            energies: list of (weight, EnergyModel) tuples
                     E_total = sum(weight * energy_model.energy(...))
        """
        self.energies = energies
    
    def energy(self, z: np.ndarray, y: Optional[int] = None, obs: Optional[np.ndarray] = None) -> float:
        total = 0.0
        for weight, model in self.energies:
            total += weight * model.energy(z, y, obs)
        return total
    
    def local_delta(
        self,
        z: np.ndarray,
        i: int,
        j: int,
        new_token: int,
        y: Optional[int] = None,
        obs: Optional[np.ndarray] = None
    ) -> float:
        total = 0.0
        for weight, model in self.energies:
            total += weight * model.local_delta(z, i, j, new_token, y, obs)
        return total


class ObservationEnergy(EnergyModel):
    """
    Energy penalizing deviation from observed/partial data.
    
    E = λ * ||decode(z) - obs||^2  (at observed positions)
    
    Useful for:
    - Inpainting (fill masked regions while matching observed pixels)
    - Denoising
    - Reconstruction under constraints
    """
    
    def __init__(
        self,
        codebook: np.ndarray,
        patch_size: int,
        obs_mask: Optional[np.ndarray] = None,
        weight: float = 1.0,
    ):
        """
        Args:
            codebook: (K, D) VQ codebook for decoding
            patch_size: patch size for decoding
            obs_mask: (H, W) boolean mask, True where observation is valid
            weight: weight for observation term
        """
        self.codebook = codebook
        self.patch_size = patch_size
        self.obs_mask = obs_mask
        self.weight = weight
    
    def energy(self, z: np.ndarray, y: Optional[int] = None, obs: Optional[np.ndarray] = None) -> float:
        if obs is None:
            return 0.0
        
        # Handle both package and direct imports
        try:
            from route_c.boundary import decode_tokens
        except ImportError:
            from ..boundary import decode_tokens
        reconstructed = decode_tokens(z, self.codebook, self.patch_size)
        
        diff = (reconstructed.astype(np.float32) - obs.astype(np.float32)) ** 2
        
        if self.obs_mask is not None:
            diff = diff * self.obs_mask
        
        return self.weight * np.mean(diff)


# ============================================================================
# ENERGY-BASED INFERENCE
# ============================================================================

def gibbs_fill_energy(
    token_grid: np.ndarray,
    mask: np.ndarray,
    energy_model: EnergyModel,
    vocab_size: int,
    n_steps: int = 10,
    seed: int = 42,
    y: Optional[int] = None,
    obs: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fill masked tokens using Gibbs sampling with energy-based acceptance.
    
    At each masked position, samples from:
        P(token) ∝ exp(-E(z with token at position))
    
    Args:
        token_grid: (H, W) current token grid
        mask: (H, W) boolean, True where to sample
        energy_model: EnergyModel instance
        vocab_size: number of token types
        n_steps: number of Gibbs sweeps
        seed: random seed
        y: optional target class for ClassifierEnergy
        obs: optional observation for ObservationEnergy
        verbose: print progress
    
    Returns:
        filled_grid: filled token grid
        stats: dictionary with energies, etc.
    """
    rng = np.random.default_rng(seed)
    grid = token_grid.copy()
    
    # Initialize masked positions randomly
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if mask[i, j]:
                grid[i, j] = rng.integers(vocab_size)
    
    stats = {'energies': [], 'acceptance_rates': []}
    
    H, W = grid.shape
    positions = [(i, j) for i in range(H) for j in range(W) if mask[i, j]]
    
    for step in range(n_steps):
        rng.shuffle(positions)
        
        for i, j in positions:
            # Compute energy for each possible token
            energies = np.zeros(vocab_size)
            for k in range(vocab_size):
                delta = energy_model.local_delta(grid, i, j, k, y, obs)
                old_e = 0  # Relative energies
                energies[k] = delta
            
            # Sample from Boltzmann distribution
            # Shift for numerical stability
            energies = energies - energies.min()
            probs = np.exp(-energies)
            probs = probs / (probs.sum() + 1e-10)
            
            grid[i, j] = rng.choice(vocab_size, p=probs)
        
        current_energy = energy_model.energy(grid, y, obs)
        stats['energies'].append(current_energy)
        
        if verbose and (step % max(1, n_steps // 5) == 0 or step == n_steps - 1):
            print(f"  Gibbs step {step}: E = {current_energy:.3f}")
    
    return grid, stats


def metropolis_fill_energy(
    token_grid: np.ndarray,
    mask: np.ndarray,
    energy_model: EnergyModel,
    vocab_size: int,
    n_steps: int = 50,
    temperature: float = 1.0,
    seed: int = 42,
    y: Optional[int] = None,
    obs: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fill masked tokens using Metropolis-Hastings with energy-based acceptance.
    
    Args:
        token_grid: (H, W) current token grid
        mask: (H, W) boolean, True where to sample
        energy_model: EnergyModel instance
        vocab_size: number of token types
        n_steps: number of MH sweeps
        temperature: temperature (higher = more random)
        seed: random seed
        y: optional target class
        obs: optional observation
        verbose: print progress
    
    Returns:
        filled_grid: filled token grid
        stats: dictionary with energies, acceptance rates, delta_E distribution
    """
    rng = np.random.default_rng(seed)
    grid = token_grid.copy()
    
    # Initialize masked positions randomly
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if mask[i, j]:
                grid[i, j] = rng.integers(vocab_size)
    
    stats = {'energies': [], 'acceptance_rates': [], 'delta_energies': []}
    
    H, W = grid.shape
    positions = [(i, j) for i in range(H) for j in range(W) if mask[i, j]]
    
    for step in range(n_steps):
        rng.shuffle(positions)
        n_accepted = 0
        
        for i, j in positions:
            old_token = grid[i, j]
            new_token = rng.integers(vocab_size)
            
            if new_token == old_token:
                continue
            
            delta_e = energy_model.local_delta(grid, i, j, new_token, y, obs)
            stats['delta_energies'].append(delta_e)
            
            # Metropolis acceptance
            if delta_e < 0:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta_e / temperature)
            
            if accept:
                grid[i, j] = new_token
                n_accepted += 1
        
        acc_rate = n_accepted / max(1, len(positions))
        stats['acceptance_rates'].append(acc_rate)
        
        current_energy = energy_model.energy(grid, y, obs)
        stats['energies'].append(current_energy)
        
        if verbose and (step % max(1, n_steps // 5) == 0 or step == n_steps - 1):
            print(f"  MH step {step}: E = {current_energy:.3f}, acc = {acc_rate:.1%}")
    
    return grid, stats
