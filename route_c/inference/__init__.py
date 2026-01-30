"""
Route C Inference Module
========================
Inference-time dynamics for masked token completion:
- Gibbs sampling
- Metropolis-Hastings
- Block proposals
- Energy-based inference (see energy.py)
"""

import numpy as np
import sys
import os
from typing import Tuple, Optional, List, Callable

# Handle both package and direct imports
try:
    from ..learning import TokenConditionalModel
except ImportError:
    try:
        from learning import TokenConditionalModel
    except ImportError:
        # Add parent to path
        _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        from learning import TokenConditionalModel


# ============================================================================
# GIBBS SAMPLING
# ============================================================================

def gibbs_step(
    token_grid: np.ndarray,
    mask: np.ndarray,
    model: TokenConditionalModel,
    rng: np.random.Generator,
    order: str = "random",
) -> np.ndarray:
    """
    Single Gibbs sweep over masked positions.
    
    Args:
        token_grid: (H, W) current token grid
        mask: (H, W) boolean, True where tokens are masked (to be sampled)
        model: TokenConditionalModel for P(token | context)
        rng: random number generator
        order: "random" or "sequential"
    
    Returns:
        updated token_grid (modified in place)
    """
    H, W = token_grid.shape
    
    # Get masked positions
    positions = [(i, j) for i in range(H) for j in range(W) if mask[i, j]]
    
    if order == "random":
        rng.shuffle(positions)
    
    for i, j in positions:
        context = model._extract_context(token_grid, i, j)
        probs = model.predict_proba(context)
        token_grid[i, j] = rng.choice(model.vocab_size, p=probs)
    
    return token_grid


def gibbs_fill(
    token_grid: np.ndarray,
    mask: np.ndarray,
    model: TokenConditionalModel,
    n_steps: int = 10,
    seed: int = 42,
    verbose: bool = False,
    return_history: bool = False,
) -> np.ndarray:
    """
    Fill masked tokens using Gibbs sampling.
    
    Args:
        token_grid: (H, W) token grid with masked positions
        mask: (H, W) boolean, True where tokens are masked
        model: TokenConditionalModel
        n_steps: number of Gibbs sweeps
        seed: random seed
        verbose: print progress
        return_history: if True, return list of grids at each step
    
    Returns:
        filled_grid: token grid with filled masked positions
        (optionally) history: list of intermediate grids
    """
    rng = np.random.default_rng(seed)
    grid = token_grid.copy()
    
    # Initialize masked positions with random tokens
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if mask[i, j]:
                grid[i, j] = rng.integers(model.vocab_size)
    
    history = [grid.copy()] if return_history else None
    
    for step in range(n_steps):
        grid = gibbs_step(grid, mask, model, rng)
        
        if verbose and (step % max(1, n_steps // 5) == 0 or step == n_steps - 1):
            ll = model.log_likelihood(grid, mask)
            print(f"  Gibbs step {step}: masked LL = {ll:.2f}")
        
        if return_history:
            history.append(grid.copy())
    
    if return_history:
        return grid, history
    return grid


# ============================================================================
# METROPOLIS-HASTINGS
# ============================================================================

def compute_local_energy(
    token_grid: np.ndarray,
    i: int,
    j: int,
    model: TokenConditionalModel,
) -> float:
    """
    Compute local energy (negative log-prob) at position (i, j).
    Includes the token's own probability given its context.
    """
    context = model._extract_context(token_grid, i, j)
    token = token_grid[i, j]
    probs = model.predict_proba(context)
    return -np.log(probs[token] + 1e-10)


def compute_delta_energy(
    token_grid: np.ndarray,
    i: int,
    j: int,
    new_token: int,
    model: TokenConditionalModel,
) -> float:
    """
    Compute change in energy from flipping token at (i, j).
    
    This is an approximation that considers:
    1. The token's own probability given its context
    2. The effect on neighbors' probabilities
    """
    H, W = token_grid.shape
    old_token = token_grid[i, j]
    
    # Energy contribution from this position
    context = model._extract_context(token_grid, i, j)
    old_probs = model.predict_proba(context)
    old_energy = -np.log(old_probs[old_token] + 1e-10)
    new_energy = -np.log(old_probs[new_token] + 1e-10)
    
    delta = new_energy - old_energy
    
    # Effect on neighbors (optional, for more accurate energy)
    # Temporarily change token
    token_grid[i, j] = new_token
    
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            
            ni, nj = i + di, j + dj
            if model.boundary == "torus":
                ni, nj = ni % H, nj % W
            elif not (0 <= ni < H and 0 <= nj < W):
                continue
            
            # Neighbor's energy changes because its context changed
            neighbor_token = token_grid[ni, nj]
            
            # New context (with new token)
            context_new = model._extract_context(token_grid, ni, nj)
            probs_new = model.predict_proba(context_new)
            
            # Old context (restore to compute)
            token_grid[i, j] = old_token
            context_old = model._extract_context(token_grid, ni, nj)
            probs_old = model.predict_proba(context_old)
            token_grid[i, j] = new_token  # Switch back
            
            delta += -np.log(probs_new[neighbor_token] + 1e-10) + np.log(probs_old[neighbor_token] + 1e-10)
    
    # Restore original token
    token_grid[i, j] = old_token
    
    return delta


def metropolis_step(
    token_grid: np.ndarray,
    mask: np.ndarray,
    model: TokenConditionalModel,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, int, List[float]]:
    """
    Single Metropolis sweep over masked positions.
    
    Args:
        token_grid: (H, W) current token grid
        mask: (H, W) boolean, True where tokens are masked
        model: TokenConditionalModel
        rng: random number generator
        temperature: temperature for acceptance (higher = more random)
    
    Returns:
        updated token_grid
        n_accepted: number of accepted moves
        delta_energies: list of ΔE values for analysis
    """
    H, W = token_grid.shape
    positions = [(i, j) for i in range(H) for j in range(W) if mask[i, j]]
    rng.shuffle(positions)
    
    n_accepted = 0
    delta_energies = []
    
    for i, j in positions:
        old_token = token_grid[i, j]
        
        # Propose new token (uniform proposal)
        new_token = rng.integers(model.vocab_size)
        if new_token == old_token:
            continue
        
        # Compute energy change
        delta_e = compute_delta_energy(token_grid, i, j, new_token, model)
        delta_energies.append(delta_e)
        
        # Accept/reject
        if delta_e < 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-delta_e / temperature)
        
        if accept:
            token_grid[i, j] = new_token
            n_accepted += 1
    
    return token_grid, n_accepted, delta_energies


def metropolis_fill(
    token_grid: np.ndarray,
    mask: np.ndarray,
    model: TokenConditionalModel,
    n_steps: int = 50,
    temperature: float = 1.0,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Fill masked tokens using Metropolis-Hastings sampling.
    
    Returns:
        filled_grid: token grid with filled masked positions
        stats: dictionary with acceptance rates, energies, etc.
    """
    rng = np.random.default_rng(seed)
    grid = token_grid.copy()
    
    # Initialize masked positions randomly
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if mask[i, j]:
                grid[i, j] = rng.integers(model.vocab_size)
    
    stats = {
        'acceptance_rates': [],
        'delta_energies': [],
        'log_likelihoods': [],
    }
    
    n_masked = mask.sum()
    
    for step in range(n_steps):
        grid, n_accepted, deltas = metropolis_step(grid, mask, model, rng, temperature)
        
        acc_rate = n_accepted / max(1, n_masked)
        stats['acceptance_rates'].append(acc_rate)
        stats['delta_energies'].extend(deltas)
        
        if verbose and (step % max(1, n_steps // 5) == 0 or step == n_steps - 1):
            ll = model.log_likelihood(grid, mask)
            stats['log_likelihoods'].append(ll)
            print(f"  Metropolis step {step}: accept={acc_rate:.2%}, masked LL = {ll:.2f}")
    
    return grid, stats


# ============================================================================
# BLOCK PROPOSALS
# ============================================================================

def block_gibbs_step(
    token_grid: np.ndarray,
    mask: np.ndarray,
    model: TokenConditionalModel,
    rng: np.random.Generator,
    block_size: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    """
    Gibbs sweep with block proposals (update blocks of tokens together).
    """
    H, W = token_grid.shape
    bh, bw = block_size
    
    # Iterate over blocks
    for bi in range(0, H, bh):
        for bj in range(0, W, bw):
            # Check if any position in block is masked
            block_masked = False
            for di in range(min(bh, H - bi)):
                for dj in range(min(bw, W - bj)):
                    if mask[bi + di, bj + dj]:
                        block_masked = True
                        break
                if block_masked:
                    break
            
            if not block_masked:
                continue
            
            # Sample each masked position in block (sequentially within block)
            for di in range(min(bh, H - bi)):
                for dj in range(min(bw, W - bj)):
                    i, j = bi + di, bj + dj
                    if mask[i, j]:
                        context = model._extract_context(token_grid, i, j)
                        probs = model.predict_proba(context)
                        token_grid[i, j] = rng.choice(model.vocab_size, p=probs)
    
    return token_grid


def block_gibbs_fill(
    token_grid: np.ndarray,
    mask: np.ndarray,
    model: TokenConditionalModel,
    n_steps: int = 10,
    block_size: Tuple[int, int] = (2, 2),
    seed: int = 42,
    verbose: bool = False,
) -> np.ndarray:
    """Fill masked tokens using block Gibbs sampling."""
    rng = np.random.default_rng(seed)
    grid = token_grid.copy()
    
    # Initialize masked positions randomly
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if mask[i, j]:
                grid[i, j] = rng.integers(model.vocab_size)
    
    for step in range(n_steps):
        grid = block_gibbs_step(grid, mask, model, rng, block_size)
        
        if verbose and (step % max(1, n_steps // 5) == 0 or step == n_steps - 1):
            ll = model.log_likelihood(grid, mask)
            print(f"  Block Gibbs step {step}: masked LL = {ll:.2f}")
    
    return grid


# ============================================================================
# TEMPLATE-BASED PROPOSALS
# ============================================================================

def template_proposal_fill(
    token_grid: np.ndarray,
    mask: np.ndarray,
    templates: np.ndarray,
    model: TokenConditionalModel,
    n_candidates: int = 10,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Fill masked region using template candidates from training data.
    
    Args:
        token_grid: (H, W) token grid with masked region
        mask: (H, W) boolean mask
        templates: (N, H, W) candidate template grids
        model: TokenConditionalModel for scoring
        n_candidates: number of random templates to try
        seed: random seed
    
    Returns:
        filled_grid: best filled grid
        best_idx: index of best template
    """
    rng = np.random.default_rng(seed)
    
    N = templates.shape[0]
    candidate_indices = rng.choice(N, size=min(n_candidates, N), replace=False)
    
    best_ll = -np.inf
    best_grid = token_grid.copy()
    best_idx = -1
    
    for idx in candidate_indices:
        # Create candidate by copying template values into masked positions
        candidate = token_grid.copy()
        candidate[mask] = templates[idx][mask]
        
        # Score candidate
        ll = model.log_likelihood(candidate, mask)
        
        if ll > best_ll:
            best_ll = ll
            best_grid = candidate
            best_idx = idx
    
    if verbose:
        print(f"  Template proposal: best LL = {best_ll:.2f} from template {best_idx}")
    
    return best_grid, best_idx


# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def analyze_delta_energies(deltas: List[float]) -> dict:
    """Analyze distribution of ΔE values."""
    deltas = np.array(deltas)
    return {
        'mean': np.mean(deltas),
        'std': np.std(deltas),
        'min': np.min(deltas),
        'max': np.max(deltas),
        'median': np.median(deltas),
        'pct_negative': (deltas < 0).mean(),
        'n_samples': len(deltas),
    }


def compute_token_accuracy(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute accuracy of predicted tokens at masked positions."""
    return (predicted[mask] == ground_truth[mask]).mean()
