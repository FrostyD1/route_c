"""
Route C Core Utilities
======================
Symmetry, topology, and neighborhood extraction for discrete world modeling.
"""

import numpy as np
from typing import Tuple, List, Optional, Callable

# ============================================================================
# NEIGHBORHOOD EXTRACTION
# ============================================================================

def get_neighbors_2d(
    grid: np.ndarray,
    i: int,
    j: int,
    neighborhood: str = "moore",
    boundary: str = "standard",
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Extract neighborhood around position (i, j) in a 2D grid.
    
    Args:
        grid: 2D array of shape (H, W) or (H, W, C)
        i, j: center position
        neighborhood: "moore" (3x3) or "von_neumann" (cross)
        boundary: "standard" (clip/ignore edges) or "torus" (wrap-around)
    
    Returns:
        values: array of neighbor values
        coords: list of (i, j) coordinates
    """
    H, W = grid.shape[:2]
    
    if neighborhood == "moore":
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),          (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
    elif neighborhood == "von_neumann":
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    elif neighborhood == "moore_with_center":
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),  (0, 0),  (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
    else:
        raise ValueError(f"Unknown neighborhood: {neighborhood}")
    
    values = []
    coords = []
    
    for di, dj in offsets:
        ni, nj = i + di, j + dj
        
        if boundary == "torus":
            ni = ni % H
            nj = nj % W
            values.append(grid[ni, nj])
            coords.append((ni, nj))
        elif boundary == "standard":
            if 0 <= ni < H and 0 <= nj < W:
                values.append(grid[ni, nj])
                coords.append((ni, nj))
        else:
            raise ValueError(f"Unknown boundary: {boundary}")
    
    return np.array(values), coords


def extract_context_window(
    grid: np.ndarray,
    i: int,
    j: int,
    window_size: int = 3,
    boundary: str = "standard",
    pad_value: int = 0,
) -> np.ndarray:
    """
    Extract a fixed-size context window centered at (i, j).
    
    Returns a (window_size, window_size) array with padding for edges
    if boundary is "standard", or wrap-around if "torus".
    """
    H, W = grid.shape[:2]
    half = window_size // 2
    
    window = np.full((window_size, window_size), pad_value, dtype=grid.dtype)
    
    for di in range(-half, half + 1):
        for dj in range(-half, half + 1):
            ni, nj = i + di, j + dj
            wi, wj = di + half, dj + half
            
            if boundary == "torus":
                ni = ni % H
                nj = nj % W
                window[wi, wj] = grid[ni, nj]
            elif boundary == "standard":
                if 0 <= ni < H and 0 <= nj < W:
                    window[wi, wj] = grid[ni, nj]
    
    return window


# ============================================================================
# SYMMETRY (D4 GROUP)
# ============================================================================

def get_d4_transforms() -> List[Callable]:
    """Return list of D4 group transforms (rotations + reflections)."""
    transforms = [
        lambda x: x,                          # identity
        lambda x: np.rot90(x, 1),             # 90 deg
        lambda x: np.rot90(x, 2),             # 180 deg
        lambda x: np.rot90(x, 3),             # 270 deg
        lambda x: np.fliplr(x),               # horizontal flip
        lambda x: np.flipud(x),               # vertical flip
        lambda x: np.rot90(np.fliplr(x), 1),  # flip + 90
        lambda x: np.rot90(np.fliplr(x), 3),  # flip + 270
    ]
    return transforms


def canonicalize_d4(patch: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Return the canonical (lexicographically smallest) D4 representative.
    
    Returns:
        canonical_patch: the canonical form
        transform_idx: which D4 transform was applied
    """
    transforms = get_d4_transforms()
    candidates = [(t(patch), idx) for idx, t in enumerate(transforms)]
    
    # Sort by flattened representation
    def patch_key(item):
        return tuple(item[0].flatten().tolist())
    
    candidates.sort(key=patch_key)
    return candidates[0]


def context_to_tuple(context: np.ndarray, exclude_center: bool = True) -> tuple:
    """
    Convert 3x3 context window to a hashable tuple.
    If exclude_center, returns 8-tuple (neighbors only).
    """
    flat = context.flatten()
    if exclude_center and len(flat) == 9:
        # Remove center element (index 4)
        return tuple(flat[:4]) + tuple(flat[5:])
    return tuple(flat)


def tuple_to_context(t: tuple, center_value: int = 0) -> np.ndarray:
    """Convert 8-tuple back to 3x3 context with given center."""
    if len(t) == 8:
        flat = list(t[:4]) + [center_value] + list(t[4:])
    else:
        flat = list(t)
    return np.array(flat).reshape(3, 3)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def grid_to_flat_index(i: int, j: int, W: int) -> int:
    """Convert 2D grid coordinates to flat index."""
    return i * W + j


def flat_index_to_grid(idx: int, W: int) -> Tuple[int, int]:
    """Convert flat index to 2D grid coordinates."""
    return idx // W, idx % W


def make_random_mask(shape: Tuple[int, int], mask_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Create a random binary mask with given ratio of masked (True) positions."""
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    n_mask = int(H * W * mask_ratio)
    indices = rng.choice(H * W, size=n_mask, replace=False)
    for idx in indices:
        i, j = flat_index_to_grid(idx, W)
        mask[i, j] = True
    return mask


def make_block_mask(
    shape: Tuple[int, int],
    block_size: Tuple[int, int],
    block_pos: Optional[Tuple[int, int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Create a block mask at given or random position."""
    H, W = shape
    bh, bw = block_size
    
    if block_pos is None:
        if rng is None:
            rng = np.random.default_rng(42)
        i = rng.integers(0, max(1, H - bh + 1))
        j = rng.integers(0, max(1, W - bw + 1))
    else:
        i, j = block_pos
    
    mask = np.zeros((H, W), dtype=bool)
    mask[i:i+bh, j:j+bw] = True
    return mask
