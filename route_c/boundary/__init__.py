"""
Route C Boundary Translation
============================
Continuous ↔ Discrete boundary translation:
- ADC/DAC (thresholding, dequantization)
- Patchification (image -> patches -> token grid)
- VQ Codebook (k-means based)
"""

import numpy as np
from typing import Tuple, Optional, List


# ============================================================================
# BASIC ADC/DAC (Threshold-based)
# ============================================================================

def adc_threshold(image: np.ndarray, threshold: float = 128.0) -> np.ndarray:
    """Simple ADC: binarize image with threshold."""
    return (image > threshold).astype(np.uint8)


def adc_multi_threshold(image: np.ndarray, thresholds: List[float]) -> np.ndarray:
    """Multi-level ADC: quantize to len(thresholds)+1 levels."""
    result = np.zeros_like(image, dtype=np.uint8)
    for t in thresholds:
        result += (image > t).astype(np.uint8)
    return result


def dac_binary(binary: np.ndarray, low: float = 0.0, high: float = 255.0) -> np.ndarray:
    """Simple DAC: convert binary to continuous values."""
    return np.where(binary, high, low).astype(np.float32)


# ============================================================================
# PATCHIFICATION
# ============================================================================

def patchify(
    image: np.ndarray,
    patch_size: int,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Extract non-overlapping or strided patches from image.
    
    Args:
        image: 2D array (H, W) or 3D array (H, W, C)
        patch_size: size of square patches
        stride: step between patches (default: patch_size for non-overlapping)
    
    Returns:
        patches: array of shape (n_patches, patch_size, patch_size) or (n_patches, patch_size, patch_size, C)
        grid_shape: (n_rows, n_cols) of the patch grid
    """
    if stride is None:
        stride = patch_size
    
    H, W = image.shape[:2]
    n_rows = (H - patch_size) // stride + 1
    n_cols = (W - patch_size) // stride + 1
    
    patches = []
    for i in range(n_rows):
        for j in range(n_cols):
            y = i * stride
            x = j * stride
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    
    patches = np.array(patches)
    return patches, (n_rows, n_cols)


def unpatchify(
    patches: np.ndarray,
    grid_shape: Tuple[int, int],
    patch_size: int,
    stride: Optional[int] = None,
) -> np.ndarray:
    """
    Reconstruct image from patches (simple tiling, no overlap averaging).
    
    Args:
        patches: array of shape (n_patches, patch_size, patch_size, ...)
        grid_shape: (n_rows, n_cols)
        patch_size: size of square patches
        stride: step between patches (default: patch_size)
    
    Returns:
        image: reconstructed image
    """
    if stride is None:
        stride = patch_size
    
    n_rows, n_cols = grid_shape
    H = (n_rows - 1) * stride + patch_size
    W = (n_cols - 1) * stride + patch_size
    
    # Determine output shape
    if patches.ndim == 3:
        image = np.zeros((H, W), dtype=patches.dtype)
    else:
        image = np.zeros((H, W) + patches.shape[3:], dtype=patches.dtype)
    
    # For overlapping patches, we use last-write-wins
    # A proper implementation would average overlapping regions
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            y = i * stride
            x = j * stride
            image[y:y+patch_size, x:x+patch_size] = patches[idx]
            idx += 1
    
    return image


# ============================================================================
# K-MEANS VQ CODEBOOK
# ============================================================================

def kmeans_codebook_fit(
    patches: np.ndarray,
    K: int,
    n_iters: int = 100,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Fit k-means codebook to patches (pure numpy implementation).
    
    Args:
        patches: (N, ...) array of patches (will be flattened)
        K: number of codebook entries
        n_iters: number of k-means iterations
        seed: random seed
        verbose: print progress
    
    Returns:
        codebook: (K, D) array of centroids
        assignments: (N,) array of cluster assignments
        losses: list of per-iteration losses (MSE)
    """
    rng = np.random.default_rng(seed)
    
    # Flatten patches to (N, D)
    N = patches.shape[0]
    original_shape = patches.shape[1:]
    X = patches.reshape(N, -1).astype(np.float32)
    D = X.shape[1]
    
    # Initialize codebook with k-means++ style
    codebook = np.zeros((K, D), dtype=np.float32)
    
    # First centroid: random
    idx = rng.integers(N)
    codebook[0] = X[idx]
    
    # Remaining centroids: k-means++ initialization
    for k in range(1, K):
        # Compute distances to nearest existing centroid
        dists = np.min([np.sum((X - codebook[j:j+1])**2, axis=1) for j in range(k)], axis=0)
        probs = dists / (dists.sum() + 1e-10)
        idx = rng.choice(N, p=probs)
        codebook[k] = X[idx]
    
    losses = []
    assignments = np.zeros(N, dtype=np.int32)
    
    for it in range(n_iters):
        # E-step: assign to nearest centroid
        # Compute distances: (N, K)
        dists = np.zeros((N, K), dtype=np.float32)
        for k in range(K):
            dists[:, k] = np.sum((X - codebook[k:k+1])**2, axis=1)
        
        assignments = np.argmin(dists, axis=1)
        loss = np.mean(np.min(dists, axis=1))
        losses.append(loss)
        
        if verbose and (it % 10 == 0 or it == n_iters - 1):
            print(f"  K-means iter {it}: loss = {loss:.4f}")
        
        # M-step: update centroids
        new_codebook = np.zeros_like(codebook)
        counts = np.zeros(K)
        
        for k in range(K):
            mask = assignments == k
            if mask.sum() > 0:
                new_codebook[k] = X[mask].mean(axis=0)
                counts[k] = mask.sum()
            else:
                # Reinitialize empty cluster
                new_codebook[k] = X[rng.integers(N)]
                counts[k] = 0
        
        codebook = new_codebook
        
        # Check convergence
        if it > 0 and abs(losses[-1] - losses[-2]) < 1e-6:
            if verbose:
                print(f"  Converged at iteration {it}")
            break
    
    return codebook, assignments, losses


def encode_tokens(
    image: np.ndarray,
    codebook: np.ndarray,
    patch_size: int,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Encode image into token grid using codebook.
    
    Args:
        image: 2D array (H, W)
        codebook: (K, D) array of centroids
        patch_size: size of patches
        stride: step between patches
    
    Returns:
        token_grid: 2D array of token indices (grid_H, grid_W)
        grid_shape: (n_rows, n_cols)
    """
    patches, grid_shape = patchify(image, patch_size, stride)
    N = patches.shape[0]
    X = patches.reshape(N, -1).astype(np.float32)
    K = codebook.shape[0]
    
    # Find nearest codebook entry for each patch
    dists = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        dists[:, k] = np.sum((X - codebook[k:k+1])**2, axis=1)
    
    tokens = np.argmin(dists, axis=1)
    token_grid = tokens.reshape(grid_shape)
    
    return token_grid, grid_shape


def decode_tokens(
    token_grid: np.ndarray,
    codebook: np.ndarray,
    patch_size: int,
    stride: Optional[int] = None,
) -> np.ndarray:
    """
    Decode token grid back to image using codebook.
    
    Args:
        token_grid: 2D array of token indices
        codebook: (K, D) array of centroids
        patch_size: size of patches
        stride: step between patches
    
    Returns:
        image: reconstructed image
    """
    grid_shape = token_grid.shape
    n_rows, n_cols = grid_shape
    
    # Reshape codebook entries to patches
    K, D = codebook.shape
    codebook_patches = codebook.reshape(K, patch_size, patch_size)
    
    # Get patches from tokens
    tokens_flat = token_grid.flatten()
    patches = codebook_patches[tokens_flat]
    
    # Reconstruct image
    image = unpatchify(patches, grid_shape, patch_size, stride)
    
    return image


# ============================================================================
# BATCH OPERATIONS
# ============================================================================

def encode_batch(
    images: np.ndarray,
    codebook: np.ndarray,
    patch_size: int,
    stride: Optional[int] = None,
) -> np.ndarray:
    """
    Encode batch of images to token grids.
    
    Args:
        images: (N, H, W) array
        codebook: (K, D) centroids
        patch_size: patch size
        stride: stride
    
    Returns:
        token_grids: (N, grid_H, grid_W) array
    """
    N = images.shape[0]
    token_grids = []
    
    for i in range(N):
        token_grid, _ = encode_tokens(images[i], codebook, patch_size, stride)
        token_grids.append(token_grid)
    
    return np.stack(token_grids)


def decode_batch(
    token_grids: np.ndarray,
    codebook: np.ndarray,
    patch_size: int,
    stride: Optional[int] = None,
) -> np.ndarray:
    """
    Decode batch of token grids to images.
    
    Args:
        token_grids: (N, grid_H, grid_W) array
        codebook: (K, D) centroids
        patch_size: patch size
        stride: stride
    
    Returns:
        images: (N, H, W) array
    """
    N = token_grids.shape[0]
    images = []
    
    for i in range(N):
        image = decode_tokens(token_grids[i], codebook, patch_size, stride)
        images.append(image)
    
    return np.stack(images)
