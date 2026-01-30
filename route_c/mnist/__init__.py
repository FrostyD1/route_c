"""
Route C MNIST Module
====================
Data loading and utilities for MNIST experiments.
"""

import numpy as np
from typing import Tuple, Optional
import os


def load_mnist_numpy(data_dir: str = "./data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset using torchvision (for download) then convert to numpy.
    
    Returns:
        train_images: (60000, 28, 28) uint8 array
        train_labels: (60000,) int array
        test_images: (10000, 28, 28) uint8 array
        test_labels: (10000,) int array
    """
    try:
        from torchvision import datasets
        import torch
        
        # Download and load
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True)
        
        # Convert to numpy
        train_images = train_dataset.data.numpy()
        train_labels = train_dataset.targets.numpy()
        test_images = test_dataset.data.numpy()
        test_labels = test_dataset.targets.numpy()
        
        return train_images, train_labels, test_images, test_labels
        
    except ImportError:
        raise ImportError("torchvision required for MNIST download. Install with: pip install torchvision")


def subsample_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    n_samples: int,
    seed: int = 42,
    stratified: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample dataset, optionally maintaining class balance.
    
    Args:
        images: (N, H, W) array
        labels: (N,) array
        n_samples: number of samples to keep
        seed: random seed
        stratified: if True, maintain class proportions
    
    Returns:
        subsampled images and labels
    """
    rng = np.random.default_rng(seed)
    N = len(images)
    
    if n_samples >= N:
        return images, labels
    
    if stratified:
        n_classes = len(np.unique(labels))
        per_class = n_samples // n_classes
        remainder = n_samples % n_classes
        
        indices = []
        for c in range(n_classes):
            class_indices = np.where(labels == c)[0]
            n_take = per_class + (1 if c < remainder else 0)
            n_take = min(n_take, len(class_indices))
            indices.extend(rng.choice(class_indices, size=n_take, replace=False))
        
        indices = np.array(indices)
        rng.shuffle(indices)
    else:
        indices = rng.choice(N, size=n_samples, replace=False)
    
    return images[indices], labels[indices]


def normalize_images(images: np.ndarray, scale: float = 255.0) -> np.ndarray:
    """Normalize images to [0, 1] range."""
    return images.astype(np.float32) / scale


def create_occluded_images(
    images: np.ndarray,
    occlusion_size: Tuple[int, int] = (14, 14),
    occlusion_pos: Optional[Tuple[int, int]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create occluded versions of images with a rectangular block set to zero.
    
    Args:
        images: (N, H, W) array
        occlusion_size: (height, width) of occlusion block
        occlusion_pos: if None, random position for each image
        seed: random seed
    
    Returns:
        occluded_images: (N, H, W) with occlusion
        occlusion_masks: (N, H, W) boolean, True where occluded
    """
    rng = np.random.default_rng(seed)
    N, H, W = images.shape
    oh, ow = occlusion_size
    
    occluded = images.copy()
    masks = np.zeros((N, H, W), dtype=bool)
    
    for i in range(N):
        if occlusion_pos is None:
            y = rng.integers(0, max(1, H - oh + 1))
            x = rng.integers(0, max(1, W - ow + 1))
        else:
            y, x = occlusion_pos
        
        occluded[i, y:y+oh, x:x+ow] = 0
        masks[i, y:y+oh, x:x+ow] = True
    
    return occluded, masks


def pixel_mask_to_token_mask(
    pixel_mask: np.ndarray,
    patch_size: int,
    stride: Optional[int] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert pixel-level mask to token-level mask.
    A token is masked if more than threshold fraction of its pixels are masked.
    
    Args:
        pixel_mask: (H, W) boolean mask
        patch_size: size of patches
        stride: step between patches
        threshold: fraction threshold
    
    Returns:
        token_mask: (grid_H, grid_W) boolean mask
    """
    if stride is None:
        stride = patch_size
    
    H, W = pixel_mask.shape
    n_rows = (H - patch_size) // stride + 1
    n_cols = (W - patch_size) // stride + 1
    
    token_mask = np.zeros((n_rows, n_cols), dtype=bool)
    
    for i in range(n_rows):
        for j in range(n_cols):
            y = i * stride
            x = j * stride
            patch_mask = pixel_mask[y:y+patch_size, x:x+patch_size]
            if patch_mask.mean() > threshold:
                token_mask[i, j] = True
    
    return token_mask


def compute_reconstruction_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> dict:
    """
    Compute reconstruction quality metrics.
    
    Returns dict with:
        mse: mean squared error
        mae: mean absolute error
        psnr: peak signal-to-noise ratio
    """
    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    mae = np.mean(np.abs(original.astype(np.float32) - reconstructed.astype(np.float32)))
    
    if mse > 0:
        max_val = 255.0 if original.max() > 1 else 1.0
        psnr = 10 * np.log10(max_val ** 2 / mse)
    else:
        psnr = np.inf
    
    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr,
    }
