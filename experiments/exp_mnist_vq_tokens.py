#!/usr/bin/env python3
"""
Route C Experiment: MNIST VQ Tokens + Discrete Inference
=========================================================

This experiment implements the Route C framework:
1. Boundary Translation: Image -> VQ Token Grid (learned k-means codebook)
2. Discrete Core Closure: Token conditional model + Gibbs/Metropolis inference
3. Classification: Spatial histogram features + log-odds classifier

Key experiments:
A) Tokenization quality (reconstruction MSE)
B) Masked token completion (token accuracy after inference)
C) Occlusion robustness (classification before/after inference)
D) Inference dynamics ablation (Gibbs vs Metropolis, different proposals)
"""

import numpy as np
import sys
import os
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    make_random_mask, make_block_mask,
    canonicalize_d4, extract_context_window,
)
from boundary import (
    patchify, unpatchify, 
    kmeans_codebook_fit, encode_tokens, decode_tokens,
    encode_batch, decode_batch,
)
from learning import (
    TokenConditionalModel, LogOddsClassifier,
    extract_features_batch, token_grid_to_spatial_histogram,
    export_conditional_rules, print_rules,
)
from inference import (
    gibbs_fill, metropolis_fill, block_gibbs_fill, template_proposal_fill,
    compute_token_accuracy, analyze_delta_energies,
)
from mnist import (
    load_mnist_numpy, subsample_dataset, normalize_images,
    create_occluded_images, pixel_mask_to_token_mask,
    compute_reconstruction_metrics,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Experiment configuration."""
    # Tokenization
    patch_size: int = 4
    codebook_size: int = 64  # K in paper
    kmeans_iters: int = 50
    
    # Model
    smoothing: float = 1.0
    boundary: str = "standard"  # or "torus"
    
    # Inference
    gibbs_steps: int = 20
    metropolis_steps: int = 50
    temperature: float = 1.0
    
    # Classification
    spatial_bins: tuple = (2, 2)  # For 7x7 token grid, 2x2 bins
    
    # Data
    train_samples: int = 10000  # Subsample for speed
    test_samples: int = 2000
    
    # Masking
    mask_ratio: float = 0.25  # 25% masked tokens
    occlusion_size: tuple = (14, 14)  # 14x14 pixel block
    
    # Seeds
    seed: int = 42


def print_config(cfg: Config):
    """Print configuration."""
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    for attr in dir(cfg):
        if not attr.startswith('_'):
            print(f"  {attr}: {getattr(cfg, attr)}")
    print("="*60 + "\n")


# ============================================================================
# VISUALIZATION
# ============================================================================

def save_visualizations(
    output_dir: str,
    original_images: np.ndarray,
    token_grids: np.ndarray,
    reconstructed_images: np.ndarray,
    codebook: np.ndarray,
    patch_size: int,
    n_samples: int = 5,
    prefix: str = "tokenization",
):
    """Save visualization of tokenization results."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples))
    
    for i in range(n_samples):
        # Original
        axes[i, 0].imshow(original_images[i], cmap='gray')
        axes[i, 0].set_title(f'Original')
        axes[i, 0].axis('off')
        
        # Token grid
        axes[i, 1].imshow(token_grids[i], cmap='tab20')
        axes[i, 1].set_title(f'Token Grid')
        axes[i, 1].axis('off')
        
        # Reconstructed
        axes[i, 2].imshow(reconstructed_images[i], cmap='gray')
        axes[i, 2].set_title(f'Reconstructed')
        axes[i, 2].axis('off')
        
        # Difference
        diff = np.abs(original_images[i].astype(float) - reconstructed_images[i].astype(float))
        axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title(f'Difference')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_samples.png", dpi=150)
    plt.close()
    
    # Save codebook visualization
    K = codebook.shape[0]
    grid_side = int(np.ceil(np.sqrt(K)))
    
    fig, axes = plt.subplots(grid_side, grid_side, figsize=(12, 12))
    for k in range(K):
        ax = axes[k // grid_side, k % grid_side]
        patch = codebook[k].reshape(patch_size, patch_size)
        ax.imshow(patch, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'{k}', fontsize=8)
        ax.axis('off')
    
    # Hide empty subplots
    for k in range(K, grid_side * grid_side):
        axes[k // grid_side, k % grid_side].axis('off')
    
    plt.suptitle(f'Codebook (K={K})')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_codebook.png", dpi=150)
    plt.close()


def save_inference_visualization(
    output_dir: str,
    original_grid: np.ndarray,
    masked_grid: np.ndarray,
    mask: np.ndarray,
    filled_grid: np.ndarray,
    codebook: np.ndarray,
    patch_size: int,
    prefix: str = "inference",
):
    """Save visualization of inference results."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Token grids
    axes[0, 0].imshow(original_grid, cmap='tab20')
    axes[0, 0].set_title('Original Tokens')
    axes[0, 0].axis('off')
    
    # Masked grid (with mask overlay)
    masked_vis = masked_grid.copy().astype(float)
    masked_vis[mask] = -1
    axes[0, 1].imshow(masked_vis, cmap='tab20')
    axes[0, 1].set_title(f'Masked ({mask.sum()} tokens)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(filled_grid, cmap='tab20')
    axes[0, 2].set_title('Filled Tokens')
    axes[0, 2].axis('off')
    
    # Correctness
    correct = (filled_grid == original_grid)
    axes[0, 3].imshow(correct & mask, cmap='RdYlGn')
    axes[0, 3].set_title(f'Correct at Masked (acc={compute_token_accuracy(filled_grid, original_grid, mask):.1%})')
    axes[0, 3].axis('off')
    
    # Reconstructed images
    codebook_patches = codebook.reshape(-1, patch_size, patch_size)
    
    orig_img = decode_tokens(original_grid, codebook, patch_size)
    axes[1, 0].imshow(orig_img, cmap='gray')
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    # Masked image
    masked_img = orig_img.copy()
    # Compute which pixels are masked
    H, W = original_grid.shape
    for i in range(H):
        for j in range(W):
            if mask[i, j]:
                y, x = i * patch_size, j * patch_size
                masked_img[y:y+patch_size, x:x+patch_size] = 0
    axes[1, 1].imshow(masked_img, cmap='gray')
    axes[1, 1].set_title('Masked Image')
    axes[1, 1].axis('off')
    
    filled_img = decode_tokens(filled_grid, codebook, patch_size)
    axes[1, 2].imshow(filled_img, cmap='gray')
    axes[1, 2].set_title('Filled Image')
    axes[1, 2].axis('off')
    
    diff = np.abs(orig_img.astype(float) - filled_img.astype(float))
    axes[1, 3].imshow(diff, cmap='hot')
    axes[1, 3].set_title('Difference')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}.png", dpi=150)
    plt.close()


# ============================================================================
# EXPERIMENTS
# ============================================================================

def experiment_tokenization(
    train_images: np.ndarray,
    test_images: np.ndarray,
    cfg: Config,
    output_dir: str,
) -> dict:
    """
    Experiment A: Tokenization quality
    - Learn codebook from training images
    - Encode/decode and measure reconstruction quality
    """
    print("\n" + "="*60)
    print("EXPERIMENT A: TOKENIZATION QUALITY")
    print("="*60)
    
    # Collect all patches from training images
    print(f"Extracting patches from {len(train_images)} training images...")
    all_patches = []
    for img in train_images:
        patches, _ = patchify(img, cfg.patch_size)
        all_patches.append(patches)
    all_patches = np.vstack(all_patches)
    print(f"  Total patches: {len(all_patches)}")
    
    # Fit codebook
    print(f"Fitting k-means codebook (K={cfg.codebook_size})...")
    t0 = time.time()
    codebook, _, losses = kmeans_codebook_fit(
        all_patches, cfg.codebook_size, cfg.kmeans_iters, cfg.seed, verbose=True
    )
    t_fit = time.time() - t0
    print(f"  Codebook fit time: {t_fit:.2f}s")
    print(f"  Final k-means loss: {losses[-1]:.4f}")
    
    # Encode training and test sets
    print("Encoding images to token grids...")
    train_tokens = encode_batch(train_images, codebook, cfg.patch_size)
    test_tokens = encode_batch(test_images, codebook, cfg.patch_size)
    
    # Decode back
    print("Decoding token grids to images...")
    train_recon = decode_batch(train_tokens, codebook, cfg.patch_size)
    test_recon = decode_batch(test_tokens, codebook, cfg.patch_size)
    
    # Compute metrics
    train_metrics = compute_reconstruction_metrics(train_images, train_recon)
    test_metrics = compute_reconstruction_metrics(test_images, test_recon)
    
    print("\nReconstruction Quality:")
    print(f"  Train MSE: {train_metrics['mse']:.2f}, PSNR: {train_metrics['psnr']:.2f} dB")
    print(f"  Test MSE:  {test_metrics['mse']:.2f}, PSNR: {test_metrics['psnr']:.2f} dB")
    
    # Save visualizations
    save_visualizations(
        output_dir, test_images[:5], test_tokens[:5], test_recon[:5],
        codebook, cfg.patch_size, n_samples=5, prefix="exp_a_tokenization"
    )
    print(f"  Saved visualizations to {output_dir}/exp_a_tokenization_*.png")
    
    return {
        'codebook': codebook,
        'train_tokens': train_tokens,
        'test_tokens': test_tokens,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'kmeans_losses': losses,
    }


def experiment_masked_completion(
    train_tokens: np.ndarray,
    test_tokens: np.ndarray,
    codebook: np.ndarray,
    cfg: Config,
    output_dir: str,
) -> dict:
    """
    Experiment B: Masked token completion
    - Train conditional model on token grids
    - Randomly mask tokens and fill with Gibbs
    - Report token accuracy
    """
    print("\n" + "="*60)
    print("EXPERIMENT B: MASKED TOKEN COMPLETION")
    print("="*60)
    
    # Train conditional model
    print(f"Training TokenConditionalModel...")
    t0 = time.time()
    model = TokenConditionalModel(
        vocab_size=cfg.codebook_size,
        context_size=8,
        smoothing=cfg.smoothing,
        boundary=cfg.boundary,
    )
    model.fit(train_tokens, verbose=True)
    t_fit = time.time() - t0
    print(f"  Model fit time: {t_fit:.2f}s")
    print(f"  Unique contexts: {len(model.counts)}")
    
    # Export and print top rules
    rules = export_conditional_rules(model, min_count=50, top_k=3)
    print_rules(rules, max_rules=10)
    
    # Evaluate masked completion on test set
    print(f"\nEvaluating masked token completion (mask ratio={cfg.mask_ratio})...")
    rng = np.random.default_rng(cfg.seed)
    
    n_test = min(200, len(test_tokens))  # Limit for speed
    accuracies_random = []
    accuracies_gibbs = []
    
    for i in range(n_test):
        original = test_tokens[i]
        H, W = original.shape
        
        # Create random mask
        mask = make_random_mask((H, W), cfg.mask_ratio, rng)
        
        # Masked grid (copy tokens, will be overwritten at masked positions)
        masked = original.copy()
        
        # Random baseline (just random tokens)
        random_fill = original.copy()
        random_fill[mask] = rng.integers(cfg.codebook_size, size=mask.sum())
        
        # Gibbs fill
        filled = gibbs_fill(masked, mask, model, n_steps=cfg.gibbs_steps, seed=cfg.seed + i)
        
        # Compute accuracies
        acc_random = compute_token_accuracy(random_fill, original, mask)
        acc_gibbs = compute_token_accuracy(filled, original, mask)
        
        accuracies_random.append(acc_random)
        accuracies_gibbs.append(acc_gibbs)
    
    mean_random = np.mean(accuracies_random)
    std_random = np.std(accuracies_random)
    mean_gibbs = np.mean(accuracies_gibbs)
    std_gibbs = np.std(accuracies_gibbs)
    
    print(f"\nMasked Token Completion Results (n={n_test}):")
    print(f"  Random baseline: {mean_random:.1%} ± {std_random:.1%}")
    print(f"  Gibbs fill:      {mean_gibbs:.1%} ± {std_gibbs:.1%}")
    print(f"  Improvement:     +{(mean_gibbs - mean_random)*100:.1f}pp")
    
    # Save example visualization
    example_idx = 0
    original = test_tokens[example_idx]
    mask = make_random_mask(original.shape, cfg.mask_ratio, np.random.default_rng(cfg.seed))
    filled = gibbs_fill(original.copy(), mask, model, n_steps=cfg.gibbs_steps, seed=cfg.seed)
    
    save_inference_visualization(
        output_dir, original, original, mask, filled,
        codebook, cfg.patch_size, prefix="exp_b_masked_completion"
    )
    print(f"  Saved visualization to {output_dir}/exp_b_masked_completion.png")
    
    return {
        'model': model,
        'random_acc': (mean_random, std_random),
        'gibbs_acc': (mean_gibbs, std_gibbs),
        'rules': rules,
    }


def experiment_classification(
    train_tokens: np.ndarray,
    train_labels: np.ndarray,
    test_tokens: np.ndarray,
    test_labels: np.ndarray,
    cfg: Config,
    output_dir: str,
) -> dict:
    """
    Experiment C: Classification with spatial histogram features
    """
    print("\n" + "="*60)
    print("EXPERIMENT C: CLASSIFICATION")
    print("="*60)
    
    # Extract features
    print("Extracting spatial histogram features...")
    train_features = extract_features_batch(
        train_tokens, cfg.codebook_size, cfg.spatial_bins, include_global=True
    )
    test_features = extract_features_batch(
        test_tokens, cfg.codebook_size, cfg.spatial_bins, include_global=True
    )
    print(f"  Feature dimension: {train_features.shape[1]}")
    
    # Train log-odds classifier
    print("Training log-odds classifier...")
    classifier = LogOddsClassifier(train_features.shape[1], 10, smoothing=cfg.smoothing)
    classifier.fit(train_features, train_labels, verbose=True)
    
    # Evaluate
    train_acc = classifier.score(train_features, train_labels)
    test_acc = classifier.score(test_features, test_labels)
    
    print(f"\nClassification Results:")
    print(f"  Train accuracy: {train_acc:.1%}")
    print(f"  Test accuracy:  {test_acc:.1%}")
    
    return {
        'classifier': classifier,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'feature_dim': train_features.shape[1],
    }


def experiment_occlusion_robustness(
    test_images: np.ndarray,
    test_labels: np.ndarray,
    codebook: np.ndarray,
    model: TokenConditionalModel,
    classifier: LogOddsClassifier,
    cfg: Config,
    output_dir: str,
) -> dict:
    """
    Experiment D: Occlusion robustness
    - Compare pixel-level occlusion vs token masking
    - Evaluate classification before and after inference
    """
    print("\n" + "="*60)
    print("EXPERIMENT D: OCCLUSION ROBUSTNESS")
    print("="*60)
    
    rng = np.random.default_rng(cfg.seed)
    n_test = min(500, len(test_images))
    
    # Create occluded images
    print(f"Creating occluded images (block size {cfg.occlusion_size})...")
    occluded_images, pixel_masks = create_occluded_images(
        test_images[:n_test], cfg.occlusion_size, seed=cfg.seed
    )
    
    # Encode original and occluded
    print("Encoding images...")
    original_tokens = encode_batch(test_images[:n_test], codebook, cfg.patch_size)
    occluded_tokens = encode_batch(occluded_images, codebook, cfg.patch_size)
    
    # Convert pixel masks to token masks
    token_masks = []
    for i in range(n_test):
        tmask = pixel_mask_to_token_mask(pixel_masks[i], cfg.patch_size, threshold=0.25)
        token_masks.append(tmask)
    token_masks = np.stack(token_masks)
    
    print(f"  Average tokens masked per image: {token_masks.sum(axis=(1,2)).mean():.1f}")
    
    # Evaluate: Clean tokens (baseline)
    clean_features = extract_features_batch(
        original_tokens, cfg.codebook_size, cfg.spatial_bins, include_global=True
    )
    clean_acc = classifier.score(clean_features, test_labels[:n_test])
    
    # Evaluate: Occluded tokens (no inference)
    occluded_features = extract_features_batch(
        occluded_tokens, cfg.codebook_size, cfg.spatial_bins, include_global=True
    )
    occluded_acc = classifier.score(occluded_features, test_labels[:n_test])
    
    # Evaluate: After Gibbs inference
    print(f"Running Gibbs inference on occluded tokens...")
    filled_tokens = []
    for i in range(n_test):
        if i % 100 == 0:
            print(f"  Processing {i}/{n_test}...")
        
        filled = gibbs_fill(
            occluded_tokens[i].copy(),
            token_masks[i],
            model,
            n_steps=cfg.gibbs_steps,
            seed=cfg.seed + i
        )
        filled_tokens.append(filled)
    filled_tokens = np.stack(filled_tokens)
    
    filled_features = extract_features_batch(
        filled_tokens, cfg.codebook_size, cfg.spatial_bins, include_global=True
    )
    filled_acc = classifier.score(filled_features, test_labels[:n_test])
    
    # Token accuracy on masked positions
    token_accs = []
    for i in range(n_test):
        acc = compute_token_accuracy(filled_tokens[i], original_tokens[i], token_masks[i])
        token_accs.append(acc)
    mean_token_acc = np.mean(token_accs)
    
    print(f"\nOcclusion Robustness Results (n={n_test}):")
    print(f"  Clean accuracy (baseline):     {clean_acc:.1%}")
    print(f"  Occluded accuracy (no infer):  {occluded_acc:.1%}")
    print(f"  After Gibbs inference:         {filled_acc:.1%}")
    print(f"  Δ from occlusion:              {(filled_acc - occluded_acc)*100:+.1f}pp")
    print(f"  Token recovery accuracy:       {mean_token_acc:.1%}")
    
    # Save visualization
    example_idx = 5
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original
    axes[0, 0].imshow(test_images[example_idx], cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Occluded
    axes[0, 1].imshow(occluded_images[example_idx], cmap='gray')
    axes[0, 1].set_title('Occluded Image')
    axes[0, 1].axis('off')
    
    # Token grids
    axes[0, 2].imshow(original_tokens[example_idx], cmap='tab20')
    axes[0, 2].set_title('Original Tokens')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(occluded_tokens[example_idx], cmap='tab20')
    axes[0, 3].set_title('Occluded Tokens')
    axes[0, 3].axis('off')
    
    # Mask
    axes[1, 0].imshow(token_masks[example_idx], cmap='Reds')
    axes[1, 0].set_title('Token Mask')
    axes[1, 0].axis('off')
    
    # Filled
    axes[1, 1].imshow(filled_tokens[example_idx], cmap='tab20')
    axes[1, 1].set_title('Filled Tokens')
    axes[1, 1].axis('off')
    
    # Reconstructed
    filled_img = decode_tokens(filled_tokens[example_idx], codebook, cfg.patch_size)
    axes[1, 2].imshow(filled_img, cmap='gray')
    axes[1, 2].set_title('Reconstructed')
    axes[1, 2].axis('off')
    
    # Correctness
    correct = (filled_tokens[example_idx] == original_tokens[example_idx])
    axes[1, 3].imshow(correct, cmap='RdYlGn')
    axes[1, 3].set_title(f'Correct Tokens')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/exp_d_occlusion.png", dpi=150)
    plt.close()
    print(f"  Saved visualization to {output_dir}/exp_d_occlusion.png")
    
    return {
        'clean_acc': clean_acc,
        'occluded_acc': occluded_acc,
        'filled_acc': filled_acc,
        'token_recovery_acc': mean_token_acc,
    }


def experiment_inference_ablation(
    test_tokens: np.ndarray,
    model: TokenConditionalModel,
    codebook: np.ndarray,
    cfg: Config,
    output_dir: str,
) -> dict:
    """
    Experiment E: Inference dynamics ablation
    - Compare Gibbs vs Metropolis
    - Analyze ΔE distributions
    """
    print("\n" + "="*60)
    print("EXPERIMENT E: INFERENCE ABLATION")
    print("="*60)
    
    rng = np.random.default_rng(cfg.seed)
    n_samples = 50
    
    results = {
        'gibbs': {'token_acc': [], 'time': []},
        'metropolis': {'token_acc': [], 'time': [], 'delta_e': []},
        'block_gibbs': {'token_acc': [], 'time': []},
    }
    
    for i in range(n_samples):
        original = test_tokens[i]
        mask = make_random_mask(original.shape, cfg.mask_ratio, rng)
        
        # Gibbs
        t0 = time.time()
        filled_gibbs = gibbs_fill(original.copy(), mask, model, cfg.gibbs_steps, seed=cfg.seed+i)
        results['gibbs']['time'].append(time.time() - t0)
        results['gibbs']['token_acc'].append(compute_token_accuracy(filled_gibbs, original, mask))
        
        # Metropolis
        t0 = time.time()
        filled_metro, stats = metropolis_fill(original.copy(), mask, model, cfg.metropolis_steps, cfg.temperature, seed=cfg.seed+i)
        results['metropolis']['time'].append(time.time() - t0)
        results['metropolis']['token_acc'].append(compute_token_accuracy(filled_metro, original, mask))
        results['metropolis']['delta_e'].extend(stats['delta_energies'])
        
        # Block Gibbs
        t0 = time.time()
        filled_block = block_gibbs_fill(original.copy(), mask, model, cfg.gibbs_steps, block_size=(2, 2), seed=cfg.seed+i)
        results['block_gibbs']['time'].append(time.time() - t0)
        results['block_gibbs']['token_acc'].append(compute_token_accuracy(filled_block, original, mask))
    
    print(f"\nInference Ablation Results (n={n_samples}, mask={cfg.mask_ratio}):")
    print(f"{'Method':<15} {'Token Acc':>12} {'Time (ms)':>12}")
    print("-" * 40)
    
    for method in ['gibbs', 'metropolis', 'block_gibbs']:
        acc_mean = np.mean(results[method]['token_acc'])
        acc_std = np.std(results[method]['token_acc'])
        time_mean = np.mean(results[method]['time']) * 1000
        print(f"{method:<15} {acc_mean:>10.1%} ± {acc_std:.1%}  {time_mean:>8.1f}")
    
    # Analyze ΔE distribution
    delta_e_stats = analyze_delta_energies(results['metropolis']['delta_e'])
    print(f"\nΔE Distribution (Metropolis):")
    print(f"  Mean: {delta_e_stats['mean']:.3f}")
    print(f"  Std:  {delta_e_stats['std']:.3f}")
    print(f"  % negative: {delta_e_stats['pct_negative']:.1%}")
    
    # Plot ΔE distribution
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    
    deltas = np.array(results['metropolis']['delta_e'])
    deltas_clipped = np.clip(deltas, -5, 5)
    ax.hist(deltas_clipped, bins=50, density=True, alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', label='ΔE=0')
    ax.set_xlabel('ΔE (clipped to [-5, 5])')
    ax.set_ylabel('Density')
    ax.set_title(f'ΔE Distribution (Metropolis, n={len(deltas)})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/exp_e_delta_e_dist.png", dpi=150)
    plt.close()
    print(f"  Saved ΔE distribution to {output_dir}/exp_e_delta_e_dist.png")
    
    return {
        'results': results,
        'delta_e_stats': delta_e_stats,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all experiments."""
    # Setup
    cfg = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print_config(cfg)
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\nLoading MNIST data...")
    train_images, train_labels, test_images, test_labels = load_mnist_numpy("./data")
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Subsample for speed
    train_images, train_labels = subsample_dataset(
        train_images, train_labels, cfg.train_samples, cfg.seed
    )
    test_images, test_labels = subsample_dataset(
        test_images, test_labels, cfg.test_samples, cfg.seed
    )
    print(f"  Subsampled: Train {len(train_images)}, Test {len(test_images)}")
    
    # Run experiments
    t_start = time.time()
    
    # A: Tokenization
    results_A = experiment_tokenization(train_images, test_images, cfg, output_dir)
    codebook = results_A['codebook']
    train_tokens = results_A['train_tokens']
    test_tokens = results_A['test_tokens']
    
    # B: Masked completion
    results_B = experiment_masked_completion(
        train_tokens, test_tokens, codebook, cfg, output_dir
    )
    model = results_B['model']
    
    # C: Classification
    results_C = experiment_classification(
        train_tokens, train_labels, test_tokens, test_labels, cfg, output_dir
    )
    classifier = results_C['classifier']
    
    # D: Occlusion robustness
    results_D = experiment_occlusion_robustness(
        test_images, test_labels, codebook, model, classifier, cfg, output_dir
    )
    
    # E: Inference ablation
    results_E = experiment_inference_ablation(
        test_tokens, model, codebook, cfg, output_dir
    )
    
    # Final summary
    t_total = time.time() - t_start
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\nExperiment A (Tokenization):")
    print(f"  Test reconstruction MSE: {results_A['test_metrics']['mse']:.2f}")
    print(f"  Test PSNR: {results_A['test_metrics']['psnr']:.2f} dB")
    
    print(f"\nExperiment B (Masked Completion):")
    print(f"  Random baseline: {results_B['random_acc'][0]:.1%}")
    print(f"  Gibbs fill: {results_B['gibbs_acc'][0]:.1%}")
    
    print(f"\nExperiment C (Classification):")
    print(f"  Test accuracy: {results_C['test_acc']:.1%}")
    
    print(f"\nExperiment D (Occlusion Robustness):")
    print(f"  Clean: {results_D['clean_acc']:.1%}")
    print(f"  Occluded (no infer): {results_D['occluded_acc']:.1%}")
    print(f"  After Gibbs: {results_D['filled_acc']:.1%}")
    print(f"  Δ: {(results_D['filled_acc'] - results_D['occluded_acc'])*100:+.1f}pp")
    
    print(f"\nExperiment E (Inference Ablation):")
    for method in ['gibbs', 'metropolis', 'block_gibbs']:
        acc = np.mean(results_E['results'][method]['token_acc'])
        print(f"  {method}: {acc:.1%}")
    
    print(f"\nTotal runtime: {t_total:.1f}s")
    print(f"Results saved to: {output_dir}/")
    print("="*60)
    
    # Save summary to file
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write("Route C Experiment Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Configuration:\n")
        for attr in dir(cfg):
            if not attr.startswith('_'):
                f.write(f"  {attr}: {getattr(cfg, attr)}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Tokenization MSE: {results_A['test_metrics']['mse']:.2f}\n")
        f.write(f"  Tokenization PSNR: {results_A['test_metrics']['psnr']:.2f}\n")
        f.write(f"  Masked completion (random): {results_B['random_acc'][0]:.3f}\n")
        f.write(f"  Masked completion (Gibbs): {results_B['gibbs_acc'][0]:.3f}\n")
        f.write(f"  Classification test acc: {results_C['test_acc']:.3f}\n")
        f.write(f"  Occlusion clean: {results_D['clean_acc']:.3f}\n")
        f.write(f"  Occlusion no infer: {results_D['occluded_acc']:.3f}\n")
        f.write(f"  Occlusion with Gibbs: {results_D['filled_acc']:.3f}\n")
    
    print(f"Summary saved to {output_dir}/summary.txt")


if __name__ == "__main__":
    main()
