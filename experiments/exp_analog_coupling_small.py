#!/usr/bin/env python3
"""
Route C Experiment: Analog Coupling (Small Scale)
=================================================

Why Analog Coupling is Needed
-----------------------------
Previous experiments showed that pure discrete conditional inference (Gibbs fill)
achieves excellent token recovery (42%) but HURTS classification (-2.8pp).

Root cause: E_conditional optimizes P(z|context) which learns "statistically common"
token patterns, NOT "discriminatively useful" patterns. The filled tokens drift
toward blurry averages that harm classification boundaries.

Solution: Analog Coupling
-------------------------
Add E_observation that couples discrete inference to the OBSERVED continuous pixels:

    E(z) = λ_cond * E_conditional(z)        # discrete structure prior
         + λ_obs  * ||M ⊙ (decode(z) - o)||² # analog observation coupling
         + λ_cls  * E_classifier(z, y)       # task-aware objective

This preserves Route C principles:
1. Continuous enters ONLY via E_obs on observed pixels (boundary translation)
2. Inference remains DISCRETE and ITERATIVE in z-space (Gibbs/MH)
3. local_delta becomes stronger and more aligned with pixel reality

Key benefit: E_obs creates strong ΔE signal when token reconstruction deviates
from observed pixels, preventing drift toward wrong averages.

Experiment Design
-----------------
Small scale:
- train_samples = 2000
- test_samples = 500  
- occlusion_eval = 200

Compare inference modes:
- A:  λ_cond=1.0, λ_obs=0.0, λ_cls=0.0  (pure conditional)
- B1: λ_cond=1.0, λ_obs=0.1, λ_cls=0.0  (light analog coupling)
- B2: λ_cond=1.0, λ_obs=1.0, λ_cls=0.0  (strong analog coupling)  
- C:  λ_cond=0.1, λ_obs=0.1, λ_cls=1.0  (task-coupled)
"""

import numpy as np
import os
import sys
import time
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

# Optional imports
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# Check for GPU (torch)
try:
    import torch
    HAS_TORCH = True
    HAS_GPU = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
except ImportError:
    HAS_TORCH = False
    HAS_GPU = False
    DEVICE = None

# Add parent directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from core import make_random_mask
from boundary import (
    patchify, kmeans_codebook_fit, encode_tokens, decode_tokens,
    encode_batch, decode_batch,
)
from learning import (
    TokenConditionalModel, LogOddsClassifier,
    token_grid_to_spatial_histogram,
)
from mnist import load_mnist_numpy, subsample_dataset


# ============================================================================
# ENHANCED OBSERVATION ENERGY WITH FAST LOCAL_DELTA
# ============================================================================

class ObservationEnergyFast:
    """
    Observation energy with efficient local_delta computation.
    
    E_obs(z) = (1/N_obs) * Σ_{observed pixels} (decode(z) - o_obs)²
    
    Key optimization: changing token at (i,j) only affects patch region
    [i*p : (i+1)*p, j*p : (j+1)*p], so local_delta only recomputes that patch.
    """
    
    def __init__(
        self,
        codebook: np.ndarray,
        patch_size: int,
        pixel_obs_mask: np.ndarray,  # (H_img, W_img) True where observed
    ):
        """
        Args:
            codebook: (K, patch_size*patch_size) VQ codebook
            patch_size: size of each patch
            pixel_obs_mask: (H_img, W_img) boolean, True = observed pixel
        """
        self.codebook = codebook  # (K, D)
        self.patch_size = patch_size
        self.pixel_obs_mask = pixel_obs_mask.astype(np.float32)
        
        # Precompute codebook patches for fast indexing
        K = codebook.shape[0]
        self.codebook_patches = codebook.reshape(K, patch_size, patch_size)
        
        # Count observed pixels for normalization
        self.n_observed = max(1, pixel_obs_mask.sum())
    
    def _get_patch_region(self, i: int, j: int) -> Tuple[int, int, int, int]:
        """Get pixel region for token at (i, j)."""
        p = self.patch_size
        return i * p, (i + 1) * p, j * p, (j + 1) * p
    
    def _patch_error(
        self, 
        token: int, 
        obs_patch: np.ndarray, 
        mask_patch: np.ndarray
    ) -> float:
        """Compute squared error for a single patch."""
        recon_patch = self.codebook_patches[token]
        diff_sq = ((recon_patch - obs_patch) ** 2) * mask_patch
        return diff_sq.sum()
    
    def energy(
        self, 
        z: np.ndarray, 
        y: Optional[int] = None, 
        obs: Optional[np.ndarray] = None
    ) -> float:
        """Compute full observation energy."""
        if obs is None:
            return 0.0
        
        obs = obs.astype(np.float32)
        H_tok, W_tok = z.shape
        total_error = 0.0
        
        for i in range(H_tok):
            for j in range(W_tok):
                y0, y1, x0, x1 = self._get_patch_region(i, j)
                obs_patch = obs[y0:y1, x0:x1]
                mask_patch = self.pixel_obs_mask[y0:y1, x0:x1]
                total_error += self._patch_error(z[i, j], obs_patch, mask_patch)
        
        return total_error / self.n_observed
    
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
        Efficient local energy change.
        
        Only recompute the affected patch region's squared error.
        """
        if obs is None:
            return 0.0
        
        old_token = z[i, j]
        if old_token == new_token:
            return 0.0
        
        obs = obs.astype(np.float32)
        y0, y1, x0, x1 = self._get_patch_region(i, j)
        obs_patch = obs[y0:y1, x0:x1]
        mask_patch = self.pixel_obs_mask[y0:y1, x0:x1]
        
        old_error = self._patch_error(old_token, obs_patch, mask_patch)
        new_error = self._patch_error(new_token, obs_patch, mask_patch)
        
        return (new_error - old_error) / self.n_observed


class TokenConditionalEnergyFast:
    """Simplified token conditional energy for this experiment."""
    
    def __init__(self, model, token_mask: np.ndarray):
        """
        Args:
            model: TokenConditionalModel
            token_mask: (H_tok, W_tok) True where token is unknown/masked
        """
        self.model = model
        self.token_mask = token_mask
    
    def energy(self, z: np.ndarray, y=None, obs=None) -> float:
        H, W = z.shape
        total = 0.0
        for i in range(H):
            for j in range(W):
                context = self.model._extract_context(z, i, j)
                probs = self.model.predict_proba(context)
                total += -np.log(probs[z[i, j]] + 1e-10)
        return total
    
    def local_delta(self, z: np.ndarray, i: int, j: int, new_token: int, y=None, obs=None) -> float:
        old_token = z[i, j]
        if old_token == new_token:
            return 0.0
        
        H, W = z.shape
        
        # Energy at (i,j) itself
        context = self.model._extract_context(z, i, j)
        probs = self.model.predict_proba(context)
        delta = -np.log(probs[new_token] + 1e-10) + np.log(probs[old_token] + 1e-10)
        
        # Effect on neighbors
        z[i, j] = new_token
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if not (0 <= ni < H and 0 <= nj < W):
                    continue
                
                neighbor_token = z[ni, nj]
                context_new = self.model._extract_context(z, ni, nj)
                probs_new = self.model.predict_proba(context_new)
                
                z[i, j] = old_token
                context_old = self.model._extract_context(z, ni, nj)
                probs_old = self.model.predict_proba(context_old)
                z[i, j] = new_token
                
                delta += -np.log(probs_new[neighbor_token] + 1e-10) + np.log(probs_old[neighbor_token] + 1e-10)
        
        z[i, j] = old_token
        return delta


class ClassifierEnergyFast:
    """Classification energy for task-coupled inference."""
    
    def __init__(self, classifier, vocab_size: int, spatial_bins: Tuple[int, int], target_class: int):
        self.classifier = classifier
        self.vocab_size = vocab_size
        self.spatial_bins = spatial_bins
        self.target_class = target_class
    
    def _features(self, z: np.ndarray) -> np.ndarray:
        return token_grid_to_spatial_histogram(z, self.vocab_size, self.spatial_bins)
    
    def energy(self, z: np.ndarray, y=None, obs=None) -> float:
        probs = self.classifier.predict_proba(self._features(z))
        return -np.log(probs[self.target_class] + 1e-10)
    
    def local_delta(self, z: np.ndarray, i: int, j: int, new_token: int, y=None, obs=None) -> float:
        old_token = z[i, j]
        old_e = self.energy(z)
        z[i, j] = new_token
        new_e = self.energy(z)
        z[i, j] = old_token
        return new_e - old_e


class CombinedEnergyFast:
    """Weighted combination of energy models."""
    
    def __init__(self, energies: List[Tuple[float, Any]]):
        self.energies = [(w, e) for w, e in energies if w > 0]
    
    def energy(self, z: np.ndarray, y=None, obs=None) -> float:
        return sum(w * e.energy(z, y, obs) for w, e in self.energies)
    
    def local_delta(self, z: np.ndarray, i: int, j: int, new_token: int, y=None, obs=None) -> float:
        return sum(w * e.local_delta(z, i, j, new_token, y, obs) for w, e in self.energies)


# ============================================================================
# GIBBS INFERENCE
# ============================================================================

def gibbs_fill(
    token_grid: np.ndarray,
    token_mask: np.ndarray,
    energy_model,
    vocab_size: int,
    n_steps: int,
    obs: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Gibbs sampling for masked token completion.
    
    At each masked position, sample from Boltzmann:
        P(k) ∝ exp(-ΔE(k))
    where ΔE(k) is energy change from setting token to k.
    """
    rng = np.random.default_rng(seed)
    grid = token_grid.copy()
    
    # Get masked positions
    positions = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if token_mask[i, j]]
    
    # Initialize randomly
    for i, j in positions:
        grid[i, j] = rng.integers(vocab_size)
    
    stats = {'energies': [], 'delta_e_samples': []}
    
    for step in range(n_steps):
        rng.shuffle(positions)
        
        for i, j in positions:
            # Compute ΔE for each possible token
            deltas = np.zeros(vocab_size)
            for k in range(vocab_size):
                deltas[k] = energy_model.local_delta(grid, i, j, k, obs=obs)
            
            # Sample from Boltzmann
            deltas = deltas - deltas.min()  # Shift for stability
            probs = np.exp(-deltas)
            probs = probs / (probs.sum() + 1e-10)
            
            grid[i, j] = rng.choice(vocab_size, p=probs)
            
            # Record some ΔE samples
            if step == 0 and len(stats['delta_e_samples']) < 100:
                stats['delta_e_samples'].extend(deltas[:10].tolist())
        
        stats['energies'].append(energy_model.energy(grid, obs=obs))
    
    return grid, stats


# ============================================================================
# EXPERIMENT HELPERS
# ============================================================================

def create_occlusion(
    image: np.ndarray,
    occlusion_size: Tuple[int, int] = (14, 14),
    position: Optional[Tuple[int, int]] = None,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create occluded image and pixel mask.
    
    Returns:
        occluded_image: image with black rectangle
        pixel_obs_mask: True where pixel is observed (NOT occluded)
    """
    H, W = image.shape
    oh, ow = occlusion_size
    
    if position is None:
        y = rng.integers(0, max(1, H - oh + 1))
        x = rng.integers(0, max(1, W - ow + 1))
    else:
        y, x = position
    
    occluded = image.copy()
    occluded[y:y+oh, x:x+ow] = 0
    
    pixel_obs_mask = np.ones((H, W), dtype=bool)
    pixel_obs_mask[y:y+oh, x:x+ow] = False
    
    return occluded, pixel_obs_mask


def pixel_mask_to_token_mask(
    pixel_mask: np.ndarray,
    patch_size: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert pixel observation mask to token mask.
    
    A token is "masked" (unknown) if more than threshold of its pixels are unobserved.
    
    Args:
        pixel_mask: (H, W) True where pixel is OBSERVED
        patch_size: size of patches
        threshold: fraction threshold
    
    Returns:
        token_mask: (H_tok, W_tok) True where token is UNKNOWN
    """
    H, W = pixel_mask.shape
    H_tok = H // patch_size
    W_tok = W // patch_size
    
    token_mask = np.zeros((H_tok, W_tok), dtype=bool)
    
    for i in range(H_tok):
        for j in range(W_tok):
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            patch_obs = pixel_mask[y0:y1, x0:x1]
            # Token is unknown if most pixels are unobserved
            if patch_obs.mean() < threshold:
                token_mask[i, j] = True
    
    return token_mask


def compute_token_accuracy(predicted: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray) -> float:
    """Accuracy on masked positions only."""
    if mask.sum() == 0:
        return 0.0
    return (predicted[mask] == ground_truth[mask]).mean()


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

@dataclass
class Config:
    # Data
    train_samples: int = 2000
    test_samples: int = 500
    occlusion_eval: int = 200
    
    # Tokenization
    patch_size: int = 4
    codebook_size: int = 64
    kmeans_iters: int = 30
    
    # Classification
    spatial_bins: Tuple[int, int] = (4, 4)  # Better features
    
    # Occlusion
    occlusion_size: Tuple[int, int] = (14, 14)
    
    # Inference
    gibbs_steps: int = 20
    
    # Seed
    seed: int = 42


def run_experiment():
    """Run analog coupling experiment."""
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    
    print("=" * 70)
    print("ROUTE C: ANALOG COUPLING EXPERIMENT (SMALL SCALE)")
    print("=" * 70)
    print(f"\nConfig: train={cfg.train_samples}, test={cfg.test_samples}, eval={cfg.occlusion_eval}")
    print(f"        patch={cfg.patch_size}, K={cfg.codebook_size}, bins={cfg.spatial_bins}")
    print(f"        tqdm={'yes' if HAS_TQDM else 'no'}, torch={'yes' if HAS_TORCH else 'no'}, GPU={'yes' if HAS_GPU else 'no'}")
    
    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("\n[1/5] Loading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist_numpy("./data")
    
    train_images, train_labels = subsample_dataset(
        train_images, train_labels, cfg.train_samples, cfg.seed
    )
    test_images, test_labels = subsample_dataset(
        test_images, test_labels, cfg.test_samples, cfg.seed
    )
    print(f"       Train: {len(train_images)}, Test: {len(test_images)}")
    
    # ========================================================================
    # 2. LEARN CODEBOOK
    # ========================================================================
    print("\n[2/5] Learning VQ codebook...")
    all_patches = []
    for img in train_images:
        patches, _ = patchify(img, cfg.patch_size)
        all_patches.append(patches)
    all_patches = np.concatenate(all_patches, axis=0)
    
    codebook, _, _ = kmeans_codebook_fit(
        all_patches, cfg.codebook_size, cfg.kmeans_iters, cfg.seed, verbose=False
    )
    print(f"       Codebook shape: {codebook.shape}")
    
    # Encode training data
    train_tokens = encode_batch(train_images, codebook, cfg.patch_size)
    test_tokens = encode_batch(test_images, codebook, cfg.patch_size)
    print(f"       Token grid shape: {train_tokens[0].shape}")
    
    # ========================================================================
    # 3. TRAIN TOKEN CONDITIONAL MODEL
    # ========================================================================
    print("\n[3/5] Training token conditional model...")
    cond_model = TokenConditionalModel(
        vocab_size=cfg.codebook_size,
        smoothing=1.0,
        boundary="standard",
    )
    cond_model.fit(train_tokens, verbose=False)
    print(f"       Unique contexts: {len(cond_model.counts)}")
    
    # ========================================================================
    # 4. TRAIN CLASSIFIER
    # ========================================================================
    print("\n[4/5] Training classifier...")
    
    def extract_features(tokens):
        N = tokens.shape[0]
        feats = []
        for i in range(N):
            f = token_grid_to_spatial_histogram(tokens[i], cfg.codebook_size, cfg.spatial_bins)
            feats.append(f)
        return np.stack(feats)
    
    train_feats = extract_features(train_tokens)
    test_feats = extract_features(test_tokens)
    
    classifier = LogOddsClassifier(
        n_features=train_feats.shape[1],
        n_classes=10,
        smoothing=1.0,
    )
    classifier.fit(train_feats, train_labels)
    
    clean_acc = classifier.score(test_feats, test_labels)
    print(f"       Clean test accuracy: {clean_acc:.1%}")
    
    # ========================================================================
    # 5. OCCLUSION EXPERIMENT
    # ========================================================================
    print("\n[5/5] Running occlusion experiments...")
    print(f"       Evaluating {cfg.occlusion_eval} samples with {cfg.occlusion_size} occlusion")
    
    # Select eval subset
    eval_indices = rng.choice(len(test_images), cfg.occlusion_eval, replace=False)
    
    # Results storage
    results = {
        'none': {'correct': 0, 'token_acc': []},
        'A': {'correct': 0, 'token_acc': [], 'delta_e_mag': []},
        'B1': {'correct': 0, 'token_acc': [], 'delta_e_mag': []},
        'B2': {'correct': 0, 'token_acc': [], 'delta_e_mag': []},
        'C': {'correct': 0, 'token_acc': [], 'delta_e_mag': []},
    }
    
    t_start = time.time()
    
    # Use tqdm if available
    iterator = tqdm(enumerate(eval_indices), total=len(eval_indices), desc="       Inference") if HAS_TQDM else enumerate(eval_indices)
    
    for idx_i, idx in iterator:
        if not HAS_TQDM and idx_i % 50 == 0:
            print(f"       Progress: {idx_i}/{cfg.occlusion_eval}")
        
        original_image = test_images[idx]
        original_tokens = test_tokens[idx]
        label = test_labels[idx]
        
        # Create occlusion
        occluded_image, pixel_obs_mask = create_occlusion(
            original_image, cfg.occlusion_size, rng=rng
        )
        
        # Get token mask (which tokens are unknown)
        token_mask = pixel_mask_to_token_mask(pixel_obs_mask, cfg.patch_size, threshold=0.3)
        n_masked = token_mask.sum()
        
        if n_masked == 0:
            # No tokens affected, skip
            continue
        
        # Encode occluded image
        occluded_tokens, _ = encode_tokens(occluded_image, codebook, cfg.patch_size)
        
        # Baseline: no inference
        feat_occ = token_grid_to_spatial_histogram(occluded_tokens, cfg.codebook_size, cfg.spatial_bins)
        pred_none = classifier.predict(feat_occ)
        results['none']['correct'] += int(pred_none == label)
        
        # Mode A: Conditional only (λ_cond=1.0, λ_obs=0, λ_cls=0)
        energy_A = TokenConditionalEnergyFast(cond_model, token_mask)
        filled_A, stats_A = gibbs_fill(
            occluded_tokens, token_mask, energy_A, cfg.codebook_size,
            cfg.gibbs_steps, occluded_image, cfg.seed + idx
        )
        feat_A = token_grid_to_spatial_histogram(filled_A, cfg.codebook_size, cfg.spatial_bins)
        pred_A = classifier.predict(feat_A)
        results['A']['correct'] += int(pred_A == label)
        results['A']['token_acc'].append(compute_token_accuracy(filled_A, original_tokens, token_mask))
        results['A']['delta_e_mag'].extend([abs(d) for d in stats_A.get('delta_e_samples', [])])
        
        # Mode B1: Analog-coupled light (λ_cond=1.0, λ_obs=0.1)
        energy_obs = ObservationEnergyFast(codebook, cfg.patch_size, pixel_obs_mask)
        energy_B1 = CombinedEnergyFast([
            (1.0, TokenConditionalEnergyFast(cond_model, token_mask)),
            (0.1, energy_obs),
        ])
        filled_B1, stats_B1 = gibbs_fill(
            occluded_tokens, token_mask, energy_B1, cfg.codebook_size,
            cfg.gibbs_steps, occluded_image, cfg.seed + idx
        )
        feat_B1 = token_grid_to_spatial_histogram(filled_B1, cfg.codebook_size, cfg.spatial_bins)
        pred_B1 = classifier.predict(feat_B1)
        results['B1']['correct'] += int(pred_B1 == label)
        results['B1']['token_acc'].append(compute_token_accuracy(filled_B1, original_tokens, token_mask))
        
        # Mode B2: Analog-coupled strong (λ_cond=1.0, λ_obs=1.0)
        energy_B2 = CombinedEnergyFast([
            (1.0, TokenConditionalEnergyFast(cond_model, token_mask)),
            (1.0, energy_obs),
        ])
        filled_B2, stats_B2 = gibbs_fill(
            occluded_tokens, token_mask, energy_B2, cfg.codebook_size,
            cfg.gibbs_steps, occluded_image, cfg.seed + idx
        )
        feat_B2 = token_grid_to_spatial_histogram(filled_B2, cfg.codebook_size, cfg.spatial_bins)
        pred_B2 = classifier.predict(feat_B2)
        results['B2']['correct'] += int(pred_B2 == label)
        results['B2']['token_acc'].append(compute_token_accuracy(filled_B2, original_tokens, token_mask))
        
        # Mode C: Task-coupled (λ_cond=0.1, λ_obs=0.1, λ_cls=1.0)
        energy_cls = ClassifierEnergyFast(classifier, cfg.codebook_size, cfg.spatial_bins, label)
        energy_C = CombinedEnergyFast([
            (0.1, TokenConditionalEnergyFast(cond_model, token_mask)),
            (0.1, energy_obs),
            (1.0, energy_cls),
        ])
        filled_C, stats_C = gibbs_fill(
            occluded_tokens, token_mask, energy_C, cfg.codebook_size,
            cfg.gibbs_steps, occluded_image, cfg.seed + idx
        )
        feat_C = token_grid_to_spatial_histogram(filled_C, cfg.codebook_size, cfg.spatial_bins)
        pred_C = classifier.predict(feat_C)
        results['C']['correct'] += int(pred_C == label)
        results['C']['token_acc'].append(compute_token_accuracy(filled_C, original_tokens, token_mask))
    
    t_elapsed = time.time() - t_start
    
    # ========================================================================
    # REPORT RESULTS
    # ========================================================================
    n_eval = cfg.occlusion_eval
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nClean test accuracy (no occlusion): {clean_acc:.1%}")
    print(f"\nOcclusion experiment ({n_eval} samples, {cfg.occlusion_size} block):")
    print()
    print(f"| {'Mode':<6} | {'λ_cond':>6} | {'λ_obs':>6} | {'λ_cls':>6} | {'Token Rec':>10} | {'Occl Acc':>10} |")
    print(f"|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*12}|{'-'*12}|")
    
    # None (baseline)
    acc_none = results['none']['correct'] / n_eval
    print(f"| {'none':<6} | {'-':>6} | {'-':>6} | {'-':>6} | {'-':>10} | {acc_none:>9.1%} |")
    
    # Mode A
    acc_A = results['A']['correct'] / n_eval
    tok_A = np.mean(results['A']['token_acc']) if results['A']['token_acc'] else 0
    print(f"| {'A':<6} | {'1.0':>6} | {'0.0':>6} | {'0.0':>6} | {tok_A:>9.1%} | {acc_A:>9.1%} |")
    
    # Mode B1
    acc_B1 = results['B1']['correct'] / n_eval
    tok_B1 = np.mean(results['B1']['token_acc']) if results['B1']['token_acc'] else 0
    print(f"| {'B1':<6} | {'1.0':>6} | {'0.1':>6} | {'0.0':>6} | {tok_B1:>9.1%} | {acc_B1:>9.1%} |")
    
    # Mode B2
    acc_B2 = results['B2']['correct'] / n_eval
    tok_B2 = np.mean(results['B2']['token_acc']) if results['B2']['token_acc'] else 0
    print(f"| {'B2':<6} | {'1.0':>6} | {'1.0':>6} | {'0.0':>6} | {tok_B2:>9.1%} | {acc_B2:>9.1%} |")
    
    # Mode C
    acc_C = results['C']['correct'] / n_eval
    tok_C = np.mean(results['C']['token_acc']) if results['C']['token_acc'] else 0
    print(f"| {'C':<6} | {'0.1':>6} | {'0.1':>6} | {'1.0':>6} | {tok_C:>9.1%} | {acc_C:>9.1%} |")
    
    print()
    print(f"Runtime: {t_elapsed:.1f}s")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    delta_A = acc_A - acc_none
    delta_B1 = acc_B1 - acc_none
    delta_B2 = acc_B2 - acc_none
    delta_C = acc_C - acc_none
    
    print(f"\nAccuracy changes vs baseline (no inference):")
    print(f"  Mode A  (cond only):      {delta_A:+.1%}")
    print(f"  Mode B1 (analog light):   {delta_B1:+.1%}")
    print(f"  Mode B2 (analog strong):  {delta_B2:+.1%}")
    print(f"  Mode C  (task-coupled):   {delta_C:+.1%}")
    
    print(f"\nToken recovery comparison:")
    print(f"  A:  {tok_A:.1%}")
    print(f"  B1: {tok_B1:.1%}")
    print(f"  B2: {tok_B2:.1%}")
    print(f"  C:  {tok_C:.1%}")
    
    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    if delta_B1 > delta_A or delta_B2 > delta_A:
        print("✓ Analog coupling (E_obs) improves classification vs pure conditional")
    else:
        print("✗ Analog coupling did not improve over pure conditional")
    
    if delta_C > delta_B1:
        print("✓ Task coupling (E_cls) provides additional benefit")
    else:
        print("✗ Task coupling did not outperform analog coupling alone")
    
    if tok_B2 > tok_A:
        print("✓ Strong analog coupling improves token recovery")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    run_experiment()
