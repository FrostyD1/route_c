#!/usr/bin/env python3
"""
Route C: Analog Coupling Experiment (FAST VERSION)
===================================================
Key optimization: Vectorized local_delta - compute all K deltas in ONE call
"""

import numpy as np
import os
import sys
import time
from typing import Tuple, List, Any
from dataclasses import dataclass

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw): return x

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boundary import patchify, kmeans_codebook_fit, encode_tokens, encode_batch
from learning import TokenConditionalModel, LogOddsClassifier, token_grid_to_spatial_histogram
from mnist import load_mnist_numpy, subsample_dataset


# ============================================================================
# VECTORIZED ENERGY - KEY OPTIMIZATION
# ============================================================================

class VectorizedEnergy:
    """All energies combined, vectorized to compute all K deltas at once."""
    
    def __init__(
        self,
        codebook: np.ndarray,
        patch_size: int,
        cond_model: TokenConditionalModel,
        pixel_obs_mask: np.ndarray,
        classifier: LogOddsClassifier = None,
        spatial_bins: Tuple[int, int] = None,
        target_class: int = None,
        lambda_cond: float = 1.0,
        lambda_obs: float = 0.0,
        lambda_cls: float = 0.0,
    ):
        self.patch_size = patch_size
        self.cond_model = cond_model
        self.K = cond_model.vocab_size
        
        # Precompute codebook patches (K, p, p)
        self.codebook_patches = codebook.reshape(self.K, patch_size, patch_size).astype(np.float32)
        
        # Observation mask
        self.pixel_obs_mask = pixel_obs_mask.astype(np.float32)
        self.n_observed = max(1.0, pixel_obs_mask.sum())
        
        # Classifier (optional)
        self.classifier = classifier
        self.spatial_bins = spatial_bins
        self.target_class = target_class
        
        # Weights
        self.lambda_cond = lambda_cond
        self.lambda_obs = lambda_obs
        self.lambda_cls = lambda_cls
    
    def delta_all(self, z: np.ndarray, i: int, j: int, obs: np.ndarray) -> np.ndarray:
        """
        Compute ΔE for ALL K tokens at position (i,j) in ONE call.
        Returns shape (K,) array.
        """
        p = self.patch_size
        old_token = z[i, j]
        deltas = np.zeros(self.K, dtype=np.float32)
        
        # === E_cond: -log P(token | context) ===
        if self.lambda_cond > 0:
            context = self.cond_model._extract_context(z, i, j)
            probs = self.cond_model.predict_proba(context)
            nll_all = -np.log(probs + 1e-10)  # (K,)
            nll_old = nll_all[old_token]
            deltas += self.lambda_cond * (nll_all - nll_old)
        
        # === E_obs: ||patch - obs||^2 on observed pixels ===
        if self.lambda_obs > 0:
            y0, y1, x0, x1 = i*p, (i+1)*p, j*p, (j+1)*p
            obs_patch = obs[y0:y1, x0:x1].astype(np.float32)
            mask_patch = self.pixel_obs_mask[y0:y1, x0:x1]
            
            # Vectorized: (K, p, p) - (1, p, p) -> (K, p, p)
            diff = self.codebook_patches - obs_patch[None, :, :]
            sq_err = (diff ** 2) * mask_patch[None, :, :]
            errors = sq_err.sum(axis=(1, 2)) / self.n_observed  # (K,)
            
            old_err = errors[old_token]
            deltas += self.lambda_obs * (errors - old_err)
        
        # === E_cls: -log P(y | z) - expensive, skip if lambda=0 ===
        if self.lambda_cls > 0 and self.classifier is not None:
            # Only compute for old token and randomly sample a few others
            feat_old = token_grid_to_spatial_histogram(z, self.K, self.spatial_bins)
            prob_old = self.classifier.predict_proba(feat_old)[self.target_class]
            e_old = -np.log(prob_old + 1e-10)
            
            # Approximate: assume histogram change is small, compute exactly for a subset
            cls_deltas = np.zeros(self.K)
            sample_tokens = [old_token, 0, self.K//2, self.K-1]
            for k in sample_tokens:
                z[i, j] = k
                feat_k = token_grid_to_spatial_histogram(z, self.K, self.spatial_bins)
                prob_k = self.classifier.predict_proba(feat_k)[self.target_class]
                cls_deltas[k] = -np.log(prob_k + 1e-10) - e_old
            z[i, j] = old_token
            
            deltas += self.lambda_cls * cls_deltas
        
        return deltas


def fast_gibbs(
    z: np.ndarray,
    token_mask: np.ndarray,
    energy: VectorizedEnergy,
    n_steps: int,
    obs: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Gibbs sampling with vectorized energy computation."""
    rng = np.random.default_rng(seed)
    grid = z.copy()
    K = energy.K
    
    positions = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if token_mask[i, j]]
    
    # Random init
    for i, j in positions:
        grid[i, j] = rng.integers(K)
    
    for _ in range(n_steps):
        rng.shuffle(positions)
        for i, j in positions:
            # Get ALL K deltas in one call
            deltas = energy.delta_all(grid, i, j, obs)
            
            # Boltzmann sampling
            deltas = deltas - deltas.min()
            probs = np.exp(-deltas)
            probs /= probs.sum() + 1e-10
            
            grid[i, j] = rng.choice(K, p=probs)
    
    return grid


# ============================================================================
# EXPERIMENT
# ============================================================================

@dataclass 
class Config:
    train_samples: int = 2000
    test_samples: int = 500
    occlusion_eval: int = 200
    patch_size: int = 4
    codebook_size: int = 64
    kmeans_iters: int = 30
    spatial_bins: Tuple[int, int] = (4, 4)
    occlusion_size: Tuple[int, int] = (14, 14)
    gibbs_steps: int = 15
    seed: int = 42


def create_occlusion(image, occ_size, rng):
    H, W = image.shape
    oh, ow = occ_size
    y = rng.integers(0, max(1, H - oh + 1))
    x = rng.integers(0, max(1, W - ow + 1))
    
    occluded = image.copy()
    occluded[y:y+oh, x:x+ow] = 0
    
    pixel_mask = np.ones((H, W), dtype=bool)
    pixel_mask[y:y+oh, x:x+ow] = False
    
    return occluded, pixel_mask


def pixel_to_token_mask(pixel_mask, patch_size, threshold=0.3):
    H, W = pixel_mask.shape
    H_t, W_t = H // patch_size, W // patch_size
    token_mask = np.zeros((H_t, W_t), dtype=bool)
    
    for i in range(H_t):
        for j in range(W_t):
            patch = pixel_mask[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            if patch.mean() < threshold:
                token_mask[i, j] = True
    return token_mask


def token_accuracy(pred, gt, mask):
    if mask.sum() == 0:
        return 0.0
    return (pred[mask] == gt[mask]).mean()


def run():
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    
    print("="*60)
    print("ROUTE C: ANALOG COUPLING (VECTORIZED)")
    print("="*60)
    print(f"Config: train={cfg.train_samples}, eval={cfg.occlusion_eval}, gibbs_steps={cfg.gibbs_steps}")
    
    # Load data
    print("\n[1/5] Loading MNIST...")
    train_img, train_lbl, test_img, test_lbl = load_mnist_numpy("./data")
    train_img, train_lbl = subsample_dataset(train_img, train_lbl, cfg.train_samples, cfg.seed)
    test_img, test_lbl = subsample_dataset(test_img, test_lbl, cfg.test_samples, cfg.seed)
    print(f"       Train: {len(train_img)}, Test: {len(test_img)}")
    
    # Codebook
    print("\n[2/5] Learning codebook...")
    patches = np.concatenate([patchify(img, cfg.patch_size)[0] for img in train_img])
    codebook, _, _ = kmeans_codebook_fit(patches, cfg.codebook_size, cfg.kmeans_iters, cfg.seed)
    print(f"       Codebook: {codebook.shape}")
    
    train_tokens = encode_batch(train_img, codebook, cfg.patch_size)
    test_tokens = encode_batch(test_img, codebook, cfg.patch_size)
    print(f"       Token grid: {train_tokens[0].shape}")
    
    # Conditional model
    print("\n[3/5] Training conditional model...")
    cond_model = TokenConditionalModel(cfg.codebook_size, smoothing=1.0)
    cond_model.fit(train_tokens)
    print(f"       Contexts: {len(cond_model.counts)}")
    
    # Classifier
    print("\n[4/5] Training classifier...")
    def feats(tokens):
        return np.stack([token_grid_to_spatial_histogram(t, cfg.codebook_size, cfg.spatial_bins) for t in tokens])
    
    train_feats, test_feats = feats(train_tokens), feats(test_tokens)
    classifier = LogOddsClassifier(train_feats.shape[1], 10, 1.0)
    classifier.fit(train_feats, train_lbl)
    clean_acc = classifier.score(test_feats, test_lbl)
    print(f"       Clean accuracy: {clean_acc:.1%}")
    
    # Occlusion experiment
    print(f"\n[5/5] Occlusion experiment ({cfg.occlusion_eval} samples)...")
    
    eval_idx = rng.choice(len(test_img), cfg.occlusion_eval, replace=False)
    results = {k: {'correct': 0, 'tok_acc': []} for k in ['none', 'A', 'B1', 'B2', 'C']}
    
    t0 = time.time()
    iterator = tqdm(eval_idx, desc="       Inference") if HAS_TQDM else eval_idx
    
    for idx in iterator:
        orig_img, orig_tok, label = test_img[idx], test_tokens[idx], test_lbl[idx]
        
        occ_img, pix_mask = create_occlusion(orig_img, cfg.occlusion_size, rng)
        tok_mask = pixel_to_token_mask(pix_mask, cfg.patch_size)
        
        if tok_mask.sum() == 0:
            continue
        
        occ_tok, _ = encode_tokens(occ_img, codebook, cfg.patch_size)
        
        # Baseline: no inference
        f = token_grid_to_spatial_histogram(occ_tok, cfg.codebook_size, cfg.spatial_bins)
        results['none']['correct'] += int(classifier.predict(f) == label)
        
        # Mode A: λ_cond=1.0, λ_obs=0, λ_cls=0
        energy_A = VectorizedEnergy(codebook, cfg.patch_size, cond_model, pix_mask,
                                     lambda_cond=1.0, lambda_obs=0.0, lambda_cls=0.0)
        filled_A = fast_gibbs(occ_tok, tok_mask, energy_A, cfg.gibbs_steps, occ_img, cfg.seed+idx)
        f = token_grid_to_spatial_histogram(filled_A, cfg.codebook_size, cfg.spatial_bins)
        results['A']['correct'] += int(classifier.predict(f) == label)
        results['A']['tok_acc'].append(token_accuracy(filled_A, orig_tok, tok_mask))
        
        # Mode B1: λ_cond=1.0, λ_obs=0.1
        energy_B1 = VectorizedEnergy(codebook, cfg.patch_size, cond_model, pix_mask,
                                      lambda_cond=1.0, lambda_obs=0.1, lambda_cls=0.0)
        filled_B1 = fast_gibbs(occ_tok, tok_mask, energy_B1, cfg.gibbs_steps, occ_img, cfg.seed+idx)
        f = token_grid_to_spatial_histogram(filled_B1, cfg.codebook_size, cfg.spatial_bins)
        results['B1']['correct'] += int(classifier.predict(f) == label)
        results['B1']['tok_acc'].append(token_accuracy(filled_B1, orig_tok, tok_mask))
        
        # Mode B2: λ_cond=1.0, λ_obs=1.0
        energy_B2 = VectorizedEnergy(codebook, cfg.patch_size, cond_model, pix_mask,
                                      lambda_cond=1.0, lambda_obs=1.0, lambda_cls=0.0)
        filled_B2 = fast_gibbs(occ_tok, tok_mask, energy_B2, cfg.gibbs_steps, occ_img, cfg.seed+idx)
        f = token_grid_to_spatial_histogram(filled_B2, cfg.codebook_size, cfg.spatial_bins)
        results['B2']['correct'] += int(classifier.predict(f) == label)
        results['B2']['tok_acc'].append(token_accuracy(filled_B2, orig_tok, tok_mask))
        
        # Mode C: λ_cond=0.1, λ_obs=0.1, λ_cls=1.0
        energy_C = VectorizedEnergy(codebook, cfg.patch_size, cond_model, pix_mask,
                                     classifier, cfg.spatial_bins, label,
                                     lambda_cond=0.1, lambda_obs=0.1, lambda_cls=1.0)
        filled_C = fast_gibbs(occ_tok, tok_mask, energy_C, cfg.gibbs_steps, occ_img, cfg.seed+idx)
        f = token_grid_to_spatial_histogram(filled_C, cfg.codebook_size, cfg.spatial_bins)
        results['C']['correct'] += int(classifier.predict(f) == label)
        results['C']['tok_acc'].append(token_accuracy(filled_C, orig_tok, tok_mask))
    
    elapsed = time.time() - t0
    n = cfg.occlusion_eval
    
    # Report
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nClean accuracy: {clean_acc:.1%}")
    print(f"\n| Mode | λ_cond | λ_obs | λ_cls | Token Rec | Occl Acc |")
    print(f"|------|--------|-------|-------|-----------|----------|")
    
    acc_none = results['none']['correct'] / n
    print(f"| none |   -    |   -   |   -   |     -     | {acc_none:>7.1%} |")
    
    for mode, lc, lo, lcls in [('A', '1.0', '0.0', '0.0'), 
                                ('B1', '1.0', '0.1', '0.0'),
                                ('B2', '1.0', '1.0', '0.0'),
                                ('C', '0.1', '0.1', '1.0')]:
        acc = results[mode]['correct'] / n
        tok = np.mean(results[mode]['tok_acc']) if results[mode]['tok_acc'] else 0
        print(f"| {mode:<4} |  {lc}  | {lo}  | {lcls}  | {tok:>8.1%} | {acc:>7.1%} |")
    
    print(f"\nRuntime: {elapsed:.1f}s ({elapsed/n:.2f}s per sample)")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    base = acc_none
    print(f"\nAccuracy delta vs baseline (no inference):")
    for mode in ['A', 'B1', 'B2', 'C']:
        acc = results[mode]['correct'] / n
        print(f"  {mode}: {acc-base:+.1%}")
    
    best = max(['A', 'B1', 'B2', 'C'], key=lambda m: results[m]['correct'])
    print(f"\nBest mode: {best}")
    
    if results['B1']['correct'] > results['A']['correct'] or results['B2']['correct'] > results['A']['correct']:
        print("✓ Analog coupling (E_obs) helps!")
    else:
        print("✗ Analog coupling did not help")
    
    print("="*60)


if __name__ == "__main__":
    run()
