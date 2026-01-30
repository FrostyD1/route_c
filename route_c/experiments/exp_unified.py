#!/usr/bin/env python3
"""
Route C Unified Experiment: Token Grid + Energy-Based Inference + DNF
======================================================================

This experiment implements the full Route C framework:
1. Boundary Translation: Image -> VQ Token Grid
2. Discrete Core Closure: Energy-based inference in token space
3. Classification: Spatial histogram + log-odds OR DNF head
4. Rule Extraction: Teacher -> DNF distillation with logic optimization

Latest Interpretation Summary:
------------------------------
1. Route C = Boundary Translation + Discrete Core Closure
   - Continuous world is interface only: ADC/DAC translate to/from discrete latent
   - Core reasoning/inference is CLOSED in discrete domain (z-space)
   
2. Expressivity Lower Bound:
   - Discretization is NOT inherently weaker
   - If discrete core includes add/mul, it can emulate ResNet
   - Observed weakness = effective capacity / learnability issue, NOT fundamental limit
   
3. Training vs Inference:
   - Training: differentiable surrogates OK (STE/Gumbel)
   - Inference: discrete iterative solving (Gibbs/Metropolis/ICM)
   - Proposal scale MUST match energy scale (block/token > pixel)
   
4. DNF and Log-Odds:
   - Same "readout layer" slot, different parameterizations
   - DNF valuable for interpretability, logic hardening, rule extraction
   - DNF needs (a) spatial features, (b) stronger learning, (c) logic optimization
"""

import numpy as np
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Add parent directory to path for direct execution
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Use imports that work both as package and direct
from core import make_random_mask, make_block_mask
from boundary import (
    patchify, kmeans_codebook_fit, encode_tokens, decode_tokens,
    encode_batch, decode_batch,
)
from learning import (
    TokenConditionalModel, LogOddsClassifier,
    extract_features_batch, token_grid_to_spatial_histogram,
)
from learning.dnf import (
    TeacherDistillation, binarize_features, create_dnf_features_from_histogram,
)
from inference import gibbs_fill, compute_token_accuracy
from inference.energy import (
    EnergyModel, TokenConditionalEnergy, ClassifierEnergy, CombinedEnergy,
    gibbs_fill_energy, metropolis_fill_energy,
)
from logic import optimize_dnf, dnf_to_str, OptimizationStats
from mnist import (
    load_mnist_numpy, subsample_dataset,
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
    codebook_size: int = 64
    kmeans_iters: int = 50
    
    # Model
    smoothing: float = 1.0
    boundary: str = "standard"
    
    # Inference
    gibbs_steps: int = 20
    metropolis_steps: int = 50
    temperature: float = 1.0
    
    # Energy combination weights
    lambda_conditional: float = 1.0
    lambda_classifier: float = 0.1
    
    # Classification
    spatial_bins: Tuple[int, int] = (2, 2)
    
    # DNF
    dnf_max_depth: int = 6
    dnf_min_samples_leaf: int = 20
    dnf_top_k_features: int = 100
    
    # Data
    train_samples: int = 10000
    test_samples: int = 2000
    
    # Masking
    mask_ratio: float = 0.25
    occlusion_size: Tuple[int, int] = (14, 14)
    
    # Seeds
    seed: int = 42


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def experiment_tokenization(
    train_images: np.ndarray,
    test_images: np.ndarray,
    cfg: Config,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Experiment A: VQ Tokenization Quality
    
    - Learn codebook via k-means
    - Encode images to token grids
    - Evaluate reconstruction MSE/PSNR
    """
    if verbose:
        print("\n" + "="*60)
        print("EXPERIMENT A: TOKENIZATION QUALITY")
        print("="*60)
    
    # Extract patches from training images
    all_patches = []
    for img in train_images:
        patches, _ = patchify(img, cfg.patch_size)
        all_patches.append(patches)
    all_patches = np.concatenate(all_patches, axis=0)
    
    if verbose:
        print(f"  Extracted {len(all_patches)} patches from {len(train_images)} images")
    
    # Learn codebook
    codebook, _, losses = kmeans_codebook_fit(
        all_patches, cfg.codebook_size, cfg.kmeans_iters, cfg.seed, verbose=verbose
    )
    
    if verbose:
        print(f"  Codebook shape: {codebook.shape}")
        print(f"  Final k-means loss: {losses[-1]:.2f}")
    
    # Encode train and test
    train_tokens = encode_batch(train_images, codebook, cfg.patch_size)
    test_tokens = encode_batch(test_images, codebook, cfg.patch_size)
    
    if verbose:
        print(f"  Token grid shape: {train_tokens[0].shape}")
    
    # Evaluate reconstruction
    test_recon = decode_batch(test_tokens, codebook, cfg.patch_size)
    metrics = compute_reconstruction_metrics(test_images, test_recon)
    
    if verbose:
        print(f"\n  Reconstruction Metrics:")
        print(f"    MSE:  {metrics['mse']:.2f}")
        print(f"    PSNR: {metrics['psnr']:.2f} dB")
    
    return {
        'codebook': codebook,
        'train_tokens': train_tokens,
        'test_tokens': test_tokens,
        'test_recon': test_recon,
        'metrics': metrics,
    }


def experiment_conditional_model(
    train_tokens: np.ndarray,
    test_tokens: np.ndarray,
    cfg: Config,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Experiment B: Token Conditional Model + Masked Completion
    
    - Fit count-based conditional model P(token | context)
    - Evaluate masked token completion with Gibbs sampling
    """
    if verbose:
        print("\n" + "="*60)
        print("EXPERIMENT B: CONDITIONAL MODEL + MASKED COMPLETION")
        print("="*60)
    
    # Fit model
    model = TokenConditionalModel(
        vocab_size=cfg.codebook_size,
        smoothing=cfg.smoothing,
        boundary=cfg.boundary,
    )
    model.fit(train_tokens, verbose=verbose)
    
    if verbose:
        print(f"  Unique contexts: {len(model.counts)}")
    
    # Evaluate masked completion
    rng = np.random.default_rng(cfg.seed)
    n_eval = min(100, len(test_tokens))
    
    random_accs = []
    gibbs_accs = []
    
    for i in range(n_eval):
        original = test_tokens[i].copy()
        mask = make_random_mask(original.shape, cfg.mask_ratio, rng)
        
        # Random baseline
        random_fill = original.copy()
        random_fill[mask] = rng.integers(cfg.codebook_size, size=mask.sum())
        random_acc = compute_token_accuracy(random_fill, original, mask)
        random_accs.append(random_acc)
        
        # Gibbs fill
        masked = original.copy()
        masked[mask] = 0
        filled = gibbs_fill(masked, mask, model, cfg.gibbs_steps, seed=cfg.seed+i)
        gibbs_acc = compute_token_accuracy(filled, original, mask)
        gibbs_accs.append(gibbs_acc)
    
    if verbose:
        print(f"\n  Masked Completion Results (mask={cfg.mask_ratio:.0%}):")
        print(f"    Random baseline: {np.mean(random_accs):.1%} ± {np.std(random_accs):.1%}")
        print(f"    Gibbs fill:      {np.mean(gibbs_accs):.1%} ± {np.std(gibbs_accs):.1%}")
    
    return {
        'model': model,
        'random_acc': (np.mean(random_accs), np.std(random_accs)),
        'gibbs_acc': (np.mean(gibbs_accs), np.std(gibbs_accs)),
    }


def experiment_classification(
    train_tokens: np.ndarray,
    train_labels: np.ndarray,
    test_tokens: np.ndarray,
    test_labels: np.ndarray,
    cfg: Config,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Experiment C: Classification with Log-Odds Head
    
    - Extract spatial histogram features
    - Train log-odds classifier
    - Evaluate accuracy
    """
    if verbose:
        print("\n" + "="*60)
        print("EXPERIMENT C: CLASSIFICATION (LOG-ODDS)")
        print("="*60)
    
    # Extract features
    train_features = extract_features_batch(
        train_tokens, cfg.codebook_size, cfg.spatial_bins, include_global=False
    )
    test_features = extract_features_batch(
        test_tokens, cfg.codebook_size, cfg.spatial_bins, include_global=False
    )
    
    if verbose:
        print(f"  Feature dimension: {train_features.shape[1]}")
    
    # Train classifier
    classifier = LogOddsClassifier(
        n_features=train_features.shape[1],
        n_classes=10,
        smoothing=cfg.smoothing,
    )
    classifier.fit(train_features, train_labels, verbose=verbose)
    
    # Evaluate
    train_acc = classifier.score(train_features, train_labels)
    test_acc = classifier.score(test_features, test_labels)
    
    if verbose:
        print(f"\n  Classification Results:")
        print(f"    Train accuracy: {train_acc:.1%}")
        print(f"    Test accuracy:  {test_acc:.1%}")
    
    return {
        'classifier': classifier,
        'train_features': train_features,
        'test_features': test_features,
        'train_acc': train_acc,
        'test_acc': test_acc,
    }


def experiment_energy_inference(
    test_images: np.ndarray,
    test_labels: np.ndarray,
    codebook: np.ndarray,
    model: TokenConditionalModel,
    classifier: LogOddsClassifier,
    cfg: Config,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Experiment D: Energy-Based Inference for Occlusion Robustness
    
    - Compare TokenConditionalEnergy vs ClassifierEnergy vs Combined
    - Evaluate classification before/after inference
    """
    if verbose:
        print("\n" + "="*60)
        print("EXPERIMENT D: ENERGY-BASED INFERENCE")
        print("="*60)
    
    # Create occluded images
    occluded_images, pixel_masks = create_occluded_images(
        test_images, cfg.occlusion_size, seed=cfg.seed
    )
    
    # Encode
    original_tokens = encode_batch(test_images, codebook, cfg.patch_size)
    occluded_tokens = encode_batch(occluded_images, codebook, cfg.patch_size)
    
    # Convert pixel masks to token masks
    token_masks = np.array([
        pixel_mask_to_token_mask(pm, cfg.patch_size) 
        for pm in pixel_masks
    ])
    
    if verbose:
        print(f"  Occlusion: {cfg.occlusion_size[0]}x{cfg.occlusion_size[1]} pixels")
        print(f"  Avg tokens masked: {np.mean([m.sum() for m in token_masks]):.1f}")
    
    # Classification on clean
    clean_features = extract_features_batch(
        original_tokens, cfg.codebook_size, cfg.spatial_bins, include_global=False
    )
    clean_acc = classifier.score(clean_features, test_labels)
    
    # Classification on occluded (no inference)
    occluded_features = extract_features_batch(
        occluded_tokens, cfg.codebook_size, cfg.spatial_bins, include_global=False
    )
    occluded_acc = classifier.score(occluded_features, test_labels)
    
    if verbose:
        print(f"\n  Before inference:")
        print(f"    Clean accuracy:    {clean_acc:.1%}")
        print(f"    Occluded accuracy: {occluded_acc:.1%}")
    
    # Energy-based inference
    n_eval = min(200, len(test_images))
    
    results = {
        'conditional': {'accs': [], 'token_accs': []},
        'classifier': {'accs': [], 'token_accs': []},
        'combined': {'accs': [], 'token_accs': []},
    }
    
    for i in range(n_eval):
        token_grid = occluded_tokens[i].copy()
        mask = token_masks[i]
        original = original_tokens[i]
        label = test_labels[i]
        
        if mask.sum() == 0:
            continue
        
        # E1: Token Conditional Energy
        energy_cond = TokenConditionalEnergy(model, mask)
        filled_cond, _ = gibbs_fill_energy(
            token_grid.copy(), mask, energy_cond, cfg.codebook_size,
            cfg.gibbs_steps, cfg.seed + i, verbose=False
        )
        
        feat = token_grid_to_spatial_histogram(filled_cond, cfg.codebook_size, cfg.spatial_bins)
        pred = classifier.predict(feat)
        results['conditional']['accs'].append(pred == label)
        results['conditional']['token_accs'].append(compute_token_accuracy(filled_cond, original, mask))
        
        # E2: Classifier Energy
        energy_class = ClassifierEnergy(
            classifier, cfg.codebook_size, cfg.spatial_bins, target_class=label
        )
        filled_class, _ = gibbs_fill_energy(
            token_grid.copy(), mask, energy_class, cfg.codebook_size,
            cfg.gibbs_steps, cfg.seed + i, verbose=False
        )
        
        feat = token_grid_to_spatial_histogram(filled_class, cfg.codebook_size, cfg.spatial_bins)
        pred = classifier.predict(feat)
        results['classifier']['accs'].append(pred == label)
        results['classifier']['token_accs'].append(compute_token_accuracy(filled_class, original, mask))
        
        # Combined Energy
        energy_combined = CombinedEnergy([
            (cfg.lambda_conditional, energy_cond),
            (cfg.lambda_classifier, energy_class),
        ])
        filled_combined, _ = gibbs_fill_energy(
            token_grid.copy(), mask, energy_combined, cfg.codebook_size,
            cfg.gibbs_steps, cfg.seed + i, verbose=False
        )
        
        feat = token_grid_to_spatial_histogram(filled_combined, cfg.codebook_size, cfg.spatial_bins)
        pred = classifier.predict(feat)
        results['combined']['accs'].append(pred == label)
        results['combined']['token_accs'].append(compute_token_accuracy(filled_combined, original, mask))
    
    if verbose:
        print(f"\n  After inference (n={n_eval}):")
        print(f"  {'Energy Type':<20} {'Class Acc':>12} {'Token Acc':>12}")
        print(f"  {'-'*45}")
        for name in ['conditional', 'classifier', 'combined']:
            acc = np.mean(results[name]['accs'])
            token_acc = np.mean(results[name]['token_accs'])
            delta = acc - occluded_acc
            print(f"  {name:<20} {acc:>10.1%} ({delta:+.1%}) {token_acc:>10.1%}")
    
    return {
        'clean_acc': clean_acc,
        'occluded_acc': occluded_acc,
        'results': results,
    }


def experiment_dnf_distillation(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    classifier: LogOddsClassifier,
    cfg: Config,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Experiment E: DNF Distillation from Teacher
    
    - Use log-odds classifier as teacher
    - Distill to decision trees
    - Extract DNF rules with logic optimization
    - Compare accuracy and rule complexity
    """
    if verbose:
        print("\n" + "="*60)
        print("EXPERIMENT E: DNF DISTILLATION")
        print("="*60)
    
    # Get teacher predictions
    teacher_train_pred = classifier.predict(train_features)
    teacher_test_pred = classifier.predict(test_features)
    
    teacher_train_acc = (teacher_train_pred == train_labels).mean()
    teacher_test_acc = (teacher_test_pred == test_labels).mean()
    
    if verbose:
        print(f"  Teacher (log-odds):")
        print(f"    Train acc: {teacher_train_acc:.1%}")
        print(f"    Test acc:  {teacher_test_acc:.1%}")
    
    # Binarize features for DNF
    binary_train, selected = create_dnf_features_from_histogram(
        train_features, top_k=cfg.dnf_top_k_features
    )
    binary_test = binarize_features(
        test_features[:, selected], n_bins=2
    )
    
    if verbose:
        print(f"\n  Binary features: {binary_train.shape[1]} (from top {cfg.dnf_top_k_features})")
    
    # Distill
    distiller = TeacherDistillation(
        n_classes=10,
        max_depth=cfg.dnf_max_depth,
        min_samples_leaf=cfg.dnf_min_samples_leaf,
    )
    
    rules = distiller.distill(binary_train, teacher_train_pred, optimize=True, verbose=verbose)
    
    # Evaluate
    dnf_train_acc = distiller.score(binary_train, train_labels)
    dnf_test_acc = distiller.score(binary_test, test_labels)
    
    complexity = distiller.get_rule_complexity()
    
    if verbose:
        print(f"\n  DNF (distilled):")
        print(f"    Train acc: {dnf_train_acc:.1%}")
        print(f"    Test acc:  {dnf_test_acc:.1%}")
        print(f"\n  Rule Complexity:")
        print(f"    Total clauses:  {complexity['total_clauses']}")
        print(f"    Total literals: {complexity['total_literals']}")
        print(f"    Avg literals/clause: {complexity['avg_literals_per_clause']:.1f}")
    
    # Show sample rules
    if verbose:
        print(f"\n  Sample rules (class 0):")
        rules_0 = rules.get(0, [])[:3]
        for i, clause in enumerate(rules_0):
            print(f"    Clause {i+1}: {len(clause)} literals")
    
    return {
        'distiller': distiller,
        'rules': rules,
        'teacher_train_acc': teacher_train_acc,
        'teacher_test_acc': teacher_test_acc,
        'dnf_train_acc': dnf_train_acc,
        'dnf_test_acc': dnf_test_acc,
        'complexity': complexity,
    }


def experiment_logic_optimization(
    rules: Dict[int, list],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Experiment F: Logic Optimization Statistics
    
    - Apply full optimization pipeline
    - Report size reduction and sharing stats
    """
    if verbose:
        print("\n" + "="*60)
        print("EXPERIMENT F: LOGIC OPTIMIZATION")
        print("="*60)
    
    all_stats = []
    
    for c, class_rules in rules.items():
        if not class_rules:
            continue
        
        optimized, stats = optimize_dnf(class_rules, verbose=False)
        all_stats.append(stats)
        
        if verbose and c < 3:  # Show first 3 classes
            print(f"\n  Class {c}:")
            print(f"    Clauses: {stats.original_clauses} -> {stats.final_clauses} ({stats.clause_reduction:.1%} reduction)")
            print(f"    Literals: {stats.original_literals} -> {stats.final_literals} ({stats.literal_reduction:.1%} reduction)")
            print(f"    Subsumed: {stats.removed_subsumed}, Factored: {stats.factored}, Shared: {stats.shared_nodes}")
    
    # Aggregate stats
    total_original = sum(s.original_clauses for s in all_stats)
    total_final = sum(s.final_clauses for s in all_stats)
    total_shared = sum(s.shared_nodes for s in all_stats)
    
    if verbose:
        print(f"\n  Overall:")
        print(f"    Total clauses: {total_original} -> {total_final}")
        print(f"    Total shared nodes: {total_shared}")
    
    return {
        'stats': all_stats,
        'total_original': total_original,
        'total_final': total_final,
        'total_shared': total_shared,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all experiments."""
    cfg = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("ROUTE C: UNIFIED DISCRETE WORLD MODELING")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Configuration:")
    for attr in ['patch_size', 'codebook_size', 'spatial_bins', 'mask_ratio', 'gibbs_steps']:
        print(f"  {attr}: {getattr(cfg, attr)}")
    
    # Load data
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist_numpy("./data")
    
    train_images, train_labels = subsample_dataset(
        train_images, train_labels, cfg.train_samples, cfg.seed
    )
    test_images, test_labels = subsample_dataset(
        test_images, test_labels, cfg.test_samples, cfg.seed
    )
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")
    
    t_start = time.time()
    
    # Run experiments
    results_A = experiment_tokenization(train_images, test_images, cfg)
    codebook = results_A['codebook']
    train_tokens = results_A['train_tokens']
    test_tokens = results_A['test_tokens']
    
    results_B = experiment_conditional_model(train_tokens, test_tokens, cfg)
    model = results_B['model']
    
    results_C = experiment_classification(
        train_tokens, train_labels, test_tokens, test_labels, cfg
    )
    classifier = results_C['classifier']
    
    results_D = experiment_energy_inference(
        test_images, test_labels, codebook, model, classifier, cfg
    )
    
    results_E = experiment_dnf_distillation(
        results_C['train_features'], train_labels,
        results_C['test_features'], test_labels,
        classifier, cfg
    )
    
    results_F = experiment_logic_optimization(results_E['rules'])
    
    t_total = time.time() - t_start
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\n{'Metric':<40} {'Value':>15}")
    print("-"*55)
    print(f"{'Tokenization MSE':<40} {results_A['metrics']['mse']:>15.2f}")
    print(f"{'Tokenization PSNR':<40} {results_A['metrics']['psnr']:>13.2f} dB")
    print(f"{'Masked completion (random)':<40} {results_B['random_acc'][0]:>14.1%}")
    print(f"{'Masked completion (Gibbs)':<40} {results_B['gibbs_acc'][0]:>14.1%}")
    print(f"{'Classification (clean)':<40} {results_C['test_acc']:>14.1%}")
    print(f"{'Classification (occluded)':<40} {results_D['occluded_acc']:>14.1%}")
    print(f"{'Classification (E_cond infer)':<40} {np.mean(results_D['results']['conditional']['accs']):>14.1%}")
    print(f"{'Classification (E_combined infer)':<40} {np.mean(results_D['results']['combined']['accs']):>14.1%}")
    print(f"{'DNF distilled (test)':<40} {results_E['dnf_test_acc']:>14.1%}")
    print(f"{'DNF total clauses':<40} {results_E['complexity']['total_clauses']:>15}")
    print(f"{'Logic opt reduction':<40} {1-results_F['total_final']/max(1,results_F['total_original']):>14.1%}")
    
    print(f"\nTotal runtime: {t_total:.1f}s")
    print("="*60)
    
    # Save summary
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write("Route C Experiment Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Tokenization MSE: {results_A['metrics']['mse']:.2f}\n")
        f.write(f"Tokenization PSNR: {results_A['metrics']['psnr']:.2f}\n")
        f.write(f"Masked completion (random): {results_B['random_acc'][0]:.3f}\n")
        f.write(f"Masked completion (Gibbs): {results_B['gibbs_acc'][0]:.3f}\n")
        f.write(f"Classification (clean): {results_C['test_acc']:.3f}\n")
        f.write(f"Classification (occluded): {results_D['occluded_acc']:.3f}\n")
        f.write(f"Classification (E_combined): {np.mean(results_D['results']['combined']['accs']):.3f}\n")
        f.write(f"DNF test acc: {results_E['dnf_test_acc']:.3f}\n")
        f.write(f"DNF clauses: {results_E['complexity']['total_clauses']}\n")
    
    print(f"Summary saved to {output_dir}/summary.txt")
    
    return {
        'A': results_A,
        'B': results_B,
        'C': results_C,
        'D': results_D,
        'E': results_E,
        'F': results_F,
    }


if __name__ == "__main__":
    main()
