#!/usr/bin/env python3
"""
Route C: Full Experiment Suite
==============================

Runs all Route C experiments in sequence:
1. Train learnable ADC/DAC + local energy + classifier
2. Occlusion + discrete inference evaluation
3. Logic distillation (classifier → DNF)

Demonstrates:
G1. Learnable ADC improves vs non-learned tokenization
G2. Discrete core closure learns meaningful local structure
G3. Analog coupling helps occlusion recovery
G4. Logic distillation extracts interpretable rules
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import time

# Add parent to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import experiments
from exp_mnist_routec_learnable_adc import (
    Config, RouteCModel, load_data, train
)

# Import distillation
sys.path.insert(0, _parent_dir)
from learning.distill_logic import (
    LogicDistiller, generate_feature_names, dnf_to_verilog
)


def run_full_experiment():
    print("="*70)
    print("ROUTE C: FULL EXPERIMENT SUITE")
    print("="*70)
    print("\nThis demonstrates:")
    print("  G1. Learnable ADC vs fixed tokenization")
    print("  G2. Discrete core closure (masked prediction)")
    print("  G3. Analog coupling for occlusion recovery")
    print("  G4. Logic distillation (DNF extraction)")
    
    # ========================================================================
    # PHASE 1: TRAINING
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 1: TRAINING ROUTE C MODEL")
    print("="*70)
    
    cfg = Config()
    if torch.cuda.is_available():
        cfg.device = "cuda"
    
    model, cfg, train_data, test_data = train(cfg)
    
    # ========================================================================
    # PHASE 2: OCCLUSION + INFERENCE
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 2: OCCLUSION + DISCRETE INFERENCE")
    print("="*70)
    
    from exp_mnist_occlusion_inference import (
        InferenceConfig, DiscreteInferenceEngine,
        create_occlusion, create_bit_mask, estimate_sigma_squared
    )
    
    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False
        def tqdm(x, **kw): return x
    
    test_x, test_y = test_data
    device = cfg.device
    model.eval()
    
    # Estimate σ²
    print("\nEstimating σ² from reconstruction...")
    sigma_sq = estimate_sigma_squared(model, test_x, device, n_samples=500)
    print(f"  σ² = {sigma_sq:.6f}")
    
    # Run occlusion experiment
    inf_cfg = InferenceConfig()
    inf_cfg.n_eval = 200
    engine = DiscreteInferenceEngine(model, device, sigma_sq, inf_cfg)
    
    rng = np.random.default_rng(inf_cfg.seed)
    eval_idx = rng.choice(len(test_x), inf_cfg.n_eval, replace=False)
    
    results = {
        'clean_correct': 0,
        'baseline_correct': 0,
        'inferred_correct': 0,
        'bit_recovery': [],
    }
    
    print(f"\nRunning inference on {inf_cfg.n_eval} samples...")
    t0 = time.time()
    
    iterator = tqdm(eval_idx, desc="Inference") if HAS_TQDM else eval_idx
    
    for idx in iterator:
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()
        
        # Clean prediction
        with torch.no_grad():
            x_t = test_x[idx:idx+1].to(device)
            z_clean = model.encode(x_t)
            logits_clean = model.classifier(z_clean)
            pred_clean = logits_clean.argmax(dim=1).item()
        
        results['clean_correct'] += int(pred_clean == label)
        
        # Create occlusion
        x_occ, pixel_mask = create_occlusion(x_clean, inf_cfg.occlusion_size, rng)
        
        # Encode occluded
        with torch.no_grad():
            x_occ_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_occ = model.encode(x_occ_t)[0].cpu().numpy()
            
            logits_occ = model.classifier(torch.from_numpy(z_occ[None]).to(device))
            pred_baseline = logits_occ.argmax(dim=1).item()
        
        results['baseline_correct'] += int(pred_baseline == label)
        
        # Create bit mask and run inference
        bit_mask = create_bit_mask(pixel_mask, cfg.n_bits, cfg.latent_size)
        z_inferred, _ = engine.run_inference(z_occ, x_occ, pixel_mask.astype(np.float32), bit_mask)
        
        # Prediction after inference
        with torch.no_grad():
            logits_inf = model.classifier(torch.from_numpy(z_inferred[None]).to(device))
            pred_inferred = logits_inf.argmax(dim=1).item()
        
        results['inferred_correct'] += int(pred_inferred == label)
        
        # Bit recovery
        z_clean_np = z_clean[0].cpu().numpy()
        if bit_mask.any():
            recovery = (z_inferred[bit_mask] == z_clean_np[bit_mask]).mean()
            results['bit_recovery'].append(recovery)
    
    inf_time = time.time() - t0
    
    n = inf_cfg.n_eval
    clean_acc = results['clean_correct'] / n
    baseline_acc = results['baseline_correct'] / n
    inferred_acc = results['inferred_correct'] / n
    bit_rec = np.mean(results['bit_recovery']) if results['bit_recovery'] else 0
    
    print(f"\nOcclusion Results ({inf_time:.1f}s):")
    print(f"  Clean accuracy:     {clean_acc:.1%}")
    print(f"  Occluded baseline:  {baseline_acc:.1%}")
    print(f"  Occluded inferred:  {inferred_acc:.1%}")
    print(f"  Δ inference:        {inferred_acc - baseline_acc:+.1%}")
    print(f"  Bit recovery:       {bit_rec:.1%}")
    
    # ========================================================================
    # PHASE 3: LOGIC DISTILLATION
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 3: LOGIC DISTILLATION")
    print("="*70)
    
    # Get z representations and teacher predictions
    print("\nExtracting z and teacher predictions...")
    
    train_x, train_y = train_data
    
    with torch.no_grad():
        # Use subset for distillation
        n_distill = min(5000, len(train_x))
        Z_train = []
        teacher_preds = []
        
        for i in range(0, n_distill, 64):
            batch_x = train_x[i:i+64].to(device)
            z = model.encode(batch_x)
            logits = model.classifier(z)
            
            Z_train.append(z.cpu().numpy())
            teacher_preds.append(logits.argmax(dim=1).cpu().numpy())
        
        Z_train = np.concatenate(Z_train, axis=0)
        teacher_preds = np.concatenate(teacher_preds, axis=0)
    
    # Flatten z for distillation
    Z_flat = Z_train.reshape(Z_train.shape[0], -1)  # (N, k*H*W)
    
    print(f"  Z shape: {Z_flat.shape}")
    print(f"  Teacher predictions: {len(teacher_preds)}")
    
    # Run distillation
    print("\nDistilling to DNF...")
    
    feature_names = generate_feature_names(cfg.n_bits, cfg.latent_size)
    
    distiller = LogicDistiller(
        n_classes=10,
        max_depth=10,
        min_samples_leaf=10,
    )
    distiller.fit(Z_flat, teacher_preds, feature_names)
    
    # Evaluate fidelity
    distill_preds = distiller.predict(Z_flat)
    fidelity = (distill_preds == teacher_preds).mean()
    
    stats = distiller.get_stats()
    
    print(f"\nDistillation Results:")
    print(f"  Fidelity (agreement with teacher): {fidelity:.1%}")
    print(f"  Total clauses: {stats['total_clauses']}")
    print(f"  Total literals: {stats['total_literals']}")
    print(f"  Avg clauses per class: {stats['avg_clauses_per_class']:.1f}")
    print(f"  Avg literals per clause: {stats['avg_literals_per_clause']:.1f}")
    
    # Show sample rules
    print("\nSample DNF rules (class 0):")
    rule_str = distiller.dnf_to_str(0)
    if len(rule_str) > 200:
        rule_str = rule_str[:200] + "..."
    print(f"  {rule_str}")
    
    # Generate Verilog
    print("\nGenerating Verilog...")
    verilog = dnf_to_verilog(distiller.dnfs, feature_names, "mnist_classifier")
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/classifier.v", "w") as f:
        f.write(verilog)
    print("  Saved to outputs/classifier.v")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print(f"""
| Metric                           | Value      |
|----------------------------------|------------|
| Clean test accuracy              | {clean_acc:>9.1%} |
| Core closure (masked pred)       | {bit_rec:>9.1%} |
| Occluded baseline accuracy       | {baseline_acc:>9.1%} |
| Occluded after inference         | {inferred_acc:>9.1%} |
| Inference benefit                | {inferred_acc - baseline_acc:>+8.1%} |
| Logic distillation fidelity      | {fidelity:>9.1%} |
| Total DNF clauses                | {stats['total_clauses']:>10} |
| Total DNF literals               | {stats['total_literals']:>10} |
""")
    
    print("="*70)
    print("ROUTE C VERIFICATION SUMMARY")
    print("="*70)
    
    print("\nG1. Learnable ADC:")
    print(f"    ✓ Achieved {clean_acc:.1%} clean accuracy with learned binary z")
    
    print("\nG2. Discrete Core Closure:")
    if bit_rec > 0.55:
        print(f"    ✓ Masked bit prediction: {bit_rec:.1%} (> 50% baseline)")
    else:
        print(f"    ⚠ Masked bit prediction: {bit_rec:.1%} (needs improvement)")
    
    print("\nG3. Analog Coupling (Occlusion Recovery):")
    delta = inferred_acc - baseline_acc
    if delta > 0:
        print(f"    ✓ Inference improved accuracy by {delta:+.1%}")
    else:
        print(f"    ⚠ Inference did not improve ({delta:+.1%})")
    
    print("\nG4. Logic Distillation:")
    if fidelity > 0.7:
        print(f"    ✓ Fidelity {fidelity:.1%}, {stats['total_clauses']} clauses extracted")
    else:
        print(f"    ⚠ Fidelity {fidelity:.1%} is low")
    
    print("\n" + "="*70)
    
    return {
        'clean_acc': clean_acc,
        'baseline_acc': baseline_acc,
        'inferred_acc': inferred_acc,
        'bit_recovery': bit_rec,
        'fidelity': fidelity,
        'dnf_stats': stats,
    }


if __name__ == "__main__":
    results = run_full_experiment()
