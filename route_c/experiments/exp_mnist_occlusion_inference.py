#!/usr/bin/env python3
"""
Route C: Occlusion + Discrete Inference Experiment
==================================================

This experiment demonstrates:
G3. Analog coupling acts as boundary condition for occlusion recovery
G4. Inference remains discrete (block Gibbs/MH in z-space)

Setup:
1. Load trained Route C model
2. Apply 14×14 occlusion to test images
3. Run discrete inference to recover z in occluded region
4. Compare accuracy before/after inference

Energy for inference:
E(z) = λ_core * E_core(z) + λ_obs * E_obs(z, o)

Where σ² is estimated from clean reconstruction residuals.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Any

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw): return x

# Add parent to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from exp_mnist_routec_learnable_adc import (
    Config, RouteCModel, load_data
)


# ============================================================================
# INFERENCE CONFIG
# ============================================================================

@dataclass
class InferenceConfig:
    n_steps: int = 30
    block_size: Tuple[int, int] = (2, 2)
    temperature: float = 1.0
    lambda_core: float = 1.0
    lambda_obs: float = 1.0
    occlusion_size: Tuple[int, int] = (14, 14)
    n_eval: int = 200
    seed: int = 42


# ============================================================================
# DISCRETE INFERENCE
# ============================================================================

class DiscreteInferenceEngine:
    """Block Gibbs inference in z-space."""
    
    def __init__(
        self,
        model: RouteCModel,
        device: str,
        sigma_sq: float,
        inf_cfg: InferenceConfig,
    ):
        self.model = model
        self.device = device
        self.sigma_sq = sigma_sq
        self.cfg = inf_cfg
        self.model.eval()
    
    def decode_np(self, z: np.ndarray) -> np.ndarray:
        """z: (k, H, W) → o_hat: (28, 28)"""
        with torch.no_grad():
            z_t = torch.from_numpy(z.astype(np.float32))[None].to(self.device)
            o_hat = self.model.decode(z_t)
            return o_hat[0, 0].cpu().numpy()
    
    def energy_core(self, z: np.ndarray) -> float:
        """E_core = -Σ log p(z_i | neigh)"""
        with torch.no_grad():
            z_t = torch.from_numpy(z.astype(np.float32))[None].to(self.device)
            logits = self.model.local_pred(z_t)
            loss = F.binary_cross_entropy_with_logits(logits, z_t, reduction='sum')
            return loss.item()
    
    def energy_obs(self, z: np.ndarray, o_obs: np.ndarray, mask: np.ndarray) -> float:
        """E_obs = (1/(2σ²)) ||M ⊙ (D(z) - o)||²"""
        o_hat = self.decode_np(z)
        diff = (o_hat - o_obs) * mask
        mse = (diff ** 2).sum() / max(1, mask.sum())
        return mse / (2 * self.sigma_sq)
    
    def total_energy(self, z: np.ndarray, o_obs: np.ndarray, mask: np.ndarray) -> float:
        E = 0.0
        if self.cfg.lambda_core > 0:
            E += self.cfg.lambda_core * self.energy_core(z)
        if self.cfg.lambda_obs > 0:
            E += self.cfg.lambda_obs * self.energy_obs(z, o_obs, mask)
        return E
    
    def run_inference(
        self,
        z_init: np.ndarray,
        o_obs: np.ndarray,
        pixel_mask: np.ndarray,
        bit_mask: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run block Gibbs inference.
        
        Args:
            z_init: initial z from encoding occluded image
            o_obs: observed image (with occlusion)
            pixel_mask: (28, 28) True where pixel is observed
            bit_mask: (k, H, W) True where bit is unknown
        
        Returns:
            z_final: optimized z
            stats: energy trace
        """
        rng = np.random.default_rng(self.cfg.seed)
        z = z_init.copy()
        k, H, W = z.shape
        bh, bw = self.cfg.block_size
        
        # Initialize masked bits randomly
        z[bit_mask] = rng.integers(0, 2, size=bit_mask.sum())
        
        stats = {'energies': []}
        
        for step in range(self.cfg.n_steps):
            # Iterate over spatial blocks
            for bi in range(0, H, bh):
                for bj in range(0, W, bw):
                    i_end = min(bi + bh, H)
                    j_end = min(bj + bw, W)
                    
                    # Get masked bits in this block
                    block_mask = bit_mask[:, bi:i_end, bj:j_end]
                    masked_pos = np.argwhere(block_mask)
                    
                    if len(masked_pos) == 0:
                        continue
                    
                    n_bits = len(masked_pos)
                    
                    if n_bits <= 6:  # Enumerate up to 64 configs
                        # Enumerate all configurations
                        energies = []
                        for config in range(2 ** n_bits):
                            for idx, (b, i, j) in enumerate(masked_pos):
                                z[b, bi + i, bj + j] = (config >> idx) & 1
                            E = self.total_energy(z, o_obs, pixel_mask)
                            energies.append(E)
                        
                        # Sample from Boltzmann
                        energies = np.array(energies)
                        energies = (energies - energies.min()) / self.cfg.temperature
                        probs = np.exp(-energies)
                        probs = probs / probs.sum()
                        
                        chosen = rng.choice(2 ** n_bits, p=probs)
                        
                        for idx, (b, i, j) in enumerate(masked_pos):
                            z[b, bi + i, bj + j] = (chosen >> idx) & 1
                    
                    else:
                        # MH with random flips
                        E_current = self.total_energy(z, o_obs, pixel_mask)
                        
                        for _ in range(3):
                            n_flip = rng.integers(1, min(4, n_bits) + 1)
                            flip_idx = rng.choice(n_bits, size=n_flip, replace=False)
                            
                            z_prop = z.copy()
                            for idx in flip_idx:
                                b, i, j = masked_pos[idx]
                                z_prop[b, bi + i, bj + j] = 1 - z_prop[b, bi + i, bj + j]
                            
                            E_prop = self.total_energy(z_prop, o_obs, pixel_mask)
                            
                            dE = (E_prop - E_current) / self.cfg.temperature
                            if dE < 0 or rng.random() < np.exp(-dE):
                                z = z_prop
                                E_current = E_prop
            
            stats['energies'].append(self.total_energy(z, o_obs, pixel_mask))
        
        return z, stats


# ============================================================================
# OCCLUSION HELPERS
# ============================================================================

def create_occlusion(
    image: np.ndarray,
    occ_size: Tuple[int, int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create occluded image and pixel mask."""
    H, W = image.shape
    oh, ow = occ_size
    
    y = rng.integers(0, max(1, H - oh + 1))
    x = rng.integers(0, max(1, W - ow + 1))
    
    occluded = image.copy()
    occluded[y:y+oh, x:x+ow] = 0
    
    pixel_mask = np.ones((H, W), dtype=bool)
    pixel_mask[y:y+oh, x:x+ow] = False
    
    return occluded, pixel_mask


def create_bit_mask(
    pixel_mask: np.ndarray,
    n_bits: int,
    latent_size: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Create bit mask from pixel mask."""
    patch_size = 28 // latent_size
    bit_mask = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    
    for i in range(latent_size):
        for j in range(latent_size):
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            
            obs_ratio = pixel_mask[y0:y1, x0:x1].mean()
            if obs_ratio < threshold:
                bit_mask[:, i, j] = True
    
    return bit_mask


def estimate_sigma_squared(
    model: RouteCModel,
    images: torch.Tensor,
    device: str,
    n_samples: int = 500,
) -> float:
    """Estimate σ² from reconstruction residuals."""
    model.eval()
    residuals = []
    
    with torch.no_grad():
        for i in range(min(n_samples, len(images))):
            x = images[i:i+1].to(device)
            z = model.encode(x)
            x_hat = model.decode(z)
            
            res = ((x_hat - x) ** 2).mean().item()
            residuals.append(res)
    
    return np.mean(residuals)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_occlusion_experiment():
    print("="*60)
    print("ROUTE C: OCCLUSION + DISCRETE INFERENCE")
    print("="*60)
    
    # Load model
    print("\n[1/4] Loading trained model...")
    
    if not os.path.exists("outputs/routec_model.pt"):
        print("       Model not found. Training first...")
        from exp_mnist_routec_learnable_adc import main as train_main
        model, model_cfg, train_data, test_data = train_main()
    else:
        checkpoint = torch.load("outputs/routec_model.pt", map_location='cpu')
        model_cfg = checkpoint['config']
        model = RouteCModel(model_cfg)
        model.load_state_dict(checkpoint['model'])
        
        # Load test data
        _, _, _, test_data = load_data(model_cfg)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    test_x, test_y = test_data
    
    print(f"       Model loaded, device={device}")
    
    # Estimate σ²
    print("\n[2/4] Estimating σ² from reconstruction...")
    sigma_sq = estimate_sigma_squared(model, test_x, device, n_samples=500)
    print(f"       σ² = {sigma_sq:.6f}")
    
    # Setup inference
    inf_cfg = InferenceConfig()
    engine = DiscreteInferenceEngine(model, device, sigma_sq, inf_cfg)
    
    # Run occlusion experiment
    print(f"\n[3/4] Running occlusion experiment ({inf_cfg.n_eval} samples)...")
    
    rng = np.random.default_rng(inf_cfg.seed)
    eval_idx = rng.choice(len(test_x), inf_cfg.n_eval, replace=False)
    
    results = {
        'baseline_correct': 0,
        'inferred_correct': 0,
        'clean_correct': 0,
        'bit_recovery': [],
        'energy_reduction': [],
    }
    
    t0 = time.time()
    iterator = tqdm(eval_idx, desc="       Inference") if HAS_TQDM else eval_idx
    
    for idx in iterator:
        # Get clean image
        x_clean = test_x[idx].numpy()[0]  # (28, 28)
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
        
        # Encode occluded image
        with torch.no_grad():
            x_occ_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_occ = model.encode(x_occ_t)[0].cpu().numpy()
            
            # Baseline prediction (no inference)
            logits_occ = model.classifier(torch.from_numpy(z_occ[None]).to(device))
            pred_baseline = logits_occ.argmax(dim=1).item()
        
        results['baseline_correct'] += int(pred_baseline == label)
        
        # Create bit mask
        bit_mask = create_bit_mask(pixel_mask, model_cfg.n_bits, model_cfg.latent_size)
        
        # Run discrete inference
        z_inferred, stats = engine.run_inference(z_occ, x_occ, pixel_mask.astype(np.float32), bit_mask)
        
        # Prediction after inference
        with torch.no_grad():
            logits_inf = model.classifier(torch.from_numpy(z_inferred[None]).to(device))
            pred_inferred = logits_inf.argmax(dim=1).item()
        
        results['inferred_correct'] += int(pred_inferred == label)
        
        # Bit recovery (compare with clean z on masked positions)
        z_clean_np = z_clean[0].cpu().numpy()
        if bit_mask.any():
            recovery = (z_inferred[bit_mask] == z_clean_np[bit_mask]).mean()
            results['bit_recovery'].append(recovery)
        
        # Energy reduction
        if len(stats['energies']) > 1:
            reduction = stats['energies'][0] - stats['energies'][-1]
            results['energy_reduction'].append(reduction)
    
    elapsed = time.time() - t0
    
    # Report
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    n = inf_cfg.n_eval
    clean_acc = results['clean_correct'] / n
    baseline_acc = results['baseline_correct'] / n
    inferred_acc = results['inferred_correct'] / n
    
    print(f"\nClassification Accuracy:")
    print(f"  Clean (no occlusion):  {clean_acc:.1%}")
    print(f"  Occluded (baseline):   {baseline_acc:.1%}")
    print(f"  Occluded (inferred):   {inferred_acc:.1%}")
    print(f"  Δ (inference benefit): {inferred_acc - baseline_acc:+.1%}")
    
    bit_rec = np.mean(results['bit_recovery']) if results['bit_recovery'] else 0
    print(f"\nBit Recovery (on masked positions): {bit_rec:.1%}")
    
    avg_reduction = np.mean(results['energy_reduction']) if results['energy_reduction'] else 0
    print(f"\nAverage Energy Reduction: {avg_reduction:.2f}")
    
    print(f"\nRuntime: {elapsed:.1f}s ({elapsed/n:.2f}s per sample)")
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"\n| Metric                  | Value    |")
    print(f"|-------------------------|----------|")
    print(f"| Clean accuracy          | {clean_acc:>7.1%} |")
    print(f"| Occluded baseline       | {baseline_acc:>7.1%} |")
    print(f"| Occluded after inference| {inferred_acc:>7.1%} |")
    print(f"| Inference benefit       | {inferred_acc - baseline_acc:>+6.1%} |")
    print(f"| Bit recovery            | {bit_rec:>7.1%} |")
    print(f"| Energy reduction        | {avg_reduction:>7.2f} |")
    
    return results


def main():
    results = run_occlusion_experiment()
    return results


if __name__ == "__main__":
    main()
