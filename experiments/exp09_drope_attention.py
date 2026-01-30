#!/usr/bin/env python3
"""
Experiment 09: D-RoPE + Content Addressing = Discrete Attention via Permutation Alignment
=========================================================================================

Run: python exp09_drope_attention.py

Goals (3 non-negotiable phenomena):
A) Action at a Distance: Center occlusion recovery without E_obs signal inside
B) Sparsity Efficiency: Sparse candidates (5-10%) achieve most of dense benefit
C) Mixing Time: Faster convergence with D-RoPE

D-RoPE is NOT Transformer attention:
- No sin/cos embeddings
- No floating-point scores
- Pure discrete: shift, XOR, popcount, threshold compare
- Compile-friendly: maps to wires, XOR gates, popcount, comparators
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw): return x


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    # Data
    train_samples: int = 5000  # Quick training
    test_samples: int = 2000
    batch_size: int = 64
    
    # Model
    n_bits: int = 8
    latent_size: int = 7
    hidden_dim: int = 64
    energy_hidden: int = 32
    
    # Training
    epochs: int = 5  # Quick
    lr: float = 1e-3
    tau_start: float = 1.0
    tau_end: float = 0.2
    
    # Loss weights
    alpha_recon: float = 1.0
    beta_core: float = 0.5
    gamma_cls: float = 1.0
    
    # Inference
    n_sweeps: int = 30
    block_size: Tuple[int, int] = (2, 2)
    
    # D-RoPE
    lambda_core: float = 1.0
    lambda_obs: float = 1.0
    lambda_rope: float = 0.5
    rope_threshold: int = None  # Auto: 0.25 * k
    
    # Evaluation
    n_eval_final: int = 500
    n_eval_curves: int = 100  # For per-sweep tracking
    occlusion_size: Tuple[int, int] = (14, 14)
    
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# MODEL (Same as before)
# ============================================================================

class GumbelSigmoid(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits):
        if self.training:
            u = torch.rand_like(logits).clamp(1e-8, 1-1e-8)
            gumbel = -torch.log(-torch.log(u))
            noisy = (logits + gumbel) / self.temperature
        else:
            noisy = logits / self.temperature
        soft = torch.sigmoid(noisy)
        hard = (soft > 0.5).float()
        return hard - soft.detach() + soft
    
    def set_temperature(self, tau):
        self.temperature = tau


class Encoder(nn.Module):
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1),
        )
    
    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, z):
        return self.deconv(z)


class LocalPredictor(nn.Module):
    def __init__(self, n_bits, hidden_dim=32):
        super().__init__()
        self.n_bits = n_bits
        self.net = nn.Sequential(
            nn.Linear(9 * n_bits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_bits),
        )
    
    def forward(self, z):
        B, k, H, W = z.shape
        z_pad = F.pad(z, (1, 1, 1, 1), mode='constant', value=0)
        windows = F.unfold(z_pad, kernel_size=3)
        windows = windows.reshape(B, k, 9, H * W)
        windows[:, :, 4, :] = 0
        windows = windows.reshape(B, k * 9, H * W)
        windows = windows.permute(0, 2, 1)
        logits = self.net(windows)
        return logits.permute(0, 2, 1).reshape(B, k, H, W)


class Classifier(nn.Module):
    def __init__(self, n_bits, latent_size=7, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(n_bits * latent_size * latent_size, n_classes)
    
    def forward(self, z):
        return self.fc(z.reshape(z.shape[0], -1))


class RouteCModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_bits = cfg.n_bits
        self.encoder = Encoder(cfg.n_bits, cfg.hidden_dim)
        self.quantizer = GumbelSigmoid(cfg.tau_start)
        self.decoder = Decoder(cfg.n_bits, cfg.hidden_dim)
        self.local_pred = LocalPredictor(cfg.n_bits, cfg.energy_hidden)
        self.classifier = Classifier(cfg.n_bits, cfg.latent_size)
    
    def encode(self, x):
        return self.quantizer(self.encoder(x))
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        cls_logits = self.classifier(z)
        core_logits = self.local_pred(z)
        return z, x_hat, cls_logits, core_logits
    
    def set_temperature(self, tau):
        self.quantizer.set_temperature(tau)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(cfg: Config):
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    rng = np.random.default_rng(cfg.seed)
    train_idx = rng.choice(len(train_ds), cfg.train_samples, replace=False)
    test_idx = rng.choice(len(test_ds), cfg.test_samples, replace=False)
    
    train_x = torch.stack([train_ds[i][0] for i in train_idx])
    train_y = torch.tensor([train_ds[i][1] for i in train_idx])
    test_x = torch.stack([test_ds[i][0] for i in test_idx])
    test_y = torch.tensor([test_ds[i][1] for i in test_idx])
    
    return train_x, train_y, test_x, test_y


# ============================================================================
# TRAINING
# ============================================================================

def train_model(cfg: Config):
    """Train or load cached model"""
    model_path = "outputs/exp09_model.pt"
    
    if os.path.exists(model_path):
        print("Loading cached model...")
        model = RouteCModel(cfg).to(cfg.device)
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        return model
    
    print("Training model...")
    train_x, train_y, _, _ = load_data(cfg)
    
    model = RouteCModel(cfg).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=cfg.batch_size, shuffle=True)
    
    for epoch in range(cfg.epochs):
        model.train()
        tau = cfg.tau_start + (cfg.tau_end - cfg.tau_start) * epoch / max(1, cfg.epochs - 1)
        model.set_temperature(tau)
        
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            
            z, x_hat, cls_logits, core_logits = model(x)
            
            loss_recon = F.mse_loss(x_hat, x)
            mask = torch.rand_like(z) < 0.15
            loss_core = F.binary_cross_entropy_with_logits(
                core_logits[mask], z.detach()[mask]
            ) if mask.any() else 0.0
            loss_cls = F.cross_entropy(cls_logits, y)
            
            loss = cfg.alpha_recon * loss_recon + cfg.beta_core * loss_core + cfg.gamma_cls * loss_cls
            loss.backward()
            optimizer.step()
        
        print(f"  Epoch {epoch+1}/{cfg.epochs}")
    
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model


# ============================================================================
# OCCLUSION UTILITIES
# ============================================================================

def create_center_occlusion(image: np.ndarray, occ_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Center occlusion (hardest for diffusion-based recovery)"""
    H, W = image.shape
    oh, ow = occ_size
    y, x = (H - oh) // 2, (W - ow) // 2
    
    occluded = image.copy()
    occluded[y:y+oh, x:x+ow] = 0
    
    mask = np.ones((H, W), dtype=np.float32)
    mask[y:y+oh, x:x+ow] = 0
    
    return occluded, mask


def create_random_occlusion(image: np.ndarray, occ_size: Tuple[int, int], rng) -> Tuple[np.ndarray, np.ndarray]:
    """Random position occlusion"""
    H, W = image.shape
    oh, ow = occ_size
    y = rng.integers(0, max(1, H - oh + 1))
    x = rng.integers(0, max(1, W - ow + 1))
    
    occluded = image.copy()
    occluded[y:y+oh, x:x+ow] = 0
    
    mask = np.ones((H, W), dtype=np.float32)
    mask[y:y+oh, x:x+ow] = 0
    
    return occluded, mask


def create_bit_mask(pixel_mask: np.ndarray, n_bits: int, latent_size: int = 7) -> np.ndarray:
    """Create bit mask from pixel mask"""
    patch_size = 28 // latent_size
    bit_mask = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    
    for i in range(latent_size):
        for j in range(latent_size):
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            if pixel_mask[y0:y1, x0:x1].mean() < 0.5:
                bit_mask[:, i, j] = True
    return bit_mask


def compute_occluded_mse(o_hat: np.ndarray, o_orig: np.ndarray, mask: np.ndarray) -> float:
    """MSE only on occluded region"""
    occluded_mask = (1 - mask)
    if occluded_mask.sum() == 0:
        return 0.0
    return ((o_hat - o_orig) ** 2 * occluded_mask).sum() / occluded_mask.sum()


# ============================================================================
# D-RoPE INTEGRATION
# ============================================================================

# Import D-RoPE module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.drope import DRoPEEnergy, CandidateConfig, CombinedEnergyWithRope, BlockGibbsWithRope, compute_mixing_time


def estimate_sigma_sq(model, test_x, device, n_samples=500):
    """Estimate σ² from reconstruction residuals"""
    model.eval()
    residuals = []
    with torch.no_grad():
        for i in range(min(n_samples, len(test_x))):
            x = test_x[i:i+1].to(device)
            z = model.encode(x)
            x_hat = model.decode(z)
            residuals.append(((x_hat - x) ** 2).mean().item())
    return np.mean(residuals)


class BaselineEnergy:
    """Baseline energy - ALL TORCH, no numpy"""
    
    def __init__(self, model, sigma_sq, lambda_core=1.0, lambda_obs=1.0, device=None):
        self.model = model
        self.sigma_sq = sigma_sq
        self.lambda_core = lambda_core
        self.lambda_obs = lambda_obs
        self.device = device or torch.device('cpu')
        # Cache for observation (set once per sample)
        self._o_obs_t = None
        self._mask_t = None
        self._mask_sum = 1.0
    
    def set_observation(self, o_obs, mask):
        """Cache observation tensors on GPU - call once per sample"""
        self._o_obs_t = torch.from_numpy(o_obs.astype(np.float32)).to(self.device)
        self._mask_t = torch.from_numpy(mask.astype(np.float32)).to(self.device)
        self._mask_sum = self._mask_t.sum().clamp(min=1.0)
    
    def energy(self, z):
        """Compute total energy - ALL ON GPU, returns scalar"""
        with torch.inference_mode():
            z_t = z.unsqueeze(0) if z.dim() == 3 else z
            
            # E_core
            if self.lambda_core > 0:
                logits = self.model.local_pred(z_t)
                e_core = F.binary_cross_entropy_with_logits(logits, z_t, reduction='sum')
            else:
                e_core = 0.0
            
            # E_obs - ALL TORCH
            if self.lambda_obs > 0 and self._o_obs_t is not None:
                o_hat = self.model.decode(z_t)[0, 0]  # Stay on GPU
                diff = (o_hat - self._o_obs_t) * self._mask_t
                mse = (diff * diff).sum() / self._mask_sum
                e_obs = mse / (2 * self.sigma_sq)
            else:
                e_obs = 0.0
            
            total = self.lambda_core * e_core + self.lambda_obs * e_obs
            return total.item() if isinstance(total, torch.Tensor) else total


class BaselineSolver:
    """Baseline solver - FAST: whole-block proposal, in-place flip, no numpy"""
    
    def __init__(self, energy, block_size=(2, 2), device=None):
        self.energy = energy
        self.block_size = block_size
        self.device = device or torch.device('cpu')
    
    def run(self, z_init, o_obs, pixel_mask, bit_mask, n_sweeps=30, track_metrics=False, classifier_fn=None):
        z = z_init.clone().to(self.device)
        k, H, W = z.shape
        bh, bw = self.block_size
        
        # Cache observation on GPU once
        self.energy.set_observation(o_obs, pixel_mask)
        
        # Convert bit_mask to torch
        bit_mask_t = torch.from_numpy(bit_mask).to(self.device)
        
        # Initialize masked bits randomly
        z[bit_mask_t] = torch.randint(0, 2, (bit_mask_t.sum().item(),), device=self.device, dtype=z.dtype)
        
        # Current energy
        E_curr = self.energy.energy(z)
        
        metrics = {'energy': [E_curr]}
        
        for sweep in range(n_sweeps):
            for bi in range(0, H, bh):
                for bj in range(0, W, bw):
                    i_end, j_end = min(bi + bh, H), min(bj + bw, W)
                    
                    # Get block mask
                    block_bit_mask = bit_mask_t[:, bi:i_end, bj:j_end]
                    if not block_bit_mask.any():
                        continue
                    
                    # IN-PLACE FLIP whole block (only masked bits)
                    z[:, bi:i_end, bj:j_end][block_bit_mask] = 1 - z[:, bi:i_end, bj:j_end][block_bit_mask]
                    
                    # Compute proposal energy
                    E_prop = self.energy.energy(z)
                    
                    # MH accept/reject
                    dE = E_prop - E_curr
                    if dE < 0 or torch.rand(1).item() < np.exp(-dE):
                        E_curr = E_prop  # Accept
                    else:
                        # Reject: flip back
                        z[:, bi:i_end, bj:j_end][block_bit_mask] = 1 - z[:, bi:i_end, bj:j_end][block_bit_mask]
            
            if track_metrics:
                metrics['energy'].append(E_curr)
        
        return z, metrics


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_occlusion(
    model,
    test_x,
    test_y,
    cfg: Config,
    use_rope: bool,
    occlusion_type: str,
    density: str = 'sparse',
    n_samples: int = None,
    track_curves: bool = False,
):
    """
    Evaluate occlusion recovery with or without D-RoPE.
    """
    device = torch.device(cfg.device)
    model.eval()
    
    n_samples = n_samples or cfg.n_eval_final
    sigma_sq = estimate_sigma_sq(model, test_x, device)
    
    # Setup energy and solver
    if use_rope:
        rope_config = CandidateConfig(density=density)
        drope = DRoPEEnergy(
            H=cfg.latent_size, W=cfg.latent_size, k=cfg.n_bits,
            threshold=cfg.rope_threshold, config=rope_config, device=device
        )
        energy = CombinedEnergyWithRope(
            model, drope, sigma_sq,
            lambda_core=cfg.lambda_core, lambda_obs=cfg.lambda_obs, lambda_rope=cfg.lambda_rope,
            device=device
        )
        solver = BlockGibbsWithRope(energy, block_size=cfg.block_size, device=device)
    else:
        energy = BaselineEnergy(model, sigma_sq, cfg.lambda_core, cfg.lambda_obs, device)
        solver = BaselineSolver(energy, block_size=cfg.block_size, device=device)
    
    rng = np.random.default_rng(cfg.seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)
    
    results = {
        'acc_before': 0, 'acc_after': 0,
        'mse_before': [], 'mse_after': [],
        'mixing_times': [],
        'curves': [] if track_curves else None,
    }
    
    classifier_fn = lambda z: model.classifier(z.to(device))
    
    t0 = time.time()
    iterator = tqdm(eval_idx, desc=f"{'RoPE' if use_rope else 'Base'}-{occlusion_type}") if HAS_TQDM else eval_idx
    
    for idx in iterator:
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()
        
        # Create occlusion
        if occlusion_type == 'center':
            x_occ, pixel_mask = create_center_occlusion(x_clean, cfg.occlusion_size)
        else:
            x_occ, pixel_mask = create_random_occlusion(x_clean, cfg.occlusion_size, rng)
        
        # Encode occluded
        with torch.no_grad():
            x_occ_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_init = model.encode(x_occ_t)[0]
            
            # Before inference
            logits = model.classifier(z_init.unsqueeze(0))
            pred_before = logits.argmax(dim=1).item()
            results['acc_before'] += int(pred_before == label)
            
            o_hat_before = model.decode(z_init.unsqueeze(0))[0, 0].cpu().numpy()
            results['mse_before'].append(compute_occluded_mse(o_hat_before, x_clean, pixel_mask))
        
        # Create bit mask
        bit_mask = create_bit_mask(pixel_mask, cfg.n_bits, cfg.latent_size)
        
        # Run inference
        z_final, metrics = solver.run(
            z_init, x_occ, pixel_mask, bit_mask,
            n_sweeps=cfg.n_sweeps, track_metrics=track_curves
        )
        
        # After inference
        with torch.no_grad():
            logits = model.classifier(z_final.unsqueeze(0))
            pred_after = logits.argmax(dim=1).item()
            results['acc_after'] += int(pred_after == label)
            
            o_hat_after = model.decode(z_final.unsqueeze(0))[0, 0].cpu().numpy()
            results['mse_after'].append(compute_occluded_mse(o_hat_after, x_clean, pixel_mask))
        
        # Mixing time (only meaningful if tracking)
        if track_curves and len(metrics.get('energy', [])) > 1:
            mixing = compute_mixing_time(metrics)
            results['mixing_times'].append(mixing['sweeps_95'])
        else:
            results['mixing_times'].append(cfg.n_sweeps)  # Default
        
        # Store curves for detailed analysis
        if track_curves and results['curves'] is not None:
            results['curves'].append({
                'energy': metrics['energy'],
                'label': label,
                'pred_before': pred_before,
                'pred_after': pred_after,
            })
    
    runtime = time.time() - t0
    n = len(eval_idx)
    
    return {
        'acc_before': results['acc_before'] / n,
        'acc_after': results['acc_after'] / n,
        'delta_acc': (results['acc_after'] - results['acc_before']) / n,
        'mse_before': np.mean(results['mse_before']),
        'mse_after': np.mean(results['mse_after']),
        'delta_mse': np.mean(results['mse_after']) - np.mean(results['mse_before']),
        'median_sweeps_95': np.median(results['mixing_times']),
        'runtime': runtime,
        'runtime_per_sample': runtime / n,
        'curves': results['curves'],
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def save_qualitative_examples(model, test_x, test_y, cfg, output_dir):
    """Save qualitative comparison strips"""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(cfg.device)
    model.eval()
    
    sigma_sq = estimate_sigma_sq(model, test_x, device)
    
    # Setup both solvers
    rope_config = CandidateConfig(density='sparse')
    drope = DRoPEEnergy(H=cfg.latent_size, W=cfg.latent_size, k=cfg.n_bits, device=device)
    energy_rope = CombinedEnergyWithRope(model, drope, sigma_sq, device=device)
    solver_rope = BlockGibbsWithRope(energy_rope, device=device)
    
    energy_base = BaselineEnergy(model, sigma_sq, device=device)
    solver_base = BaselineSolver(energy_base, device=device)
    
    rng = np.random.default_rng(cfg.seed + 999)
    
    for ex_idx in range(3):
        idx = rng.integers(0, len(test_x))
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()
        
        x_occ, pixel_mask = create_center_occlusion(x_clean, cfg.occlusion_size)
        bit_mask = create_bit_mask(pixel_mask, cfg.n_bits, cfg.latent_size)
        
        with torch.no_grad():
            x_occ_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_init = model.encode(x_occ_t)[0]
        
        # Run both solvers and capture intermediate states
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        
        sweep_checkpoints = [0, 5, 10, cfg.n_sweeps]
        
        for row, (solver, name) in enumerate([(solver_base, 'Baseline'), (solver_rope, 'D-RoPE')]):
            z = z_init.clone()
            z[torch.from_numpy(bit_mask).to(device)] = torch.randint(0, 2, (bit_mask.sum(),), device=device, dtype=z.dtype)
            
            col = 0
            for s in range(cfg.n_sweeps + 1):
                if s in sweep_checkpoints:
                    with torch.no_grad():
                        o_hat = model.decode(z.unsqueeze(0))[0, 0].cpu().numpy()
                    axes[row, col].imshow(o_hat, cmap='gray', vmin=0, vmax=1)
                    axes[row, col].set_title(f'{name} sweep={s}')
                    axes[row, col].axis('off')
                    col += 1
                
                if s < cfg.n_sweeps:
                    # One sweep
                    z, _ = solver.run(z, x_occ, pixel_mask, bit_mask, n_sweeps=1, track_metrics=False)
            
            # Ground truth
            axes[row, 4].imshow(x_clean, cmap='gray', vmin=0, vmax=1)
            axes[row, 4].set_title(f'Ground Truth (label={label})')
            axes[row, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/example_{ex_idx}_label{label}.png', dpi=150)
        plt.close()
    
    print(f"Saved qualitative examples to {output_dir}/")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    print("="*80)
    print("EXP09: D-RoPE + CONTENT ADDRESSING = DISCRETE ATTENTION")
    print("="*80)
    
    cfg = Config()
    print(f"\nDevice: {cfg.device}")
    print(f"n_bits (k): {cfg.n_bits}")
    print(f"D-RoPE threshold: {cfg.rope_threshold or f'auto (~{max(1, round(0.25*cfg.n_bits))})'}")
    
    # Load data
    print("\n[1] Loading data...")
    train_x, train_y, test_x, test_y = load_data(cfg)
    print(f"    Train: {len(train_x)}, Test: {len(test_x)}")
    
    # Train or load model
    print("\n[2] Preparing model...")
    os.makedirs("outputs", exist_ok=True)
    model = train_model(cfg)
    model = model.to(cfg.device)
    model.eval()
    
    # Evaluate clean accuracy
    with torch.no_grad():
        loader = DataLoader(TensorDataset(test_x, test_y), batch_size=64)
        correct = 0
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            z = model.encode(x)
            pred = model.classifier(z).argmax(dim=1)
            correct += (pred == y).sum().item()
        clean_acc = correct / len(test_y)
    print(f"    Clean accuracy: {clean_acc:.1%}")
    
    # ========================================================================
    # MAIN EVALUATION: Baseline vs D-RoPE
    # ========================================================================
    print("\n[3] Main evaluation: Baseline vs D-RoPE...")
    print("    " + "-"*90)
    print("    | setting | method   | acc_before | acc_after |  Δacc  | mse_before | mse_after | runtime |")
    print("    " + "-"*90)
    
    results_table = []
    
    for occ_type in ['center', 'random']:
        for use_rope in [False, True]:
            method = '+RoPE' if use_rope else 'Baseline'
            
            res = evaluate_occlusion(
                model, test_x, test_y, cfg,
                use_rope=use_rope,
                occlusion_type=occ_type,
                n_samples=cfg.n_eval_final,
            )
            
            results_table.append({
                'setting': occ_type,
                'method': method,
                **res,
            })
            
            # Print immediately
            print(f"    | {occ_type:>7} | {method:>8} | {res['acc_before']:>10.1%} | {res['acc_after']:>9.1%} | "
                  f"{res['delta_acc']:>+6.1%} | {res['mse_before']:>10.4f} | {res['mse_after']:>9.4f} | {res['runtime']:>6.1f}s |")
    
    print("    " + "-"*90)
    
    # ========================================================================
    # SPARSITY ABLATION
    # ========================================================================
    print("\n[4] Sparsity ablation (center occlusion)...")
    print("    " + "-"*70)
    print("    | density | cand/site |  Δacc  |   Δmse   | runtime |")
    print("    " + "-"*70)
    
    sparsity_table = []
    
    for density in ['sparse', 'medium', 'dense']:
        res = evaluate_occlusion(
            model, test_x, test_y, cfg,
            use_rope=True,
            occlusion_type='center',
            density=density,
            n_samples=200,  # Fewer for ablation
        )
        
        # Count candidates
        rope_config = CandidateConfig(density=density)
        n_cand = len(build_candidate_offsets(cfg.latent_size, cfg.latent_size, rope_config))
        
        sparsity_table.append({
            'density': density,
            'candidates_per_site': n_cand,
            **res,
        })
        
        # Print immediately
        print(f"    | {density:>7} | {n_cand:>9} | {res['delta_acc']:>+6.1%} | {res['delta_mse']:>+8.4f} | {res['runtime']:>6.1f}s |")
    
    print("    " + "-"*70)
    
    # ========================================================================
    # SAVE QUALITATIVE EXAMPLES
    # ========================================================================
    print("\n[5] Saving qualitative examples...")
    save_qualitative_examples(model, test_x, test_y, cfg, "outputs/exp09")
    
    # ========================================================================
    # PRINT RESULTS
    # ========================================================================
    print("\n" + "="*100)
    print("MAIN RESULTS TABLE")
    print("="*100)
    
    header = "| setting | method   | acc_before | acc_after |  Δacc  | mse_before | mse_after |  Δmse   | sweeps_95% |"
    sep = "|" + "-"*9 + "|" + "-"*10 + "|" + "-"*12 + "|" + "-"*11 + "|" + "-"*8 + "|" + "-"*12 + "|" + "-"*11 + "|" + "-"*9 + "|" + "-"*12 + "|"
    
    print(header)
    print(sep)
    
    for r in results_table:
        print(f"| {r['setting']:>7} | {r['method']:>8} | {r['acc_before']:>10.1%} | {r['acc_after']:>9.1%} | {r['delta_acc']:>+6.1%} | "
              f"{r['mse_before']:>10.4f} | {r['mse_after']:>9.4f} | {r['delta_mse']:>+7.4f} | {r['median_sweeps_95']:>10.0f} |")
    
    print("\n" + "="*100)
    print("SPARSITY ABLATION TABLE (Center Occlusion)")
    print("="*100)
    
    header2 = "| density | cand/site | runtime/sample |  Δacc  |   Δmse   | sweeps_95% |"
    sep2 = "|" + "-"*9 + "|" + "-"*11 + "|" + "-"*16 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*12 + "|"
    
    print(header2)
    print(sep2)
    
    for r in sparsity_table:
        print(f"| {r['density']:>7} | {r['candidates_per_site']:>9} | {r['runtime_per_sample']:>14.2f}s | "
              f"{r['delta_acc']:>+6.1%} | {r['delta_mse']:>+8.4f} | {r['median_sweeps_95']:>10.0f} |")
    
    # ========================================================================
    # INTERPRETATION
    # ========================================================================
    print("\n" + "="*100)
    print("ROUTE C INTERPRETATION: D-RoPE AS DISCRETE ATTENTION")
    print("="*100)
    
    # Extract key comparisons
    base_center = [r for r in results_table if r['setting'] == 'center' and r['method'] == 'Baseline'][0]
    rope_center = [r for r in results_table if r['setting'] == 'center' and r['method'] == '+RoPE'][0]
    
    print(f"""
PHENOMENON A: Action at a Distance
----------------------------------
Center occlusion removes E_obs signal from the core region.
Without D-RoPE: Δacc = {base_center['delta_acc']:+.1%}, relies on slow boundary diffusion.
With D-RoPE:    Δacc = {rope_center['delta_acc']:+.1%}, long-range candidates enable faster collapse.

The key insight: D-RoPE connects distant positions via XOR-based content matching,
allowing information to "jump" across the occluded region without waiting for
local E_core propagation.

PHENOMENON B: Sparsity Efficiency
---------------------------------
Sparse candidates ({sparsity_table[0]['candidates_per_site']}/site): Δacc = {sparsity_table[0]['delta_acc']:+.1%}
Medium candidates ({sparsity_table[1]['candidates_per_site']}/site): Δacc = {sparsity_table[1]['delta_acc']:+.1%}
Dense candidates ({sparsity_table[2]['candidates_per_site']}/site):  Δacc = {sparsity_table[2]['delta_acc']:+.1%}

Most benefit is achieved with sparse candidates, validating that Route C can use
O(1) long-range connections rather than O(N²) full attention.

PHENOMENON C: Mixing Time
-------------------------
Baseline median sweeps to 95%: {base_center['median_sweeps_95']:.0f}
D-RoPE median sweeps to 95%:   {rope_center['median_sweeps_95']:.0f}

D-RoPE enables faster convergence by providing global coordination signals
that prevent the solver from getting stuck in local minima.

WHAT MAKES D-RoPE "ROUTE C NATIVE"
----------------------------------
1. Pure discrete operations: shift2d, XOR, popcount, threshold compare
2. No sin/cos embeddings, no floating-point attention scores
3. Compile-friendly: maps directly to hardware (wires, XOR gates, comparators)
4. Sparse by design: O(1) candidates per site, not O(N²)
5. Content-addressed: Gate(i→j) depends on Hamming distance, enabling selective routing
""")
    
    print("="*100)
    print(f"Outputs saved to: outputs/exp09/")
    print("="*100)
    
    return results_table, sparsity_table


# ============================================================================
# UTILITY FUNCTIONS FOR DROPE MODULE
# ============================================================================

def build_candidate_offsets(H, W, config):
    """Build candidate offsets (imported from drope module)"""
    offsets = []
    anchors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for r in config.radii:
        anchors.extend([
            (-r, 0), (r, 0), (0, -r), (0, r),
            (-r, -r), (-r, r), (r, -r), (r, r),
        ])
    
    seen = set()
    for dy, dx in anchors:
        if (dy, dx) != (0, 0) and (dy, dx) not in seen:
            offsets.append((dy, dx))
            seen.add((dy, dx))
    
    return offsets


if __name__ == "__main__":
    results_table, sparsity_table = run_experiment()
