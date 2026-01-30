#!/usr/bin/env python3
"""
Route C: Learnable ADC Training on MNIST
========================================

This experiment demonstrates:
G1. Learnable ADC improves downstream task vs non-learned tokenization
G2. Discrete core closure learns meaningful local structure in z
G3. Clean evaluation baseline before occlusion experiments

Training:
- ADC/DAC: Encoder→Gumbel-Sigmoid→Decoder
- Local energy predictor: masked bit prediction
- Classifier: linear on flattened z

Loss:
L = α * L_recon + β * L_core + γ * L_cls + λ * L_sparsity
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
from typing import Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw): return x

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    # Data
    train_samples: int = 10000
    test_samples: int = 2000
    batch_size: int = 64
    
    # Architecture
    n_bits: int = 8          # k: bits per position
    latent_size: int = 7     # H=W of latent grid
    hidden_dim: int = 64     # CNN hidden channels
    energy_hidden: int = 32  # Local predictor hidden
    
    # Training
    epochs: int = 8
    lr: float = 1e-3
    
    # Loss weights
    alpha_recon: float = 1.0
    beta_core: float = 0.5
    gamma_cls: float = 1.0
    lambda_sparsity: float = 1e-4
    
    # Temperature annealing
    tau_start: float = 1.0
    tau_end: float = 0.2
    
    # Core training
    mask_ratio: float = 0.15
    
    # Misc
    seed: int = 42
    device: str = "cpu"


# ============================================================================
# MODEL COMPONENTS
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
    def __init__(self, n_bits=8, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1),  # 28→14
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),  # 14→7
            nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1),
        )
    
    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, n_bits=8, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1),  # 7→14
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # 14→28
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, z):
        return self.deconv(z)


class LocalPredictor(nn.Module):
    """Predict masked bits from neighbors."""
    def __init__(self, n_bits=8, hidden_dim=32):
        super().__init__()
        self.n_bits = n_bits
        # Input: 3×3 window × k bits (masked at center)
        self.net = nn.Sequential(
            nn.Linear(9 * n_bits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_bits),
        )
    
    def forward(self, z):
        """
        Args:
            z: (B, k, H, W)
        Returns:
            logits: (B, k, H, W) predictions for each bit
        """
        B, k, H, W = z.shape
        
        # Pad and extract windows
        z_pad = F.pad(z, (1, 1, 1, 1), mode='constant', value=0)
        
        # Unfold to get all 3×3 windows
        windows = F.unfold(z_pad, kernel_size=3)  # (B, k*9, H*W)
        windows = windows.reshape(B, k, 9, H * W)
        
        # Mask center (position 4)
        windows[:, :, 4, :] = 0
        
        windows = windows.reshape(B, k * 9, H * W)
        windows = windows.permute(0, 2, 1)  # (B, H*W, k*9)
        
        logits = self.net(windows)  # (B, H*W, k)
        logits = logits.permute(0, 2, 1).reshape(B, k, H, W)
        
        return logits


class Classifier(nn.Module):
    def __init__(self, n_bits=8, latent_size=7, n_classes=10):
        super().__init__()
        input_dim = n_bits * latent_size * latent_size
        self.fc = nn.Linear(input_dim, n_classes)
    
    def forward(self, z):
        return self.fc(z.reshape(z.shape[0], -1))


class RouteCModel(nn.Module):
    """Complete Route C model."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = Encoder(cfg.n_bits, cfg.hidden_dim)
        self.quantizer = GumbelSigmoid(cfg.tau_start)
        self.decoder = Decoder(cfg.n_bits, cfg.hidden_dim)
        self.local_pred = LocalPredictor(cfg.n_bits, cfg.energy_hidden)
        self.classifier = Classifier(cfg.n_bits, cfg.latent_size)
        self.cfg = cfg
    
    def encode(self, x):
        logits = self.encoder(x)
        return self.quantizer(logits)
    
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
    """Load MNIST subset."""
    # Try torchvision first
    try:
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # Subsample
        rng = np.random.default_rng(cfg.seed)
        
        train_idx = rng.choice(len(train_dataset), cfg.train_samples, replace=False)
        test_idx = rng.choice(len(test_dataset), cfg.test_samples, replace=False)
        
        train_x = torch.stack([train_dataset[i][0] for i in train_idx])
        train_y = torch.tensor([train_dataset[i][1] for i in train_idx])
        
        test_x = torch.stack([test_dataset[i][0] for i in test_idx])
        test_y = torch.tensor([test_dataset[i][1] for i in test_idx])
        
    except ImportError:
        # Fallback to our own loader
        from mnist import load_mnist_numpy, subsample_dataset
        
        train_img, train_lbl, test_img, test_lbl = load_mnist_numpy('./data')
        train_img, train_lbl = subsample_dataset(train_img, train_lbl, cfg.train_samples, cfg.seed)
        test_img, test_lbl = subsample_dataset(test_img, test_lbl, cfg.test_samples, cfg.seed)
        
        train_x = torch.from_numpy(train_img[:, None, :, :].astype(np.float32) / 255.0)
        train_y = torch.from_numpy(train_lbl.astype(np.int64))
        test_x = torch.from_numpy(test_img[:, None, :, :].astype(np.float32) / 255.0)
        test_y = torch.from_numpy(test_lbl.astype(np.int64))
    
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=cfg.batch_size,
    )
    
    return train_loader, test_loader, (train_x, train_y), (test_x, test_y)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, cfg, epoch):
    model.train()
    device = cfg.device
    
    total_loss = 0
    total_recon = 0
    total_core = 0
    total_cls = 0
    n_batches = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        z, x_hat, cls_logits, core_logits = model(x)
        
        # Reconstruction loss
        loss_recon = F.mse_loss(x_hat, x)
        
        # Core loss: masked prediction
        mask = torch.rand_like(z) < cfg.mask_ratio
        z_target = z.detach()
        loss_core = F.binary_cross_entropy_with_logits(
            core_logits[mask], z_target[mask]
        ) if mask.any() else torch.tensor(0.0)
        
        # Classification loss
        loss_cls = F.cross_entropy(cls_logits, y)
        
        # Sparsity: encourage z to be near target density (e.g., 0.3)
        loss_sparse = (z.mean() - 0.3).pow(2)
        
        # Total loss
        loss = (cfg.alpha_recon * loss_recon +
                cfg.beta_core * loss_core +
                cfg.gamma_cls * loss_cls +
                cfg.lambda_sparsity * loss_sparse)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += loss_recon.item()
        total_core += loss_core.item() if isinstance(loss_core, torch.Tensor) else loss_core
        total_cls += loss_cls.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'core': total_core / n_batches,
        'cls': total_cls / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    device = cfg.device
    
    correct = 0
    total = 0
    total_mse = 0
    core_correct = 0
    core_total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        z, x_hat, cls_logits, core_logits = model(x)
        
        # Classification accuracy
        pred = cls_logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        # Reconstruction MSE
        total_mse += F.mse_loss(x_hat, x).item() * y.size(0)
        
        # Core prediction accuracy (on random mask)
        mask = torch.rand_like(z) < 0.2
        if mask.any():
            core_pred = (torch.sigmoid(core_logits) > 0.5).float()
            core_correct += (core_pred[mask] == z[mask]).sum().item()
            core_total += mask.sum().item()
    
    return {
        'accuracy': correct / total,
        'mse': total_mse / total,
        'core_acc': core_correct / max(1, core_total),
    }


def train(cfg: Config):
    print("="*60)
    print("ROUTE C: LEARNABLE ADC TRAINING")
    print("="*60)
    print(f"Config: {cfg.train_samples} train, {cfg.test_samples} test")
    print(f"        {cfg.n_bits} bits, {cfg.latent_size}×{cfg.latent_size} grid")
    print(f"        {cfg.epochs} epochs, lr={cfg.lr}")
    
    # Load data
    print("\n[1/3] Loading data...")
    train_loader, test_loader, train_data, test_data = load_data(cfg)
    print(f"       Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    print("\n[2/3] Creating model...")
    model = RouteCModel(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"       Parameters: {n_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    # Training
    print("\n[3/3] Training...")
    t0 = time.time()
    
    for epoch in range(cfg.epochs):
        # Anneal temperature
        tau = cfg.tau_start + (cfg.tau_end - cfg.tau_start) * epoch / (cfg.epochs - 1)
        model.set_temperature(tau)
        
        # Train
        train_stats = train_epoch(model, train_loader, optimizer, cfg, epoch)
        
        # Evaluate
        test_stats = evaluate(model, test_loader, cfg)
        
        print(f"  Epoch {epoch+1}/{cfg.epochs}: "
              f"loss={train_stats['loss']:.4f}, "
              f"τ={tau:.2f}, "
              f"test_acc={test_stats['accuracy']:.1%}, "
              f"core_acc={test_stats['core_acc']:.1%}")
    
    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed:.1f}s")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    final_stats = evaluate(model, test_loader, cfg)
    print(f"\nTest Accuracy: {final_stats['accuracy']:.1%}")
    print(f"Reconstruction MSE: {final_stats['mse']:.4f}")
    print(f"Core Closure Acc: {final_stats['core_acc']:.1%}")
    
    # Analyze z statistics
    model.eval()
    with torch.no_grad():
        sample_x = test_data[0][:100].to(cfg.device)
        z = model.encode(sample_x)
        z_mean = z.mean().item()
        z_std = z.std().item()
        print(f"\nLatent z statistics:")
        print(f"  Mean: {z_mean:.3f}")
        print(f"  Std:  {z_std:.3f}")
        print(f"  Density: {z_mean:.1%} (target: 30%)")
    
    return model, cfg, train_data, test_data


def main():
    cfg = Config()
    
    # Check for GPU
    if torch.cuda.is_available():
        cfg.device = "cuda"
        print("Using GPU")
    else:
        print("Using CPU")
    
    model, cfg, train_data, test_data = train(cfg)
    
    # Save model
    os.makedirs("outputs", exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'config': cfg,
    }, "outputs/routec_model.pt")
    print("\nModel saved to outputs/routec_model.pt")
    
    return model, cfg, train_data, test_data


if __name__ == "__main__":
    main()
