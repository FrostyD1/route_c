"""
Route C: Local Energy Model (Discrete Core)
============================================

E_core(z) = -Σ_i log pθ(z_i | neigh(z)_i)

This is the "discrete closed-core" component that learns local structure in z.
The predictor pθ can be distilled into logic rules.

Implementation:
- Small MLP predicts each bit from its spatial neighbors
- Trained via masked prediction (self-supervised on z)
- At inference time, provides energy gradients for Gibbs/MH
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class LocalBitPredictor(nn.Module):
    """
    Predict each bit z_{i,j}^b from neighboring bits.
    
    Input: z ∈ {0,1}^{k×H×W}
    For position (i,j) and bit b, predict P(z_{i,j}^b | neigh(z)_{i,j})
    
    Neighborhood: 3×3 spatial window (excluding center bit b)
    Features: all k bits at 8 neighbor positions + (k-1) other bits at center
             = 8*k + (k-1) = 9k - 1 features
    """
    
    def __init__(
        self,
        n_bits: int = 8,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.n_bits = n_bits
        
        # Context size: 3×3 spatial × k bits - 1 (exclude target bit)
        # = 9*k - 1
        self.context_size = 9 * n_bits - 1
        
        # Small MLP for each bit prediction
        # Shared across positions (translation invariant)
        self.predictor = nn.Sequential(
            nn.Linear(self.context_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_bits),  # Predict all k bits at once
        )
    
    def _extract_context(
        self,
        z: torch.Tensor,
        i: int,
        j: int,
        b: int,
    ) -> torch.Tensor:
        """
        Extract context features for predicting bit b at position (i,j).
        
        Args:
            z: (B, k, H, W)
            i, j: spatial position
            b: bit index to predict
        
        Returns:
            context: (B, context_size)
        """
        B, k, H, W = z.shape
        device = z.device
        
        # Pad z with zeros for boundary handling
        z_pad = F.pad(z, (1, 1, 1, 1), mode='constant', value=0)
        
        # Extract 3×3 window centered at (i+1, j+1) in padded tensor
        window = z_pad[:, :, i:i+3, j:j+3]  # (B, k, 3, 3)
        
        # Flatten window
        flat = window.reshape(B, -1)  # (B, 9*k)
        
        # Remove the target bit (center position, bit b)
        # Center is at index (1,1) in 3×3, which is position 4 in flattened 3×3
        # For bit b, it's at index 4*k + b in the flattened tensor
        target_idx = 4 * k + b
        
        # Create context by removing target
        context = torch.cat([flat[:, :target_idx], flat[:, target_idx+1:]], dim=1)
        
        return context  # (B, 9*k - 1)
    
    def forward_position(
        self,
        z: torch.Tensor,
        i: int,
        j: int,
    ) -> torch.Tensor:
        """
        Predict all k bits at position (i,j).
        
        Returns:
            logits: (B, k) logits for each bit
        """
        B, k, H, W = z.shape
        
        # For simplicity, extract context for bit 0 and predict all
        # (This is an approximation - ideally each bit has different context)
        context = self._extract_context(z, i, j, b=0)
        
        # Actually, let's be more careful: use full 3×3 window
        z_pad = F.pad(z, (1, 1, 1, 1), mode='constant', value=0)
        window = z_pad[:, :, i:i+3, j:j+3]  # (B, k, 3, 3)
        
        # Mask out center position entirely for prediction
        mask = torch.ones(3, 3, device=z.device)
        mask[1, 1] = 0
        window = window * mask[None, None, :, :]
        
        flat = window.reshape(B, -1)  # (B, 9*k)
        
        # Pad to expected size (add dummy features if needed)
        if flat.shape[1] < self.context_size:
            flat = F.pad(flat, (0, self.context_size - flat.shape[1]))
        elif flat.shape[1] > self.context_size:
            flat = flat[:, :self.context_size]
        
        logits = self.predictor(flat)
        return logits
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict all bits at all positions (parallel).
        
        Args:
            z: (B, k, H, W)
        
        Returns:
            logits: (B, k, H, W) prediction logits
        """
        B, k, H, W = z.shape
        
        # Use conv-based implementation for efficiency
        # This is equivalent to the MLP but parallelized
        
        # Pad input
        z_pad = F.pad(z, (1, 1, 1, 1), mode='constant', value=0)
        
        # Extract all 3×3 windows using unfold
        # (B, k, H+2, W+2) → (B, k*9, H, W)
        windows = F.unfold(z_pad, kernel_size=3, padding=0)  # (B, k*9, H*W)
        windows = windows.reshape(B, k * 9, H, W)
        
        # Mask out center (position 4 in 3×3 for each channel)
        # Create mask for 9 positions, replicated for k channels
        mask = torch.ones(k, 9, device=z.device)
        mask[:, 4] = 0  # Center position
        mask = mask.reshape(k * 9, 1, 1)
        
        windows = windows * mask
        
        # Reshape for MLP: (B, H, W, k*9) → predict (B, H, W, k)
        windows = windows.permute(0, 2, 3, 1)  # (B, H, W, k*9)
        
        # Truncate or pad to context_size
        if windows.shape[-1] < self.context_size:
            windows = F.pad(windows, (0, self.context_size - windows.shape[-1]))
        else:
            windows = windows[..., :self.context_size]
        
        # Apply MLP
        logits = self.predictor(windows)  # (B, H, W, k)
        logits = logits.permute(0, 3, 1, 2)  # (B, k, H, W)
        
        return logits
    
    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute E_core(z) = -Σ log p(z_i | neigh).
        
        Args:
            z: (B, k, H, W) binary {0,1}
        
        Returns:
            energy: (B,) total energy per sample
        """
        logits = self.forward(z)  # (B, k, H, W)
        
        # Cross-entropy with z as target
        # BCE: -[z*log(σ(l)) + (1-z)*log(1-σ(l))]
        loss = F.binary_cross_entropy_with_logits(
            logits, z, reduction='none'
        )  # (B, k, H, W)
        
        # Sum over all bits
        energy = loss.sum(dim=(1, 2, 3))  # (B,)
        
        return energy
    
    def masked_prediction_loss(
        self,
        z: torch.Tensor,
        mask_ratio: float = 0.15,
    ) -> Tuple[torch.Tensor, float]:
        """
        Masked prediction training loss.
        
        Randomly mask some bits and predict them.
        
        Returns:
            loss: scalar
            accuracy: masked prediction accuracy
        """
        B, k, H, W = z.shape
        device = z.device
        
        # Random mask
        mask = torch.rand(B, k, H, W, device=device) < mask_ratio
        
        # Create input with masked bits set to 0.5 (ambiguous)
        z_masked = z.clone()
        z_masked[mask] = 0.5
        
        # Predict
        logits = self.forward(z_masked)
        
        # Loss only on masked positions
        loss = F.binary_cross_entropy_with_logits(
            logits[mask], z[mask]
        )
        
        # Accuracy
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds[mask] == z[mask]).float().mean().item()
        
        return loss, acc


class LocalEnergyModel:
    """
    Numpy interface for local energy computation during discrete inference.
    """
    
    def __init__(self, predictor: LocalBitPredictor, device: str = 'cpu'):
        self.predictor = predictor
        self.device = device
        self.predictor.eval()
        self.predictor.to(device)
    
    def energy_np(self, z: np.ndarray) -> float:
        """
        Compute energy for numpy z.
        
        Args:
            z: (k, H, W) binary array
        
        Returns:
            energy: scalar
        """
        with torch.no_grad():
            z_t = torch.from_numpy(z.astype(np.float32))[None].to(self.device)
            e = self.predictor.energy(z_t)
            return e.item()
    
    def delta_energy_np(
        self,
        z: np.ndarray,
        i: int,
        j: int,
        b: int,
    ) -> float:
        """
        Compute energy change from flipping bit b at position (i,j).
        
        ΔE = E(z_flipped) - E(z_current)
        
        Negative ΔE means flipping reduces energy (good).
        """
        z_flip = z.copy()
        z_flip[b, i, j] = 1 - z_flip[b, i, j]
        
        e_old = self.energy_np(z)
        e_new = self.energy_np(z_flip)
        
        return e_new - e_old
    
    def local_probabilities_np(
        self,
        z: np.ndarray,
        i: int,
        j: int,
    ) -> np.ndarray:
        """
        Get predicted probabilities for all bits at position (i,j).
        
        Returns:
            probs: (k,) array of P(z_b=1 | neighbors)
        """
        with torch.no_grad():
            z_t = torch.from_numpy(z.astype(np.float32))[None].to(self.device)
            logits = self.predictor.forward_position(z_t, i, j)  # (1, k)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return probs


# ============================================================================
# SIMPLE CLASSIFIER HEAD
# ============================================================================

class SimpleClassifier(nn.Module):
    """
    Simple linear classifier on flattened z.
    
    z ∈ {0,1}^{k×H×W} → flatten → linear → 10 classes
    
    No task-specific features, just learn from z directly.
    """
    
    def __init__(
        self,
        n_bits: int = 8,
        latent_size: int = 7,
        n_classes: int = 10,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        
        input_dim = n_bits * latent_size * latent_size
        
        if hidden_dim is None:
            # Linear classifier
            self.classifier = nn.Linear(input_dim, n_classes)
        else:
            # Small MLP
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_classes),
            )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, k, H, W)
        Returns:
            logits: (B, n_classes)
        """
        B = z.shape[0]
        flat = z.reshape(B, -1)
        return self.classifier(flat)
    
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """Return predicted class."""
        return self.forward(z).argmax(dim=1)


# ============================================================================
# COMBINED ROUTE C MODEL
# ============================================================================

class RouteCModel(nn.Module):
    """
    Full Route C model combining:
    - ADC/DAC (learnable quantization)
    - Local energy predictor (discrete core)
    - Optional classifier head
    """
    
    def __init__(
        self,
        n_bits: int = 8,
        latent_size: int = 7,
        hidden_dim: int = 64,
        energy_hidden: int = 32,
        n_classes: int = 10,
        quantizer_type: str = "gumbel",
    ):
        super().__init__()
        
        self.n_bits = n_bits
        self.latent_size = latent_size
        
        # Import here to avoid circular dependency
        from .quantizer import LearnableADC_DAC
        
        # ADC/DAC
        self.adc_dac = LearnableADC_DAC(
            n_bits=n_bits,
            latent_size=latent_size,
            hidden_dim=hidden_dim,
            quantizer_type=quantizer_type,
        )
        
        # Local energy predictor
        self.energy_predictor = LocalBitPredictor(
            n_bits=n_bits,
            hidden_dim=energy_hidden,
        )
        
        # Classifier
        self.classifier = SimpleClassifier(
            n_bits=n_bits,
            latent_size=latent_size,
            n_classes=n_classes,
        )
    
    def forward(
        self,
        o: torch.Tensor,
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Full forward pass.
        
        Args:
            o: (B, 1, 28, 28) input images
            return_all: if True, return (z, o_hat, logits)
        
        Returns:
            logits: (B, n_classes) classification logits
            or (z, o_hat, logits) if return_all
        """
        z, o_hat = self.adc_dac(o)
        logits = self.classifier(z)
        
        if return_all:
            return z, o_hat, logits
        return logits
    
    def encode(self, o: torch.Tensor) -> torch.Tensor:
        return self.adc_dac.encode(o)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.adc_dac.decode(z)
    
    def set_temperature(self, tau: float):
        self.adc_dac.set_temperature(tau)
