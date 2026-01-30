"""
Route C: Learnable Quantizer (ADC)
==================================

ADC: z = Qτ(Eφ(o))

Implements:
- Gumbel-Sigmoid: differentiable relaxation with temperature
- Straight-Through Estimator (STE): hard forward, soft backward
- Bernoulli STE: sample hard {0,1}, gradient through sigmoid

Math:
  logits L = Encoder(o)           # shape (B, k, H, W)
  soft_z = sigmoid(L / τ)         # relaxed
  hard_z = (soft_z > 0.5).float() # discrete
  z = hard_z - soft_z.detach() + soft_z  # STE trick
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GumbelSigmoid(nn.Module):
    """
    Gumbel-Sigmoid for binary latent variables.
    
    Forward: hard {0,1} samples
    Backward: gradient through soft sigmoid
    """
    
    def __init__(self, temperature: float = 1.0, hard: bool = True):
        super().__init__()
        self.temperature = temperature
        self.hard = hard
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, k, H, W) raw logits from encoder
        Returns:
            z: (B, k, H, W) binary {0,1} tensor
        """
        if self.training:
            # Add Gumbel noise for exploration
            u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            gumbel_noise = -torch.log(-torch.log(u))
            noisy_logits = (logits + gumbel_noise) / self.temperature
        else:
            noisy_logits = logits / self.temperature
        
        soft = torch.sigmoid(noisy_logits)
        
        if self.hard:
            hard = (soft > 0.5).float()
            # STE: hard forward, soft backward
            return hard - soft.detach() + soft
        else:
            return soft
    
    def set_temperature(self, tau: float):
        self.temperature = tau


class BernoulliSTE(nn.Module):
    """
    Bernoulli sampling with Straight-Through Estimator.
    
    Forward: sample from Bernoulli(sigmoid(logits))
    Backward: gradient through sigmoid (ignore sampling)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        if self.training:
            # Sample from Bernoulli
            hard = torch.bernoulli(probs)
        else:
            # Deterministic at test time
            hard = (probs > 0.5).float()
        
        # STE trick
        return hard - probs.detach() + probs


class DeterministicBinarize(nn.Module):
    """
    Deterministic binarization with STE.
    
    Forward: hard threshold at 0.5
    Backward: gradient through sigmoid
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        soft = torch.sigmoid(logits / self.temperature)
        hard = (soft > 0.5).float()
        return hard - soft.detach() + soft
    
    def set_temperature(self, tau: float):
        self.temperature = tau


# ============================================================================
# ENCODER (Small CNN → Logits)
# ============================================================================

class BinaryEncoder(nn.Module):
    """
    Small CNN encoder: o → L ∈ R^{k×H×W}
    
    Input: (B, 1, 28, 28) normalized images
    Output: (B, k, H, W) logits for binary latent
    
    Default: k=8, H=W=7 → 392 bits per image
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        n_bits: int = 8,          # k: bits per position
        latent_size: int = 7,     # H=W
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.latent_size = latent_size
        
        # 28×28 → 14×14 → 7×7
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, stride=2, padding=1),  # 28→14
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),   # 14→7
            nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1),                 # 7×7, k channels
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 28, 28)
        Returns:
            logits: (B, k, H, W)
        """
        return self.conv(x)


# ============================================================================
# DECODER (z → o_hat)
# ============================================================================

class BinaryDecoder(nn.Module):
    """
    Small decoder: z ∈ {0,1}^{k×H×W} → o_hat ∈ R^{1×28×28}
    
    Uses transposed convolutions to upsample 7×7 → 28×28
    """
    
    def __init__(
        self,
        n_bits: int = 8,
        latent_size: int = 7,
        hidden_dim: int = 64,
        out_channels: int = 1,
    ):
        super().__init__()
        
        # 7×7 → 14×14 → 28×28
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1),  # 7→14
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # 14→28
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, k, H, W) binary latent
        Returns:
            o_hat: (B, 1, 28, 28) reconstructed image
        """
        return self.deconv(z)


# ============================================================================
# FULL ADC/DAC MODULE
# ============================================================================

class LearnableADC_DAC(nn.Module):
    """
    Complete ADC/DAC system for Route C.
    
    ADC: o → z = Qτ(Eφ(o))
    DAC: z → o_hat = Dψ(z)
    """
    
    def __init__(
        self,
        n_bits: int = 8,
        latent_size: int = 7,
        hidden_dim: int = 64,
        quantizer_type: str = "gumbel",  # "gumbel", "bernoulli", "deterministic"
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.n_bits = n_bits
        self.latent_size = latent_size
        
        # Encoder
        self.encoder = BinaryEncoder(
            n_bits=n_bits,
            latent_size=latent_size,
            hidden_dim=hidden_dim,
        )
        
        # Quantizer
        if quantizer_type == "gumbel":
            self.quantizer = GumbelSigmoid(temperature=temperature)
        elif quantizer_type == "bernoulli":
            self.quantizer = BernoulliSTE()
        else:
            self.quantizer = DeterministicBinarize(temperature=temperature)
        
        # Decoder
        self.decoder = BinaryDecoder(
            n_bits=n_bits,
            latent_size=latent_size,
            hidden_dim=hidden_dim,
        )
    
    def encode(self, o: torch.Tensor) -> torch.Tensor:
        """ADC: o → z"""
        logits = self.encoder(o)
        z = self.quantizer(logits)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """DAC: z → o_hat"""
        return self.decoder(z)
    
    def forward(self, o: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Returns:
            z: (B, k, H, W) binary latent
            o_hat: (B, 1, 28, 28) reconstruction
        """
        z = self.encode(o)
        o_hat = self.decode(z)
        return z, o_hat
    
    def set_temperature(self, tau: float):
        """Anneal quantizer temperature."""
        if hasattr(self.quantizer, 'set_temperature'):
            self.quantizer.set_temperature(tau)
    
    def get_logits(self, o: torch.Tensor) -> torch.Tensor:
        """Get raw logits before quantization (for analysis)."""
        return self.encoder(o)


# ============================================================================
# NUMPY INTERFACE (for inference without torch)
# ============================================================================

def torch_to_numpy_adc(model: LearnableADC_DAC, device: str = 'cpu'):
    """
    Extract numpy-compatible weights for discrete inference.
    
    Returns functions that work with numpy arrays.
    """
    model.eval()
    model.to(device)
    
    def encode_np(o: np.ndarray) -> np.ndarray:
        """o: (B, 28, 28) or (28, 28) → z: (B, k, H, W) or (k, H, W)"""
        squeeze = o.ndim == 2
        if squeeze:
            o = o[None, None, :, :]
        elif o.ndim == 3:
            o = o[:, None, :, :]
        
        with torch.no_grad():
            o_t = torch.from_numpy(o.astype(np.float32)).to(device)
            z_t = model.encode(o_t)
            z = z_t.cpu().numpy()
        
        if squeeze:
            z = z[0]
        return z
    
    def decode_np(z: np.ndarray) -> np.ndarray:
        """z: (B, k, H, W) or (k, H, W) → o_hat: (B, 28, 28) or (28, 28)"""
        squeeze = z.ndim == 3
        if squeeze:
            z = z[None, :, :, :]
        
        with torch.no_grad():
            z_t = torch.from_numpy(z.astype(np.float32)).to(device)
            o_t = model.decode(z_t)
            o = o_t.cpu().numpy()
        
        if squeeze:
            o = o[0, 0]
        else:
            o = o[:, 0]
        return o
    
    return encode_np, decode_np
