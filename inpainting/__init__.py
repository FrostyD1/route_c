"""
Route C: Amortized Inpainting
==============================
Replaces test-time MCMC with a trained mask-prediction network.

Key components:
1. InpaintNet: CNN that predicts masked bits from observed context
2. InpaintTrainer: Self-supervised training on random masks
3. iterative_decode: MaskGIT-style iterative refinement (optional)

References:
- MaskGIT (Chang et al., 2022): parallel iterative decoding with confidence-based unmasking
- Modern Hopfield Networks (Ramsauer et al., 2021): attention as single-step associative retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import time


# ============================================================================
# INPAINTING NETWORK
# ============================================================================

class InpaintNet(nn.Module):
    """
    CNN-based inpainting network for discrete binary latent grids.

    Input:  z_masked (k, H, W) + mask (1, H, W) → concatenated (k+1, H, W)
    Output: logits (k, H, W) — per-bit prediction logits

    Architecture: 3-layer residual CNN with circular padding (torus topology).
    Small and fast: designed for 7×7 latent grids with k=8 bits.
    """

    def __init__(self, k: int = 8, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        self.k = k

        layers = []
        in_ch = k + 1  # z_masked + mask indicator

        for i in range(n_layers):
            out_ch = hidden if i < n_layers - 1 else k
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='circular'))
            if i < n_layers - 1:
                layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch

        self.net = nn.Sequential(*layers)

        # Residual skip: project input z to output space
        self.skip = nn.Conv2d(k + 1, k, 1)

    def forward(self, z_masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_masked: (B, k, H, W) — input with masked bits set to 0
            mask: (B, 1, H, W) — 1 where masked (to predict), 0 where observed

        Returns:
            logits: (B, k, H, W) — prediction logits for all positions
        """
        x = torch.cat([z_masked, mask], dim=1)  # (B, k+1, H, W)
        out = self.net(x) + self.skip(x)  # Residual connection
        return out


# ============================================================================
# TRAINING
# ============================================================================

@dataclass
class InpaintConfig:
    """Configuration for inpainting training."""
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    mask_ratio_min: float = 0.1
    mask_ratio_max: float = 0.7
    hidden: int = 64
    n_layers: int = 3
    seed: int = 42
    # Energy-aware training (reviewer feedback: amortized net learns MAP inference,
    # not a pure generative model — must include E_core + E_obs + E_cls consistency)
    alpha_core: float = 0.1   # weight for E_core consistency loss
    alpha_obs: float = 0.1    # weight for E_obs reconstruction consistency loss
    gamma_cls: float = 0.5    # weight for classification loss (reviewer v2.1 §1:
                               # only bit-BCE learns "looks like" fill, not "useful for acc" fill)


class InpaintTrainer:
    """
    Self-supervised training of InpaintNet on binary latent grids.

    Training procedure:
    1. Encode images to binary latents z via the Route C encoder
    2. Sample random mask M with ratio ~ U(mask_ratio_min, mask_ratio_max)
    3. Input: z ⊙ (1-M) with mask indicator M
    4. Target: z at masked positions
    5. Loss: BCE on masked bits only

    This is the Route C analog of MaskGIT's masked token modeling,
    but operates on binary latent grids instead of VQ tokens.
    """

    def __init__(
        self,
        routec_model,
        config: InpaintConfig = None,
        device: torch.device = None,
    ):
        self.routec_model = routec_model
        self.cfg = config or InpaintConfig()
        self.device = device or torch.device('cpu')

        self.inpaint_net = InpaintNet(
            k=routec_model.n_bits,
            hidden=self.cfg.hidden,
            n_layers=self.cfg.n_layers,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.inpaint_net.parameters(), lr=self.cfg.lr
        )

    def _random_bit_mask(self, B: int, k: int, H: int, W: int) -> torch.Tensor:
        """
        Generate random per-position masks (all bits at a position are masked together).

        Returns:
            mask: (B, 1, H, W) — 1 = masked, 0 = observed
        """
        ratio = torch.rand(B, 1, 1, 1, device=self.device)
        ratio = self.cfg.mask_ratio_min + ratio * (self.cfg.mask_ratio_max - self.cfg.mask_ratio_min)
        mask = (torch.rand(B, 1, H, W, device=self.device) < ratio).float()
        return mask

    def train_epoch(self, train_x: torch.Tensor) -> float:
        """
        One epoch of energy-aware mask prediction training.

        Loss = L_mask + α_core·L_core + α_obs·L_obs
        where:
        - L_mask = BCE on masked bits (MaskGIT-style)
        - L_core = local predictor consistency (predicted z should satisfy neighborhood rules)
        - L_obs  = reconstruction consistency (decode(z_predicted) ≈ x on unmasked pixels)

        This makes the amortized net learn approximate MAP inference,
        not a pure generative model. (Reviewer feedback §5)

        Args:
            train_x: (N, 1, 28, 28) images

        Returns:
            average loss over epoch
        """
        self.routec_model.eval()
        self.inpaint_net.train()

        N = len(train_x)
        perm = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, N, self.cfg.batch_size):
            idx = perm[i:i+self.cfg.batch_size]
            x = train_x[idx].to(self.device)

            # Encode to binary latent
            with torch.no_grad():
                z = self.routec_model.encode(x)  # (B, k, H, W)
                z_hard = (z > 0.5).float()

            B, k, H, W = z_hard.shape

            # Random mask
            mask = self._random_bit_mask(B, k, H, W)  # (B, 1, H, W)
            mask_expanded = mask.expand(-1, k, -1, -1)  # (B, k, H, W)

            # Masked input
            z_masked = z_hard * (1 - mask_expanded)

            # Forward
            logits = self.inpaint_net(z_masked, mask)  # (B, k, H, W)

            # L_mask: BCE only on masked positions
            loss_mask = F.binary_cross_entropy_with_logits(
                logits[mask_expanded.bool()],
                z_hard[mask_expanded.bool()],
            )

            loss = loss_mask

            # L_core: predicted z should be consistent with local neighborhood rules
            if self.cfg.alpha_core > 0:
                z_pred = (torch.sigmoid(logits) > 0.5).float()
                # Compose: keep observed, fill predicted
                z_composed = z_hard * (1 - mask_expanded) + z_pred * mask_expanded
                with torch.no_grad():
                    core_logits = self.routec_model.local_pred(z_composed)
                # Soft prediction version for gradient flow
                z_soft = z_hard * (1 - mask_expanded) + torch.sigmoid(logits) * mask_expanded
                core_logits_soft = self.routec_model.local_pred(z_soft)
                loss_core = F.binary_cross_entropy_with_logits(
                    core_logits_soft[mask_expanded.bool()],
                    z_hard[mask_expanded.bool()],
                )
                loss = loss + self.cfg.alpha_core * loss_core

            # L_obs: reconstruction consistency
            if self.cfg.alpha_obs > 0:
                z_soft = z_hard * (1 - mask_expanded) + torch.sigmoid(logits) * mask_expanded
                x_hat = self.routec_model.decode(z_soft)
                loss_obs = F.mse_loss(x_hat, x)
                loss = loss + self.cfg.alpha_obs * loss_obs

            # L_cls: classification loss — ensures inpainted z is useful for downstream task
            # (Reviewer v2.1 §1: "only bit-BCE very likely learns fill that
            #  looks-like but is useless for classification")
            if self.cfg.gamma_cls > 0 and hasattr(self, '_train_y') and self._train_y is not None:
                z_soft = z_hard * (1 - mask_expanded) + torch.sigmoid(logits) * mask_expanded
                cls_logits = self.routec_model.classifier(z_soft)
                y_batch = self._train_y[idx].to(self.device)
                loss_cls = F.cross_entropy(cls_logits, y_batch)
                loss = loss + self.cfg.gamma_cls * loss_cls

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, train_x: torch.Tensor, train_y: torch.Tensor = None, verbose: bool = True) -> Dict:
        """Full training loop.

        Args:
            train_x: (N, 1, 28, 28) images
            train_y: (N,) labels — if provided, enables L_cls training
        """
        self._train_y = train_y  # Store for use in train_epoch
        losses = []
        for epoch in range(self.cfg.epochs):
            loss = self.train_epoch(train_x)
            losses.append(loss)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  InpaintNet epoch {epoch+1}/{self.cfg.epochs}: loss={loss:.4f}")
        self._train_y = None
        return {'losses': losses}


# ============================================================================
# INFERENCE: AMORTIZED + OPTIONAL ITERATIVE REFINEMENT
# ============================================================================

def amortized_inpaint(
    inpaint_net: InpaintNet,
    z_init: torch.Tensor,
    bit_mask: np.ndarray,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Single-pass amortized inpainting (no MCMC).

    Args:
        inpaint_net: trained InpaintNet
        z_init: (k, H, W) — initial binary latent (with occluded region encoded from occluded image)
        bit_mask: (k, H, W) bool — True where bits should be inpainted
        device: torch device

    Returns:
        z_inpainted: (k, H, W) binary tensor
    """
    if device is None:
        device = next(inpaint_net.parameters()).device

    inpaint_net.eval()

    k, H, W = z_init.shape
    z = z_init.clone().to(device)

    # Create mask tensor
    bm = torch.from_numpy(bit_mask).float().to(device)  # (k, H, W)
    mask = bm.max(dim=0, keepdim=True)[0].unsqueeze(0)  # (1, 1, H, W) — per-position mask
    bm_expanded = bm.unsqueeze(0)  # (1, k, H, W)

    # Zero out masked bits
    z_masked = z.unsqueeze(0) * (1 - bm_expanded)

    with torch.no_grad():
        logits = inpaint_net(z_masked, mask)  # (1, k, H, W)
        predictions = (torch.sigmoid(logits) > 0.5).float()

    # Fill in only masked positions
    z_result = z.clone()
    z_result[torch.from_numpy(bit_mask).to(device)] = predictions[0][torch.from_numpy(bit_mask).to(device)]

    return z_result


def iterative_inpaint(
    inpaint_net: InpaintNet,
    z_init: torch.Tensor,
    bit_mask: np.ndarray,
    n_steps: int = 4,
    device: torch.device = None,
) -> torch.Tensor:
    """
    MaskGIT-style iterative inpainting with confidence-based unmasking.

    At each step:
    1. Predict all masked bits
    2. Compute confidence = |σ(logit) - 0.5| * 2
    3. Unmask highest-confidence fraction (cosine schedule)
    4. Re-predict remaining

    Args:
        inpaint_net: trained InpaintNet
        z_init: (k, H, W) initial binary latent
        bit_mask: (k, H, W) bool — True where to inpaint
        n_steps: number of refinement iterations
        device: torch device

    Returns:
        z_inpainted: (k, H, W) binary tensor
    """
    if device is None:
        device = next(inpaint_net.parameters()).device

    inpaint_net.eval()

    k, H, W = z_init.shape
    z = z_init.clone().to(device)

    # Current mask (True = still masked)
    current_mask = torch.from_numpy(bit_mask).bool().to(device)  # (k, H, W)
    total_masked = current_mask.sum().item()

    if total_masked == 0:
        return z

    with torch.no_grad():
        for step in range(n_steps):
            # How many to keep masked after this step (cosine schedule)
            progress = (step + 1) / n_steps
            fraction_to_unmask = 1.0 - np.cos(progress * np.pi / 2)
            n_to_unmask = int(fraction_to_unmask * total_masked)
            n_to_unmask = min(n_to_unmask, current_mask.sum().item())

            if current_mask.sum().item() == 0:
                break

            # Prepare input
            mask_spatial = current_mask.float().max(dim=0, keepdim=True)[0].unsqueeze(0)
            z_masked = z.unsqueeze(0) * (1 - current_mask.float().unsqueeze(0))

            # Predict
            logits = inpaint_net(z_masked, mask_spatial)  # (1, k, H, W)
            probs = torch.sigmoid(logits[0])  # (k, H, W)

            # Predictions and confidence
            predictions = (probs > 0.5).float()
            confidence = (probs - 0.5).abs() * 2  # (k, H, W), in [0, 1]

            # Only consider currently masked positions
            confidence_masked = confidence.clone()
            confidence_masked[~current_mask] = -1.0

            # Find top-n_to_unmask positions by confidence
            flat_conf = confidence_masked.flatten()
            flat_mask = current_mask.flatten()

            if n_to_unmask > 0 and flat_mask.sum() > 0:
                # Sort by confidence (descending)
                sorted_indices = flat_conf.argsort(descending=True)

                count = 0
                for idx in sorted_indices:
                    if count >= n_to_unmask:
                        break
                    if flat_mask[idx]:
                        # Unmask this position
                        flat_mask[idx] = False
                        # Write prediction
                        z.view(-1)[idx] = predictions.view(-1)[idx]
                        count += 1

                current_mask = flat_mask.reshape(k, H, W)

        # Final pass: fill any remaining masked positions
        if current_mask.sum().item() > 0:
            mask_spatial = current_mask.float().max(dim=0, keepdim=True)[0].unsqueeze(0)
            z_masked = z.unsqueeze(0) * (1 - current_mask.float().unsqueeze(0))
            logits = inpaint_net(z_masked, mask_spatial)
            predictions = (torch.sigmoid(logits[0]) > 0.5).float()
            z[current_mask] = predictions[current_mask]

    return z
