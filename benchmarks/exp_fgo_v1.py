#!/usr/bin/env python3
"""
FGO v1: Frequency Global Operator
===================================
Phase 1: Build a global mixing primitive using FFT/DCT inside InpaintNet.

The hypothesis: local 3×3 conv is sufficient for contiguous masks (center),
but sparse/scattered masks (multi_hole, random_sparse) need global information
flow. FFT provides O(N log N) global mixing vs O(N²) for attention.

Three FGO variants:
  1. Low-pass only: keep low frequencies (global shape/structure)
  2. Multi-band gate: low/mid/high frequency 3-band gating (learned)
  3. Data-adaptive gate: gate weights generated from input (content-aware)

Architecture: InpaintNet = local_conv → FGO → local_conv (sandwich)

Test masks: multi_hole, random_sparse, center (sanity check)
Test grids: 7×7, 14×14

Usage:
    python3 -u benchmarks/exp_fgo_v1.py --device cuda --seed 42
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import time
import csv
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))


# ============================================================================
# CORE MODEL (same base as other experiments)
# ============================================================================

class GumbelSigmoid(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    def forward(self, logits):
        if self.training:
            u = torch.rand_like(logits).clamp(1e-8, 1-1e-8)
            noisy = (logits - torch.log(-torch.log(u))) / self.temperature
        else:
            noisy = logits / self.temperature
        soft = torch.sigmoid(noisy)
        hard = (soft > 0.5).float()
        return hard - soft.detach() + soft
    def set_temperature(self, tau): self.temperature = tau


class BinaryEncoder(nn.Module):
    def __init__(self, n_bits, in_channels=1, hidden_dim=64, latent_size=7):
        super().__init__()
        layers = [nn.Conv2d(in_channels, hidden_dim, 3, stride=2, padding=1), nn.ReLU()]
        if latent_size == 7:
            layers += [nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.ReLU()]
        layers += [nn.Conv2d(hidden_dim, n_bits, 3, padding=1)]
        self.conv = nn.Sequential(*layers)
    def forward(self, x): return self.conv(x)


class BinaryDecoder(nn.Module):
    def __init__(self, n_bits, out_channels=1, hidden_dim=64, latent_size=7):
        super().__init__()
        layers = [nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1), nn.ReLU()]
        if latent_size == 7:
            layers += [nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), nn.ReLU()]
        else:
            layers += [nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU()]
        layers += [nn.Conv2d(hidden_dim, out_channels, 3, padding=1), nn.Sigmoid()]
        self.deconv = nn.Sequential(*layers)
    def forward(self, z): return self.deconv(z)


class LocalPredictor(nn.Module):
    def __init__(self, n_bits, hidden_dim=32):
        super().__init__()
        self.n_bits = n_bits
        self.net = nn.Sequential(
            nn.Linear(9 * n_bits, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_bits),
        )
    def forward(self, z):
        B, k, H, W = z.shape
        z_pad = F.pad(z, (1,1,1,1), mode='constant', value=0)
        windows = F.unfold(z_pad, kernel_size=3)
        windows = windows.reshape(B, k, 9, H*W)
        windows[:, :, 4, :] = 0
        windows = windows.reshape(B, k*9, H*W).permute(0,2,1)
        logits = self.net(windows)
        return logits.permute(0,2,1).reshape(B, k, H, W)


class Classifier(nn.Module):
    def __init__(self, n_bits, latent_size=7, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(n_bits * latent_size * latent_size, n_classes)
    def forward(self, z): return self.fc(z.reshape(z.shape[0], -1))


class RouteCModel(nn.Module):
    def __init__(self, n_bits=8, latent_size=7, hidden_dim=64, energy_hidden=32, tau=1.0):
        super().__init__()
        self.n_bits = n_bits
        self.latent_size = latent_size
        self.encoder = BinaryEncoder(n_bits, 1, hidden_dim, latent_size)
        self.decoder = BinaryDecoder(n_bits, 1, hidden_dim, latent_size)
        self.quantizer = GumbelSigmoid(tau)
        self.local_pred = LocalPredictor(n_bits, energy_hidden)
        self.classifier = Classifier(n_bits, latent_size)
    def encode(self, x): return self.quantizer(self.encoder(x))
    def decode(self, z): return self.decoder(z)
    def set_temperature(self, tau): self.quantizer.set_temperature(tau)
    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z), self.classifier(z), self.local_pred(z)


# ============================================================================
# FGO: FREQUENCY GLOBAL OPERATOR (three variants)
# ============================================================================

class FGO_LowPass(nn.Module):
    """
    FGO v1a: Low-pass frequency filter.
    Keep only low-frequency components (global shape/structure).
    Cutoff is a learned parameter.
    """
    def __init__(self, channels, init_cutoff_ratio=0.5):
        super().__init__()
        self.channels = channels
        # Learnable cutoff (as fraction of max frequency)
        self.cutoff_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid → ~0.5
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape

        # 2D FFT
        X_fft = torch.fft.rfft2(x)  # (B, C, H, W//2+1) complex

        # Create frequency mask (low-pass)
        cutoff = torch.sigmoid(self.cutoff_logit)  # in (0, 1)
        freq_h = torch.fft.fftfreq(H, device=x.device)  # (H,)
        freq_w = torch.fft.rfftfreq(W, device=x.device)  # (W//2+1,)
        freq_grid = torch.sqrt(freq_h[:, None]**2 + freq_w[None, :]**2)  # (H, W//2+1)
        max_freq = freq_grid.max()
        mask = (freq_grid / (max_freq + 1e-8) < cutoff).float()  # binary low-pass
        # Smooth the mask edge
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W//2+1)

        # Apply filter
        X_filtered = X_fft * mask * self.scale

        # Inverse FFT
        out = torch.fft.irfft2(X_filtered, s=(H, W))
        return out


class FGO_MultiBand(nn.Module):
    """
    FGO v1b: Multi-band frequency gating.
    Three bands: low (structure), mid (edges), high (texture).
    Each band has a learned gate weight.
    """
    def __init__(self, channels, n_bands=3):
        super().__init__()
        self.channels = channels
        self.n_bands = n_bands
        # Per-band, per-channel gate weights
        self.band_gates = nn.Parameter(torch.ones(n_bands, channels))
        # Band boundaries (learnable)
        self.band_bounds = nn.Parameter(torch.tensor([0.33, 0.67]))

    def forward(self, x):
        B, C, H, W = x.shape

        X_fft = torch.fft.rfft2(x)

        freq_h = torch.fft.fftfreq(H, device=x.device)
        freq_w = torch.fft.rfftfreq(W, device=x.device)
        freq_grid = torch.sqrt(freq_h[:, None]**2 + freq_w[None, :]**2)
        max_freq = freq_grid.max() + 1e-8
        freq_norm = freq_grid / max_freq  # (H, W//2+1) in [0, 1]

        bounds = torch.sigmoid(self.band_bounds).sort()[0]  # ensure ordered
        gates = torch.sigmoid(self.band_gates)  # (n_bands, C)

        # Build composite mask
        composite = torch.zeros(C, H, W // 2 + 1, device=x.device)

        # Band 0: [0, bounds[0])
        mask_low = (freq_norm < bounds[0]).float()
        composite += gates[0].view(C, 1, 1) * mask_low.unsqueeze(0)

        # Band 1: [bounds[0], bounds[1])
        mask_mid = ((freq_norm >= bounds[0]) & (freq_norm < bounds[1])).float()
        composite += gates[1].view(C, 1, 1) * mask_mid.unsqueeze(0)

        # Band 2: [bounds[1], 1.0]
        mask_high = (freq_norm >= bounds[1]).float()
        composite += gates[2].view(C, 1, 1) * mask_high.unsqueeze(0)

        X_filtered = X_fft * composite.unsqueeze(0)  # (B, C, H, W//2+1)
        out = torch.fft.irfft2(X_filtered, s=(H, W))
        return out


class FGO_Adaptive(nn.Module):
    """
    FGO v1c: Data-adaptive frequency gating.
    Gate weights are generated from the input via a small network.
    Content-aware but still task-agnostic.
    """
    def __init__(self, channels, n_bands=3):
        super().__init__()
        self.channels = channels
        self.n_bands = n_bands
        # Gate generator: global average pool → MLP → per-band gates
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, n_bands * channels),  # (B, n_bands * C)
        )
        self.band_bounds = nn.Parameter(torch.tensor([0.33, 0.67]))

    def forward(self, x):
        B, C, H, W = x.shape

        # Generate gates from input content
        gate_logits = self.gate_net(x)  # (B, n_bands * C)
        gates = torch.sigmoid(gate_logits).view(B, self.n_bands, C)  # (B, n_bands, C)

        X_fft = torch.fft.rfft2(x)

        freq_h = torch.fft.fftfreq(H, device=x.device)
        freq_w = torch.fft.rfftfreq(W, device=x.device)
        freq_grid = torch.sqrt(freq_h[:, None]**2 + freq_w[None, :]**2)
        max_freq = freq_grid.max() + 1e-8
        freq_norm = freq_grid / max_freq

        bounds = torch.sigmoid(self.band_bounds).sort()[0]

        # Build per-sample composite mask
        mask_low = (freq_norm < bounds[0]).float()        # (H, W//2+1)
        mask_mid = ((freq_norm >= bounds[0]) & (freq_norm < bounds[1])).float()
        mask_high = (freq_norm >= bounds[1]).float()

        # (B, C, H, W//2+1)
        composite = (
            gates[:, 0, :, None, None] * mask_low[None, None, :, :] +
            gates[:, 1, :, None, None] * mask_mid[None, None, :, :] +
            gates[:, 2, :, None, None] * mask_high[None, None, :, :]
        )

        X_filtered = X_fft * composite
        out = torch.fft.irfft2(X_filtered, s=(H, W))
        return out


# ============================================================================
# INPAINT NETS (local-only vs FGO variants)
# ============================================================================

class InpaintNetLocal(nn.Module):
    """Baseline: local-only InpaintNet."""
    def __init__(self, k=8, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(k+1, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(hidden, k, 3, padding=1, padding_mode='circular'),
        )
        self.skip = nn.Conv2d(k+1, k, 1)
    def forward(self, z_masked, mask):
        x = torch.cat([z_masked, mask], dim=1)
        return self.net(x) + self.skip(x)


class InpaintNetFGO(nn.Module):
    """
    InpaintNet with FGO: local_conv → FGO → local_conv (sandwich).

    The FGO layer sits between local conv layers, providing global mixing
    at O(N log N) cost via FFT.
    """
    def __init__(self, k=8, hidden=64, fgo_type='lowpass'):
        super().__init__()
        self.k = k
        self.fgo_type = fgo_type

        # Layer 1: local conv (extract local features)
        self.local_pre = nn.Sequential(
            nn.Conv2d(k+1, hidden, 3, padding=1, padding_mode='circular'),
            nn.ReLU(),
        )

        # Layer 2: FGO (global mixing in frequency domain)
        if fgo_type == 'lowpass':
            self.fgo = FGO_LowPass(hidden)
        elif fgo_type == 'multiband':
            self.fgo = FGO_MultiBand(hidden)
        elif fgo_type == 'adaptive':
            self.fgo = FGO_Adaptive(hidden)
        else:
            raise ValueError(f"Unknown fgo_type: {fgo_type}")

        # FGO residual projection
        self.fgo_proj = nn.Conv2d(hidden, hidden, 1)

        # Layer 3: local conv (combine local + global)
        self.local_post = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(hidden, k, 3, padding=1, padding_mode='circular'),
        )

        self.skip = nn.Conv2d(k+1, k, 1)

    def forward(self, z_masked, mask):
        x = torch.cat([z_masked, mask], dim=1)  # (B, k+1, H, W)

        # Local pre-processing
        h = self.local_pre(x)  # (B, hidden, H, W)

        # Global mixing via FGO (with residual)
        h_global = self.fgo(h)
        h = h + self.fgo_proj(h_global)  # residual connection

        # Local post-processing
        out = self.local_post(h) + self.skip(x)
        return out


# ============================================================================
# MASKS
# ============================================================================

def make_center_mask(H=28, W=28):
    m = np.ones((H,W), dtype=np.float32); m[7:21, 7:21] = 0; return m

def make_multi_hole_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((H,W), dtype=np.float32)
    for _ in range(rng.integers(3,8)):
        s = rng.integers(2,6)
        y, x = rng.integers(0, max(1,H-s+1)), rng.integers(0, max(1,W-s+1))
        m[y:y+s, x:x+s] = 0
    return m

def make_random_sparse_mask(H=28, W=28, rng=None, ratio=0.3):
    """Random pixel-wise sparse mask: ratio of pixels are occluded."""
    if rng is None: rng = np.random.default_rng()
    m = np.ones((H,W), dtype=np.float32)
    m[rng.random((H,W)) < ratio] = 0
    return m

def make_stripe_mask(H=28, W=28):
    m = np.ones((H,W), dtype=np.float32)
    for y in range(0, H, 6): m[y:min(y+2,H), :] = 0
    return m

def make_random_stripe_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    w = rng.integers(1, 4); g = rng.integers(4, 10); p = rng.integers(0, g)
    m = np.ones((H,W), dtype=np.float32)
    if rng.random() < 0.5:
        for y in range(p, H, g): m[y:min(y+w,H), :] = 0
    else:
        for x in range(p, W, g): m[:, x:min(x+w,W)] = 0
    return m

def make_random_block_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((H,W), dtype=np.float32)
    oh, ow = rng.integers(8,18), rng.integers(8,18)
    y, x = rng.integers(0, max(1,H-oh+1)), rng.integers(0, max(1,W-ow+1))
    m[y:y+oh, x:x+ow] = 0; return m

def sample_training_mask(H=28, W=28, rng=None):
    """Mask mixture for sleep-phase training."""
    if rng is None: rng = np.random.default_rng()
    p = rng.random()
    if p < 0.25: return make_random_block_mask(H,W,rng)
    elif p < 0.45: return make_center_mask(H,W)
    elif p < 0.65: return make_random_stripe_mask(H,W,rng)
    elif p < 0.80: return make_multi_hole_mask(H,W,rng)
    else: return make_random_sparse_mask(H,W,rng, ratio=rng.uniform(0.2, 0.5))

def pixel_to_bit_mask(pixel_mask, n_bits, latent_size=7):
    H, W = pixel_mask.shape
    patch_h, patch_w = H // latent_size, W // latent_size
    bm = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    for i in range(latent_size):
        for j in range(latent_size):
            y0, y1 = i*patch_h, (i+1)*patch_h
            x0, x1 = j*patch_w, (j+1)*patch_w
            if pixel_mask[y0:y1, x0:x1].mean() < 1.0 - 1e-6:
                bm[:, i, j] = True
    return bm

def apply_noise(image, noise_type, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    if noise_type == 'noise':
        return np.clip(image + rng.normal(0, 0.1, image.shape).astype(np.float32), 0, 1)
    return image.copy()

def load_dataset(name, train_n=2000, test_n=500, seed=42):
    from torchvision import datasets, transforms
    ds_map = {
        'mnist': datasets.MNIST,
        'fmnist': datasets.FashionMNIST,
    }
    ds_cls = ds_map[name]
    tr = ds_cls('./data', train=True, download=True, transform=transforms.ToTensor())
    te = ds_cls('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(tr), train_n, replace=False)
    si = rng.choice(len(te), test_n, replace=False)
    return (torch.stack([tr[i][0] for i in ti]), torch.tensor([tr[i][1] for i in ti]),
            torch.stack([te[i][0] for i in si]), torch.tensor([te[i][1] for i in si]))


# ============================================================================
# TRAINING
# ============================================================================

def train_model(train_x, train_y, device, latent_size=7, epochs=5, lr=1e-3, batch_size=64):
    model = RouteCModel(latent_size=latent_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        tau = 1.0 + (0.2-1.0) * epoch / max(1, epochs-1)
        model.set_temperature(tau)
        el, nb = 0., 0
        for x, y in loader:
            x = x.to(device); opt.zero_grad()
            z, x_hat, _, cl = model(x)
            lr_ = F.binary_cross_entropy(x_hat.clamp(1e-6,1-1e-6), x)
            m = torch.rand_like(z) < 0.15
            lc = F.binary_cross_entropy_with_logits(cl[m], z.detach()[m]) if m.any() else torch.tensor(0., device=device)
            (lr_ + 0.5*lc).backward(); opt.step()
            el += (lr_+0.5*lc).item(); nb += 1
        print(f"      Epoch {epoch+1}/{epochs}: loss={el/max(nb,1):.4f}")
    # Frozen probe
    for p in model.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True
    co = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    for _ in range(3):
        model.classifier.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): z = model.encode(x)
            F.cross_entropy(model.classifier(z), y).backward(); co.step(); co.zero_grad()
    for p in model.parameters(): p.requires_grad = True
    return model


def train_inpaint(model, train_x, device, net, epochs=20, batch_size=64, lr=1e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    model.eval()
    N = len(train_x)
    rng = np.random.default_rng(42)
    ls = model.latent_size

    for epoch in range(epochs):
        net.train(); perm = torch.randperm(N); tl, nb = 0., 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]; x = train_x[idx].to(device)
            with torch.no_grad():
                z = model.encode(x); z_hard = (z > 0.5).float()
            B, k, H, W = z_hard.shape
            masks = []
            for b in range(B):
                pm = sample_training_mask(28, 28, rng)
                bm = pixel_to_bit_mask(pm, k, ls)
                masks.append(torch.from_numpy(bm).float())
            bit_masks = torch.stack(masks).to(device)
            pos_masks = bit_masks[:, 0:1, :, :]
            z_masked = z_hard * (1 - bit_masks)
            logits = net(z_masked, pos_masks)
            loss = F.binary_cross_entropy_with_logits(logits[bit_masks.bool()], z_hard[bit_masks.bool()])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        if (epoch+1) % 5 == 0:
            print(f"        epoch {epoch+1}/{epochs}: loss={tl/max(nb,1):.4f}")
    return net


# ============================================================================
# EVALUATION
# ============================================================================

def amortized_inpaint(net, z_init, bit_mask, device):
    net.eval()
    z = z_init.clone().to(device)
    bm = torch.from_numpy(bit_mask).float().to(device)
    mask = bm.max(dim=0, keepdim=True)[0].unsqueeze(0)
    z_masked = z.unsqueeze(0) * (1 - bm.unsqueeze(0))
    with torch.no_grad():
        logits = net(z_masked, mask)
        preds = (torch.sigmoid(logits) > 0.5).float()
    bm_bool = torch.from_numpy(bit_mask).to(device)
    z_result = z.clone()
    z_result[bm_bool] = preds[0][bm_bool]
    return z_result


def evaluate(model, net, mask_type, noise_type, test_x, test_y, device,
             n_samples=100, seed=42):
    model.eval(); net.eval()
    ls = model.latent_size

    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    # Generate masks per sample for stochastic mask types
    stochastic_masks = mask_type in ['multi_hole', 'random_sparse']

    if not stochastic_masks:
        if mask_type == 'center':
            pm = make_center_mask()
        elif mask_type == 'stripes':
            pm = make_stripe_mask()
        else:
            raise ValueError(f"Unknown fixed mask: {mask_type}")

    cb, ca, bce_b, bce_a, rts, ratios = [], [], [], [], [], []

    for idx in eval_idx:
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()
        x_noisy = apply_noise(x_clean, noise_type, rng)

        if stochastic_masks:
            if mask_type == 'multi_hole':
                pm = make_multi_hole_mask(rng=rng)
            else:  # random_sparse
                pm = make_random_sparse_mask(rng=rng, ratio=0.3)

        occ = 1 - pm
        x_occ = x_noisy * pm

        with torch.no_grad():
            x_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_init = model.encode(x_t)[0]
            pred_b = model.classifier(z_init.unsqueeze(0)).argmax(1).item()
            o_hat_b = model.decode(z_init.unsqueeze(0))[0, 0]

        bit_mask = pixel_to_bit_mask(pm, model.n_bits, ls)
        ratio = bit_mask.any(axis=0).mean()

        t0 = time.time()
        z_final = amortized_inpaint(net, z_init, bit_mask, device)
        rt = (time.time() - t0) * 1000

        with torch.no_grad():
            pred_a = model.classifier(z_final.unsqueeze(0)).argmax(1).item()
            o_hat_a = model.decode(z_final.unsqueeze(0))[0, 0]

        xt = torch.from_numpy(x_clean).to(device)
        ot = torch.from_numpy(occ).to(device)
        os_ = ot.sum().clamp(min=1.0).item()

        def ob(oh):
            l = oh.clamp(1e-6,1-1e-6)
            return (-(xt*torch.log(l)+(1-xt)*torch.log(1-l))*ot).sum().item() / os_

        cb.append(int(pred_b == label)); ca.append(int(pred_a == label))
        bce_b.append(ob(o_hat_b)); bce_a.append(ob(o_hat_a))
        rts.append(rt); ratios.append(float(ratio))

    n = len(eval_idx)
    return {
        'acc_before': np.mean(cb),
        'acc_after': np.mean(ca),
        'delta_acc': (np.sum(ca) - np.sum(cb)) / n,
        'bce_before': np.mean(bce_b),
        'bce_after': np.mean(bce_a),
        'runtime_ms': np.mean(rts),
        'repair_ratio': np.mean(ratios),
        'n_samples': n,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--dataset', default='fmnist', choices=['mnist', 'fmnist'])
    parser.add_argument('--latent_size', type=int, default=7)
    parser.add_argument('--output_dir', default='outputs/exp_fgo_v1')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("FGO v1: FREQUENCY GLOBAL OPERATOR EXPERIMENT")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}, Latent: {args.latent_size}×{args.latent_size}")
    print()

    print("[1] Loading dataset...")
    train_x, train_y, test_x, test_y = load_dataset(args.dataset, 2000, 500, args.seed)

    print(f"[2] Training Route C model (latent={args.latent_size})...")
    model = train_model(train_x, train_y, device, latent_size=args.latent_size)
    model.eval()
    with torch.no_grad():
        z = model.encode(test_x[:500].to(device))
        acc = (model.classifier(z).argmax(1).cpu() == test_y[:500]).float().mean().item()
    print(f"    Clean probe accuracy: {acc:.1%}")

    # Train all four InpaintNet variants
    k = model.n_bits
    variants = {
        'local': InpaintNetLocal(k=k).to(device),
        'fgo_lowpass': InpaintNetFGO(k=k, fgo_type='lowpass').to(device),
        'fgo_multiband': InpaintNetFGO(k=k, fgo_type='multiband').to(device),
        'fgo_adaptive': InpaintNetFGO(k=k, fgo_type='adaptive').to(device),
    }

    for name, net in variants.items():
        print(f"\n[3] Training InpaintNet ({name})...")
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        train_inpaint(model, train_x, device, net)

    # Evaluate on key mask types
    mask_types = ['center', 'multi_hole', 'random_sparse']
    noise_types = ['clean']

    all_results = []

    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    print("=" * 100)
    print(f"{'variant':<18} {'mask':<15} {'noise':<6} "
          f"{'Δacc':>7} {'bce_aft':>8} {'ratio':>6} {'ms':>6}")
    print("-" * 75)

    for name, net in variants.items():
        for mt in mask_types:
            for nt in noise_types:
                r = evaluate(model, net, mt, nt, test_x, test_y, device,
                            n_samples=args.eval_samples, seed=args.seed)
                r.update({'variant': name, 'mask_type': mt, 'noise_type': nt})
                all_results.append(r)
                print(f"{name:<18} {mt:<15} {nt:<6} "
                      f"{r['delta_acc']:>+7.1%} {r['bce_after']:>8.2f} "
                      f"{r['repair_ratio']:>6.2f} {r['runtime_ms']:>6.1f}")

    # Summary: FGO gap per mask type
    print("\n" + "=" * 80)
    print("FGO GAP vs LOCAL-ONLY (key comparison)")
    print("=" * 80)

    for mt in mask_types:
        local_r = [r for r in all_results if r['variant'] == 'local'
                   and r['mask_type'] == mt and r['noise_type'] == 'clean']
        if not local_r:
            continue
        local_dacc = local_r[0]['delta_acc']
        print(f"\n  {mt}:")
        print(f"    local:         Δacc={local_dacc:+.1%}")
        for fgo_name in ['fgo_lowpass', 'fgo_multiband', 'fgo_adaptive']:
            fgo_r = [r for r in all_results if r['variant'] == fgo_name
                     and r['mask_type'] == mt and r['noise_type'] == 'clean']
            if fgo_r:
                gap = fgo_r[0]['delta_acc'] - local_dacc
                print(f"    {fgo_name:<18} Δacc={fgo_r[0]['delta_acc']:+.1%}  "
                      f"gap={gap:+.1%}")

    # Paradigm verdict
    print("\n" + "=" * 80)
    print("PARADIGM VERDICT: Is FGO necessary for sparse masks?")
    print("=" * 80)

    for mt in ['multi_hole', 'random_sparse']:
        local_r = [r for r in all_results if r['variant'] == 'local'
                   and r['mask_type'] == mt and r['noise_type'] == 'clean']
        best_fgo = None
        best_gap = -999
        for fgo_name in ['fgo_lowpass', 'fgo_multiband', 'fgo_adaptive']:
            fgo_r = [r for r in all_results if r['variant'] == fgo_name
                     and r['mask_type'] == mt and r['noise_type'] == 'clean']
            if fgo_r and local_r:
                gap = fgo_r[0]['delta_acc'] - local_r[0]['delta_acc']
                if gap > best_gap:
                    best_gap = gap
                    best_fgo = fgo_name
        if best_fgo and local_r:
            verdict = "CONFIRMED" if best_gap > 0.02 else "NOT CONFIRMED"
            print(f"  {mt}: best_fgo={best_fgo}, gap={best_gap:+.1%} → {verdict}")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "fgo_v1_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            keys = sorted(set().union(*(r.keys() for r in all_results)))
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in all_results:
                w.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    print("\n" + "=" * 100)
    print("FGO v1 experiment complete.")
    print("=" * 100)

    return all_results


if __name__ == "__main__":
    main()
