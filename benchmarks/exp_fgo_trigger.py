#!/usr/bin/env python3
"""
FGO-Trigger: Test FGO in its correct operating regime
======================================================
Previous FGO experiments tested in wrong conditions:
- Small grid (7×7, 14×14): local conv covers entire grid
- Center mask: rewards local propagation, not global mixing
- Random-sparse: destroys all evidence → noise amplification

This experiment tests 3 trigger conditions:
  1. Grid=28×28 (784 tokens), mask=two-block (far apart, evidence preserved)
  2. Grid=28×28, mask=checkerboard-with-anchors (every 4×4 has visible anchor)
  3. Grid=28×28, mask=missing-quadrant (one quadrant gone, rest intact)

FGO should win when: large grid + long-range dependency + preserved evidence.
If gap≈0 on ALL 3 configs → FGO not a critical paradigm component.

Usage:
    python3 -u benchmarks/exp_fgo_trigger.py --device cuda --seed 42
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
# MODEL for 28×28 latent grid (no spatial downsampling)
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


class Encoder28(nn.Module):
    """28×28 → 28×28 (no downsampling, each pixel = one token)."""
    def __init__(self, n_bits, in_channels=1, hidden_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 1),  # 1×1 to bits
        )
    def forward(self, x): return self.conv(x)


class Decoder28(nn.Module):
    """28×28 → 28×28 (no upsampling)."""
    def __init__(self, n_bits, out_channels=1, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, hidden_dim, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, 1), nn.Sigmoid(),
        )
    def forward(self, z): return self.net(z)


class Encoder7(nn.Module):
    """28×28 → 7×7 (for comparison baseline)."""
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1),
        )
    def forward(self, x): return self.conv(x)


class Decoder7(nn.Module):
    """7×7 → 28×28."""
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z): return self.deconv(z)


class LocalPredictor(nn.Module):
    def __init__(self, n_bits, hidden_dim=32):
        super().__init__()
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
    def __init__(self, n_bits, latent_size, n_classes=10):
        super().__init__()
        # For large grids, use adaptive pooling to keep classifier manageable
        if latent_size > 14:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.fc = nn.Linear(n_bits * 7 * 7, n_classes)
        else:
            self.pool = None
            self.fc = nn.Linear(n_bits * latent_size * latent_size, n_classes)
    def forward(self, z):
        if self.pool is not None:
            z = self.pool(z)
        return self.fc(z.reshape(z.shape[0], -1))


class RouteCModel(nn.Module):
    def __init__(self, n_bits=8, latent_size=28, hidden_dim=32, energy_hidden=32, tau=1.0):
        super().__init__()
        self.n_bits = n_bits
        self.latent_size = latent_size
        if latent_size == 28:
            self.encoder = Encoder28(n_bits, 1, hidden_dim)
            self.decoder = Decoder28(n_bits, 1, hidden_dim)
        elif latent_size == 7:
            self.encoder = Encoder7(n_bits, 64)
            self.decoder = Decoder7(n_bits, 64)
        else:
            raise ValueError(f"Unsupported latent_size={latent_size}")
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
# FGO (from exp_fgo_v1, only best variants)
# ============================================================================

class FGO_MultiBand(nn.Module):
    """Multi-band frequency gating (3 bands: low/mid/high)."""
    def __init__(self, channels, n_bands=3):
        super().__init__()
        self.channels = channels
        self.n_bands = n_bands
        self.band_gates = nn.Parameter(torch.ones(n_bands, channels))
        self.band_bounds = nn.Parameter(torch.tensor([0.33, 0.67]))

    def forward(self, x):
        B, C, H, W = x.shape
        X_fft = torch.fft.rfft2(x)
        freq_h = torch.fft.fftfreq(H, device=x.device)
        freq_w = torch.fft.rfftfreq(W, device=x.device)
        freq_grid = torch.sqrt(freq_h[:, None]**2 + freq_w[None, :]**2)
        max_freq = freq_grid.max() + 1e-8
        freq_norm = freq_grid / max_freq
        bounds = torch.sigmoid(self.band_bounds).sort()[0]
        gates = torch.sigmoid(self.band_gates)
        mask_low = (freq_norm < bounds[0]).float()
        mask_mid = ((freq_norm >= bounds[0]) & (freq_norm < bounds[1])).float()
        mask_high = (freq_norm >= bounds[1]).float()
        composite = (gates[0].view(C,1,1) * mask_low.unsqueeze(0) +
                     gates[1].view(C,1,1) * mask_mid.unsqueeze(0) +
                     gates[2].view(C,1,1) * mask_high.unsqueeze(0))
        X_filtered = X_fft * composite.unsqueeze(0)
        return torch.fft.irfft2(X_filtered, s=(H, W))


class FGO_Adaptive(nn.Module):
    """Data-adaptive frequency gating (content-aware)."""
    def __init__(self, channels, n_bands=3):
        super().__init__()
        self.channels = channels
        self.n_bands = n_bands
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels), nn.ReLU(),
            nn.Linear(channels, n_bands * channels),
        )
        self.band_bounds = nn.Parameter(torch.tensor([0.33, 0.67]))

    def forward(self, x):
        B, C, H, W = x.shape
        gate_logits = self.gate_net(x)
        gates = torch.sigmoid(gate_logits).view(B, self.n_bands, C)
        X_fft = torch.fft.rfft2(x)
        freq_h = torch.fft.fftfreq(H, device=x.device)
        freq_w = torch.fft.rfftfreq(W, device=x.device)
        freq_grid = torch.sqrt(freq_h[:, None]**2 + freq_w[None, :]**2)
        max_freq = freq_grid.max() + 1e-8
        freq_norm = freq_grid / max_freq
        bounds = torch.sigmoid(self.band_bounds).sort()[0]
        mask_low = (freq_norm < bounds[0]).float()
        mask_mid = ((freq_norm >= bounds[0]) & (freq_norm < bounds[1])).float()
        mask_high = (freq_norm >= bounds[1]).float()
        composite = (gates[:, 0, :, None, None] * mask_low[None, None] +
                     gates[:, 1, :, None, None] * mask_mid[None, None] +
                     gates[:, 2, :, None, None] * mask_high[None, None])
        X_filtered = X_fft * composite
        return torch.fft.irfft2(X_filtered, s=(H, W))


# ============================================================================
# INPAINT NETS
# ============================================================================

class InpaintNetLocal(nn.Module):
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
    """local_conv → FGO → local_conv (sandwich)."""
    def __init__(self, k=8, hidden=64, fgo_type='multiband'):
        super().__init__()
        self.local_pre = nn.Sequential(
            nn.Conv2d(k+1, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
        )
        if fgo_type == 'multiband':
            self.fgo = FGO_MultiBand(hidden)
        elif fgo_type == 'adaptive':
            self.fgo = FGO_Adaptive(hidden)
        self.fgo_proj = nn.Conv2d(hidden, hidden, 1)
        self.local_post = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(hidden, k, 3, padding=1, padding_mode='circular'),
        )
        self.skip = nn.Conv2d(k+1, k, 1)

    def forward(self, z_masked, mask):
        x = torch.cat([z_masked, mask], dim=1)
        h = self.local_pre(x)
        h = h + self.fgo_proj(self.fgo(h))
        return self.local_post(h) + self.skip(x)


# ============================================================================
# TRIGGER MASKS (designed for long-range dependency + preserved evidence)
# ============================================================================

def make_two_block_mask(H=28, W=28):
    """Two 7×7 blocks on opposite corners. Forces cross-grid information flow."""
    m = np.ones((H, W), dtype=np.float32)
    m[2:9, 2:9] = 0      # top-left block
    m[19:26, 19:26] = 0   # bottom-right block
    return m

def make_checkerboard_anchored_mask(H=28, W=28, block_size=4):
    """
    Checkerboard: alternate 4×4 blocks masked/visible.
    Every masked block has adjacent visible blocks → evidence density preserved.
    ~50% occlusion but with guaranteed nearby evidence.
    """
    m = np.ones((H, W), dtype=np.float32)
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            # Checkerboard pattern: mask if (i//block + j//block) is even
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                m[i:min(i+block_size, H), j:min(j+block_size, W)] = 0
    return m

def make_missing_quadrant_mask(H=28, W=28):
    """Bottom-right quadrant entirely missing. Tests long-range completion."""
    m = np.ones((H, W), dtype=np.float32)
    m[H//2:, W//2:] = 0
    return m

def make_center_mask(H=28, W=28):
    """Sanity check only."""
    m = np.ones((H,W), dtype=np.float32); m[7:21, 7:21] = 0; return m

# Training masks (mixture including trigger masks)
def make_random_block_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((H,W), dtype=np.float32)
    oh, ow = rng.integers(6,16), rng.integers(6,16)
    y, x = rng.integers(0, max(1,H-oh+1)), rng.integers(0, max(1,W-ow+1))
    m[y:y+oh, x:x+ow] = 0; return m

def make_random_stripe_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    w = rng.integers(1, 4); g = rng.integers(4, 10); p = rng.integers(0, g)
    m = np.ones((H,W), dtype=np.float32)
    if rng.random() < 0.5:
        for y in range(p, H, g): m[y:min(y+w,H), :] = 0
    else:
        for x in range(p, W, g): m[:, x:min(x+w,W)] = 0
    return m

def make_multi_hole_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((H,W), dtype=np.float32)
    for _ in range(rng.integers(3,8)):
        s = rng.integers(2,6)
        y, x = rng.integers(0, max(1,H-s+1)), rng.integers(0, max(1,W-s+1))
        m[y:y+s, x:x+s] = 0
    return m

def sample_training_mask(H=28, W=28, rng=None):
    """Extended mixture including trigger masks for better coverage."""
    if rng is None: rng = np.random.default_rng()
    p = rng.random()
    if p < 0.20: return make_random_block_mask(H, W, rng)
    elif p < 0.35: return make_center_mask(H, W)
    elif p < 0.50: return make_random_stripe_mask(H, W, rng)
    elif p < 0.65: return make_multi_hole_mask(H, W, rng)
    elif p < 0.75: return make_two_block_mask(H, W)
    elif p < 0.85: return make_checkerboard_anchored_mask(H, W)
    else: return make_missing_quadrant_mask(H, W)

def pixel_to_bit_mask(pixel_mask, n_bits, latent_size):
    H, W = pixel_mask.shape
    if latent_size == H:
        # 1:1 mapping (28×28 grid = 28×28 image)
        bm = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
        bm[:, pixel_mask < 0.5] = True
        return bm
    patch_h, patch_w = H // latent_size, W // latent_size
    bm = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    for i in range(latent_size):
        for j in range(latent_size):
            y0, y1 = i*patch_h, (i+1)*patch_h
            x0, x1 = j*patch_w, (j+1)*patch_w
            if pixel_mask[y0:y1, x0:x1].mean() < 1.0 - 1e-6:
                bm[:, i, j] = True
    return bm


# ============================================================================
# DATA
# ============================================================================

def load_dataset(name, train_n=2000, test_n=500, seed=42):
    from torchvision import datasets, transforms
    ds_map = {'mnist': datasets.MNIST, 'fmnist': datasets.FashionMNIST}
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

def train_model(train_x, train_y, device, latent_size=28, epochs=5, lr=1e-3, batch_size=64):
    n_bits = 4 if latent_size == 28 else 8  # fewer bits for 28×28 to keep params manageable
    hidden = 32 if latent_size == 28 else 64
    model = RouteCModel(n_bits=n_bits, latent_size=latent_size, hidden_dim=hidden).to(device)
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
    k = model.n_bits

    for epoch in range(epochs):
        net.train(); perm = torch.randperm(N); tl, nb = 0., 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]; x = train_x[idx].to(device)
            with torch.no_grad():
                z = model.encode(x); z_hard = (z > 0.5).float()
            B = z_hard.shape[0]
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


def evaluate(model, net, mask_type, test_x, test_y, device, n_samples=100, seed=42):
    model.eval(); net.eval()
    ls = model.latent_size

    mask_fn = {
        'two_block': make_two_block_mask,
        'checkerboard': make_checkerboard_anchored_mask,
        'missing_quadrant': make_missing_quadrant_mask,
        'center': make_center_mask,
    }[mask_type]

    pm = mask_fn()
    occ = 1 - pm
    evidence_density = pm.mean()  # fraction of pixels observed

    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    cb, ca, bce_b, bce_a, rts = [], [], [], [], []

    for idx in eval_idx:
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()
        x_occ = x_clean * pm

        with torch.no_grad():
            x_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_init = model.encode(x_t)[0]
            pred_b = model.classifier(z_init.unsqueeze(0)).argmax(1).item()
            o_hat_b = model.decode(z_init.unsqueeze(0))[0, 0]

        bit_mask = pixel_to_bit_mask(pm, model.n_bits, ls)

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
            l = oh.clamp(1e-6, 1-1e-6)
            return (-(xt*torch.log(l)+(1-xt)*torch.log(1-l))*ot).sum().item() / os_

        cb.append(int(pred_b == label)); ca.append(int(pred_a == label))
        bce_b.append(ob(o_hat_b)); bce_a.append(ob(o_hat_a))
        rts.append(rt)

    n = len(eval_idx)
    bit_mask = pixel_to_bit_mask(pm, model.n_bits, ls)
    return {
        'acc_before': np.mean(cb),
        'acc_after': np.mean(ca),
        'delta_acc': (np.sum(ca) - np.sum(cb)) / n,
        'bce_before': np.mean(bce_b),
        'bce_after': np.mean(bce_a),
        'runtime_ms': np.mean(rts),
        'evidence_density': evidence_density,
        'repair_ratio': bit_mask.any(axis=0).mean(),
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
    parser.add_argument('--output_dir', default='outputs/exp_fgo_trigger')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("FGO-TRIGGER: Testing FGO in its correct operating regime")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print()

    print("[1] Loading dataset...")
    train_x, train_y, test_x, test_y = load_dataset(args.dataset, 2000, 500, args.seed)

    # ---- 28×28 grid (main test: 784 tokens, FGO should matter) ----
    print("\n" + "=" * 80)
    print("  GRID: 28×28 (N=784 tokens) — FGO trigger regime")
    print("=" * 80)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print("\n[2] Training Route C model (latent=28, k=4)...")
    model28 = train_model(train_x, train_y, device, latent_size=28)
    model28.eval()
    with torch.no_grad():
        z = model28.encode(test_x[:500].to(device))
        acc = (model28.classifier(z).argmax(1).cpu() == test_y[:500]).float().mean().item()
    print(f"    Clean probe accuracy: {acc:.1%}")
    print(f"    z shape: {z.shape}")

    k = model28.n_bits
    hidden = 32  # smaller hidden for 28×28 to keep training fast

    variants_28 = {
        'local': InpaintNetLocal(k=k, hidden=hidden).to(device),
        'fgo_multiband': InpaintNetFGO(k=k, hidden=hidden, fgo_type='multiband').to(device),
        'fgo_adaptive': InpaintNetFGO(k=k, hidden=hidden, fgo_type='adaptive').to(device),
    }

    for name, net in variants_28.items():
        print(f"\n[3] Training InpaintNet ({name})...")
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        train_inpaint(model28, train_x, device, net)

    # Trigger masks
    trigger_masks = ['two_block', 'checkerboard', 'missing_quadrant', 'center']

    all_results = []

    print("\n" + "=" * 100)
    print("28×28 RESULTS")
    print("=" * 100)
    print(f"{'variant':<18} {'mask':<20} {'Δacc':>7} {'bce_aft':>8} "
          f"{'evid%':>6} {'repair%':>8} {'ms':>6}")
    print("-" * 80)

    for name, net in variants_28.items():
        for mt in trigger_masks:
            r = evaluate(model28, net, mt, test_x, test_y, device,
                        n_samples=args.eval_samples, seed=args.seed)
            r.update({'variant': name, 'mask_type': mt, 'grid': '28x28'})
            all_results.append(r)
            print(f"{name:<18} {mt:<20} {r['delta_acc']:>+7.1%} "
                  f"{r['bce_after']:>8.3f} {r['evidence_density']:>6.1%} "
                  f"{r['repair_ratio']:>8.1%} {r['runtime_ms']:>6.1f}")

    # ---- FGO gap analysis ----
    print("\n" + "=" * 80)
    print("FGO GAP vs LOCAL-ONLY (28×28)")
    print("=" * 80)

    for mt in trigger_masks:
        local_r = [r for r in all_results if r['variant'] == 'local'
                   and r['mask_type'] == mt and r['grid'] == '28x28']
        if not local_r: continue
        local_dacc = local_r[0]['delta_acc']
        print(f"\n  {mt} (evidence={local_r[0]['evidence_density']:.0%}):")
        print(f"    local:           Δacc={local_dacc:+.1%}")
        for fgo_name in ['fgo_multiband', 'fgo_adaptive']:
            fgo_r = [r for r in all_results if r['variant'] == fgo_name
                     and r['mask_type'] == mt and r['grid'] == '28x28']
            if fgo_r:
                gap = fgo_r[0]['delta_acc'] - local_dacc
                print(f"    {fgo_name:<18} Δacc={fgo_r[0]['delta_acc']:+.1%}  "
                      f"gap={gap:+.1%}")

    # ---- Paradigm verdict ----
    print("\n" + "=" * 80)
    print("PARADIGM VERDICT: Does FGO become necessary at 28×28 with trigger masks?")
    print("=" * 80)

    significant_gaps = []
    for mt in ['two_block', 'checkerboard', 'missing_quadrant']:
        local_r = [r for r in all_results if r['variant'] == 'local'
                   and r['mask_type'] == mt and r['grid'] == '28x28']
        best_gap = -999
        best_fgo = None
        for fn in ['fgo_multiband', 'fgo_adaptive']:
            fgo_r = [r for r in all_results if r['variant'] == fn
                     and r['mask_type'] == mt and r['grid'] == '28x28']
            if fgo_r and local_r:
                gap = fgo_r[0]['delta_acc'] - local_r[0]['delta_acc']
                if gap > best_gap:
                    best_gap = gap
                    best_fgo = fn
        confirmed = best_gap > 0.02
        significant_gaps.append(best_gap)
        verdict = "CONFIRMED" if confirmed else "NOT CONFIRMED"
        print(f"  {mt}: best_fgo={best_fgo}, gap={best_gap:+.1%} → {verdict}")

    any_confirmed = any(g > 0.02 for g in significant_gaps)
    print(f"\n  OVERALL: FGO trigger hypothesis {'CONFIRMED' if any_confirmed else 'NOT CONFIRMED'}")
    if not any_confirmed:
        print("  → Frequency global operator is NOT a critical paradigm component")
        print("  → Focus resources on evidence-driven repair and observation protocol")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "fgo_trigger_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            keys = sorted(set().union(*(r.keys() for r in all_results)))
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in all_results:
                w.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    print("\n" + "=" * 100)
    print("FGO-Trigger experiment complete.")
    print("=" * 100)

    return all_results


if __name__ == "__main__":
    main()
