#!/usr/bin/env python3
"""
Adaptive Bit-Mask Policy Experiment
=====================================
Smart mask results showed a clear trade-off:
  - `any` policy: best for center (+12-13%), worst for stripes (-17%)
  - `majority`/`confidence`: fix stripes (Δacc=0%), but lose center gains

This experiment implements an ADAPTIVE policy that auto-selects based on
the observed mask characteristics (pixel occlusion ratio).

Key idea: the pixel_mask itself tells us what kind of occlusion we have.
  - Heavy occlusion (>40% pixels masked) → use `any` (aggressive repair needed)
  - Light occlusion (<40% pixels masked) → use `confidence` (selective repair)

The threshold is tuned to separate center (~25% visible in center region) from
stripes (~83% visible overall).

Additional experiment: per-position adaptive threshold for hybrid policy.

Usage:
    python3 -u benchmarks/exp_adaptive_policy.py --device cuda --seed 42
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
# MODEL (same as exp_smart_mask.py)
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
    def set_temperature(self, tau):
        self.temperature = tau

class Encoder(nn.Module):
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1),
        )
    def forward(self, x): return self.conv(x)

class Decoder(nn.Module):
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
    def __init__(self, n_bits=8, hidden_dim=64, energy_hidden=32,
                 latent_size=7, tau=1.0):
        super().__init__()
        self.n_bits = n_bits
        self.latent_size = latent_size
        self.encoder = Encoder(n_bits, hidden_dim)
        self.quantizer = GumbelSigmoid(tau)
        self.decoder = Decoder(n_bits, hidden_dim)
        self.local_pred = LocalPredictor(n_bits, energy_hidden)
        self.classifier = Classifier(n_bits, latent_size)
    def encode(self, x): return self.quantizer(self.encoder(x))
    def decode(self, z): return self.decoder(z)
    def set_temperature(self, tau): self.quantizer.set_temperature(tau)
    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z), self.classifier(z), self.local_pred(z)


# ============================================================================
# INPAINTING
# ============================================================================

class InpaintNet(nn.Module):
    def __init__(self, k=8, hidden=64):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Conv2d(k+1, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(hidden, k, 3, padding=1, padding_mode='circular'),
        )
        self.skip = nn.Conv2d(k+1, k, 1)
    def forward(self, z_masked, mask):
        x = torch.cat([z_masked, mask], dim=1)
        return self.net(x) + self.skip(x)


# ============================================================================
# BIT MASK POLICIES (from exp_smart_mask.py + new adaptive)
# ============================================================================

def pixel_to_bit_mask_any(pixel_mask, n_bits=8, latent_size=7):
    """Any occlusion → repair. Aggressive."""
    patch_size = 28 // latent_size
    bm = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    for i in range(latent_size):
        for j in range(latent_size):
            y0, y1 = i * patch_size, (i+1) * patch_size
            x0, x1 = j * patch_size, (j+1) * patch_size
            if pixel_mask[y0:y1, x0:x1].mean() < 1.0 - 1e-6:
                bm[:, i, j] = True
    return bm


def pixel_to_bit_mask_majority(pixel_mask, n_bits=8, latent_size=7):
    """Majority occluded → repair."""
    patch_size = 28 // latent_size
    bm = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    for i in range(latent_size):
        for j in range(latent_size):
            y0, y1 = i * patch_size, (i+1) * patch_size
            x0, x1 = j * patch_size, (j+1) * patch_size
            if pixel_mask[y0:y1, x0:x1].mean() < 0.5:
                bm[:, i, j] = True
    return bm


def pixel_to_bit_mask_confidence(pixel_mask, z, model, n_bits=8, latent_size=7,
                                  disagreement_threshold=0.5, device=None):
    """Repair only where local_pred DISAGREES with encoder."""
    pixel_candidates = pixel_to_bit_mask_any(pixel_mask, n_bits, latent_size)
    with torch.no_grad():
        z_t = z.unsqueeze(0)
        z_hard = (z_t > 0.5).float()
        core_logits = model.local_pred(z_t)
        core_pred = (core_logits > 0).float()
        disagree = (z_hard != core_pred).float()
        disagree_rate = disagree[0].mean(dim=0)

    bm = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    for i in range(latent_size):
        for j in range(latent_size):
            if pixel_candidates[0, i, j] and disagree_rate[i, j].item() > disagreement_threshold:
                bm[:, i, j] = True
    return bm


def pixel_to_bit_mask_adaptive(pixel_mask, z, model, n_bits=8, latent_size=7,
                                occlusion_threshold=0.40, disagree_threshold=0.5,
                                device=None):
    """
    ADAPTIVE policy: auto-select based on global pixel occlusion ratio.

    - If overall occlusion > threshold → heavy mask → use `any` (aggressive)
    - If overall occlusion <= threshold → light mask → use `confidence` (selective)

    The threshold is set to separate:
      center: ~50% pixels masked → heavy → any
      stripes: ~17% pixels masked → light → confidence
    """
    # Global occlusion ratio
    occlusion_ratio = 1.0 - pixel_mask.mean()

    if occlusion_ratio > occlusion_threshold:
        # Heavy occlusion (center-like) → aggressive repair
        return pixel_to_bit_mask_any(pixel_mask, n_bits, latent_size)
    else:
        # Light occlusion (stripes-like) → selective repair
        return pixel_to_bit_mask_confidence(pixel_mask, z, model, n_bits, latent_size,
                                             disagreement_threshold=disagree_threshold,
                                             device=device)


def pixel_to_bit_mask_adaptive_per_pos(pixel_mask, z, model, n_bits=8, latent_size=7,
                                        heavy_threshold=0.3, disagree_threshold=0.4,
                                        device=None):
    """
    Per-position adaptive: each latent position decides independently.

    For each position (i,j):
    - If patch occlusion > heavy_threshold → always repair (data too corrupted)
    - If patch occlusion > 0 AND local_pred disagrees → repair (encoder confused)
    - Otherwise → don't repair

    This is like `hybrid` from exp_smart_mask but with tuned thresholds.
    """
    patch_size = 28 // latent_size

    with torch.no_grad():
        z_t = z.unsqueeze(0)
        z_hard = (z_t > 0.5).float()
        core_logits = model.local_pred(z_t)
        core_pred = (core_logits > 0).float()
        disagree = (z_hard != core_pred).float()
        disagree_rate = disagree[0].mean(dim=0)

    bm = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    for i in range(latent_size):
        for j in range(latent_size):
            y0, y1 = i * patch_size, (i+1) * patch_size
            x0, x1 = j * patch_size, (j+1) * patch_size
            visible_ratio = pixel_mask[y0:y1, x0:x1].mean()

            if visible_ratio >= 1.0 - 1e-6:
                continue  # Fully visible → never repair
            elif visible_ratio < heavy_threshold:
                bm[:, i, j] = True  # Heavy occlusion → always repair
            elif disagree_rate[i, j].item() > disagree_threshold:
                bm[:, i, j] = True  # Light + inconsistent → repair

    return bm


def pixel_to_bit_mask_adaptive_sweep(pixel_mask, z, model, n_bits=8, latent_size=7,
                                      occlusion_threshold=0.40, device=None):
    """Adaptive with sweep-specific threshold (for threshold ablation)."""
    occlusion_ratio = 1.0 - pixel_mask.mean()
    if occlusion_ratio > occlusion_threshold:
        return pixel_to_bit_mask_any(pixel_mask, n_bits, latent_size)
    else:
        return pixel_to_bit_mask_confidence(pixel_mask, z, model, n_bits, latent_size,
                                             device=device)


# ============================================================================
# MASKS + DATA
# ============================================================================

def make_center_mask(H=28, W=28):
    m = np.ones((H,W), dtype=np.float32)
    m[7:21, 7:21] = 0
    return m

def make_stripe_mask(H=28, W=28):
    m = np.ones((H,W), dtype=np.float32)
    for y in range(0, H, 6):
        m[y:min(y+2,H), :] = 0
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
    m[y:y+oh, x:x+ow] = 0
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
    if rng is None: rng = np.random.default_rng()
    p = rng.random()
    if p < 0.4: return make_random_block_mask(H,W,rng)
    elif p < 0.6: return make_center_mask(H,W)
    elif p < 0.8: return make_random_stripe_mask(H,W,rng)
    else: return make_multi_hole_mask(H,W,rng)

def apply_noise(image, noise_type, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    if noise_type == 'noise':
        return np.clip(image + rng.normal(0, 0.1, image.shape).astype(np.float32), 0, 1)
    return image.copy()

def load_data(train_n=2000, test_n=1000, seed=42):
    from torchvision import datasets, transforms
    tr = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    te = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    ti, si = rng.choice(len(tr), train_n, replace=False), rng.choice(len(te), test_n, replace=False)
    return (torch.stack([tr[i][0] for i in ti]), torch.tensor([tr[i][1] for i in ti]),
            torch.stack([te[i][0] for i in si]), torch.tensor([te[i][1] for i in si]))


# ============================================================================
# TRAINING
# ============================================================================

def train_model(train_x, train_y, device, epochs=5, lr=1e-3, batch_size=64):
    model = RouteCModel().to(device)
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
        print(f"    Epoch {epoch+1}/{epochs}: loss={el/max(nb,1):.4f}")
    # Probe
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


def train_inpaint_mixture(model, train_x, device, epochs=20, batch_size=64, lr=1e-3):
    """Mask mixture training."""
    net = InpaintNet(k=model.n_bits).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    model.eval()
    N = len(train_x)
    rng = np.random.default_rng(42)

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
                bm = pixel_to_bit_mask_any(pm, k, H)
                masks.append(torch.from_numpy(bm).float())
            bit_masks = torch.stack(masks).to(device)
            pos_masks = bit_masks[:, 0:1, :, :]
            z_masked = z_hard * (1 - bit_masks)
            logits = net(z_masked, pos_masks)
            loss = F.binary_cross_entropy_with_logits(logits[bit_masks.bool()], z_hard[bit_masks.bool()])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        if (epoch+1) % 5 == 0:
            print(f"      epoch {epoch+1}/{epochs}: loss={tl/max(nb,1):.4f}")
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


def evaluate_policy(model, inpaint_net, policy_name, policy_fn,
                    mask_type, noise_type, test_x, test_y, device,
                    n_samples=100, seed=42):
    model.eval(); inpaint_net.eval()
    pm = make_center_mask() if mask_type == 'center' else make_stripe_mask()
    occ = 1 - pm

    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    cb, ca, bce_b, bce_a, mse_a, rts, bit_ratios = [], [], [], [], [], [], []

    for idx in eval_idx:
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()
        x_noisy = apply_noise(x_clean, noise_type, rng)
        x_occ = x_noisy * pm

        with torch.no_grad():
            x_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_init = model.encode(x_t)[0]
            pred_b = model.classifier(z_init.unsqueeze(0)).argmax(1).item()
            o_hat_b = model.decode(z_init.unsqueeze(0))[0, 0]

        bit_mask = policy_fn(pm, z_init, model, device)
        bit_ratios.append(float(bit_mask[0].mean()))

        if not bit_mask.any():
            pred_a = pred_b
            o_hat_a = o_hat_b
            rt = 0.0
        else:
            t0 = time.time()
            z_final = amortized_inpaint(inpaint_net, z_init, bit_mask, device)
            rt = (time.time() - t0) * 1000
            with torch.no_grad():
                pred_a = model.classifier(z_final.unsqueeze(0)).argmax(1).item()
                o_hat_a = model.decode(z_final.unsqueeze(0))[0, 0]

        xt = torch.from_numpy(x_clean).to(device)
        ot = torch.from_numpy(occ).to(device)
        os_ = ot.sum().clamp(min=1.0).item()

        def om(oh):
            d = (oh-xt)*ot; return (d*d).sum().item()/os_
        def ob(oh):
            l = oh.clamp(1e-6,1-1e-6)
            return (-(xt*torch.log(l)+(1-xt)*torch.log(1-l))*ot).sum().item()/os_

        cb.append(int(pred_b == label)); ca.append(int(pred_a == label))
        mse_a.append(om(o_hat_a))
        bce_b.append(ob(o_hat_b)); bce_a.append(ob(o_hat_a))
        rts.append(rt)

    cb, ca = np.array(cb), np.array(ca)
    n = len(eval_idx)
    return {
        'policy': policy_name,
        'mask_type': mask_type,
        'noise_type': noise_type,
        'acc_before': cb.mean(),
        'acc_after': ca.mean(),
        'delta_acc': (ca.sum() - cb.sum()) / n,
        'bce_before': np.mean(bce_b),
        'bce_after': np.mean(bce_a),
        'mse_after': np.mean(mse_a),
        'runtime_ms': np.mean(rts),
        'bit_mask_ratio': np.mean(bit_ratios),
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
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--inpaint_epochs', type=int, default=20)
    parser.add_argument('--output_dir', default='outputs/exp_adaptive_policy')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("ADAPTIVE BIT-MASK POLICY EXPERIMENT")
    print("=" * 100)
    print(f"Device: {device}")
    print()

    # Check mask occlusion ratios
    center_occ = 1 - make_center_mask().mean()
    stripe_occ = 1 - make_stripe_mask().mean()
    print(f"[0] Mask occlusion ratios:")
    print(f"    Center: {center_occ:.2%} occluded")
    print(f"    Stripes: {stripe_occ:.2%} occluded")
    print(f"    → Threshold at 0.40 cleanly separates them")
    print()

    print("[1] Loading data...")
    train_x, train_y, test_x, test_y = load_data(2000, 1000, args.seed)

    print("\n[2] Training Route C model (BCE, no L_cls)...")
    model = train_model(train_x, train_y, device, epochs=args.epochs)
    model.eval()
    with torch.no_grad():
        z = model.encode(test_x[:500].to(device))
        acc = (model.classifier(z).argmax(1).cpu() == test_y[:500]).float().mean().item()
    print(f"    Clean accuracy: {acc:.1%}")

    print("\n[3] Training InpaintNet (mask mixture)...")
    inpaint_net = train_inpaint_mixture(model, train_x, device, epochs=args.inpaint_epochs)

    configs = [
        ('center', 'clean'), ('center', 'noise'),
        ('stripes', 'clean'), ('stripes', 'noise'),
    ]

    # ======================================================================
    # Part A: Baselines + Adaptive policy (fixed threshold 0.40)
    # ======================================================================
    print("\n" + "=" * 100)
    print("PART A: Baselines + Adaptive Policy")
    print("=" * 100)

    policies = {
        'any': lambda pm, z, model, dev: pixel_to_bit_mask_any(pm),
        'confidence': lambda pm, z, model, dev: pixel_to_bit_mask_confidence(pm, z, model, device=dev),
        'adaptive_0.40': lambda pm, z, model, dev: pixel_to_bit_mask_adaptive(pm, z, model, occlusion_threshold=0.40, device=dev),
        'adaptive_pp': lambda pm, z, model, dev: pixel_to_bit_mask_adaptive_per_pos(pm, z, model, device=dev),
    }

    all_results = []
    for mt, nt in configs:
        print(f"\n  === {mt} + {nt} ===")
        for pname, pfn in policies.items():
            r = evaluate_policy(model, inpaint_net, pname, pfn,
                               mt, nt, test_x, test_y, device,
                               n_samples=args.eval_samples, seed=args.seed)
            all_results.append(r)
            print(f"    {pname:<16}: Δacc={r['delta_acc']:+.1%}, "
                  f"bit_ratio={r['bit_mask_ratio']:.3f}, "
                  f"bce={r['bce_after']:.2f}, t={r['runtime_ms']:.1f}ms")

    # ======================================================================
    # Part B: Threshold sweep for adaptive policy
    # ======================================================================
    print("\n" + "=" * 100)
    print("PART B: Adaptive Threshold Sweep")
    print("=" * 100)

    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
    sweep_results = []

    for thresh in thresholds:
        pfn = lambda pm, z, model, dev, t=thresh: pixel_to_bit_mask_adaptive_sweep(
            pm, z, model, occlusion_threshold=t, device=dev)
        pname = f"adapt_{thresh:.2f}"

        row = {'threshold': thresh}
        for mt, nt in configs:
            r = evaluate_policy(model, inpaint_net, pname, pfn,
                               mt, nt, test_x, test_y, device,
                               n_samples=args.eval_samples, seed=args.seed)
            row[f'{mt}_{nt}_dacc'] = r['delta_acc']
            row[f'{mt}_{nt}_ratio'] = r['bit_mask_ratio']

        # Compute aggregate score: sum of Δacc across all configs
        row['total_dacc'] = sum(row[f'{mt}_{nt}_dacc'] for mt, nt in configs)
        sweep_results.append(row)
        print(f"  thresh={thresh:.2f}: "
              f"center_c={row['center_clean_dacc']:+.1%} "
              f"center_n={row['center_noise_dacc']:+.1%} "
              f"stripe_c={row['stripes_clean_dacc']:+.1%} "
              f"stripe_n={row['stripes_noise_dacc']:+.1%} "
              f"TOTAL={row['total_dacc']:+.1%}")

    # Find best threshold
    best = max(sweep_results, key=lambda x: x['total_dacc'])
    print(f"\n  Best threshold: {best['threshold']:.2f} (total Δacc={best['total_dacc']:+.1%})")

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 110)
    print("FULL SUMMARY")
    print("=" * 110)
    header = (f"{'policy':<18} {'mask':<10} {'noise':<8} "
              f"{'acc_bef':>7} {'acc_aft':>7} {'Δacc':>7} "
              f"{'bit_r':>6} {'bce_aft':>8} {'ms':>6}")
    print(header)
    print("-" * 110)
    for r in all_results:
        print(f"{r['policy']:<18} {r['mask_type']:<10} {r['noise_type']:<8} "
              f"{r['acc_before']:>7.1%} {r['acc_after']:>7.1%} {r['delta_acc']:>+7.1%} "
              f"{r['bit_mask_ratio']:>6.3f} {r['bce_after']:>8.2f} {r['runtime_ms']:>6.1f}")

    # Key comparison
    print("\n" + "=" * 80)
    print("KEY: Does adaptive beat both `any` and `confidence`?")
    print("=" * 80)
    for mt, nt in configs:
        any_r = [r for r in all_results if r['policy']=='any' and r['mask_type']==mt and r['noise_type']==nt][0]
        conf_r = [r for r in all_results if r['policy']=='confidence' and r['mask_type']==mt and r['noise_type']==nt][0]
        adpt_r = [r for r in all_results if r['policy']=='adaptive_0.40' and r['mask_type']==mt and r['noise_type']==nt][0]
        pp_r = [r for r in all_results if r['policy']=='adaptive_pp' and r['mask_type']==mt and r['noise_type']==nt][0]

        best_fixed = max(any_r['delta_acc'], conf_r['delta_acc'])
        best_adpt = max(adpt_r['delta_acc'], pp_r['delta_acc'])
        winner = "ADAPTIVE" if best_adpt >= best_fixed else "FIXED"

        print(f"\n  {mt}+{nt}:")
        print(f"    any:          Δacc={any_r['delta_acc']:+.1%}")
        print(f"    confidence:   Δacc={conf_r['delta_acc']:+.1%}")
        print(f"    adaptive_0.40: Δacc={adpt_r['delta_acc']:+.1%}")
        print(f"    adaptive_pp:  Δacc={pp_r['delta_acc']:+.1%}")
        print(f"    → Winner: {winner}")

    # Pareto check
    print("\n" + "=" * 80)
    print("PARETO CHECK: center + stripes combined Δacc")
    print("=" * 80)
    for pol in policies.keys():
        center_avg = np.mean([r['delta_acc'] for r in all_results
                              if r['policy']==pol and r['mask_type']=='center'])
        stripe_avg = np.mean([r['delta_acc'] for r in all_results
                              if r['policy']==pol and r['mask_type']=='stripes'])
        total = center_avg + stripe_avg
        print(f"  {pol:<18}: center_avg={center_avg:+.1%}, stripe_avg={stripe_avg:+.1%}, total={total:+.1%}")

    # Save results
    csv_path = os.path.join(args.output_dir, "adaptive_policy_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    # Save sweep
    sweep_path = os.path.join(args.output_dir, "threshold_sweep.csv")
    with open(sweep_path, 'w', newline='') as f:
        if sweep_results:
            w = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
            w.writeheader()
            for r in sweep_results: w.writerow(r)
    print(f"Sweep CSV saved to {sweep_path}")

    print("\n" + "=" * 100)
    print("Adaptive policy experiment complete.")
    print("=" * 100)

    return all_results, sweep_results


if __name__ == "__main__":
    main()
