#!/usr/bin/env python3
"""
Smart Bit-Mask Policy Experiment
==================================
Root cause of stripes failure: `any` policy marks 71% of positions for repair,
but the encoder already handles stripe occlusion (acc_before=70-75%).
Repairing "mostly correct" positions makes things worse.

Solution: Confidence-based bit_mask policy.
Only repair positions where the encoder is UNCERTAIN about its encoding.

Three policies tested:
1. `any` (current): repair if ANY pixel in patch is occluded → bit_ratio ~0.71
2. `majority`: repair if >50% of patch pixels are occluded → bit_ratio ~0.30
3. `confidence`: repair only where local_pred disagrees with encoder → adaptive

The confidence policy uses E_core's local predictor:
  For each position, compare encoder's z_i with local_pred's prediction.
  If they agree → encoder is consistent with neighborhood → don't repair.
  If they disagree → encoder may be corrupted → repair.

This is a principled approach: we repair positions that violate E_core
local consistency, not positions that have ANY pixel occlusion.

Usage:
    python3 -u benchmarks/exp_smart_mask.py --device cuda --seed 42
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
# MODEL
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
# BIT MASK POLICIES
# ============================================================================

def pixel_to_bit_mask_any(pixel_mask, n_bits=8, latent_size=7):
    """Any occlusion → repair. Current default. Aggressive."""
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
    """Majority occluded → repair. Less aggressive."""
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
    """
    Confidence-based: repair only where local_pred DISAGREES with encoder.

    For each position:
    1. Check if ANY pixel is occluded (prerequisite)
    2. If yes, compute local_pred's prediction from neighbors
    3. If local_pred disagrees with encoder on >threshold fraction of bits → repair
    4. Otherwise → trust the encoder, don't repair

    This is principled: we repair E_core violations, not pixel occlusions.
    """
    # First get the pixel-based candidates (any policy)
    pixel_candidates = pixel_to_bit_mask_any(pixel_mask, n_bits, latent_size)

    if device is None:
        device = z.device

    with torch.no_grad():
        z_t = z.unsqueeze(0)  # (1, k, H, W)
        z_hard = (z_t > 0.5).float()
        core_logits = model.local_pred(z_t)  # (1, k, H, W)
        core_pred = (core_logits > 0).float()  # local_pred's prediction

        # Per-position disagreement: fraction of bits where encoder ≠ local_pred
        disagree = (z_hard != core_pred).float()  # (1, k, H, W)
        disagree_rate = disagree[0].mean(dim=0)  # (H, W) — avg over k bits

    bm = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    for i in range(latent_size):
        for j in range(latent_size):
            # Must be pixel-occluded AND encoder-inconsistent
            if pixel_candidates[0, i, j] and disagree_rate[i, j].item() > disagreement_threshold:
                bm[:, i, j] = True

    return bm


def pixel_to_bit_mask_hybrid(pixel_mask, z, model, n_bits=8, latent_size=7,
                              majority_threshold=0.3, disagree_threshold=0.4,
                              device=None):
    """
    Hybrid policy: repair if EITHER heavily occluded OR encoder-inconsistent.

    - Heavily occluded (>70% pixels masked): always repair (data too corrupted)
    - Lightly occluded + consistent: don't repair (encoder handled it)
    - Lightly occluded + inconsistent: repair (encoder confused)

    This handles both center (heavy occlusion → repair) and stripes (light → selective).
    """
    patch_size = 28 // latent_size

    if device is None:
        device = z.device

    with torch.no_grad():
        z_t = z.unsqueeze(0)
        z_hard = (z_t > 0.5).float()
        core_logits = model.local_pred(z_t)
        core_pred = (core_logits > 0).float()
        disagree = (z_hard != core_pred).float()
        disagree_rate = disagree[0].mean(dim=0)  # (H, W)

    bm = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    for i in range(latent_size):
        for j in range(latent_size):
            y0, y1 = i * patch_size, (i+1) * patch_size
            x0, x1 = j * patch_size, (j+1) * patch_size
            visible_ratio = pixel_mask[y0:y1, x0:x1].mean()

            if visible_ratio >= 1.0 - 1e-6:
                # Fully visible → never repair
                continue
            elif visible_ratio < majority_threshold:
                # Heavily occluded → always repair
                bm[:, i, j] = True
            else:
                # Lightly occluded → repair only if inconsistent
                if disagree_rate[i, j].item() > disagree_threshold:
                    bm[:, i, j] = True

    return bm


POLICIES = {
    'any': lambda pm, z, model, dev: pixel_to_bit_mask_any(pm),
    'majority': lambda pm, z, model, dev: pixel_to_bit_mask_majority(pm),
    'confidence': lambda pm, z, model, dev: pixel_to_bit_mask_confidence(pm, z, model, device=dev),
    'hybrid': lambda pm, z, model, dev: pixel_to_bit_mask_hybrid(pm, z, model, device=dev),
}


# ============================================================================
# MASKS + NOISE + DATA
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
    """Mask mixture training (best config from GDA v2 experiments)."""
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
    k, H, W = z_init.shape
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


def evaluate_policy(model, inpaint_net, policy_name, mask_type, noise_type,
                    test_x, test_y, device, n_samples=100, seed=42):
    model.eval(); inpaint_net.eval()
    pm = make_center_mask() if mask_type == 'center' else make_stripe_mask()
    occ = 1 - pm

    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    cb, ca, bce_b, bce_a, mse_b, mse_a, rts = [], [], [], [], [], [], []
    bit_ratios = []

    policy_fn = POLICIES[policy_name]

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

        # Apply policy to get bit_mask
        bit_mask = policy_fn(pm, z_init, model, device)
        bit_ratios.append(float(bit_mask[0].mean()))

        # If nothing to repair, skip inference
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
        mse_b.append(om(o_hat_b)); mse_a.append(om(o_hat_a))
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
    parser.add_argument('--output_dir', default='outputs/exp_smart_mask')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("SMART BIT-MASK POLICY EXPERIMENT")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Policies: any, majority, confidence, hybrid")
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
    policies = ['any', 'majority', 'confidence', 'hybrid']

    print(f"\n[4] Evaluating {len(policies)} policies × {len(configs)} configs...")
    all_results = []

    for mt, nt in configs:
        print(f"\n  === {mt} + {nt} ===")
        for pol in policies:
            r = evaluate_policy(model, inpaint_net, pol, mt, nt,
                               test_x, test_y, device,
                               n_samples=args.eval_samples, seed=args.seed)
            all_results.append(r)
            print(f"    {pol:<12}: Δacc={r['delta_acc']:+.1%}, "
                  f"bit_ratio={r['bit_mask_ratio']:.2f}, "
                  f"bce={r['bce_after']:.2f}, t={r['runtime_ms']:.1f}ms")

    # Summary
    print("\n" + "=" * 110)
    print("FULL SUMMARY")
    print("=" * 110)
    header = (f"{'policy':<12} {'mask':<10} {'noise':<8} "
              f"{'acc_bef':>7} {'acc_aft':>7} {'Δacc':>7} "
              f"{'bit_r':>6} {'bce_aft':>8} {'ms':>6}")
    print(header)
    print("-" * 110)
    for r in all_results:
        print(f"{r['policy']:<12} {r['mask_type']:<10} {r['noise_type']:<8} "
              f"{r['acc_before']:>7.1%} {r['acc_after']:>7.1%} {r['delta_acc']:>+7.1%} "
              f"{r['bit_mask_ratio']:>6.2f} {r['bce_after']:>8.2f} {r['runtime_ms']:>6.1f}")

    # Key: stripes
    print("\n" + "=" * 80)
    print("KEY: STRIPES (sorted by Δacc)")
    print("=" * 80)
    for nt in ['clean', 'noise']:
        print(f"\n  stripes + {nt}:")
        stripe_r = [(r['policy'], r['delta_acc'], r['bit_mask_ratio'])
                    for r in all_results
                    if r['mask_type'] == 'stripes' and r['noise_type'] == nt]
        stripe_r.sort(key=lambda x: -x[1])
        for pol, da, br in stripe_r:
            s = "+" if da > 0 else ("~" if da > -0.03 else "-")
            print(f"    [{s}] {pol:<12}: Δacc={da:+.1%}, bit_ratio={br:.2f}")

    # Key: center (should not regress)
    print("\n" + "=" * 80)
    print("CENTER (should not regress)")
    print("=" * 80)
    for nt in ['clean', 'noise']:
        print(f"\n  center + {nt}:")
        center_r = [(r['policy'], r['delta_acc'], r['bit_mask_ratio'])
                    for r in all_results
                    if r['mask_type'] == 'center' and r['noise_type'] == nt]
        center_r.sort(key=lambda x: -x[1])
        for pol, da, br in center_r:
            print(f"    {pol:<12}: Δacc={da:+.1%}, bit_ratio={br:.2f}")

    # Hypothesis test
    print("\n" + "=" * 80)
    print("HYPOTHESIS: Smart policy can make stripes non-negative")
    print("=" * 80)
    for nt in ['clean', 'noise']:
        best = max(
            [(r['policy'], r['delta_acc']) for r in all_results
             if r['mask_type'] == 'stripes' and r['noise_type'] == nt],
            key=lambda x: x[1]
        )
        status = "CONFIRMED" if best[1] >= 0 else f"NOT YET (best={best[0]} at {best[1]:+.1%})"
        print(f"  stripes+{nt}: {status}")

    # Save
    csv_path = os.path.join(args.output_dir, "smart_mask_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    print("\n" + "=" * 100)
    print("Smart mask experiment complete.")
    print("=" * 100)

    return all_results


if __name__ == "__main__":
    main()
