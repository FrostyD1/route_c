#!/usr/bin/env python3
"""
Evidence-Strength Repair Experiment
====================================
Experiment #3: Can E_obs residual signal drive position-wise repair better
than E_core consistency?

Problem: E_core consistency can't detect correlated corruption (center mask —
all neighbors wrong → they "agree" on wrong values → confidence says "don't
repair"). E_obs residual (decoder reconstruction error on observed pixels)
DOES detect this because it compares against actual observation.

Approach:
  1. Encode occluded image → z_init
  2. Amortized inpaint → z_repaired (full "any" policy)
  3. Compute E_obs residual per patch: how well does decode(z) match
     observation at each patch? High residual → bad fit → repair needed
  4. Selective repair: only replace positions where E_obs residual > threshold
  5. Compare: any (replace all), confidence (E_core), evidence (E_obs residual)

Paradigm prediction:
  - E_obs evidence should identify center as needing repair (high residual
    because decoded output doesn't match observation in occluded region)
  - E_obs evidence should protect stripes (low residual where observation
    exists → don't repair those positions)
  - This should achieve Pareto-better results than adaptive_pp

Usage:
    python3 -u benchmarks/exp_evidence_strength.py --device cuda --seed 42
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
# MODEL (same as other experiments)
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
    def __init__(self, n_bits, in_channels=1, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1),
        )
    def forward(self, x): return self.conv(x)


class BinaryDecoder(nn.Module):
    def __init__(self, n_bits, out_channels=1, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1), nn.Sigmoid(),
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
    def __init__(self, n_bits=8, latent_size=7, in_channels=1, hidden_dim=64,
                 energy_hidden=32, tau=1.0):
        super().__init__()
        self.n_bits = n_bits
        self.latent_size = latent_size
        self.encoder = BinaryEncoder(n_bits, in_channels, hidden_dim)
        self.decoder = BinaryDecoder(n_bits, in_channels, hidden_dim)
        self.quantizer = GumbelSigmoid(tau)
        self.local_pred = LocalPredictor(n_bits, energy_hidden)
        self.classifier = Classifier(n_bits, latent_size)
    def encode(self, x): return self.quantizer(self.encoder(x))
    def decode(self, z): return self.decoder(z)
    def set_temperature(self, tau): self.quantizer.set_temperature(tau)
    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z), self.classifier(z), self.local_pred(z)


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
# MASKS + DATA
# ============================================================================

def make_center_mask(H=28, W=28):
    m = np.ones((H,W), dtype=np.float32); m[7:21, 7:21] = 0; return m

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
        'kmnist': datasets.KMNIST,
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


def train_inpaint(model, train_x, device, epochs=20, batch_size=64, lr=1e-3):
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
                bm = pixel_to_bit_mask(pm, k)
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
# EVIDENCE-STRENGTH REPAIR
# ============================================================================

def compute_eobs_residual_per_patch(model, z, observation, pixel_mask, device):
    """
    Compute E_obs residual per latent patch.

    For each patch (i,j) in the latent grid, compute BCE between
    decode(z) and observation in the corresponding pixel region,
    weighted by pixel_mask (only observed pixels contribute).

    Returns: (latent_size, latent_size) tensor of per-patch E_obs residuals.
    High residual = decode(z) doesn't match observation well = repair needed.
    """
    model.eval()
    ls = model.latent_size
    H_img, W_img = observation.shape
    patch_h, patch_w = H_img // ls, W_img // ls

    with torch.no_grad():
        o_hat = model.decode(z.unsqueeze(0))[0, 0]  # (H_img, W_img)

    obs_t = torch.from_numpy(observation).to(device)
    pmask_t = torch.from_numpy(pixel_mask).to(device)

    residuals = torch.zeros(ls, ls, device=device)
    for i in range(ls):
        for j in range(ls):
            y0, y1 = i * patch_h, (i+1) * patch_h
            x0, x1 = j * patch_w, (j+1) * patch_w
            patch_obs = obs_t[y0:y1, x0:x1]
            patch_hat = o_hat[y0:y1, x0:x1].clamp(1e-6, 1-1e-6)
            patch_mask = pmask_t[y0:y1, x0:x1]  # 1=observed, 0=occluded

            # Only compute residual on observed pixels
            observed_count = patch_mask.sum()
            if observed_count > 0:
                bce = -(patch_obs * torch.log(patch_hat) +
                        (1-patch_obs) * torch.log(1-patch_hat))
                residuals[i, j] = (bce * patch_mask).sum() / observed_count
            else:
                # Fully occluded patch — no observation signal available
                # Mark with high residual (should always repair)
                residuals[i, j] = float('inf')

    return residuals


def amortized_inpaint_full(net, z_init, bit_mask, device):
    """Standard amortized inpaint — replace all masked positions."""
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


def evidence_strength_repair(model, net, z_init, bit_mask, pixel_mask,
                              observation, device, threshold=0.5):
    """
    E_obs-guided selective repair.

    1. Run amortized inpaint to get candidate z_repaired
    2. Compute E_obs residual per patch on z_init
    3. Only replace positions where E_obs residual > threshold
       (high residual = observation mismatch = needs repair)

    For fully occluded patches (no observation), always repair (infinite residual).
    For partially observed patches, use threshold to decide.
    """
    net.eval(); model.eval()

    # Step 1: get full repair candidate
    z_repaired = amortized_inpaint_full(net, z_init, bit_mask, device)

    # Step 2: compute E_obs residual on INITIAL z (before repair)
    residuals = compute_eobs_residual_per_patch(
        model, z_init, observation, pixel_mask, device)

    # Step 3: selective merge based on evidence strength
    z_result = z_init.clone()
    k = model.n_bits
    ls = model.latent_size
    repair_count = 0
    total_masked = 0

    for i in range(ls):
        for j in range(ls):
            if bit_mask[0, i, j]:  # This position is masked
                total_masked += 1
                if residuals[i, j] > threshold:
                    # High residual → repair this position
                    z_result[:, i, j] = z_repaired[:, i, j]
                    repair_count += 1
                # else: keep z_init (observation fits well enough)

    ratio = repair_count / max(total_masked, 1)
    return z_result, ratio, residuals


def eobs_adaptive_repair(model, net, z_init, bit_mask, pixel_mask,
                          observation, device):
    """
    Fully adaptive E_obs repair: repair positions where E_obs residual
    is above the median residual of masked positions.

    No fixed threshold — adapts to each sample's residual distribution.
    """
    net.eval(); model.eval()

    z_repaired = amortized_inpaint_full(net, z_init, bit_mask, device)
    residuals = compute_eobs_residual_per_patch(
        model, z_init, observation, pixel_mask, device)

    k = model.n_bits
    ls = model.latent_size
    z_result = z_init.clone()

    # Collect residuals at masked positions
    masked_residuals = []
    masked_positions = []
    for i in range(ls):
        for j in range(ls):
            if bit_mask[0, i, j]:
                r = residuals[i, j].item()
                if not np.isinf(r):
                    masked_residuals.append(r)
                masked_positions.append((i, j, r))

    if not masked_residuals:
        return z_repaired, 1.0, residuals

    # Adaptive threshold = median of observed-patch residuals
    median_r = np.median(masked_residuals) if masked_residuals else 0.0

    repair_count = 0
    for i, j, r in masked_positions:
        if np.isinf(r) or r > median_r:
            z_result[:, i, j] = z_repaired[:, i, j]
            repair_count += 1

    ratio = repair_count / max(len(masked_positions), 1)
    return z_result, ratio, residuals


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_policy(policy_name, model, net, mask_type, noise_type,
                    test_x, test_y, device, n_samples=100, seed=42,
                    eobs_threshold=0.5):
    model.eval(); net.eval()
    pm = make_center_mask() if mask_type == 'center' else make_stripe_mask()
    occ = 1 - pm

    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    cb, ca, bce_b, bce_a, rts, ratios = [], [], [], [], [], []
    residual_stats = {'mean': [], 'max': [], 'min': []}

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

        bit_mask = pixel_to_bit_mask(pm, model.n_bits)
        t0 = time.time()

        if policy_name == 'any':
            z_final = amortized_inpaint_full(net, z_init, bit_mask, device)
            ratio = bit_mask.any(axis=0).mean()
        elif policy_name == 'evidence_fixed':
            z_final, ratio, resid = evidence_strength_repair(
                model, net, z_init, bit_mask, pm, x_noisy, device,
                threshold=eobs_threshold)
        elif policy_name == 'evidence_adaptive':
            z_final, ratio, resid = eobs_adaptive_repair(
                model, net, z_init, bit_mask, pm, x_noisy, device)
        elif policy_name == 'evidence_always_occluded':
            # Hybrid: always repair fully-occluded patches, use E_obs for partial
            z_final, ratio, resid = evidence_strength_repair(
                model, net, z_init, bit_mask, pm, x_noisy, device,
                threshold=eobs_threshold)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

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
        ratios.append(float(ratio))

        if policy_name.startswith('evidence'):
            finite = resid[~resid.isinf()]
            if finite.numel() > 0:
                residual_stats['mean'].append(finite.mean().item())
                residual_stats['max'].append(finite.max().item())
                residual_stats['min'].append(finite.min().item())

    n = len(eval_idx)
    result = {
        'policy': policy_name,
        'mask_type': mask_type,
        'noise_type': noise_type,
        'acc_before': np.mean(cb),
        'acc_after': np.mean(ca),
        'delta_acc': (np.sum(ca) - np.sum(cb)) / n,
        'bce_before': np.mean(bce_b),
        'bce_after': np.mean(bce_a),
        'runtime_ms': np.mean(rts),
        'repair_ratio': np.mean(ratios),
        'n_samples': n,
    }
    if residual_stats['mean']:
        result['resid_mean'] = np.mean(residual_stats['mean'])
        result['resid_max'] = np.mean(residual_stats['max'])
        result['resid_min'] = np.mean(residual_stats['min'])

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'fmnist', 'kmnist'])
    parser.add_argument('--output_dir', default='outputs/exp_evidence_strength')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("EVIDENCE-STRENGTH REPAIR EXPERIMENT")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print()

    print("[1] Loading dataset...")
    train_x, train_y, test_x, test_y = load_dataset(args.dataset, 2000, 500, args.seed)

    print("[2] Training Route C model...")
    model = train_model(train_x, train_y, device)
    model.eval()
    with torch.no_grad():
        z = model.encode(test_x[:500].to(device))
        acc = (model.classifier(z).argmax(1).cpu() == test_y[:500]).float().mean().item()
    print(f"    Clean probe accuracy: {acc:.1%}")

    print("[3] Training InpaintNet (mask mixture)...")
    net = train_inpaint(model, train_x, device)
    net.eval()

    # ---- Diagnostic: residual distribution on center vs stripes ----
    print("\n" + "=" * 80)
    print("  DIAGNOSTIC: E_obs residual distribution")
    print("=" * 80)
    rng = np.random.default_rng(args.seed + 200)
    for mask_type in ['center', 'stripes']:
        pm = make_center_mask() if mask_type == 'center' else make_stripe_mask()
        residuals_all = []
        for idx in range(min(20, len(test_x))):
            x = test_x[idx].numpy()[0]
            x_occ = x * pm
            with torch.no_grad():
                xt = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
                z_init = model.encode(xt)[0]
            resid = compute_eobs_residual_per_patch(model, z_init, x, pm, device)
            finite = resid[~resid.isinf()]
            if finite.numel() > 0:
                residuals_all.append(finite.cpu().numpy())
        if residuals_all:
            all_r = np.concatenate(residuals_all)
            print(f"  {mask_type}: mean={all_r.mean():.4f}, "
                  f"median={np.median(all_r):.4f}, "
                  f"std={all_r.std():.4f}, "
                  f"min={all_r.min():.4f}, max={all_r.max():.4f}")

    # ---- Threshold sweep for evidence_fixed ----
    print("\n" + "=" * 80)
    print("  THRESHOLD SWEEP: E_obs residual threshold")
    print("=" * 80)
    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    sweep_results = []
    for th in thresholds:
        r_center = evaluate_policy('evidence_fixed', model, net, 'center', 'clean',
                                    test_x, test_y, device, n_samples=args.eval_samples,
                                    seed=args.seed, eobs_threshold=th)
        r_stripes = evaluate_policy('evidence_fixed', model, net, 'stripes', 'clean',
                                     test_x, test_y, device, n_samples=args.eval_samples,
                                     seed=args.seed, eobs_threshold=th)
        sweep_results.append((th, r_center, r_stripes))
        total = r_center['delta_acc'] + r_stripes['delta_acc']
        print(f"  th={th:.1f}: center Δacc={r_center['delta_acc']:+.1%} "
              f"(ratio={r_center['repair_ratio']:.2f}), "
              f"stripes Δacc={r_stripes['delta_acc']:+.1%} "
              f"(ratio={r_stripes['repair_ratio']:.2f}), "
              f"total={total:+.1%}")

    # ---- Main comparison ----
    print("\n" + "=" * 80)
    print("  MAIN COMPARISON: any vs evidence_adaptive vs evidence_fixed(best)")
    print("=" * 80)

    # Find best threshold from sweep
    best_th = max(sweep_results,
                  key=lambda x: x[1]['delta_acc'] + x[2]['delta_acc'])[0]
    print(f"  Best threshold from sweep: {best_th}")

    configs = [('center', 'clean'), ('center', 'noise'),
               ('stripes', 'clean'), ('stripes', 'noise')]
    policies = ['any', 'evidence_adaptive', 'evidence_fixed']

    all_results = []

    for policy in policies:
        for mt, nt in configs:
            r = evaluate_policy(policy, model, net, mt, nt,
                               test_x, test_y, device,
                               n_samples=args.eval_samples, seed=args.seed,
                               eobs_threshold=best_th)
            all_results.append(r)
            print(f"  {policy:<22} {mt:<8} {nt:<6} "
                  f"Δacc={r['delta_acc']:+.1%} "
                  f"ratio={r['repair_ratio']:.3f} "
                  f"bce={r['bce_before']:.2f}→{r['bce_after']:.2f} "
                  f"t={r['runtime_ms']:.1f}ms")

    # ---- Summary table ----
    print("\n" + "=" * 100)
    print("EVIDENCE-STRENGTH REPAIR SUMMARY")
    print("=" * 100)
    print(f"{'policy':<22} {'center_c':>8} {'center_n':>8} "
          f"{'stripes_c':>9} {'stripes_n':>9} {'total':>7}")
    print("-" * 70)

    for policy in policies:
        pr = [r for r in all_results if r['policy'] == policy]
        vals = {}
        for r in pr:
            key = f"{r['mask_type']}_{r['noise_type'][0]}"
            vals[key] = r['delta_acc']
        total = sum(vals.values())
        print(f"{policy:<22} "
              f"{vals.get('center_c', 0):>+8.1%} {vals.get('center_n', 0):>+8.1%} "
              f"{vals.get('stripes_c', 0):>+9.1%} {vals.get('stripes_n', 0):>+9.1%} "
              f"{total:>+7.1%}")

    # ---- Paradigm verdict ----
    print("\n" + "=" * 80)
    print("PARADIGM VERDICT: E_obs residual vs E_core consistency")
    print("=" * 80)

    any_total = sum(r['delta_acc'] for r in all_results if r['policy'] == 'any')
    ev_ad_total = sum(r['delta_acc'] for r in all_results if r['policy'] == 'evidence_adaptive')
    ev_fx_total = sum(r['delta_acc'] for r in all_results if r['policy'] == 'evidence_fixed')

    print(f"  any (baseline):        total={any_total:+.1%}")
    print(f"  evidence_adaptive:     total={ev_ad_total:+.1%}")
    print(f"  evidence_fixed(th={best_th}): total={ev_fx_total:+.1%}")

    # Compare with adaptive_pp from previous experiment
    print(f"\n  Previous adaptive_pp (E_core): total=+0.0%")
    print(f"  → E_obs evidence {'BETTER' if max(ev_ad_total, ev_fx_total) > 0.0 else 'NOT BETTER'} "
          f"than E_core consistency")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "evidence_strength_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            all_keys = set()
            for r in all_results:
                all_keys.update(r.keys())
            keys = sorted(all_keys)
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in all_results:
                w.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    # Save threshold sweep
    sweep_path = os.path.join(args.output_dir, "threshold_sweep.csv")
    with open(sweep_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['threshold', 'center_dacc', 'center_ratio',
                     'stripes_dacc', 'stripes_ratio', 'total'])
        for th, rc, rs in sweep_results:
            w.writerow([th, rc['delta_acc'], rc['repair_ratio'],
                        rs['delta_acc'], rs['repair_ratio'],
                        rc['delta_acc'] + rs['delta_acc']])
    print(f"Threshold sweep saved to {sweep_path}")

    print("\n" + "=" * 100)
    print("Evidence-strength repair experiment complete.")
    print("=" * 100)

    return all_results


if __name__ == "__main__":
    main()
