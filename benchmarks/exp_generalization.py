#!/usr/bin/env python3
"""
Cross-Dataset Generalization Experiment
========================================
Critical Experiment A: Does the discrete-core inference paradigm generalize
beyond MNIST with ZERO architecture changes?

Same model, same hyperparameters, same training protocol, same mask mixture.
Only the dataset changes: MNIST → FashionMNIST → KMNIST.

Paradigm predictions:
  - If correct: positive Δacc on center for all 3 datasets
  - If wrong:   Δacc degrades with visual complexity

Metrics (3 categories):
  1. Observation: E_obs (BCE on occluded region)
  2. Structure:   E_core violation rate
  3. Probe:       Δacc (classification accuracy change)

Usage:
    python3 -u benchmarks/exp_generalization.py --device cuda --seed 42
    python3 -u benchmarks/exp_generalization.py --device cuda --datasets mnist,fmnist,kmnist
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
# MODEL — Parameterized for generalization
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

class Encoder(nn.Module):
    def __init__(self, in_channels, n_bits, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1),
        )
    def forward(self, x): return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, n_bits, out_channels, hidden_dim=64):
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
    def __init__(self, in_channels=1, out_channels=1, n_bits=8, hidden_dim=64,
                 energy_hidden=32, latent_size=7, n_classes=10, tau=1.0):
        super().__init__()
        self.n_bits = n_bits
        self.latent_size = latent_size
        self.in_channels = in_channels
        self.encoder = Encoder(in_channels, n_bits, hidden_dim)
        self.quantizer = GumbelSigmoid(tau)
        self.decoder = Decoder(n_bits, out_channels, hidden_dim)
        self.local_pred = LocalPredictor(n_bits, energy_hidden)
        self.classifier = Classifier(n_bits, latent_size, n_classes)
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
# MASKS
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

def pixel_to_bit_mask_any(pixel_mask, n_bits=8, latent_size=7):
    patch_size = 28 // latent_size
    bm = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    for i in range(latent_size):
        for j in range(latent_size):
            y0, y1 = i * patch_size, (i+1) * patch_size
            x0, x1 = j * patch_size, (j+1) * patch_size
            if pixel_mask[y0:y1, x0:x1].mean() < 1.0 - 1e-6:
                bm[:, i, j] = True
    return bm

def apply_noise(image, noise_type, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    if noise_type == 'noise':
        return np.clip(image + rng.normal(0, 0.1, image.shape).astype(np.float32), 0, 1)
    return image.copy()


# ============================================================================
# DATA LOADING — multi-dataset
# ============================================================================

def load_dataset(name, train_n=2000, test_n=1000, seed=42):
    """Load dataset by name. Returns (train_x, train_y, test_x, test_y)."""
    from torchvision import datasets, transforms

    dataset_map = {
        'mnist': datasets.MNIST,
        'fmnist': datasets.FashionMNIST,
        'kmnist': datasets.KMNIST,
    }

    if name not in dataset_map:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(dataset_map.keys())}")

    cls = dataset_map[name]
    tr = cls('./data', train=True, download=True, transform=transforms.ToTensor())
    te = cls('./data', train=False, download=True, transform=transforms.ToTensor())

    rng = np.random.default_rng(seed)
    ti = rng.choice(len(tr), train_n, replace=False)
    si = rng.choice(len(te), test_n, replace=False)

    train_x = torch.stack([tr[i][0] for i in ti])
    train_y = torch.tensor([tr[i][1] for i in ti])
    test_x = torch.stack([te[i][0] for i in si])
    test_y = torch.tensor([te[i][1] for i in si])

    return train_x, train_y, test_x, test_y


# ============================================================================
# TRAINING
# ============================================================================

def train_model(train_x, train_y, device, in_channels=1, epochs=5, lr=1e-3, batch_size=64):
    """Train Route C model. BCE E_obs, no L_cls."""
    model = RouteCModel(in_channels=in_channels, out_channels=in_channels).to(device)
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


def train_inpaint_mixture(model, train_x, device, epochs=20, batch_size=64, lr=1e-3):
    """Mask mixture training (paradigm standard)."""
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
            print(f"        inpaint epoch {epoch+1}/{epochs}: loss={tl/max(nb,1):.4f}")
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


def evaluate_config(model, inpaint_net, mask_type, noise_type,
                    test_x, test_y, device, n_samples=100, seed=42):
    """Evaluate one (mask_type, noise_type) config. Returns metrics dict."""
    model.eval(); inpaint_net.eval()
    pm = make_center_mask() if mask_type == 'center' else make_stripe_mask()
    occ = 1 - pm

    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    cb, ca, bce_b, bce_a, mse_a = [], [], [], [], []
    core_viol_b, core_viol_a = [], []
    rts, bit_ratios = [], []

    for idx in eval_idx:
        x_clean = test_x[idx].numpy()  # (C, H, W)
        if x_clean.ndim == 3:
            x_clean_2d = x_clean[0]  # grayscale
        else:
            x_clean_2d = x_clean
        label = test_y[idx].item()
        x_noisy = apply_noise(x_clean_2d, noise_type, rng)
        x_occ = x_noisy * pm

        with torch.no_grad():
            x_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_init = model.encode(x_t)[0]
            pred_b = model.classifier(z_init.unsqueeze(0)).argmax(1).item()
            o_hat_b = model.decode(z_init.unsqueeze(0))[0, 0]

            # E_core violation before
            z_hard_b = (z_init.unsqueeze(0) > 0.5).float()
            core_logits_b = model.local_pred(z_hard_b)
            core_pred_b = (core_logits_b > 0).float()
            viol_b = (z_hard_b != core_pred_b).float().mean().item()
            core_viol_b.append(viol_b)

        bit_mask = pixel_to_bit_mask_any(pm, model.n_bits, model.latent_size)
        bit_ratios.append(float(bit_mask[0].mean()))

        if not bit_mask.any():
            pred_a = pred_b; o_hat_a = o_hat_b; rt = 0.0
            core_viol_a.append(viol_b)
        else:
            t0 = time.time()
            z_final = amortized_inpaint(inpaint_net, z_init, bit_mask, device)
            rt = (time.time() - t0) * 1000

            with torch.no_grad():
                pred_a = model.classifier(z_final.unsqueeze(0)).argmax(1).item()
                o_hat_a = model.decode(z_final.unsqueeze(0))[0, 0]

                z_hard_a = (z_final.unsqueeze(0) > 0.5).float()
                core_logits_a = model.local_pred(z_hard_a)
                core_pred_a = (core_logits_a > 0).float()
                viol_a = (z_hard_a != core_pred_a).float().mean().item()
                core_viol_a.append(viol_a)

        xt = torch.from_numpy(x_clean_2d).to(device)
        ot = torch.from_numpy(occ).to(device)
        os_ = ot.sum().clamp(min=1.0).item()

        def ob(oh):
            l = oh.clamp(1e-6,1-1e-6)
            return (-(xt*torch.log(l)+(1-xt)*torch.log(1-l))*ot).sum().item()/os_
        def om(oh):
            d = (oh-xt)*ot; return (d*d).sum().item()/os_

        cb.append(int(pred_b == label)); ca.append(int(pred_a == label))
        bce_b.append(ob(o_hat_b)); bce_a.append(ob(o_hat_a))
        mse_a.append(om(o_hat_a))
        rts.append(rt)

    n = len(eval_idx)
    return {
        'mask_type': mask_type,
        'noise_type': noise_type,
        'acc_before': np.mean(cb),
        'acc_after': np.mean(ca),
        'delta_acc': (np.sum(ca) - np.sum(cb)) / n,
        'bce_before': np.mean(bce_b),
        'bce_after': np.mean(bce_a),
        'mse_after': np.mean(mse_a),
        'core_viol_before': np.mean(core_viol_b),
        'core_viol_after': np.mean(core_viol_a),
        'delta_core_viol': np.mean(core_viol_a) - np.mean(core_viol_b),
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
    parser.add_argument('--train_n', type=int, default=2000)
    parser.add_argument('--test_n', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--inpaint_epochs', type=int, default=20)
    parser.add_argument('--datasets', default='mnist,fmnist,kmnist',
                        help='Comma-separated dataset names')
    parser.add_argument('--output_dir', default='outputs/exp_generalization')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    dataset_names = [d.strip() for d in args.datasets.split(',')]

    print("=" * 100)
    print("CROSS-DATASET GENERALIZATION EXPERIMENT")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Datasets: {dataset_names}")
    print(f"Protocol: BCE E_obs, no L_cls, mask mixture training, amortized inference")
    print(f"Goal: Does the paradigm generalize with ZERO architecture changes?")
    print()

    configs = [
        ('center', 'clean'), ('center', 'noise'),
        ('stripes', 'clean'), ('stripes', 'noise'),
    ]

    all_results = []

    for ds_name in dataset_names:
        print(f"\n{'='*80}")
        print(f"  DATASET: {ds_name.upper()}")
        print(f"{'='*80}")

        # Reset seed per dataset for reproducibility
        torch.manual_seed(args.seed); np.random.seed(args.seed)

        print(f"\n  [1] Loading {ds_name}...")
        train_x, train_y, test_x, test_y = load_dataset(
            ds_name, args.train_n, args.test_n, args.seed)
        in_channels = train_x.shape[1]
        print(f"      Shape: {train_x.shape}, channels={in_channels}")

        print(f"\n  [2] Training Route C model...")
        model = train_model(train_x, train_y, device,
                          in_channels=in_channels, epochs=args.epochs)
        model.eval()

        with torch.no_grad():
            z = model.encode(test_x[:500].to(device))
            acc = (model.classifier(z).argmax(1).cpu() == test_y[:500]).float().mean().item()
        print(f"      Clean probe accuracy: {acc:.1%}")

        print(f"\n  [3] Training InpaintNet (mask mixture)...")
        inpaint_net = train_inpaint_mixture(model, train_x, device,
                                           epochs=args.inpaint_epochs)

        print(f"\n  [4] Evaluating...")
        for mt, nt in configs:
            r = evaluate_config(model, inpaint_net, mt, nt,
                               test_x, test_y, device,
                               n_samples=args.eval_samples, seed=args.seed)
            r['dataset'] = ds_name
            all_results.append(r)
            print(f"      {mt:<8}+{nt:<6}: Δacc={r['delta_acc']:+.1%}, "
                  f"bce={r['bce_before']:.2f}→{r['bce_after']:.2f}, "
                  f"viol={r['core_viol_before']:.3f}→{r['core_viol_after']:.3f}, "
                  f"t={r['runtime_ms']:.1f}ms")

    # ======================================================================
    # Cross-dataset summary
    # ======================================================================
    print("\n" + "=" * 120)
    print("CROSS-DATASET SUMMARY")
    print("=" * 120)

    header = (f"{'dataset':<10} {'mask':<8} {'noise':<6} "
              f"{'acc_bef':>7} {'acc_aft':>7} {'Δacc':>7} "
              f"{'bce_bef':>8} {'bce_aft':>8} "
              f"{'viol_bef':>8} {'viol_aft':>8} "
              f"{'ms':>6}")
    print(header)
    print("-" * 120)
    for r in all_results:
        print(f"{r['dataset']:<10} {r['mask_type']:<8} {r['noise_type']:<6} "
              f"{r['acc_before']:>7.1%} {r['acc_after']:>7.1%} {r['delta_acc']:>+7.1%} "
              f"{r['bce_before']:>8.2f} {r['bce_after']:>8.2f} "
              f"{r['core_viol_before']:>8.3f} {r['core_viol_after']:>8.3f} "
              f"{r['runtime_ms']:>6.1f}")

    # ======================================================================
    # Paradigm verdict per dataset
    # ======================================================================
    print("\n" + "=" * 80)
    print("PARADIGM VERDICT: Does amortized inference produce positive Δacc on center?")
    print("=" * 80)
    for ds_name in dataset_names:
        center_results = [r for r in all_results
                         if r['dataset'] == ds_name and r['mask_type'] == 'center']
        center_avg = np.mean([r['delta_acc'] for r in center_results])
        stripe_results = [r for r in all_results
                         if r['dataset'] == ds_name and r['mask_type'] == 'stripes']
        stripe_avg = np.mean([r['delta_acc'] for r in stripe_results])

        verdict = "PASS" if center_avg > 0 else "FAIL"
        print(f"  {ds_name.upper():<10}: center Δacc={center_avg:+.1%} [{verdict}], "
              f"stripes Δacc={stripe_avg:+.1%}")

    # ======================================================================
    # Paradigm prediction check
    # ======================================================================
    print("\n" + "=" * 80)
    print("PARADIGM PREDICTIONS CHECK")
    print("=" * 80)

    # Prediction 1: positive Δacc on center for all datasets
    all_center_pass = all(
        np.mean([r['delta_acc'] for r in all_results
                 if r['dataset'] == ds and r['mask_type'] == 'center']) > 0
        for ds in dataset_names
    )
    p1 = "CONFIRMED" if all_center_pass else "FALSIFIED"
    print(f"  [1] Positive center Δacc for all datasets: {p1}")

    # Prediction 2: BCE improvement (bce_after < bce_before) on center
    all_bce_improve = all(
        np.mean([r['bce_after'] for r in all_results
                 if r['dataset'] == ds and r['mask_type'] == 'center'])
        < np.mean([r['bce_before'] for r in all_results
                   if r['dataset'] == ds and r['mask_type'] == 'center'])
        for ds in dataset_names
    )
    p2 = "CONFIRMED" if all_bce_improve else "FALSIFIED"
    print(f"  [2] E_obs improves after repair for all datasets: {p2}")

    # Prediction 3: stripes Δacc ≤ 0 (known limitation)
    all_stripes_neg = all(
        np.mean([r['delta_acc'] for r in all_results
                 if r['dataset'] == ds and r['mask_type'] == 'stripes']) <= 0.02
        for ds in dataset_names
    )
    p3 = "CONFIRMED" if all_stripes_neg else "SURPRISING (stripes improved!)"
    print(f"  [3] Stripes Δacc ≤ 0 (known limitation): {p3}")

    # ======================================================================
    # Three-metric table (observation + structure + probe)
    # ======================================================================
    print("\n" + "=" * 80)
    print("THREE-METRIC FRAMEWORK (per dataset, center+clean)")
    print("=" * 80)
    print(f"{'dataset':<10} {'Observation':>15} {'Structure':>15} {'Probe':>10}")
    print(f"{'':10} {'ΔBCE':>15} {'ΔViol':>15} {'Δacc':>10}")
    print("-" * 55)
    for ds in dataset_names:
        r = [x for x in all_results
             if x['dataset'] == ds and x['mask_type'] == 'center' and x['noise_type'] == 'clean']
        if r:
            r = r[0]
            dbce = r['bce_after'] - r['bce_before']
            dviol = r['core_viol_after'] - r['core_viol_before']
            print(f"{ds:<10} {dbce:>+15.2f} {dviol:>+15.3f} {r['delta_acc']:>+10.1%}")

    # Save
    csv_path = os.path.join(args.output_dir, "generalization_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    print("\n" + "=" * 100)
    print("Cross-dataset generalization experiment complete.")
    print("=" * 100)

    return all_results


if __name__ == "__main__":
    main()
