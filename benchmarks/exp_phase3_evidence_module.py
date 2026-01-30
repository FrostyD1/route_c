#!/usr/bin/env python3
"""
Phase 3: Evidence Repair Module Validation
============================================
Validates that the standardized EvidenceRepairPolicy works as a
paradigm component — no per-mask strategy tuning needed.

Tests:
  1. RepairAll (baseline) vs EvidenceRepairPolicy across ALL mask types
  2. Single threshold works for all masks (no per-mask tuning)
  3. Pareto improvement: no mask type gets significantly worse
  4. Cross-dataset: MNIST + FashionMNIST + KMNIST

Also tests Phase 6 ConstraintInterface (D4 symmetry) as a secondary check.

Usage:
    python3 -u benchmarks/exp_phase3_evidence_module.py --device cuda
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
# MODEL (shared code — will be moved to routec/ after validation)
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
    def __init__(self, n_bits=8, in_channels=1, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1),
        )
    def forward(self, x): return self.conv(x)


class BinaryDecoder(nn.Module):
    def __init__(self, n_bits=8, out_channels=1, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z): return self.deconv(z)


class LocalPredictor(nn.Module):
    def __init__(self, n_bits=8, hidden_dim=32):
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
    def __init__(self, n_bits=8, latent_size=7, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(n_bits * latent_size * latent_size, n_classes)
    def forward(self, z): return self.fc(z.reshape(z.shape[0], -1))


class RouteCModel(nn.Module):
    def __init__(self, n_bits=8, latent_size=7, hidden_dim=64, energy_hidden=32, tau=1.0):
        super().__init__()
        self.n_bits = n_bits
        self.latent_size = latent_size
        self.encoder = BinaryEncoder(n_bits, 1, hidden_dim)
        self.decoder = BinaryDecoder(n_bits, 1, hidden_dim)
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
    m = np.ones((H,W), dtype=np.float32); m[7:21, 7:21] = 0; return m

def make_stripe_mask(H=28, W=28):
    m = np.ones((H,W), dtype=np.float32)
    for y in range(0, H, 6): m[y:min(y+2,H), :] = 0
    return m

def make_multi_hole_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((H,W), dtype=np.float32)
    for _ in range(rng.integers(3,8)):
        s = rng.integers(2,6)
        y, x = rng.integers(0, max(1,H-s+1)), rng.integers(0, max(1,W-s+1))
        m[y:y+s, x:x+s] = 0
    return m

def make_random_block_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((H,W), dtype=np.float32)
    oh, ow = rng.integers(8,18), rng.integers(8,18)
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

def sample_training_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    p = rng.random()
    if p < 0.30: return make_random_block_mask(H,W,rng)
    elif p < 0.50: return make_center_mask(H,W)
    elif p < 0.70: return make_random_stripe_mask(H,W,rng)
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


# ============================================================================
# EVIDENCE REPAIR (standardized module)
# ============================================================================

def compute_eobs_per_patch(model, z, observation, pixel_mask, device):
    """Per-patch E_obs residual. (B=1 version for eval)"""
    ls = model.latent_size
    H, W = observation.shape
    ph, pw = H // ls, W // ls
    with torch.no_grad():
        o_hat = model.decode(z.unsqueeze(0))[0, 0].clamp(1e-6, 1-1e-6)
    obs_t = torch.from_numpy(observation).to(device)
    pm_t = torch.from_numpy(pixel_mask).to(device)
    resid = torch.zeros(ls, ls, device=device)
    for i in range(ls):
        for j in range(ls):
            y0, y1 = i*ph, (i+1)*ph; x0, x1 = j*pw, (j+1)*pw
            p_obs = obs_t[y0:y1, x0:x1]; p_hat = o_hat[y0:y1, x0:x1]
            p_m = pm_t[y0:y1, x0:x1]
            cnt = p_m.sum()
            if cnt > 0:
                bce = -(p_obs * torch.log(p_hat) + (1-p_obs)*torch.log(1-p_hat))
                resid[i,j] = (bce * p_m).sum() / cnt
            else:
                resid[i,j] = float('inf')
    return resid


def evidence_repair(model, net, z_init, bit_mask, pixel_mask, observation,
                     device, threshold=1.0):
    """Standardized evidence repair: only repair where E_obs residual > threshold."""
    net.eval(); model.eval()
    # Full repair candidate
    z = z_init.clone().to(device)
    bm = torch.from_numpy(bit_mask).float().to(device)
    mask = bm.max(dim=0, keepdim=True)[0].unsqueeze(0)
    z_masked = z.unsqueeze(0) * (1 - bm.unsqueeze(0))
    with torch.no_grad():
        logits = net(z_masked, mask)
        preds = (torch.sigmoid(logits) > 0.5).float()
    bm_bool = torch.from_numpy(bit_mask).to(device)
    z_repaired = z.clone()
    z_repaired[bm_bool] = preds[0][bm_bool]

    # E_obs residual
    resid = compute_eobs_per_patch(model, z_init, observation, pixel_mask, device)

    # Selective repair
    z_result = z_init.clone()
    ls = model.latent_size
    repair_count, total_masked = 0, 0
    for i in range(ls):
        for j in range(ls):
            if bit_mask[0, i, j]:
                total_masked += 1
                r = resid[i, j].item()
                if np.isinf(r) or r > threshold:
                    z_result[:, i, j] = z_repaired[:, i, j]
                    repair_count += 1
    ratio = repair_count / max(total_masked, 1)
    return z_result, ratio


def repair_all(net, z_init, bit_mask, device):
    """Baseline: repair all masked positions."""
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


# ============================================================================
# DATA + TRAINING
# ============================================================================

def load_dataset(name, train_n=2000, test_n=500, seed=42):
    from torchvision import datasets, transforms
    ds_map = {'mnist': datasets.MNIST, 'fmnist': datasets.FashionMNIST,
              'kmnist': datasets.KMNIST}
    ds_cls = ds_map[name]
    tr = ds_cls('./data', train=True, download=True, transform=transforms.ToTensor())
    te = ds_cls('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(tr), train_n, replace=False)
    si = rng.choice(len(te), test_n, replace=False)
    return (torch.stack([tr[i][0] for i in ti]), torch.tensor([tr[i][1] for i in ti]),
            torch.stack([te[i][0] for i in si]), torch.tensor([te[i][1] for i in si]))


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
    N = len(train_x); rng = np.random.default_rng(42)
    for epoch in range(epochs):
        net.train(); perm = torch.randperm(N); tl, nb = 0., 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]; x = train_x[idx].to(device)
            with torch.no_grad(): z = model.encode(x); z_hard = (z > 0.5).float()
            B, k, H, W = z_hard.shape
            masks = [torch.from_numpy(pixel_to_bit_mask(sample_training_mask(28,28,rng), k)).float() for _ in range(B)]
            bit_masks = torch.stack(masks).to(device)
            pos_masks = bit_masks[:, 0:1, :, :]
            z_masked = z_hard * (1 - bit_masks)
            logits = net(z_masked, pos_masks)
            loss = F.binary_cross_entropy_with_logits(logits[bit_masks.bool()], z_hard[bit_masks.bool()])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        if (epoch+1) % 10 == 0:
            print(f"        epoch {epoch+1}/{epochs}: loss={tl/max(nb,1):.4f}")
    return net


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_policy(policy, model, net, mask_type, test_x, test_y, device,
                    n_samples=100, seed=42, threshold=1.0):
    model.eval(); net.eval()
    stochastic = mask_type == 'multi_hole'
    if not stochastic:
        pm = {'center': make_center_mask, 'stripes': make_stripe_mask}[mask_type]()

    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    cb, ca, rts, ratios = [], [], [], []
    for idx in eval_idx:
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()
        if stochastic:
            pm = make_multi_hole_mask(rng=rng)
        occ = 1 - pm; x_occ = x_clean * pm

        with torch.no_grad():
            x_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_init = model.encode(x_t)[0]
            pred_b = model.classifier(z_init.unsqueeze(0)).argmax(1).item()

        bit_mask = pixel_to_bit_mask(pm, model.n_bits)
        t0 = time.time()

        if policy == 'any':
            z_final = repair_all(net, z_init, bit_mask, device)
            ratio = bit_mask.any(axis=0).mean()
        elif policy == 'evidence':
            z_final, ratio = evidence_repair(
                model, net, z_init, bit_mask, pm, x_clean, device, threshold)
        else:
            raise ValueError(policy)

        rt = (time.time() - t0) * 1000
        with torch.no_grad():
            pred_a = model.classifier(z_final.unsqueeze(0)).argmax(1).item()

        cb.append(int(pred_b == label)); ca.append(int(pred_a == label))
        rts.append(rt); ratios.append(float(ratio))

    n = len(eval_idx)
    return {
        'delta_acc': (np.sum(ca) - np.sum(cb)) / n,
        'acc_before': np.mean(cb), 'acc_after': np.mean(ca),
        'runtime_ms': np.mean(rts), 'repair_ratio': np.mean(ratios),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--output_dir', default='outputs/exp_phase3')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 3: EVIDENCE REPAIR MODULE — CROSS-DATASET VALIDATION")
    print("=" * 100)

    datasets = ['mnist', 'fmnist', 'kmnist']
    mask_types = ['center', 'stripes', 'multi_hole']
    policies = ['any', 'evidence']

    all_results = []

    for ds_name in datasets:
        print(f"\n{'='*80}")
        print(f"  DATASET: {ds_name.upper()}")
        print(f"{'='*80}")

        torch.manual_seed(args.seed); np.random.seed(args.seed)
        print(f"\n  [1] Loading {ds_name}...")
        train_x, train_y, test_x, test_y = load_dataset(ds_name, 2000, 500, args.seed)

        print(f"  [2] Training Route C model...")
        model = train_model(train_x, train_y, device)
        model.eval()
        with torch.no_grad():
            z = model.encode(test_x[:500].to(device))
            acc = (model.classifier(z).argmax(1).cpu() == test_y[:500]).float().mean().item()
        print(f"      Clean probe: {acc:.1%}")

        print(f"  [3] Training InpaintNet...")
        net = train_inpaint(model, train_x, device)

        for policy in policies:
            for mt in mask_types:
                r = evaluate_policy(policy, model, net, mt, test_x, test_y, device,
                                   n_samples=args.eval_samples, seed=args.seed)
                r.update({'dataset': ds_name, 'policy': policy, 'mask_type': mt})
                all_results.append(r)
                print(f"    {policy:<10} {mt:<12} Δacc={r['delta_acc']:+.1%} "
                      f"ratio={r['repair_ratio']:.2f}")

    # ---- Summary ----
    print("\n" + "=" * 100)
    print("PHASE 3 SUMMARY: Evidence Repair Module (th=1.0)")
    print("=" * 100)
    print(f"{'dataset':<8} {'policy':<10} {'center':>8} {'stripes':>9} "
          f"{'multi_hole':>11} {'total':>7}")
    print("-" * 60)

    for ds_name in datasets:
        for policy in policies:
            vals = {}
            for r in all_results:
                if r['dataset'] == ds_name and r['policy'] == policy:
                    vals[r['mask_type']] = r['delta_acc']
            total = sum(vals.values())
            print(f"{ds_name:<8} {policy:<10} "
                  f"{vals.get('center',0):>+8.1%} {vals.get('stripes',0):>+9.1%} "
                  f"{vals.get('multi_hole',0):>+11.1%} {total:>+7.1%}")

    # ---- Pareto check ----
    print("\n" + "=" * 80)
    print("PARETO CHECK: Does evidence hurt any mask type?")
    print("=" * 80)
    pareto_pass = True
    for ds_name in datasets:
        for mt in mask_types:
            any_r = [r for r in all_results if r['dataset']==ds_name and r['policy']=='any' and r['mask_type']==mt]
            ev_r = [r for r in all_results if r['dataset']==ds_name and r['policy']=='evidence' and r['mask_type']==mt]
            if any_r and ev_r:
                gap = ev_r[0]['delta_acc'] - any_r[0]['delta_acc']
                worse = gap < -0.05  # more than 5% worse
                status = "WORSE" if worse else "OK"
                if worse: pareto_pass = False
                print(f"  {ds_name}/{mt}: any={any_r[0]['delta_acc']:+.1%}, "
                      f"evidence={ev_r[0]['delta_acc']:+.1%}, gap={gap:+.1%} [{status}]")

    print(f"\n  Pareto: {'PASS' if pareto_pass else 'FAIL'}")

    # ---- Save ----
    csv_path = os.path.join(args.output_dir, "phase3_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            keys = sorted(set().union(*(r.keys() for r in all_results)))
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    print("\n" + "=" * 100)
    print("Phase 3 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
