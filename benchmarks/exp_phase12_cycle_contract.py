#!/usr/bin/env python3
"""
Phase 12: Cycle Contract (Protocol Verifiability)
===================================================
Core question: Is z a true "protocol" (reversible, verifiable),
or just a one-time lossy compression?

Cycle consistency test:
  x → encode → z → decode → x̂ → encode → ẑ
  Measure: Hamming(z, ẑ) — if low, z is a stable protocol

Also test: does repair make cycles MORE stable?
  x_corrupt → z_corrupt → repair → z_repaired → decode → x̂ → encode → ẑ
  Compare Hamming(z_repaired, ẑ) vs Hamming(z_corrupt, ẑ_corrupt)

Uses our standard RouteCModel (FMNIST 28×28, 7×7 latent).
No ResNet needed — this tests our own ADC/DAC protocol.

Usage:
    python3 -u benchmarks/exp_phase12_cycle_contract.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os, sys, csv, argparse

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
    def set_temperature(self, tau): self.temperature = tau

class BinaryEncoder(nn.Module):
    def __init__(self, n_bits=8, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1))
    def forward(self, x): return self.conv(x)

class BinaryDecoder(nn.Module):
    def __init__(self, n_bits=8, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1), nn.Sigmoid())
    def forward(self, z): return self.deconv(z)

class LocalPredictor(nn.Module):
    def __init__(self, n_bits=8, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9*n_bits, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_bits))
    def forward(self, z):
        B,k,H,W = z.shape
        z_pad = F.pad(z, (1,1,1,1), mode='constant', value=0)
        w = F.unfold(z_pad, kernel_size=3).reshape(B,k,9,H*W)
        w[:,:,4,:] = 0
        w = w.reshape(B,k*9,H*W).permute(0,2,1)
        return self.net(w).permute(0,2,1).reshape(B,k,H,W)

class Classifier(nn.Module):
    def __init__(self, n_bits=8, latent_size=7):
        super().__init__()
        self.fc = nn.Linear(n_bits*latent_size*latent_size, 10)
    def forward(self, z): return self.fc(z.reshape(z.shape[0],-1))

class RouteCModel(nn.Module):
    def __init__(self, n_bits=8, hidden_dim=64, tau=1.0):
        super().__init__()
        self.n_bits = n_bits; self.latent_size = 7
        self.encoder = BinaryEncoder(n_bits, hidden_dim)
        self.decoder = BinaryDecoder(n_bits, hidden_dim)
        self.quantizer = GumbelSigmoid(tau)
        self.local_pred = LocalPredictor(n_bits)
        self.classifier = Classifier(n_bits)
    def encode(self, x): return self.quantizer(self.encoder(x))
    def decode(self, z): return self.decoder(z)
    def set_temperature(self, tau): self.quantizer.set_temperature(tau)

class InpaintNet(nn.Module):
    def __init__(self, k=8, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(k+1, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(hidden, k, 3, padding=1, padding_mode='circular'))
        self.skip = nn.Conv2d(k+1, k, 1)
    def forward(self, z_masked, mask):
        x = torch.cat([z_masked, mask], dim=1)
        return self.net(x) + self.skip(x)


# ============================================================================
# DATA + TRAINING
# ============================================================================

def make_center_mask():
    m = np.ones((28,28), dtype=np.float32); m[7:21,7:21] = 0; return m
def make_random_block_mask(rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((28,28), dtype=np.float32)
    oh,ow = rng.integers(8,18),rng.integers(8,18)
    y,x = rng.integers(0,max(1,28-oh+1)),rng.integers(0,max(1,28-ow+1))
    m[y:y+oh,x:x+ow] = 0; return m
def make_random_stripe_mask(rng=None):
    if rng is None: rng = np.random.default_rng()
    w = rng.integers(1,4); g = rng.integers(4,10); p = rng.integers(0,g)
    m = np.ones((28,28), dtype=np.float32)
    if rng.random() < 0.5:
        for y in range(p,28,g): m[y:min(y+w,28),:] = 0
    else:
        for x in range(p,28,g): m[:,x:min(x+w,28)] = 0
    return m
def make_multi_hole_mask(rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((28,28), dtype=np.float32)
    for _ in range(rng.integers(3,8)):
        s = rng.integers(2,6)
        y,x = rng.integers(0,max(1,28-s+1)),rng.integers(0,max(1,28-s+1))
        m[y:y+s,x:x+s] = 0
    return m
def sample_training_mask(rng=None):
    if rng is None: rng = np.random.default_rng()
    p = rng.random()
    if p < 0.30: return make_random_block_mask(rng)
    elif p < 0.50: return make_center_mask()
    elif p < 0.70: return make_random_stripe_mask(rng)
    else: return make_multi_hole_mask(rng)
def pixel_to_bit_mask(pixel_mask, n_bits, ls=7):
    bm = np.zeros((n_bits,ls,ls), dtype=bool)
    ph,pw = 28//ls, 28//ls
    for i in range(ls):
        for j in range(ls):
            if pixel_mask[i*ph:(i+1)*ph,j*pw:(j+1)*pw].mean() < 1.0-1e-6:
                bm[:,i,j] = True
    return bm

def load_dataset(name='fmnist', train_n=2000, test_n=500, seed=42):
    from torchvision import datasets, transforms
    ds = {'mnist': datasets.MNIST, 'fmnist': datasets.FashionMNIST}[name]
    tr = ds('./data', train=True, download=True, transform=transforms.ToTensor())
    te = ds('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(tr), train_n, replace=False)
    si = rng.choice(len(te), test_n, replace=False)
    return (torch.stack([tr[i][0] for i in ti]), torch.tensor([tr[i][1] for i in ti]),
            torch.stack([te[i][0] for i in si]), torch.tensor([te[i][1] for i in si]))

def train_model(train_x, train_y, device, epochs=5, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = RouteCModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    for epoch in range(epochs):
        model.train()
        model.set_temperature(1.0 + (0.2-1.0)*epoch/max(1,epochs-1))
        for x,y in loader:
            x = x.to(device); opt.zero_grad()
            z = model.encode(x); x_hat = model.decode(z)
            cl = model.local_pred(z)
            lr_ = F.binary_cross_entropy(x_hat.clamp(1e-6,1-1e-6), x)
            m = torch.rand_like(z) < 0.15
            lc = F.binary_cross_entropy_with_logits(cl[m], z.detach()[m]) if m.any() else torch.tensor(0.,device=device)
            (lr_+0.5*lc).backward(); opt.step()
    for p in model.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True
    co = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    for _ in range(3):
        model.classifier.train()
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            with torch.no_grad(): z = model.encode(x)
            F.cross_entropy(model.classifier(z), y).backward(); co.step(); co.zero_grad()
    for p in model.parameters(): p.requires_grad = True
    return model

def train_inpaint(model, train_x, device, epochs=20):
    net = InpaintNet(k=model.n_bits).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    model.eval(); rng = np.random.default_rng(42)
    for epoch in range(epochs):
        net.train(); perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), 64):
            idx = perm[i:i+64]; x = train_x[idx].to(device)
            with torch.no_grad(): z = model.encode(x); z_hard = (z > 0.5).float()
            B,k,H,W = z_hard.shape
            masks = [torch.from_numpy(pixel_to_bit_mask(sample_training_mask(rng),k)).float() for _ in range(B)]
            bm = torch.stack(masks).to(device)
            z_masked = z_hard * (1 - bm)
            logits = net(z_masked, bm[:, 0:1])
            loss = F.binary_cross_entropy_with_logits(logits[bm.bool()], z_hard[bm.bool()])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


# ============================================================================
# CYCLE EVALUATION
# ============================================================================

def evaluate_cycle(model, test_x, device, n_samples=200, seed=42):
    """x → z → x̂ → ẑ, measure Hamming(z, ẑ)."""
    model.eval()
    rng = np.random.default_rng(seed)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    hammings, bit_flips, total_bits = [], [], []

    with torch.no_grad():
        for idx in eval_idx:
            x = test_x[idx:idx+1].to(device)
            z = model.encode(x)
            z_hard = (z > 0.5).float()
            x_hat = model.decode(z_hard).clamp(0, 1)
            z_hat = model.encode(x_hat)
            z_hat_hard = (z_hat > 0.5).float()

            # Hamming distance
            flips = (z_hard != z_hat_hard).float().sum().item()
            total = z_hard.numel()
            hammings.append(flips / total)
            bit_flips.append(flips)
            total_bits.append(total)

    return {
        'hamming_rate': np.mean(hammings),
        'hamming_std': np.std(hammings),
        'avg_bit_flips': np.mean(bit_flips),
        'total_bits': total_bits[0],
    }


def evaluate_cycle_with_repair(model, net, test_x, device, mask_type='center',
                                n_samples=200, seed=42):
    """
    Test: does repair improve cycle stability?

    Path A (no repair): x_occ → z → x̂ → ẑ → Hamming(z, ẑ)
    Path B (repair):    x_occ → z → repair → z_rep → x̂_rep → ẑ_rep → Hamming(z_rep, ẑ_rep)
    """
    model.eval(); net.eval()
    pm = make_center_mask()
    rng = np.random.default_rng(seed)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    ham_no_repair, ham_repair = [], []

    with torch.no_grad():
        for idx in eval_idx:
            x_clean = test_x[idx].numpy()[0]
            x_occ = x_clean * pm
            x_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)

            # Encode occluded
            z = model.encode(x_t)
            z_hard = (z > 0.5).float()

            # Path A: no repair cycle
            x_hat_a = model.decode(z_hard).clamp(0, 1)
            z_hat_a = (model.encode(x_hat_a) > 0.5).float()
            ham_a = (z_hard != z_hat_a).float().mean().item()
            ham_no_repair.append(ham_a)

            # Path B: repair then cycle
            bit_mask = pixel_to_bit_mask(pm, model.n_bits)
            bm = torch.from_numpy(bit_mask).float().to(device)
            mask = bm.max(dim=0, keepdim=True)[0].unsqueeze(0)
            z_masked = z_hard * (1 - bm.unsqueeze(0))
            logits = net(z_masked, mask)
            preds = (torch.sigmoid(logits) > 0.5).float()
            z_rep = z_hard.clone()
            z_rep[0][bm.bool()] = preds[0][bm.bool()]

            x_hat_b = model.decode(z_rep).clamp(0, 1)
            z_hat_b = (model.encode(x_hat_b) > 0.5).float()
            ham_b = (z_rep != z_hat_b).float().mean().item()
            ham_repair.append(ham_b)

    return {
        'mask': mask_type,
        'hamming_no_repair': np.mean(ham_no_repair),
        'hamming_repaired': np.mean(ham_repair),
        'stability_gain': np.mean(ham_no_repair) - np.mean(ham_repair),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=200)
    parser.add_argument('--output_dir', default='outputs/exp_phase12')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 12: CYCLE CONTRACT (z IS A PROTOCOL, NOT A COMPRESSION)")
    print("=" * 100)

    print("[1] Loading FMNIST...")
    train_x, train_y, test_x, test_y = load_dataset('fmnist', 2000, 500, args.seed)

    print("[2] Training model...")
    model = train_model(train_x, train_y, device, seed=args.seed)

    print("[3] Training InpaintNet...")
    net = train_inpaint(model, train_x, device)

    # Test 1: Clean cycle
    print("\n" + "=" * 80)
    print("TEST 1: CLEAN CYCLE (x → z → x̂ → ẑ)")
    print("=" * 80)
    cycle_clean = evaluate_cycle(model, test_x, device, n_samples=args.eval_samples)
    print(f"  Hamming(z, ẑ) = {cycle_clean['hamming_rate']:.4f} "
          f"(±{cycle_clean['hamming_std']:.4f})")
    print(f"  Average bit flips: {cycle_clean['avg_bit_flips']:.1f} / "
          f"{cycle_clean['total_bits']} total bits")
    print(f"  → {'STABLE' if cycle_clean['hamming_rate'] < 0.1 else 'UNSTABLE'} protocol "
          f"({'<10% flip rate' if cycle_clean['hamming_rate'] < 0.1 else '>10% flip rate'})")

    # Test 2: Repair improves stability?
    print("\n" + "=" * 80)
    print("TEST 2: REPAIR IMPROVES CYCLE STABILITY?")
    print("=" * 80)
    cycle_repair = evaluate_cycle_with_repair(
        model, net, test_x, device, n_samples=args.eval_samples)
    print(f"  Occluded (no repair): Hamming = {cycle_repair['hamming_no_repair']:.4f}")
    print(f"  Occluded (repaired):  Hamming = {cycle_repair['hamming_repaired']:.4f}")
    print(f"  Stability gain: {cycle_repair['stability_gain']:+.4f}")
    print(f"  → Repair {'IMPROVES' if cycle_repair['stability_gain'] > 0 else 'DOES NOT IMPROVE'} "
          f"cycle stability")

    # Test 3: Multi-cycle drift
    print("\n" + "=" * 80)
    print("TEST 3: MULTI-CYCLE DRIFT (x → z → x̂ → ẑ → x̂̂ → ẑ̂)")
    print("=" * 80)
    model.eval()
    rng = np.random.default_rng(args.seed)
    eval_idx = rng.choice(len(test_x), min(100, len(test_x)), replace=False)

    drift_per_cycle = {1: [], 2: [], 3: [], 5: []}
    with torch.no_grad():
        for idx in eval_idx:
            x = test_x[idx:idx+1].to(device)
            z_orig = (model.encode(x) > 0.5).float()
            z_current = z_orig.clone()
            for c in range(5):
                x_hat = model.decode(z_current).clamp(0, 1)
                z_current = (model.encode(x_hat) > 0.5).float()
                if (c + 1) in drift_per_cycle:
                    ham = (z_orig != z_current).float().mean().item()
                    drift_per_cycle[c + 1].append(ham)

    print(f"  {'Cycles':>8} {'Hamming':>10} {'Drift from c=1':>16}")
    print("-" * 40)
    ham1 = np.mean(drift_per_cycle[1])
    for c in sorted(drift_per_cycle.keys()):
        ham = np.mean(drift_per_cycle[c])
        drift = ham - ham1
        print(f"  {c:>8} {ham:>10.4f} {drift:>+16.4f}")

    stable = np.mean(drift_per_cycle[5]) < np.mean(drift_per_cycle[1]) * 2
    print(f"\n  → Multi-cycle {'STABLE (drift < 2× single)' if stable else 'DRIFTING'}")

    # Save results
    all_results = [
        {'test': 'clean_cycle', 'hamming': cycle_clean['hamming_rate'],
         'bit_flips': cycle_clean['avg_bit_flips'], 'total_bits': cycle_clean['total_bits']},
        {'test': 'repair_no_repair', 'hamming': cycle_repair['hamming_no_repair']},
        {'test': 'repair_repaired', 'hamming': cycle_repair['hamming_repaired'],
         'stability_gain': cycle_repair['stability_gain']},
    ]
    for c in sorted(drift_per_cycle.keys()):
        all_results.append({
            'test': f'drift_cycle_{c}', 'hamming': np.mean(drift_per_cycle[c])})

    csv_path = os.path.join(args.output_dir, "phase12_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(set().union(*(r.keys() for r in all_results)))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")
    print("\n" + "=" * 100)
    print("Phase 12 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
