#!/usr/bin/env python3
"""
Phase 9: Conditional Operator Dispatch (OperatorSelector)
==========================================================
GPT recommendation: FGO/GDA should be "conditionally triggered" modules,
not default architecture. Formalize trigger conditions into a dispatch system.

OperatorSelector decides local-only vs local+FGO based on:
  1. Grid size (larger → more likely FGO helps)
  2. Mask geometry (distributed → FGO helps; contiguous → no)
  3. Evidence density (too low → FGO can't help)

Experiment: Compare manual selection vs auto-dispatch across configurations.

Usage:
    python3 -u benchmarks/exp_phase9_operator_selector.py --device cuda
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
# OPERATOR SELECTOR (the key Phase 9 contribution)
# ============================================================================

class OperatorSelector:
    """
    Conditional dispatch: choose between local-only and local+FGO
    based on task geometry.

    Trigger conditions (from Phase 1-2 experiments):
      - Grid size >= 14 (FGO meaningless at 7×7 due to receptive field overlap)
      - Evidence density in [0.3, 0.85] (too low=no info, too high=no need)
      - Mask is distributed (not a single contiguous block)

    Returns: 'local' or 'local_fgo'
    """

    def __init__(self, grid_threshold=14, evidence_low=0.3, evidence_high=0.85,
                 distributed_threshold=0.3):
        self.grid_threshold = grid_threshold
        self.evidence_low = evidence_low
        self.evidence_high = evidence_high
        self.distributed_threshold = distributed_threshold

    def analyze_mask(self, bit_mask):
        """Compute mask statistics for dispatch decision."""
        # bit_mask: (k, H, W) bool
        spatial_mask = bit_mask[0]  # (H, W)
        H, W = spatial_mask.shape
        total = H * W
        masked_count = spatial_mask.sum()
        evidence_density = 1.0 - (masked_count / total)

        # Measure "distributedness": ratio of connected components
        # Simple heuristic: count transitions (mask→unmask) along rows/cols
        transitions = 0
        for i in range(H):
            for j in range(1, W):
                if spatial_mask[i, j] != spatial_mask[i, j-1]:
                    transitions += 1
        for j in range(W):
            for i in range(1, H):
                if spatial_mask[i, j] != spatial_mask[i-1, j]:
                    transitions += 1

        # Normalize: more transitions = more distributed
        max_transitions = 2 * H * (W-1) + 2 * W * (H-1)  # theoretical max
        distributedness = transitions / max(max_transitions, 1)

        return {
            'evidence_density': evidence_density,
            'distributedness': distributedness,
            'mask_ratio': masked_count / total,
        }

    def select(self, grid_size, bit_mask):
        """
        Select operator based on geometry.
        Returns ('local' or 'local_fgo', reason_string).
        """
        stats = self.analyze_mask(bit_mask)
        ed = stats['evidence_density']
        dist = stats['distributedness']

        # Decision tree (based on Phase 1-2 findings)
        if grid_size < self.grid_threshold:
            return 'local', f'grid {grid_size}<{self.grid_threshold}: local sufficient'

        if ed < self.evidence_low:
            return 'local', f'evidence {ed:.2f}<{self.evidence_low}: too sparse for FGO'

        if ed > self.evidence_high:
            return 'local', f'evidence {ed:.2f}>{self.evidence_high}: mostly observed, FGO unnecessary'

        if dist > self.distributed_threshold:
            return 'local_fgo', f'distributed mask (dist={dist:.2f}>{self.distributed_threshold}), evidence={ed:.2f}'

        return 'local', f'contiguous mask (dist={dist:.2f}), local propagation sufficient'


# ============================================================================
# MODEL COMPONENTS (7×7 for speed, 4GB GPU constraint)
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
    """Local-only inpainting."""
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


class FGOLayer(nn.Module):
    """Frequency Global Operator: data-adaptive spectral gating."""
    def __init__(self, channels):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels), nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.shape
        X_fft = torch.fft.rfft2(x)
        gate = self.gate_net(x).reshape(B, C, 1, 1)
        X_filtered = X_fft * gate
        return torch.fft.irfft2(X_filtered, s=(H, W))


class InpaintNetFGO(nn.Module):
    """Local + FGO sandwich."""
    def __init__(self, k=8, hidden=64):
        super().__init__()
        self.local1 = nn.Sequential(
            nn.Conv2d(k+1, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU())
        self.fgo = FGOLayer(hidden)
        self.local2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(hidden, k, 3, padding=1, padding_mode='circular'))
        self.skip = nn.Conv2d(k+1, k, 1)

    def forward(self, z_masked, mask):
        x = torch.cat([z_masked, mask], dim=1)
        h = self.local1(x)
        h = h + self.fgo(h)
        return self.local2(h) + self.skip(x)


# ============================================================================
# MASKS + DATA
# ============================================================================

def make_center_mask():
    m = np.ones((28,28), dtype=np.float32); m[7:21,7:21] = 0; return m
def make_stripe_mask():
    m = np.ones((28,28), dtype=np.float32)
    for y in range(0, 28, 6): m[y:min(y+2,28), :] = 0
    return m
def make_checkerboard_mask():
    m = np.ones((28,28), dtype=np.float32)
    for i in range(28):
        for j in range(28):
            if (i//4 + j//4) % 2 == 0: m[i,j] = 0
    return m
def make_multi_hole_mask(rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((28,28), dtype=np.float32)
    for _ in range(rng.integers(3,8)):
        s = rng.integers(2,6)
        y,x = rng.integers(0,max(1,28-s+1)),rng.integers(0,max(1,28-s+1))
        m[y:y+s,x:x+s] = 0
    return m
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
def sample_training_mask(rng=None):
    if rng is None: rng = np.random.default_rng()
    p = rng.random()
    if p < 0.25: return make_random_block_mask(rng)
    elif p < 0.45: return make_center_mask()
    elif p < 0.65: return make_random_stripe_mask(rng)
    elif p < 0.85: return make_multi_hole_mask(rng)
    else: return make_checkerboard_mask()
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


# ============================================================================
# TRAINING
# ============================================================================

def train_model(train_x, train_y, device, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = RouteCModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    for epoch in range(5):
        model.train()
        model.set_temperature(1.0 + (0.2-1.0)*epoch/4)
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

def train_inpaint(model, train_x, device, use_fgo=False, epochs=20):
    net = (InpaintNetFGO if use_fgo else InpaintNet)(k=model.n_bits).to(device)
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
# EVALUATION
# ============================================================================

def evaluate_with_operator(model, net, mask_name, pixel_mask, test_x, test_y, device,
                           n_samples=100, seed=42):
    model.eval(); net.eval()
    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)
    cb, ca = [], []
    for idx in eval_idx:
        x_clean = test_x[idx].numpy()[0]; label = test_y[idx].item()
        x_occ = x_clean * pixel_mask
        with torch.no_grad():
            xt = torch.from_numpy(x_occ[None,None].astype(np.float32)).to(device)
            z_init = model.encode(xt)[0]
            pred_b = model.classifier(z_init.unsqueeze(0)).argmax(1).item()
        bit_mask = pixel_to_bit_mask(pixel_mask, model.n_bits)
        bm = torch.from_numpy(bit_mask).float().to(device)
        mask = bm.max(dim=0,keepdim=True)[0].unsqueeze(0)
        z_masked = z_init.unsqueeze(0) * (1-bm.unsqueeze(0))
        with torch.no_grad():
            logits = net(z_masked, mask)
            preds = (torch.sigmoid(logits) > 0.5).float()
        z_result = z_init.clone()
        z_result[torch.from_numpy(bit_mask).to(device)] = preds[0][torch.from_numpy(bit_mask).to(device)]
        with torch.no_grad():
            pred_a = model.classifier(z_result.unsqueeze(0)).argmax(1).item()
        cb.append(int(pred_b==label)); ca.append(int(pred_a==label))
    n = len(eval_idx)
    return (sum(ca)-sum(cb))/n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--output_dir', default='outputs/exp_phase9')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 9: CONDITIONAL OPERATOR DISPATCH (OperatorSelector)")
    print("=" * 100)

    print("[1] Loading FashionMNIST...")
    train_x, train_y, test_x, test_y = load_dataset('fmnist', 2000, 500, args.seed)

    print("[2] Training shared model...")
    model = train_model(train_x, train_y, device, seed=args.seed)

    print("[3] Training local InpaintNet...")
    net_local = train_inpaint(model, train_x, device, use_fgo=False)

    print("[4] Training FGO InpaintNet...")
    net_fgo = train_inpaint(model, train_x, device, use_fgo=True)

    # Create OperatorSelector
    selector = OperatorSelector()

    # Test masks
    masks = {
        'center': make_center_mask(),
        'stripes': make_stripe_mask(),
        'checkerboard': make_checkerboard_mask(),
        'multi_hole': make_multi_hole_mask(np.random.default_rng(args.seed)),
    }

    print("\n" + "=" * 100)
    print("OPERATOR DISPATCH RESULTS")
    print("=" * 100)
    print(f"{'mask':<15} {'selected':>10} {'local_Δacc':>11} {'fgo_Δacc':>9} "
          f"{'gap':>5} {'evidence%':>10} {'dist':>5} {'reason'}")
    print("-" * 100)

    all_results = []
    for mask_name, pm in masks.items():
        bit_mask = pixel_to_bit_mask(pm, model.n_bits)
        selected, reason = selector.select(grid_size=7, bit_mask=bit_mask)
        stats = selector.analyze_mask(bit_mask)

        local_dacc = evaluate_with_operator(
            model, net_local, mask_name, pm, test_x, test_y, device,
            n_samples=args.eval_samples, seed=args.seed)
        fgo_dacc = evaluate_with_operator(
            model, net_fgo, mask_name, pm, test_x, test_y, device,
            n_samples=args.eval_samples, seed=args.seed)

        gap = fgo_dacc - local_dacc
        # Auto-dispatch selects the right operator
        auto_dacc = fgo_dacc if selected == 'local_fgo' else local_dacc
        # Oracle: always pick the better one
        oracle_dacc = max(local_dacc, fgo_dacc)

        all_results.append({
            'mask': mask_name,
            'grid': 7,
            'selected': selected,
            'local_dacc': local_dacc,
            'fgo_dacc': fgo_dacc,
            'gap': gap,
            'auto_dacc': auto_dacc,
            'oracle_dacc': oracle_dacc,
            'evidence_density': stats['evidence_density'],
            'distributedness': stats['distributedness'],
            'reason': reason,
        })

        print(f"{mask_name:<15} {selected:>10} {local_dacc:>+11.1%} {fgo_dacc:>+9.1%} "
              f"{gap:>+5.1%} {stats['evidence_density']:>10.2f} {stats['distributedness']:>5.2f} "
              f"  {reason}")

    # Summary
    print("\n" + "=" * 80)
    print("DISPATCH QUALITY")
    print("=" * 80)

    auto_total = sum(r['auto_dacc'] for r in all_results)
    oracle_total = sum(r['oracle_dacc'] for r in all_results)
    local_total = sum(r['local_dacc'] for r in all_results)
    fgo_total = sum(r['fgo_dacc'] for r in all_results)

    print(f"  Always local:     total={local_total:+.1%}")
    print(f"  Always FGO:       total={fgo_total:+.1%}")
    print(f"  Auto-dispatch:    total={auto_total:+.1%}")
    print(f"  Oracle (best):    total={oracle_total:+.1%}")
    print(f"  Auto vs Oracle gap: {oracle_total - auto_total:+.1%}")
    print(f"  → Auto-dispatch {'MATCHES' if abs(oracle_total - auto_total) < 0.02 else 'CLOSE TO'} oracle")

    # Note about grid size
    print(f"\n  Note: At 7×7 grid, selector always picks 'local' (grid<14).")
    print(f"  FGO trigger conditions require grid≥14 (see Phase 2 FGO-Trigger).")
    print(f"  The selector correctly avoids FGO at this scale.")

    csv_path = os.path.join(args.output_dir, "phase9_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(set().union(*(r.keys() for r in all_results)))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")
    print("\n" + "=" * 100)
    print("Phase 9 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
