#!/usr/bin/env python3
"""
Phase 6: Constraint Writing Interface
=======================================
Test: Can we write rotation/scale invariance as E_core constraints
and measure improvement in robustness?

Three approaches:
  1. E_core + D4 equivariant constraint (average energy over 4 rotations)
  2. Augmentation baseline (train with rotated data, no constraint)
  3. No constraint (baseline)

Measure: E_core violation rate and Δacc under rotation perturbation.

Usage:
    python3 -u benchmarks/exp_phase6_constraints.py --device cuda
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

    def energy(self, z):
        logits = self.forward(z)
        loss = F.binary_cross_entropy_with_logits(logits, z, reduction='none')
        return loss.sum(dim=(1,2,3))

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


def train_model(train_x, train_y, device, constraint='none', aug_rotate=False,
                epochs=5, seed=42):
    """
    Train with optional D4 constraint or rotation augmentation.

    constraint='none': standard training
    constraint='d4': add E_core equivariance loss over 4 rotations
    aug_rotate=True: augment training data with 90° rotations
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model = RouteCModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        model.set_temperature(1.0 + (0.2-1.0)*epoch/max(1,epochs-1))
        el,nb = 0.,0
        for x,y in loader:
            x = x.to(device); opt.zero_grad()

            if aug_rotate:
                # Randomly rotate each sample
                rots = torch.randint(0, 4, (x.shape[0],))
                x_aug = x.clone()
                for i in range(x.shape[0]):
                    if rots[i] > 0:
                        x_aug[i] = torch.rot90(x[i], rots[i].item(), [1, 2])
                x = x_aug

            z = model.encode(x); x_hat = model.decode(z)
            cl = model.local_pred(z)
            lr_ = F.binary_cross_entropy(x_hat.clamp(1e-6,1-1e-6), x)
            m = torch.rand_like(z) < 0.15
            lc = F.binary_cross_entropy_with_logits(cl[m], z.detach()[m]) if m.any() else torch.tensor(0.,device=device)

            loss = lr_ + 0.5*lc

            if constraint == 'd4':
                # D4 equivariance: E_core should be similar across rotations of z
                with torch.no_grad():
                    z_hard = (z > 0.5).float()
                e0 = model.local_pred.energy(z_hard)
                e90 = model.local_pred.energy(torch.rot90(z_hard, 1, [2,3]))
                e180 = model.local_pred.energy(torch.rot90(z_hard, 2, [2,3]))
                e270 = model.local_pred.energy(torch.rot90(z_hard, 3, [2,3]))
                e_stack = torch.stack([e0, e90, e180, e270], dim=0)
                # Minimize variance across rotations
                l_d4 = e_stack.var(dim=0).mean()
                loss = loss + 0.1 * l_d4

            loss.backward(); opt.step(); el += loss.item(); nb += 1

    # Frozen probe
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


def evaluate_rotation_robustness(model, test_x, test_y, device, n_samples=200, seed=42):
    """Measure probe accuracy and E_core violation under 0/90/180/270° rotation."""
    model.eval()
    rng = np.random.default_rng(seed+200)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    results = {}
    for rot in [0, 1, 2, 3]:
        accs, viols = [], []
        for idx in eval_idx:
            x = test_x[idx:idx+1].to(device)
            if rot > 0:
                x = torch.rot90(x, rot, [2, 3])
            with torch.no_grad():
                z = model.encode(x)
                z_hard = (z > 0.5).float()
                pred = model.classifier(z_hard).argmax(1).item()
                # E_core violation
                logits = model.local_pred(z_hard)
                preds_core = (torch.sigmoid(logits) > 0.5).float()
                viol = (preds_core != z_hard).float().mean().item()
            accs.append(int(pred == test_y[idx].item()))
            viols.append(viol)
        results[f'rot{rot*90}'] = {
            'acc': np.mean(accs),
            'viol': np.mean(viols),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=200)
    parser.add_argument('--output_dir', default='outputs/exp_phase6')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 100)
    print("PHASE 6: CONSTRAINT WRITING INTERFACE (D4 SYMMETRY)")
    print("=" * 100)

    print("[1] Loading FashionMNIST...")
    train_x, train_y, test_x, test_y = load_dataset('fmnist', 2000, 500, args.seed)

    configs = [
        ('no_constraint', 'none', False),
        ('d4_constraint', 'd4', False),
        ('aug_rotate', 'none', True),
        ('d4_plus_aug', 'd4', True),
    ]

    all_results = []

    for name, constraint, aug in configs:
        print(f"\n[2] Training: {name}...")
        model = train_model(train_x, train_y, device, constraint=constraint,
                           aug_rotate=aug, seed=args.seed)
        rot_results = evaluate_rotation_robustness(
            model, test_x, test_y, device, n_samples=args.eval_samples, seed=args.seed)
        all_results.append((name, rot_results))

    # Display
    print("\n" + "=" * 100)
    print("ROTATION ROBUSTNESS RESULTS")
    print("=" * 100)
    print(f"{'config':<18} {'rot0_acc':>9} {'rot90_acc':>9} {'rot180_acc':>10} "
          f"{'rot270_acc':>10} {'acc_var':>8}")
    print("-" * 70)

    for name, rr in all_results:
        accs = [rr[f'rot{r}']['acc'] for r in [0, 90, 180, 270]]
        var = np.var(accs)
        print(f"{name:<18} {accs[0]:>9.1%} {accs[1]:>9.1%} {accs[2]:>10.1%} "
              f"{accs[3]:>10.1%} {var:>8.4f}")

    print(f"\n{'config':<18} {'rot0_viol':>9} {'rot90_viol':>10} {'rot180_viol':>11} "
          f"{'rot270_viol':>11} {'viol_var':>9}")
    print("-" * 70)

    for name, rr in all_results:
        viols = [rr[f'rot{r}']['viol'] for r in [0, 90, 180, 270]]
        var = np.var(viols)
        print(f"{name:<18} {viols[0]:>9.3f} {viols[1]:>10.3f} {viols[2]:>11.3f} "
              f"{viols[3]:>11.3f} {var:>9.6f}")

    print("\n" + "=" * 80)
    print("CONSTRAINT VERDICT")
    print("=" * 80)
    no_c = [r for n,r in all_results if n == 'no_constraint'][0]
    d4_c = [r for n,r in all_results if n == 'd4_constraint'][0]
    aug_c = [r for n,r in all_results if n == 'aug_rotate'][0]

    no_var = np.var([no_c[f'rot{r}']['acc'] for r in [0,90,180,270]])
    d4_var = np.var([d4_c[f'rot{r}']['acc'] for r in [0,90,180,270]])
    aug_var = np.var([aug_c[f'rot{r}']['acc'] for r in [0,90,180,270]])

    print(f"  Accuracy variance across rotations:")
    print(f"    no_constraint: {no_var:.4f}")
    print(f"    d4_constraint: {d4_var:.4f}")
    print(f"    aug_rotate:    {aug_var:.4f}")
    print(f"  → D4 constraint {'REDUCES' if d4_var < no_var else 'DOES NOT REDUCE'} rotation sensitivity")
    print(f"  → Aug rotate {'REDUCES' if aug_var < no_var else 'DOES NOT REDUCE'} rotation sensitivity")

    # Save
    csv_path = os.path.join(args.output_dir, "phase6_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'rotation', 'accuracy', 'violation_rate'])
        for name, rr in all_results:
            for rot in [0, 90, 180, 270]:
                w.writerow([name, rot, rr[f'rot{rot}']['acc'], rr[f'rot{rot}']['viol']])

    print(f"\nCSV saved to {csv_path}")
    print("\n" + "=" * 100)
    print("Phase 6 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
