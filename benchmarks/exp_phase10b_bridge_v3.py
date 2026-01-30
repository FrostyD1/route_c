#!/usr/bin/env python3
"""
Phase 10B: Feature Bridge v3 (INT4 Token + VQ Codebook, Same-Domain Teacher)
==============================================================================
Phase 10/10A failed: binary quantization on ImageNet features = wrong task.

Fix #1: Same-domain teacher (small CNN trained ON FMNIST, not ImageNet ResNet18)
Fix #2: Two discrete forms beyond binary:
  Route A: INT4/INT8 QAT tokens (continuous → fake-quantized multi-level)
  Route B: VQ codebook (continuous → nearest code index, learnable clustering)
Fix #3: Logits distillation (KL) + cycle drift penalty

Teacher: 3-layer CNN → features h (7×7×64), classifier head → 10 classes.
Bridge: h → Q → discrete tokens → R → ĥ, preserving f_head(h) ≈ f_head(ĥ).

Success: agreement ≥ 70% (from current 28%), cycle drift controlled.

Usage:
    python3 -u benchmarks/exp_phase10b_bridge_v3.py --device cuda
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


# ============================================================================
# SAME-DOMAIN TEACHER (trained on FMNIST)
# ============================================================================

class FMNISTTeacher(nn.Module):
    """Small CNN teacher for FMNIST. Split into feat_extractor + head."""
    def __init__(self):
        super().__init__()
        # Feature extractor: 28×28→7×7×64
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),   # 14×14
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 7×7
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),            # 7×7
        )
        # Classifier head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        h = self.features(x)
        return self.head(h)

    def extract(self, x):
        return self.features(x)


def train_teacher(train_x, train_y, test_x, test_y, device, epochs=10):
    """Train FMNIST-native teacher to ~85%+ accuracy."""
    teacher = FMNISTTeacher().to(device)
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)

    for epoch in range(epochs):
        teacher.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(teacher(x), y).backward()
            opt.step()

    teacher.eval()
    with torch.no_grad():
        logits = teacher(test_x[:500].to(device))
        acc = (logits.argmax(1).cpu() == test_y[:500]).float().mean().item()
    print(f"    Teacher accuracy: {acc:.1%}")

    # Freeze
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


# ============================================================================
# QUANTIZATION: INT4/INT8 fake quantize (STE)
# ============================================================================

class FakeQuantINT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n_levels):
        # Per-channel quantization
        B, C, H, W = x.shape
        x_flat = x.permute(1, 0, 2, 3).reshape(C, -1)
        scale = x_flat.abs().amax(dim=1, keepdim=True) / (n_levels // 2)
        scale = scale.clamp(min=1e-8)
        x_flat_q = (x_flat / scale).round().clamp(-(n_levels//2), n_levels//2 - 1) * scale
        return x_flat_q.reshape(C, B, H, W).permute(1, 0, 2, 3)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def fake_quant(x, bits=4):
    n_levels = 2 ** bits
    return FakeQuantINT.apply(x, n_levels)


# ============================================================================
# ROUTE A: INT TOKEN BRIDGE
# ============================================================================

class INTTokenBridge(nn.Module):
    """Q: h(64ch) → k continuous → INT4/8 quantize. R: quantized → ĥ(64ch)."""
    def __init__(self, in_ch=64, k=32, quant_bits=4):
        super().__init__()
        self.quant_bits = quant_bits
        self.Q = nn.Sequential(
            nn.Conv2d(in_ch, k, 3, padding=1), nn.ReLU(),
            nn.Conv2d(k, k, 3, padding=1))
        self.R = nn.Sequential(
            nn.Conv2d(k, in_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(in_ch, in_ch, 3, padding=1))
        self.skip_q = nn.Conv2d(in_ch, k, 1)
        self.skip_r = nn.Conv2d(k, in_ch, 1)

    def encode(self, h):
        t = self.Q(h) + self.skip_q(h)
        return fake_quant(t, self.quant_bits)

    def decode(self, z):
        return self.R(z) + self.skip_r(z)

    def forward(self, h):
        z = self.encode(h)
        h_hat = self.decode(z)
        return z, h_hat


# ============================================================================
# ROUTE B: VQ CODEBOOK BRIDGE
# ============================================================================

class VQCodebook(nn.Module):
    """Vector Quantization with EMA codebook update."""
    def __init__(self, embed_dim=64, num_codes=512, decay=0.99):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_codes = num_codes
        self.decay = decay

        self.register_buffer('codebook', torch.randn(num_codes, embed_dim))
        self.register_buffer('ema_count', torch.ones(num_codes))
        self.register_buffer('ema_weight', self.codebook.clone())

    def forward(self, z_e):
        """z_e: (B, D, H, W) → z_q: (B, D, H, W), indices: (B, H, W)."""
        B, D, H, W = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)  # (BHW, D)

        # Distances
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.codebook.t()
                + self.codebook.pow(2).sum(1, keepdim=True).t())
        indices = dist.argmin(1)  # (BHW,)
        z_q = self.codebook[indices].reshape(B, H, W, D).permute(0, 3, 1, 2)

        # EMA update (training only)
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices, self.num_codes).float()
                self.ema_count.mul_(self.decay).add_(onehot.sum(0), alpha=1-self.decay)
                dw = onehot.t() @ z_flat
                self.ema_weight.mul_(self.decay).add_(dw, alpha=1-self.decay)
                n = self.ema_count.clamp(min=1)
                self.codebook.copy_(self.ema_weight / n.unsqueeze(1))

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, indices.reshape(B, H, W)


class VQBridge(nn.Module):
    """Q: h(64ch) → embedding → VQ index. R: VQ embedding → ĥ(64ch)."""
    def __init__(self, in_ch=64, embed_dim=32, num_codes=512):
        super().__init__()
        self.Q_proj = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1))
        self.vq = VQCodebook(embed_dim, num_codes)
        self.R_proj = nn.Sequential(
            nn.Conv2d(embed_dim, in_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(in_ch, in_ch, 3, padding=1))
        self.skip = nn.Conv2d(embed_dim, in_ch, 1)

    def encode(self, h):
        z_e = self.Q_proj(h)
        z_q, indices = self.vq(z_e)
        return z_q, indices, z_e

    def decode(self, z_q):
        return self.R_proj(z_q) + self.skip(z_q)

    def forward(self, h):
        z_q, indices, z_e = self.encode(h)
        h_hat = self.decode(z_q)
        # VQ commitment loss
        commit_loss = F.mse_loss(z_e, z_q.detach())
        return z_q, h_hat, commit_loss, indices


# ============================================================================
# DATA
# ============================================================================

def load_dataset(train_n=4000, test_n=500, seed=42):
    from torchvision import datasets, transforms
    tr = datasets.FashionMNIST('./data', train=True, download=True,
                                transform=transforms.ToTensor())
    te = datasets.FashionMNIST('./data', train=False, download=True,
                                transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(tr), train_n, replace=False)
    si = rng.choice(len(te), test_n, replace=False)
    return (torch.stack([tr[i][0] for i in ti]), torch.tensor([tr[i][1] for i in ti]),
            torch.stack([te[i][0] for i in si]), torch.tensor([te[i][1] for i in si]))


# ============================================================================
# TRAINING
# ============================================================================

def train_int_bridge(bridge, teacher, train_x, device, epochs=30, batch_size=64):
    opt = torch.optim.Adam(bridge.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        bridge.train()
        tl_kl, tl_mse, tl_cyc, nb = 0., 0., 0., 0
        for (x,) in loader:
            x = x.to(device); opt.zero_grad()
            with torch.no_grad():
                h = teacher.extract(x)
                logits_orig = teacher.head(h)
                probs_orig = F.softmax(logits_orig, dim=1)

            z, h_hat = bridge(h)

            # KL distillation (PRIMARY)
            logits_recon = teacher.head(h_hat)
            loss_kl = F.kl_div(F.log_softmax(logits_recon, dim=1),
                               probs_orig, reduction='batchmean')

            # Feature regularizer (SECONDARY)
            loss_mse = F.mse_loss(h_hat, h)

            # Cycle penalty: h → z → ĥ → z' → Δ
            z2, _ = bridge(h_hat.detach())
            loss_cycle = F.mse_loss(z2, z.detach())

            loss = loss_kl + 0.1 * loss_mse + 0.05 * loss_cycle
            loss.backward(); opt.step()
            tl_kl += loss_kl.item(); tl_mse += loss_mse.item()
            tl_cyc += loss_cycle.item(); nb += 1

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{epochs}: KL={tl_kl/nb:.4f} "
                  f"MSE={tl_mse/nb:.4f} cycle={tl_cyc/nb:.4f}")


def train_vq_bridge(bridge, teacher, train_x, device, epochs=30, batch_size=64):
    opt = torch.optim.Adam(bridge.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        bridge.train()
        tl_kl, tl_com, tl_cyc, nb = 0., 0., 0., 0
        for (x,) in loader:
            x = x.to(device); opt.zero_grad()
            with torch.no_grad():
                h = teacher.extract(x)
                logits_orig = teacher.head(h)
                probs_orig = F.softmax(logits_orig, dim=1)

            z_q, h_hat, commit_loss, indices = bridge(h)

            # KL distillation
            logits_recon = teacher.head(h_hat)
            loss_kl = F.kl_div(F.log_softmax(logits_recon, dim=1),
                               probs_orig, reduction='batchmean')

            # Cycle penalty
            z_q2, _, _, _ = bridge(h_hat.detach())
            loss_cycle = F.mse_loss(z_q2, z_q.detach())

            loss = loss_kl + 0.25 * commit_loss + 0.05 * loss_cycle
            loss.backward(); opt.step()
            tl_kl += loss_kl.item(); tl_com += commit_loss.item()
            tl_cyc += loss_cycle.item(); nb += 1

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{epochs}: KL={tl_kl/nb:.4f} "
                  f"commit={tl_com/nb:.4f} cycle={tl_cyc/nb:.4f}")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_bridge(bridge, teacher, test_x, device, route='int', batch_size=64):
    bridge.eval(); teacher.eval()
    agreements, kl_divs, cosines = [], [], []
    cycle_drifts = {1: [], 3: [], 5: []}

    with torch.no_grad():
        for i in range(0, len(test_x), batch_size):
            x = test_x[i:i+batch_size].to(device)
            h = teacher.extract(x)
            logits_orig = teacher.head(h)

            if route == 'int':
                z, h_hat = bridge(h)
            else:
                z_q, h_hat, _, _ = bridge(h)
                z = z_q

            logits_recon = teacher.head(h_hat)

            # Agreement
            agreements.extend(
                (logits_orig.argmax(1) == logits_recon.argmax(1)).cpu().tolist())

            # KL
            kl = F.kl_div(F.log_softmax(logits_recon, dim=1),
                           F.softmax(logits_orig, dim=1),
                           reduction='none').sum(1)
            kl_divs.extend(kl.cpu().tolist())

            # Cosine
            h_flat = h.reshape(h.shape[0], -1)
            hh_flat = h_hat.reshape(h_hat.shape[0], -1)
            cosines.extend(
                F.cosine_similarity(h_flat, hh_flat, dim=1).cpu().tolist())

            # Multi-cycle drift
            z_current = z.clone()
            h_current = h_hat.clone()
            for c in range(5):
                if route == 'int':
                    z_next, h_next = bridge(h_current)
                else:
                    z_next, h_next, _, _ = bridge(h_current)
                if (c + 1) in cycle_drifts:
                    drift = F.mse_loss(z_next, z, reduction='none').mean(dim=(1,2,3))
                    cycle_drifts[c+1].extend(drift.cpu().tolist())
                z_current = z_next
                h_current = h_next

    return {
        'agreement': np.mean(agreements),
        'kl_div': np.mean(kl_divs),
        'cosine': np.mean(cosines),
        'drift_1': np.mean(cycle_drifts[1]),
        'drift_3': np.mean(cycle_drifts[3]),
        'drift_5': np.mean(cycle_drifts[5]),
    }


def evaluate_repairability(bridge, teacher, test_x, device, route='int',
                           n_samples=200, seed=42):
    """Test: corrupt h spatially → encode → repair z → decode → function recovery."""
    bridge.eval(); teacher.eval()
    rng = np.random.default_rng(seed)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    agree_clean, agree_corrupt, agree_manual_repair = [], [], []

    with torch.no_grad():
        for idx in eval_idx:
            x = test_x[idx:idx+1].to(device)
            h = teacher.extract(x)
            logits_clean = teacher.head(h)
            pred_clean = logits_clean.argmax(1)

            # Corrupt: center block mask on features
            h_corrupt = h.clone()
            _, C, H, W = h.shape
            h4, w4 = H // 4, W // 4
            h_corrupt[:, :, h4:H-h4, w4:W-w4] = 0

            # No repair
            if route == 'int':
                z_c, h_hat_c = bridge(h_corrupt)
            else:
                z_c, h_hat_c, _, _ = bridge(h_corrupt)
            pred_corrupt = teacher.head(h_hat_c).argmax(1)

            # "Manual repair": replace corrupted z positions with clean z
            if route == 'int':
                z_clean, _ = bridge(h)
            else:
                z_clean, _, _, _ = bridge(h)

            z_repaired = z_c.clone()
            z_repaired[:, :, h4:H-h4, w4:W-w4] = z_clean[:, :, h4:H-h4, w4:W-w4]
            h_hat_rep = bridge.decode(z_repaired)
            pred_repaired = teacher.head(h_hat_rep).argmax(1)

            agree_clean.append((pred_clean == pred_clean).item())
            agree_corrupt.append((pred_corrupt == pred_clean).item())
            agree_manual_repair.append((pred_repaired == pred_clean).item())

    return {
        'agree_corrupt': np.mean(agree_corrupt),
        'agree_repaired': np.mean(agree_manual_repair),
        'repair_gain': np.mean(agree_manual_repair) - np.mean(agree_corrupt),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_phase10b')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 10B: FEATURE BRIDGE v3 (SAME-DOMAIN TEACHER + INT4/VQ)")
    print("=" * 100)

    print("[1] Loading FMNIST...")
    train_x, train_y, test_x, test_y = load_dataset(4000, 500, args.seed)

    print("[2] Training same-domain teacher (small CNN on FMNIST)...")
    teacher = train_teacher(train_x, train_y, test_x, test_y, device, epochs=15)

    # Route A: INT token bridge (sweep bits)
    configs = []

    print("\n" + "=" * 80)
    print("ROUTE A: INT TOKEN BRIDGE")
    print("=" * 80)

    for quant_bits in [8, 4]:
        for k in [32, 16]:
            name = f'INT{quant_bits}_k{k}'
            print(f"\n  [{name}] Training...")
            torch.manual_seed(args.seed)
            bridge = INTTokenBridge(in_ch=64, k=k, quant_bits=quant_bits).to(device)
            train_int_bridge(bridge, teacher, train_x, device, epochs=30)
            r = evaluate_bridge(bridge, teacher, test_x, device, route='int')
            rep = evaluate_repairability(bridge, teacher, test_x, device, route='int')
            r.update(rep)
            r['config'] = name
            r['route'] = 'INT'
            r['bits_per_pos'] = k * quant_bits
            r['params'] = sum(p.numel() for p in bridge.parameters())
            configs.append(r)
            print(f"    agree={r['agreement']:.1%} cosine={r['cosine']:.4f} "
                  f"KL={r['kl_div']:.4f} drift1={r['drift_1']:.4f} "
                  f"repair_gain={r['repair_gain']:+.1%}")
            del bridge; torch.cuda.empty_cache()

    # Route B: VQ codebook bridge
    print("\n" + "=" * 80)
    print("ROUTE B: VQ CODEBOOK BRIDGE")
    print("=" * 80)

    for num_codes in [256, 512]:
        for embed_dim in [32, 16]:
            name = f'VQ_K{num_codes}_d{embed_dim}'
            print(f"\n  [{name}] Training...")
            torch.manual_seed(args.seed)
            bridge = VQBridge(in_ch=64, embed_dim=embed_dim, num_codes=num_codes).to(device)
            train_vq_bridge(bridge, teacher, train_x, device, epochs=30)
            r = evaluate_bridge(bridge, teacher, test_x, device, route='vq')
            rep = evaluate_repairability(bridge, teacher, test_x, device, route='vq')
            r.update(rep)
            r['config'] = name
            r['route'] = 'VQ'
            bits_per_pos = np.log2(num_codes)
            r['bits_per_pos'] = bits_per_pos
            r['params'] = sum(p.numel() for p in bridge.parameters())
            configs.append(r)
            print(f"    agree={r['agreement']:.1%} cosine={r['cosine']:.4f} "
                  f"KL={r['kl_div']:.4f} drift1={r['drift_1']:.4f} "
                  f"repair_gain={r['repair_gain']:+.1%}")
            del bridge; torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 100)
    print("BRIDGE COMPARISON")
    print("=" * 100)
    print(f"{'config':<18} {'route':>5} {'bits/pos':>9} {'agree%':>8} {'cosine':>8} "
          f"{'KL':>6} {'drift1':>7} {'drift5':>7} {'repair':>8}")
    print("-" * 85)

    for r in sorted(configs, key=lambda x: -x['agreement']):
        print(f"{r['config']:<18} {r['route']:>5} {r['bits_per_pos']:>9.0f} "
              f"{r['agreement']:>7.1%} {r['cosine']:>8.4f} {r['kl_div']:>6.3f} "
              f"{r['drift_1']:>7.4f} {r['drift_5']:>7.4f} {r['repair_gain']:>+8.1%}")

    best = max(configs, key=lambda x: x['agreement'])
    print(f"\n  Best: {best['config']} → agreement={best['agreement']:.1%}")
    print(f"  vs Phase 10 binary best: 34.0%")
    print(f"  Improvement: {best['agreement'] - 0.34:+.1%}")

    viable = [r for r in configs if r['agreement'] >= 0.70]
    if viable:
        print(f"\n  {len(viable)} configs reached ≥70% agreement target!")
        for v in viable:
            print(f"    {v['config']}: {v['agreement']:.1%}")
    else:
        print(f"\n  No config reached 70%. Best is {best['agreement']:.1%}.")
        print(f"  Next steps: increase teacher capacity, more training data, or larger k/codebook.")

    csv_path = os.path.join(args.output_dir, "phase10b_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(set().union(*(r.keys() for r in configs)))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in configs: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")
    print("\n" + "=" * 100)
    print("Phase 10B experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
