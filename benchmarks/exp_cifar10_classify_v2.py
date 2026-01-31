#!/usr/bin/env python3
"""
CIFAR-10 Classification Probe v2 — Semantic Stability Diagnosis
================================================================
Improvements over v1 based on paradigm-level feedback:

1. INTERVENTION STABILITY: Measure repair's effect on z (masked vs unmasked regions)
2. MIXED PROBE: Train on clean+repaired z to diagnose distribution shift vs semantic loss
3. REPAIR-AWARE PROBE: Train directly on repaired z
4. HIERARCHICAL z_sem: Pool z → low-res z_sem for stable semantic readout
5. REPAIR AS PROJECTION: Constrain repair to minimal change (trust region)

Key question: Does repair destroy semantics, or just shift distribution?
If mixed probe recovers → distribution shift (fixable)
If mixed probe also fails → semantic destruction (structural problem)

Usage:
    python3 -u benchmarks/exp_cifar10_classify_v2.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, argparse
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# GUMBEL SIGMOID
# ============================================================================

class GumbelSigmoid(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__(); self.temperature = temperature
    def forward(self, logits):
        if self.training:
            u = torch.rand_like(logits).clamp(1e-8, 1-1e-8)
            noisy = (logits - torch.log(-torch.log(u))) / self.temperature
        else:
            noisy = logits / self.temperature
        soft = torch.sigmoid(noisy); hard = (soft > 0.5).float()
        return hard - soft.detach() + soft
    def set_temperature(self, tau): self.temperature = tau


# ============================================================================
# ENCODER / DECODER
# ============================================================================

class Encoder32(nn.Module):
    def __init__(self, n_bits):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, n_bits, 3, padding=1),
        )
        self.q = GumbelSigmoid()
    def forward(self, x):
        logits = self.net(x); return self.q(logits), logits
    def set_temperature(self, tau): self.q.set_temperature(tau)


class Decoder32(nn.Module):
    def __init__(self, n_bits):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z): return self.net(z)


# ============================================================================
# E_CORE
# ============================================================================

class LocalEnergyCore(nn.Module):
    def __init__(self, n_bits, hidden=64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(9*n_bits-1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def get_context(self, z, bi, i, j):
        B, K, H, W = z.shape; ctx = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = (i+di) % H, (j+dj) % W
                for b in range(K):
                    if di == 0 and dj == 0 and b == bi: continue
                    ctx.append(z[:, b, ni, nj])
        return torch.stack(ctx, dim=1)
    def violation_rate(self, z, n_samples=50):
        B, K, H, W = z.shape; v = []
        for _ in range(min(n_samples, H*W*K)):
            b = torch.randint(K, (1,)).item()
            i = torch.randint(H, (1,)).item()
            j = torch.randint(W, (1,)).item()
            ctx = self.get_context(z, b, i, j)
            pred = (self.predictor(ctx).squeeze(1) > 0).float()
            v.append((pred != z[:, b, i, j]).float().mean().item())
        return np.mean(v)


# ============================================================================
# DENOISER
# ============================================================================

class FreqDenoiser(nn.Module):
    def __init__(self, n_bits):
        super().__init__(); self.n_bits = n_bits
        hid = min(128, max(64, n_bits * 4))
        self.net = nn.Sequential(
            nn.Conv2d(n_bits+1, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, n_bits, 3, padding=1))
        self.skip = nn.Conv2d(n_bits, n_bits, 1)

    def forward(self, z_noisy, noise_level):
        B = z_noisy.shape[0]
        nl = noise_level.view(B, 1, 1, 1).expand(-1, 1, z_noisy.shape[2], z_noisy.shape[3])
        return self.net(torch.cat([z_noisy, nl], dim=1)) + self.skip(z_noisy)

    @torch.no_grad()
    def repair(self, z, mask, n_steps=5, temperature=0.5):
        B, K, H, W = z.shape
        z_rep = z.clone()
        for step in range(n_steps):
            nl = torch.tensor([1.0 - step/n_steps], device=z.device).expand(B)
            logits = self(z_rep, nl)
            probs = torch.sigmoid(logits / temperature)
            z_new = (torch.rand_like(z_rep) < probs).float()
            z_rep = mask * z + (1-mask) * z_new
        logits = self(z_rep, torch.zeros(B, device=z.device))
        z_final = (torch.sigmoid(logits) > 0.5).float()
        return mask * z + (1-mask) * z_final


# ============================================================================
# CLASSIFICATION PROBES
# ============================================================================

class LinearProbe(nn.Module):
    def __init__(self, z_dim, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(z_dim, n_classes)
    def forward(self, z):
        return self.fc(z.reshape(z.shape[0], -1))


class ConvProbe(nn.Module):
    def __init__(self, n_bits, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32*4*4, n_classes),
        )
    def forward(self, z):
        return self.net(z)


class HierarchicalProbe(nn.Module):
    """z_sem (pooled) + z_tex (full) dual-bus probe."""
    def __init__(self, n_bits, z_h, n_classes=10):
        super().__init__()
        # z_sem path: global pool → linear
        self.sem_fc = nn.Linear(n_bits, 64)
        # z_tex path: conv → pool
        self.tex_conv = nn.Sequential(
            nn.Conv2d(n_bits, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        self.head = nn.Linear(64 + 32*4*4, n_classes)

    def forward(self, z):
        # z_sem: global average pool (stable across repair)
        z_sem = z.mean(dim=[2, 3])  # B×K
        sem_feat = F.relu(self.sem_fc(z_sem))
        # z_tex: local spatial features
        tex_feat = self.tex_conv(z)
        return self.head(torch.cat([sem_feat, tex_feat], dim=1))


class SemOnlyProbe(nn.Module):
    """Only use z_sem (global pool) — tests if global statistics carry semantics."""
    def __init__(self, n_bits, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )
    def forward(self, z):
        z_sem = z.mean(dim=[2, 3])
        return self.net(z_sem)


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def train_adc(encoder, decoder, train_x, device, epochs=40, bs=128):
    opt = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=1e-3)
    for epoch in tqdm(range(epochs), desc="ADC"):
        encoder.train(); decoder.train()
        encoder.set_temperature(1.0 + (0.3-1.0)*epoch/(epochs-1))
        perm = torch.randperm(len(train_x)); tl, nb = 0., 0
        for i in range(0, len(train_x), bs):
            x = train_x[perm[i:i+bs]].to(device)
            opt.zero_grad(); z, _ = encoder(x); xh = decoder(z)
            loss = F.mse_loss(xh, x) + 0.5*F.binary_cross_entropy(xh, x)
            loss.backward(); opt.step(); tl += loss.item(); nb += 1
    print(f"      ADC done: loss={tl/nb:.4f}")
    encoder.eval(); decoder.eval()


def encode_dataset(encoder, data_x, device, bs=256):
    z_list = []
    with torch.no_grad():
        for i in range(0, len(data_x), bs):
            z, _ = encoder(data_x[i:i+bs].to(device)); z_list.append(z.cpu())
    return torch.cat(z_list)


def train_ecore(e_core, z_data, device, epochs=10, bs=256):
    eopt = torch.optim.Adam(e_core.parameters(), lr=1e-3)
    K, H, W = z_data.shape[1:]
    for ep in tqdm(range(epochs), desc="E_core"):
        e_core.train(); perm = torch.randperm(len(z_data))
        for i in range(0, len(z_data), bs):
            z = z_data[perm[i:i+bs]].to(device); eopt.zero_grad(); tl_ = 0.
            for _ in range(20):
                b = torch.randint(K, (1,)).item()
                ii = torch.randint(H, (1,)).item()
                jj = torch.randint(W, (1,)).item()
                ctx = e_core.get_context(z, b, ii, jj)
                tl_ += F.binary_cross_entropy_with_logits(
                    e_core.predictor(ctx).squeeze(1), z[:, b, ii, jj])
            (tl_/20).backward(); eopt.step()
    e_core.eval()


def train_denoiser(denoiser, z_data, device, epochs=30, bs=128):
    dopt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)
    for epoch in tqdm(range(epochs), desc="Denoiser"):
        denoiser.train(); perm = torch.randperm(N)
        tl, nb = 0., 0
        for i in range(0, N, bs):
            z_clean = z_data[perm[i:i+bs]].to(device); B_ = z_clean.shape[0]
            noise_level = torch.rand(B_, device=device)
            flip = (torch.rand_like(z_clean) < noise_level.view(B_, 1, 1, 1)).float()
            z_noisy = z_clean*(1-flip) + (1-z_clean)*flip
            dopt.zero_grad()
            logits = denoiser(z_noisy, noise_level)
            loss = F.binary_cross_entropy_with_logits(logits, z_clean)
            loss.backward(); dopt.step(); tl += loss.item(); nb += 1
    print(f"      Denoiser done: loss={tl/nb:.4f}")
    denoiser.eval()


def train_probe(probe, z_data, labels, device, epochs=50, bs=256, lr=1e-3):
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for epoch in tqdm(range(epochs), desc="Probe", leave=False):
        probe.train(); perm = torch.randperm(len(z_data))
        for i in range(0, len(z_data), bs):
            idx = perm[i:i+bs]
            z = z_data[idx].to(device); y = labels[idx].to(device)
            opt.zero_grad()
            loss = F.cross_entropy(probe(z), y)
            loss.backward(); opt.step()
    probe.eval()


def eval_probe(probe, z_data, labels, device, bs=256):
    probe.eval()
    nc, nb = 0, 0
    with torch.no_grad():
        for i in range(0, len(z_data), bs):
            z = z_data[i:i+bs].to(device); y = labels[i:i+bs].to(device)
            nc += (probe(z).argmax(1) == y).sum().item(); nb += len(y)
    return nc / nb


# ============================================================================
# MASKS
# ============================================================================

def make_center_mask(z):
    B, K, H, W = z.shape
    mask = torch.ones(B, K, H, W)
    h4, w4 = H//4, W//4
    mask[:, :, h4:3*h4, w4:3*w4] = 0
    return mask


def apply_repair(denoiser, z_data, device, bs=64):
    """Apply center mask repair to dataset."""
    z_repaired = []
    with torch.no_grad():
        for i in range(0, len(z_data), bs):
            z_batch = z_data[i:i+bs]
            mask = make_center_mask(z_batch).to(device)
            z_masked = z_batch.to(device) * mask
            z_rep = denoiser.repair(z_masked, mask)
            z_repaired.append(z_rep.cpu())
    return torch.cat(z_repaired)


# ============================================================================
# INTERVENTION STABILITY METRICS
# ============================================================================

@torch.no_grad()
def measure_intervention_stability(z_clean, z_repaired, mask_template):
    """Measure how much repair changes z in masked vs unmasked regions.

    Returns dict with:
    - hamming_masked: avg bit flip rate in repaired (masked) region
    - hamming_unmasked: avg bit flip rate in kept (unmasked) region
    - hamming_total: overall bit flip rate
    - change_ratio: masked_change / unmasked_change (should be >> 1 if repair is localized)
    """
    # mask_template: 1=keep, 0=repair
    B = min(len(z_clean), len(z_repaired))
    z_c = z_clean[:B]; z_r = z_repaired[:B]

    diff = (z_c != z_r).float()

    # Use first sample's mask as template (same for all since center mask)
    m = mask_template[:B]

    masked_region = (1 - m)  # where repair happened
    unmasked_region = m       # where evidence was kept

    n_masked = masked_region.sum().item()
    n_unmasked = unmasked_region.sum().item()

    hamming_masked = (diff * masked_region).sum().item() / max(n_masked, 1)
    hamming_unmasked = (diff * unmasked_region).sum().item() / max(n_unmasked, 1)
    hamming_total = diff.mean().item()

    change_ratio = hamming_masked / max(hamming_unmasked, 1e-8)

    return {
        'hamming_masked': hamming_masked,
        'hamming_unmasked': hamming_unmasked,
        'hamming_total': hamming_total,
        'change_ratio': change_ratio,  # >> 1 means repair is localized
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_cifar10_classify_v2')
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("CIFAR-10 CLASSIFICATION PROBE v2 — SEMANTIC STABILITY DIAGNOSIS")
    print("=" * 100)

    # Load CIFAR-10
    print("\n[1] Loading CIFAR-10...")
    from torchvision import datasets, transforms
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())

    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(len(train_ds), args.n_train, replace=False)
    test_idx = rng.choice(len(test_ds), args.n_test, replace=False)

    train_x = torch.stack([train_ds[i][0] for i in train_idx])
    train_y = torch.tensor([train_ds[i][1] for i in train_idx])
    test_x = torch.stack([test_ds[i][0] for i in test_idx])
    test_y = torch.tensor([test_ds[i][1] for i in test_idx])
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")

    # ========================================================================
    # Train world model (32x32x8, no freq — best from v1)
    # ========================================================================
    n_bits = 8; z_h = 32
    print(f"\n[2] Training world model: z={z_h}x{z_h}x{n_bits} = {z_h*z_h*n_bits} bits")

    torch.manual_seed(args.seed)
    encoder = Encoder32(n_bits).to(device)
    decoder = Decoder32(n_bits).to(device)
    denoiser = FreqDenoiser(n_bits).to(device)
    e_core = LocalEnergyCore(n_bits).to(device)

    print("  [A] Training ADC/DAC...")
    train_adc(encoder, decoder, train_x, device, epochs=40, bs=128)

    print("  [B] Encoding datasets...")
    z_train = encode_dataset(encoder, train_x, device)
    z_test = encode_dataset(encoder, test_x, device)
    print(f"    z_train: {z_train.shape}, usage={z_train.mean():.3f}")

    print("  [C] Training E_core...")
    train_ecore(e_core, z_train, device, epochs=10)
    viol = e_core.violation_rate(z_test[:100].to(device))
    print(f"    Violation: {viol:.4f}")

    print("  [D] Training denoiser...")
    train_denoiser(denoiser, z_train, device, epochs=30, bs=128)

    # Cycle metric
    with torch.no_grad():
        zc = z_test[:100].to(device)
        xc = decoder(zc); zcy, _ = encoder(xc)
        cycle = (zc != zcy).float().mean().item()
    print(f"    Cycle: {cycle:.4f}")

    # ========================================================================
    # PHASE 1: Intervention Stability
    # ========================================================================
    print(f"\n{'='*100}")
    print("PHASE 1: INTERVENTION STABILITY — Does repair change only masked region?")
    print("="*100)

    print("  Applying center-mask repair to test set...")
    z_test_repaired = apply_repair(denoiser, z_test, device)

    mask_template = make_center_mask(z_test)
    stability = measure_intervention_stability(z_test, z_test_repaired, mask_template)

    print(f"    Hamming (masked region):   {stability['hamming_masked']:.4f}")
    print(f"    Hamming (unmasked region): {stability['hamming_unmasked']:.4f}")
    print(f"    Hamming (total):           {stability['hamming_total']:.4f}")
    print(f"    Change ratio (masked/unmasked): {stability['change_ratio']:.1f}x")

    if stability['change_ratio'] > 5:
        print("    DIAGNOSIS: Repair is LOCALIZED (good — changes concentrated in masked region)")
    elif stability['change_ratio'] > 2:
        print("    DIAGNOSIS: Repair is PARTIALLY LOCALIZED (some spillover to unmasked region)")
    else:
        print("    DIAGNOSIS: Repair is GLOBAL (bad — rewriting entire z, not just masked region)")

    # Also apply repair to training set (needed for mixed/repair-aware probes)
    print("  Applying center-mask repair to train set...")
    z_train_repaired = apply_repair(denoiser, z_train, device)

    # ========================================================================
    # PHASE 2: Probe Variants
    # ========================================================================
    print(f"\n{'='*100}")
    print("PHASE 2: PROBE VARIANTS — Diagnose distribution shift vs semantic loss")
    print("="*100)

    # Build mixed dataset: 50% clean + 50% repaired (with duplicated labels)
    z_train_mixed = torch.cat([z_train, z_train_repaired])
    train_y_mixed = torch.cat([train_y, train_y])

    z_dim = z_h * z_h * n_bits
    results = {}

    # 5 probe types × 2 test distributions = 10 experiments
    probe_configs = [
        ("conv", "train_clean",   z_train,          train_y),
        ("conv", "train_repaired", z_train_repaired, train_y),
        ("conv", "train_mixed",   z_train_mixed,    train_y_mixed),
        ("hier", "train_clean",   z_train,          train_y),
        ("hier", "train_repaired", z_train_repaired, train_y),
        ("hier", "train_mixed",   z_train_mixed,    train_y_mixed),
        ("sem",  "train_clean",   z_train,          train_y),
        ("sem",  "train_repaired", z_train_repaired, train_y),
        ("sem",  "train_mixed",   z_train_mixed,    train_y_mixed),
        ("linear", "train_clean", z_train,          train_y),
        ("linear", "train_repaired", z_train_repaired, train_y),
        ("linear", "train_mixed", z_train_mixed,    train_y_mixed),
    ]

    for probe_type, train_mode, z_tr, y_tr in probe_configs:
        name = f"{probe_type}_{train_mode}"
        print(f"\n  --- {name} ---")

        if probe_type == "conv":
            probe = ConvProbe(n_bits).to(device)
        elif probe_type == "hier":
            probe = HierarchicalProbe(n_bits, z_h).to(device)
        elif probe_type == "sem":
            probe = SemOnlyProbe(n_bits).to(device)
        elif probe_type == "linear":
            probe = LinearProbe(z_dim).to(device)

        train_probe(probe, z_tr, y_tr, device, epochs=50, bs=256)

        # Test on both clean and repaired
        acc_clean = eval_probe(probe, z_test, test_y, device)
        acc_repair = eval_probe(probe, z_test_repaired, test_y, device)
        delta = acc_repair - acc_clean

        print(f"    test_clean={acc_clean:.3f}  test_repair={acc_repair:.3f}  Δ={delta:+.3f}")

        results[name] = {
            'probe': probe_type, 'train': train_mode,
            'acc_clean': acc_clean, 'acc_repair': acc_repair, 'delta': delta
        }

        del probe; torch.cuda.empty_cache()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print("="*100)

    print(f"\n  Intervention stability:")
    print(f"    Change ratio: {stability['change_ratio']:.1f}x  (masked={stability['hamming_masked']:.4f}, unmasked={stability['hamming_unmasked']:.4f})")

    print(f"\n  {'probe':<10} {'train_on':<18} {'test_clean':>10} {'test_repair':>12} {'delta':>8}")
    print(f"  {'-'*58}")
    for name, r in results.items():
        print(f"  {r['probe']:<10} {r['train']:<18} {r['acc_clean']:>10.3f} {r['acc_repair']:>12.3f} {r['delta']:>+8.3f}")

    # Paradigm diagnosis
    print(f"\n{'='*100}")
    print("PARADIGM DIAGNOSIS")
    print("="*100)

    # Check: does training on repaired z fix the probe?
    conv_clean_on_clean = results.get('conv_train_clean', {}).get('acc_clean', 0)
    conv_clean_on_repair = results.get('conv_train_clean', {}).get('acc_repair', 0)
    conv_repair_on_repair = results.get('conv_train_repaired', {}).get('acc_repair', 0)
    conv_mixed_on_clean = results.get('conv_train_mixed', {}).get('acc_clean', 0)
    conv_mixed_on_repair = results.get('conv_train_mixed', {}).get('acc_repair', 0)

    print(f"\n  Conv probe analysis:")
    print(f"    A) train_clean  → test_clean:    {conv_clean_on_clean:.3f}")
    print(f"    B) train_clean  → test_repair:   {conv_clean_on_repair:.3f}  (v1 showed this crashes)")
    print(f"    C) train_repair → test_repair:   {conv_repair_on_repair:.3f}")
    print(f"    D) train_mixed  → test_clean:    {conv_mixed_on_clean:.3f}")
    print(f"    E) train_mixed  → test_repair:   {conv_mixed_on_repair:.3f}")

    if conv_repair_on_repair > conv_clean_on_repair + 0.05:
        print(f"\n    C >> B → DISTRIBUTION SHIFT (repair preserves semantics, just shifts distribution)")
        print(f"    FIX: Train probe on repaired distribution, or use mixed training")
    else:
        print(f"\n    C ≈ B → SEMANTIC DESTRUCTION (repair actually destroys class-relevant info)")
        print(f"    FIX: Need semantic-stability constraint in repair (projection onto semantic subspace)")

    # Check: does z_sem (global pool) survive repair?
    sem_clean = results.get('sem_train_clean', {}).get('acc_clean', 0)
    sem_repair = results.get('sem_train_clean', {}).get('acc_repair', 0)
    hier_clean = results.get('hier_train_clean', {}).get('acc_clean', 0)
    hier_repair = results.get('hier_train_clean', {}).get('acc_repair', 0)

    print(f"\n  Hierarchical analysis:")
    print(f"    SemOnly:  clean={sem_clean:.3f}  repair={sem_repair:.3f}  Δ={sem_repair-sem_clean:+.3f}")
    print(f"    Hier:     clean={hier_clean:.3f}  repair={hier_repair:.3f}  Δ={hier_repair-hier_clean:+.3f}")
    print(f"    Conv:     clean={conv_clean_on_clean:.3f}  repair={conv_clean_on_repair:.3f}  Δ={conv_clean_on_repair-conv_clean_on_clean:+.3f}")

    if abs(sem_repair - sem_clean) < 0.05:
        print(f"\n    z_sem (global pool) is REPAIR-STABLE — global statistics survive center mask repair")
    else:
        print(f"\n    z_sem (global pool) is REPAIR-SENSITIVE — even global statistics shift under repair")

    print(f"\n{'='*100}")
    print("Done.")
    print("="*100)


if __name__ == "__main__":
    main()
