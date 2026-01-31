#!/usr/bin/env python3
"""
CIFAR-10 Classification Probe v3 — Hierarchical Protocol + Semantic Carrier
=============================================================================
Phase C2: z_sem / z_tex dual bus
  - z_tex: 32×32×K (texture/detail, writable by repair)
  - z_sem: 8×8×K' (semantic/stable, pooled from encoder, repair-frozen)
  - Repair only writes z_tex, z_sem stays intact
  - Probe reads z_sem (or z_sem + pooled z_tex)

Phase C3: Self-supervised semantic carrier (no labels!)
  - Contrastive loss on z_sem: same image under 2 augmentations → z_sem close
  - Forces z_sem to carry stable semantics without any classification loss
  - Then evaluate with mixed probe (clean + repaired)

Evaluation protocol (standardized):
  - Always report: clean_acc, repair_acc, stability_gap = |clean - repair|
  - Always use mixed probe (train on clean + repaired)

Usage:
    python3 -u benchmarks/exp_cifar10_classify_v3.py --device cuda
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
# HIERARCHICAL ENCODER / DECODER (z_tex + z_sem)
# ============================================================================

class HierEncoder(nn.Module):
    """Produces z_tex (32×32×K_tex) + z_sem (8×8×K_sem) from 32×32×3 input."""
    def __init__(self, k_tex=8, k_sem=16):
        super().__init__()
        self.k_tex = k_tex; self.k_sem = k_sem
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        # z_tex head: stay at 32×32
        self.tex_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, k_tex, 3, padding=1),
        )
        # z_sem head: downsample to 8×8
        self.sem_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),  # 16×16
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),   # 8×8
            nn.Conv2d(64, k_sem, 3, padding=1),
        )
        self.q_tex = GumbelSigmoid()
        self.q_sem = GumbelSigmoid()

    def forward(self, x):
        feat = self.backbone(x)
        z_tex = self.q_tex(self.tex_head(feat))
        z_sem = self.q_sem(self.sem_head(feat))
        return z_tex, z_sem

    def set_temperature(self, tau):
        self.q_tex.set_temperature(tau)
        self.q_sem.set_temperature(tau)


class HierDecoder(nn.Module):
    """Reconstruct from z_tex (32×32×K_tex) + z_sem (8×8×K_sem) → 32×32×3."""
    def __init__(self, k_tex=8, k_sem=16):
        super().__init__()
        # Upsample z_sem to 32×32
        self.sem_up = nn.Sequential(
            nn.ConvTranspose2d(k_sem, 32, 4, stride=2, padding=1),  # 16×16
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),     # 32×32
            nn.BatchNorm2d(32), nn.ReLU(),
        )
        # Merge z_tex + upsampled z_sem → reconstruct
        self.merge = nn.Sequential(
            nn.Conv2d(k_tex + 32, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z_tex, z_sem):
        sem_feat = self.sem_up(z_sem)  # 8×8 → 32×32
        combined = torch.cat([z_tex, sem_feat], dim=1)
        return self.merge(combined)


# ============================================================================
# FLAT ENCODER / DECODER (baseline, no hierarchy)
# ============================================================================

class FlatEncoder(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, n_bits, 3, padding=1),
        )
        self.q = GumbelSigmoid()
    def forward(self, x):
        return self.q(self.net(x)), None  # z_tex, z_sem=None
    def set_temperature(self, tau): self.q.set_temperature(tau)


class FlatDecoder(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z_tex, z_sem=None): return self.net(z_tex)


# ============================================================================
# DENOISER (repairs z_tex only)
# ============================================================================

class Denoiser(nn.Module):
    def __init__(self, n_bits):
        super().__init__()
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
    def repair(self, z_tex, mask, n_steps=5, temperature=0.5):
        B, K, H, W = z_tex.shape
        z_rep = z_tex.clone()
        for step in range(n_steps):
            nl = torch.tensor([1.0 - step/n_steps], device=z_tex.device).expand(B)
            logits = self(z_rep, nl)
            probs = torch.sigmoid(logits / temperature)
            z_new = (torch.rand_like(z_rep) < probs).float()
            z_rep = mask * z_tex + (1-mask) * z_new
        logits = self(z_rep, torch.zeros(B, device=z_tex.device))
        z_final = (torch.sigmoid(logits) > 0.5).float()
        return mask * z_tex + (1-mask) * z_final


# ============================================================================
# CONTRASTIVE LOSS (self-supervised z_sem carrier, no labels)
# ============================================================================

class ContrastiveHead(nn.Module):
    """Projects z_sem into embedding space for contrastive loss."""
    def __init__(self, k_sem, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(k_sem, 128), nn.ReLU(),
            nn.Linear(128, proj_dim),
        )
    def forward(self, z_sem):
        return F.normalize(self.net(z_sem), dim=1)


def nt_xent_loss(z1, z2, temperature=0.5):
    """NT-Xent (SimCLR) contrastive loss."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # 2B × D
    sim = torch.mm(z, z.t()) / temperature  # 2B × 2B
    # Mask out self-similarity
    mask = torch.eye(2*B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)


# ============================================================================
# AUGMENTATION for contrastive learning
# ============================================================================

def augment_batch(x):
    """Random crop + color jitter for contrastive pairs. GPU-friendly."""
    B, C, H, W = x.shape
    # Random horizontal flip
    if torch.rand(1).item() > 0.5:
        x = x.flip(-1)
    # Random crop (pad 4, then random crop back to 32×32)
    x = F.pad(x, [4, 4, 4, 4], mode='reflect')
    i = torch.randint(0, 8, (1,)).item()
    j = torch.randint(0, 8, (1,)).item()
    x = x[:, :, i:i+H, j:j+W]
    # Color jitter: brightness + contrast
    brightness = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=x.device)
    x = (x * brightness).clamp(0, 1)
    return x


# ============================================================================
# PROBES
# ============================================================================

class SemProbe(nn.Module):
    """Probe on z_sem only."""
    def __init__(self, k_sem, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(k_sem, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(32*4, n_classes),
        )
    def forward(self, z): return self.net(z)


class TexProbe(nn.Module):
    """Probe on z_tex only (for comparison)."""
    def __init__(self, k_tex, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(k_tex, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32*16, n_classes),
        )
    def forward(self, z): return self.net(z)


class DualProbe(nn.Module):
    """Probe on z_sem + z_tex (pooled)."""
    def __init__(self, k_tex, k_sem, n_classes=10):
        super().__init__()
        self.sem_conv = nn.Sequential(
            nn.Conv2d(k_sem, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )
        self.tex_conv = nn.Sequential(
            nn.Conv2d(k_tex, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )
        self.head = nn.Linear(32*4 + 32*4, n_classes)

    def forward(self, z_tex, z_sem):
        s = self.sem_conv(z_sem)
        t = self.tex_conv(z_tex)
        return self.head(torch.cat([s, t], dim=1))


class FlatConvProbe(nn.Module):
    """Conv probe on flat z (for baseline comparison)."""
    def __init__(self, n_bits, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32*16, n_classes),
        )
    def forward(self, z): return self.net(z)


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def train_hier_adc(encoder, decoder, train_x, device, epochs=40, bs=128):
    """Train hierarchical encoder/decoder."""
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    for epoch in tqdm(range(epochs), desc="ADC"):
        encoder.train(); decoder.train()
        encoder.set_temperature(1.0 + (0.3-1.0)*epoch/(epochs-1))
        perm = torch.randperm(len(train_x)); tl, nb = 0., 0
        for i in range(0, len(train_x), bs):
            x = train_x[perm[i:i+bs]].to(device)
            opt.zero_grad()
            z_tex, z_sem = encoder(x)
            xh = decoder(z_tex, z_sem)
            loss = F.mse_loss(xh, x) + 0.5 * F.binary_cross_entropy(xh, x)
            loss.backward(); opt.step(); tl += loss.item(); nb += 1
    print(f"      ADC done: loss={tl/nb:.4f}")
    encoder.eval(); decoder.eval()


def train_contrastive(encoder, contra_head, train_x, device, epochs=30, bs=128):
    """Self-supervised contrastive training on z_sem (no labels!)."""
    params = list(contra_head.parameters())
    # Optionally fine-tune sem_head of encoder too
    if hasattr(encoder, 'sem_head'):
        params += list(encoder.sem_head.parameters())
        params += list(encoder.q_sem.parameters())
    opt = torch.optim.Adam(params, lr=5e-4)

    for epoch in tqdm(range(epochs), desc="Contrastive"):
        encoder.train(); contra_head.train()
        perm = torch.randperm(len(train_x)); tl, nb = 0., 0
        for i in range(0, len(train_x), bs):
            x = train_x[perm[i:i+bs]].to(device)
            # Two augmented views
            x1 = augment_batch(x)
            x2 = augment_batch(x)
            opt.zero_grad()
            _, z_sem1 = encoder(x1)
            _, z_sem2 = encoder(x2)
            e1 = contra_head(z_sem1)
            e2 = contra_head(z_sem2)
            loss = nt_xent_loss(e1, e2, temperature=0.5)
            loss.backward(); opt.step(); tl += loss.item(); nb += 1
    print(f"      Contrastive done: loss={tl/nb:.4f}")
    encoder.eval(); contra_head.eval()


def encode_dataset_hier(encoder, data_x, device, bs=128):
    tex_list, sem_list = [], []
    with torch.no_grad():
        for i in range(0, len(data_x), bs):
            z_tex, z_sem = encoder(data_x[i:i+bs].to(device))
            tex_list.append(z_tex.cpu())
            sem_list.append(z_sem.cpu())
    return torch.cat(tex_list), torch.cat(sem_list)


def encode_dataset_flat(encoder, data_x, device, bs=128):
    z_list = []
    with torch.no_grad():
        for i in range(0, len(data_x), bs):
            z, _ = encoder(data_x[i:i+bs].to(device))
            z_list.append(z.cpu())
    return torch.cat(z_list)


def train_denoiser(denoiser, z_tex_data, device, epochs=30, bs=128):
    dopt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_tex_data)
    for epoch in tqdm(range(epochs), desc="Denoiser"):
        denoiser.train(); perm = torch.randperm(N)
        tl, nb = 0., 0
        for i in range(0, N, bs):
            z = z_tex_data[perm[i:i+bs]].to(device); B_ = z.shape[0]
            nl = torch.rand(B_, device=device)
            flip = (torch.rand_like(z) < nl.view(B_, 1, 1, 1)).float()
            z_noisy = z*(1-flip) + (1-z)*flip
            dopt.zero_grad()
            logits = denoiser(z_noisy, nl)
            loss = F.binary_cross_entropy_with_logits(logits, z)
            loss.backward(); dopt.step(); tl += loss.item(); nb += 1
    print(f"      Denoiser done: loss={tl/nb:.4f}")
    denoiser.eval()


def make_center_mask(z_tex):
    B, K, H, W = z_tex.shape
    mask = torch.ones(B, K, H, W)
    h4, w4 = H//4, W//4
    mask[:, :, h4:3*h4, w4:3*w4] = 0
    return mask


def apply_tex_repair(denoiser, z_tex_data, device, bs=64):
    """Repair z_tex only. z_sem untouched."""
    z_repaired = []
    with torch.no_grad():
        for i in range(0, len(z_tex_data), bs):
            z = z_tex_data[i:i+bs]
            mask = make_center_mask(z).to(device)
            z_masked = z.to(device) * mask
            z_rep = denoiser.repair(z_masked, mask)
            z_repaired.append(z_rep.cpu())
    return torch.cat(z_repaired)


def train_probe_simple(probe, z_data, labels, device, epochs=50, bs=256, lr=1e-3):
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


def train_dual_probe(probe, z_tex, z_sem, labels, device, epochs=50, bs=256, lr=1e-3):
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for epoch in tqdm(range(epochs), desc="Probe", leave=False):
        probe.train(); perm = torch.randperm(len(z_tex))
        for i in range(0, len(z_tex), bs):
            idx = perm[i:i+bs]
            zt = z_tex[idx].to(device); zs = z_sem[idx].to(device)
            y = labels[idx].to(device)
            opt.zero_grad()
            loss = F.cross_entropy(probe(zt, zs), y)
            loss.backward(); opt.step()
    probe.eval()


def eval_probe_simple(probe, z_data, labels, device, bs=256):
    probe.eval(); nc, nb = 0, 0
    with torch.no_grad():
        for i in range(0, len(z_data), bs):
            z = z_data[i:i+bs].to(device); y = labels[i:i+bs].to(device)
            nc += (probe(z).argmax(1) == y).sum().item(); nb += len(y)
    return nc / nb


def eval_dual_probe(probe, z_tex, z_sem, labels, device, bs=256):
    probe.eval(); nc, nb = 0, 0
    with torch.no_grad():
        for i in range(0, len(z_tex), bs):
            zt = z_tex[i:i+bs].to(device); zs = z_sem[i:i+bs].to(device)
            y = labels[i:i+bs].to(device)
            nc += (probe(zt, zs).argmax(1) == y).sum().item(); nb += len(y)
    return nc / nb


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_cifar10_classify_v3')
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("CIFAR-10 CLASSIFICATION v3 — HIERARCHICAL PROTOCOL + SEMANTIC CARRIER")
    print("=" * 100)

    # Load data
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

    results = {}

    # ========================================================================
    # CONFIG A: FLAT BASELINE (v2 reproduction, mixed probe)
    # ========================================================================
    print(f"\n{'='*100}")
    print("CONFIG A: FLAT BASELINE (32×32×8, no hierarchy)")
    print("="*100)

    torch.manual_seed(args.seed)
    k_tex_flat = 8
    enc_flat = FlatEncoder(k_tex_flat).to(device)
    dec_flat = FlatDecoder(k_tex_flat).to(device)
    den_flat = Denoiser(k_tex_flat).to(device)

    print("  [A] Training ADC...")
    train_hier_adc(enc_flat, dec_flat, train_x, device, epochs=40)

    print("  [B] Encoding...")
    ztex_train_flat = encode_dataset_flat(enc_flat, train_x, device)
    ztex_test_flat = encode_dataset_flat(enc_flat, test_x, device)
    print(f"    z_tex: {ztex_train_flat.shape}")

    print("  [C] Training denoiser...")
    train_denoiser(den_flat, ztex_train_flat, device, epochs=30)

    print("  [D] Repair...")
    ztex_train_rep_flat = apply_tex_repair(den_flat, ztex_train_flat, device)
    ztex_test_rep_flat = apply_tex_repair(den_flat, ztex_test_flat, device)

    # Mixed data
    ztex_train_mixed_flat = torch.cat([ztex_train_flat, ztex_train_rep_flat])
    y_mixed = torch.cat([train_y, train_y])

    print("  [E] Mixed probe (conv)...")
    probe_flat = FlatConvProbe(k_tex_flat).to(device)
    train_probe_simple(probe_flat, ztex_train_mixed_flat, y_mixed, device, epochs=50)
    acc_clean_flat = eval_probe_simple(probe_flat, ztex_test_flat, test_y, device)
    acc_rep_flat = eval_probe_simple(probe_flat, ztex_test_rep_flat, test_y, device)
    gap_flat = abs(acc_clean_flat - acc_rep_flat)
    print(f"    FLAT: clean={acc_clean_flat:.3f}  repair={acc_rep_flat:.3f}  gap={gap_flat:.3f}")
    results['flat_baseline'] = {'clean': acc_clean_flat, 'repair': acc_rep_flat, 'gap': gap_flat}

    del enc_flat, dec_flat, den_flat, probe_flat
    torch.cuda.empty_cache()

    # ========================================================================
    # CONFIG B: HIERARCHICAL (z_tex 32×32×8 + z_sem 8×8×16), no contrastive
    # ========================================================================
    print(f"\n{'='*100}")
    print("CONFIG B: HIERARCHICAL (z_tex 32×32×8 + z_sem 8×8×16, no contrastive)")
    print("="*100)

    torch.manual_seed(args.seed)
    k_tex, k_sem = 8, 16
    enc_hier = HierEncoder(k_tex, k_sem).to(device)
    dec_hier = HierDecoder(k_tex, k_sem).to(device)
    den_hier = Denoiser(k_tex).to(device)

    enc_p = sum(p.numel() for p in enc_hier.parameters())
    dec_p = sum(p.numel() for p in dec_hier.parameters())
    print(f"    Params: enc={enc_p:,} dec={dec_p:,}")
    total_bits = 32*32*k_tex + 8*8*k_sem
    print(f"    Total bits: {32*32*k_tex} (tex) + {8*8*k_sem} (sem) = {total_bits}")

    print("  [A] Training ADC...")
    train_hier_adc(enc_hier, dec_hier, train_x, device, epochs=40)

    print("  [B] Encoding...")
    ztex_train, zsem_train = encode_dataset_hier(enc_hier, train_x, device)
    ztex_test, zsem_test = encode_dataset_hier(enc_hier, test_x, device)
    print(f"    z_tex: {ztex_train.shape}, z_sem: {zsem_train.shape}")

    print("  [C] Training denoiser (z_tex only)...")
    train_denoiser(den_hier, ztex_train, device, epochs=30)

    print("  [D] Repair z_tex (z_sem stays intact!)...")
    ztex_train_rep = apply_tex_repair(den_hier, ztex_train, device)
    ztex_test_rep = apply_tex_repair(den_hier, ztex_test, device)
    # z_sem stays the SAME for repaired

    # Mixed data for z_tex
    ztex_train_mixed = torch.cat([ztex_train, ztex_train_rep])
    zsem_train_mixed = torch.cat([zsem_train, zsem_train])  # z_sem repeated
    y_mixed = torch.cat([train_y, train_y])

    # --- Probe: z_sem only (mixed) ---
    print("\n  [E] Probes...")
    print("    --- sem_only (mixed) ---")
    probe_sem = SemProbe(k_sem).to(device)
    train_probe_simple(probe_sem, zsem_train_mixed, y_mixed, device, epochs=50)
    acc_c = eval_probe_simple(probe_sem, zsem_test, test_y, device)
    acc_r = eval_probe_simple(probe_sem, zsem_test, test_y, device)  # z_sem unchanged!
    print(f"    sem_only: clean={acc_c:.3f}  repair={acc_r:.3f}  gap={abs(acc_c-acc_r):.3f}")
    results['hier_sem_only'] = {'clean': acc_c, 'repair': acc_r, 'gap': abs(acc_c-acc_r)}
    del probe_sem; torch.cuda.empty_cache()

    # --- Probe: z_tex only (mixed) ---
    print("    --- tex_only (mixed) ---")
    probe_tex = TexProbe(k_tex).to(device)
    ztex_mixed_all = torch.cat([ztex_train, ztex_train_rep])
    train_probe_simple(probe_tex, ztex_mixed_all, y_mixed, device, epochs=50)
    acc_c = eval_probe_simple(probe_tex, ztex_test, test_y, device)
    acc_r = eval_probe_simple(probe_tex, ztex_test_rep, test_y, device)
    print(f"    tex_only: clean={acc_c:.3f}  repair={acc_r:.3f}  gap={abs(acc_c-acc_r):.3f}")
    results['hier_tex_only'] = {'clean': acc_c, 'repair': acc_r, 'gap': abs(acc_c-acc_r)}
    del probe_tex; torch.cuda.empty_cache()

    # --- Probe: dual (z_sem + z_tex, mixed) ---
    print("    --- dual (mixed) ---")
    probe_dual = DualProbe(k_tex, k_sem).to(device)
    train_dual_probe(probe_dual, ztex_train_mixed, zsem_train_mixed, y_mixed, device, epochs=50)
    acc_c = eval_dual_probe(probe_dual, ztex_test, zsem_test, test_y, device)
    acc_r = eval_dual_probe(probe_dual, ztex_test_rep, zsem_test, test_y, device)
    print(f"    dual:     clean={acc_c:.3f}  repair={acc_r:.3f}  gap={abs(acc_c-acc_r):.3f}")
    results['hier_dual'] = {'clean': acc_c, 'repair': acc_r, 'gap': abs(acc_c-acc_r)}
    del probe_dual; torch.cuda.empty_cache()

    # ========================================================================
    # CONFIG C: HIERARCHICAL + CONTRASTIVE (z_sem gets semantic pressure)
    # ========================================================================
    print(f"\n{'='*100}")
    print("CONFIG C: HIERARCHICAL + CONTRASTIVE (SimCLR on z_sem, no labels)")
    print("="*100)

    print("  [F] Contrastive training on z_sem...")
    contra_head = ContrastiveHead(k_sem).to(device)
    train_contrastive(enc_hier, contra_head, train_x, device, epochs=30)
    del contra_head; torch.cuda.empty_cache()

    # Re-encode after contrastive fine-tuning
    print("  [G] Re-encoding with contrastive-tuned encoder...")
    ztex_train_c, zsem_train_c = encode_dataset_hier(enc_hier, train_x, device)
    ztex_test_c, zsem_test_c = encode_dataset_hier(enc_hier, test_x, device)

    # Repair z_tex again
    print("  [H] Re-repair z_tex...")
    ztex_train_rep_c = apply_tex_repair(den_hier, ztex_train_c, device)
    ztex_test_rep_c = apply_tex_repair(den_hier, ztex_test_c, device)

    ztex_train_mixed_c = torch.cat([ztex_train_c, ztex_train_rep_c])
    zsem_train_mixed_c = torch.cat([zsem_train_c, zsem_train_c])
    y_mixed = torch.cat([train_y, train_y])

    # --- Probe: z_sem only after contrastive ---
    print("\n  [I] Probes after contrastive...")
    print("    --- sem_only_contra (mixed) ---")
    probe_sem_c = SemProbe(k_sem).to(device)
    train_probe_simple(probe_sem_c, zsem_train_mixed_c, y_mixed, device, epochs=50)
    acc_c = eval_probe_simple(probe_sem_c, zsem_test_c, test_y, device)
    acc_r = eval_probe_simple(probe_sem_c, zsem_test_c, test_y, device)
    print(f"    sem_contra: clean={acc_c:.3f}  repair={acc_r:.3f}  gap={abs(acc_c-acc_r):.3f}")
    results['contra_sem_only'] = {'clean': acc_c, 'repair': acc_r, 'gap': abs(acc_c-acc_r)}
    del probe_sem_c; torch.cuda.empty_cache()

    # --- Probe: dual after contrastive ---
    print("    --- dual_contra (mixed) ---")
    probe_dual_c = DualProbe(k_tex, k_sem).to(device)
    train_dual_probe(probe_dual_c, ztex_train_mixed_c, zsem_train_mixed_c, y_mixed, device, epochs=50)
    acc_c = eval_dual_probe(probe_dual_c, ztex_test_c, zsem_test_c, test_y, device)
    acc_r = eval_dual_probe(probe_dual_c, ztex_test_rep_c, zsem_test_c, test_y, device)
    print(f"    dual_contra: clean={acc_c:.3f}  repair={acc_r:.3f}  gap={abs(acc_c-acc_r):.3f}")
    results['contra_dual'] = {'clean': acc_c, 'repair': acc_r, 'gap': abs(acc_c-acc_r)}
    del probe_dual_c; torch.cuda.empty_cache()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*100}")
    print("FINAL SUMMARY — STANDARDIZED EVALUATION (mixed probe)")
    print("="*100)

    print(f"\n  {'config':<25} {'clean':>8} {'repair':>8} {'gap':>8}")
    print(f"  {'-'*49}")
    for name, r in results.items():
        print(f"  {name:<25} {r['clean']:>8.3f} {r['repair']:>8.3f} {r['gap']:>8.3f}")

    # Paradigm diagnosis
    print(f"\n{'='*100}")
    print("PARADIGM DIAGNOSIS")
    print("="*100)

    best_clean = max(r['clean'] for r in results.values())
    best_name = [n for n, r in results.items() if r['clean'] == best_clean][0]
    best_r = results[best_name]

    print(f"\n  Best clean accuracy: {best_clean:.3f} ({best_name})")
    print(f"  vs flat baseline: {results['flat_baseline']['clean']:.3f}")
    gain = best_clean - results['flat_baseline']['clean']
    print(f"  Hierarchy gain: {gain:+.3f}")

    flat_gap = results['flat_baseline']['gap']
    best_gap = best_r['gap']
    print(f"\n  Stability gap: flat={flat_gap:.3f} → best={best_gap:.3f}")

    # Check contrastive effect
    if 'hier_sem_only' in results and 'contra_sem_only' in results:
        sem_gain = results['contra_sem_only']['clean'] - results['hier_sem_only']['clean']
        print(f"  Contrastive gain on z_sem: {sem_gain:+.3f}")

    if 'hier_dual' in results and 'contra_dual' in results:
        dual_gain = results['contra_dual']['clean'] - results['hier_dual']['clean']
        print(f"  Contrastive gain on dual: {dual_gain:+.3f}")

    # z_sem repair stability (should be 0 since z_sem is frozen during repair)
    if 'hier_sem_only' in results:
        print(f"\n  z_sem repair stability: gap={results['hier_sem_only']['gap']:.3f}")
        if results['hier_sem_only']['gap'] < 0.01:
            print("    z_sem is PERFECTLY REPAIR-STABLE (gap < 1%)")

    print(f"\n{'='*100}")
    print("Done.")
    print("="*100)


if __name__ == "__main__":
    main()
