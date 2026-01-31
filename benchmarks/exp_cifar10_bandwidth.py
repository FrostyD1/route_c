#!/usr/bin/env python3
"""
CIFAR-10 Classification — C1: Bandwidth Sweep (Protocol Capacity Diagnostic)
=============================================================================
Question: Is 45% probe accuracy bottlenecked by z capacity or encoder quality?

4 z specs (all INT4-style binary, no repair, no denoiser):
  Z-A: 16×16×16  (4096 bits, spatial downsample + high channel)
  Z-B: 32×32×8   (8192 bits, current baseline)
  Z-C: 32×32×16  (16384 bits, double channels)
  Z-D: 8×8×64    (4096 bits, extreme downsample, max abstraction)

Each config: train ADC (20 epochs, fast) → conv probe → report accuracy.
NO repair, NO denoiser — pure capacity diagnostic.

Usage:
    python3 -u benchmarks/exp_cifar10_bandwidth.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, argparse, json
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# GUMBEL SIGMOID STE
# ============================================================================

class GumbelSigmoid(nn.Module):
    def __init__(self, tau=1.0):
        super().__init__(); self.tau = tau
    def forward(self, logits):
        if self.training:
            u = torch.rand_like(logits).clamp(1e-8, 1-1e-8)
            noisy = (logits - torch.log(-torch.log(u))) / self.tau
        else:
            noisy = logits / self.tau
        soft = torch.sigmoid(noisy); hard = (soft > 0.5).float()
        return hard - soft.detach() + soft
    def set_tau(self, tau): self.tau = tau


# ============================================================================
# PARAMETERIZED ENCODER/DECODER for different z shapes
# ============================================================================

class FlexEncoder(nn.Module):
    """Encoder that outputs z at specified spatial size and channel count."""
    def __init__(self, z_h, z_w, k):
        super().__init__()
        self.z_h, self.z_w, self.k = z_h, z_w, k

        layers = [nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()]

        # Determine downsampling needed: 32 → z_h
        cur_h = 32
        ch = 64
        while cur_h > z_h:
            next_ch = min(ch * 2, 256)
            layers += [
                nn.Conv2d(ch, next_ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(next_ch), nn.ReLU(),
            ]
            ch = next_ch
            cur_h //= 2

        # Final projection to k channels
        layers += [
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, k, 3, padding=1),
        ]
        self.net = nn.Sequential(*layers)
        self.q = GumbelSigmoid()

    def forward(self, x):
        return self.q(self.net(x))

    def set_temperature(self, tau): self.q.set_tau(tau)


class FlexDecoder(nn.Module):
    """Decoder that reconstructs 32×32×3 from z at any spatial size."""
    def __init__(self, z_h, z_w, k):
        super().__init__()
        layers = [nn.Conv2d(k, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()]

        cur_h = z_h
        ch = 128
        while cur_h < 32:
            next_ch = max(ch // 2, 64)
            layers += [
                nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(next_ch), nn.ReLU(),
            ]
            ch = next_ch
            cur_h *= 2

        layers += [
            nn.Conv2d(ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z): return self.net(z)


# ============================================================================
# CONV PROBE (adapted to different z spatial sizes)
# ============================================================================

class ConvProbe(nn.Module):
    def __init__(self, k, z_h, n_classes=10):
        super().__init__()
        pool_size = min(4, z_h)  # adaptive pool target
        self.net = nn.Sequential(
            nn.Conv2d(k, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
            nn.Linear(32 * pool_size * pool_size, n_classes),
        )
    def forward(self, z): return self.net(z)


class LinearProbe(nn.Module):
    def __init__(self, k, z_h, z_w, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(k * z_h * z_w, n_classes),
        )
    def forward(self, z): return self.net(z)


# ============================================================================
# TRAINING
# ============================================================================

def train_adc(encoder, decoder, train_x, device, epochs=20, bs=128):
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    for epoch in tqdm(range(epochs), desc="ADC"):
        encoder.train(); decoder.train()
        encoder.set_temperature(1.0 + (0.3-1.0)*epoch/(max(epochs-1,1)))
        perm = torch.randperm(len(train_x)); tl, nb = 0., 0
        for i in range(0, len(train_x), bs):
            x = train_x[perm[i:i+bs]].to(device)
            opt.zero_grad()
            z = encoder(x)
            xh = decoder(z)
            loss = F.mse_loss(xh, x) + 0.5 * F.binary_cross_entropy(xh, x)
            loss.backward(); opt.step(); tl += loss.item(); nb += 1
    encoder.eval(); decoder.eval()
    return tl / nb


def encode_all(encoder, data, device, bs=128):
    zs = []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            zs.append(encoder(data[i:i+bs].to(device)).cpu())
    return torch.cat(zs)


def train_probe(probe, z_data, labels, device, epochs=50, bs=256):
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
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
    probe.eval(); nc, nt = 0, 0
    with torch.no_grad():
        for i in range(0, len(z_data), bs):
            z = z_data[i:i+bs].to(device); y = labels[i:i+bs].to(device)
            nc += (probe(z).argmax(1) == y).sum().item(); nt += len(y)
    return nc / nt


# ============================================================================
# Z STATISTICS
# ============================================================================

def z_stats(z):
    """Basic z statistics: mean activation, per-channel entropy, spatial correlation."""
    mean_act = z.float().mean().item()
    # Per-channel entropy
    p = z.float().mean(dim=(0, 2, 3)).clamp(1e-6, 1-1e-6)
    entropy = -(p * p.log() + (1-p) * (1-p).log()).mean().item()
    # Channel utilization (fraction of channels not dead)
    ch_active = ((p > 0.05) & (p < 0.95)).float().mean().item()
    # Spatial correlation (adjacent bit agreement)
    if z.shape[2] > 1:
        agree_h = (z[:,:,:-1,:] == z[:,:,1:,:]).float().mean().item()
        agree_w = (z[:,:,:,:-1] == z[:,:,:,1:]).float().mean().item()
        spatial_corr = (agree_h + agree_w) / 2
    else:
        spatial_corr = 0.0
    return {
        'mean_act': mean_act, 'entropy': entropy,
        'ch_util': ch_active, 'spatial_corr': spatial_corr,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--adc_epochs', type=int, default=25)
    parser.add_argument('--probe_epochs', type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("CIFAR-10 — C1: BANDWIDTH SWEEP (Protocol Capacity Diagnostic)")
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

    # Z configs: (name, z_h, z_w, k, total_bits)
    configs = [
        ("Z-A_16x16x16", 16, 16, 16, 16*16*16),
        ("Z-B_32x32x8",  32, 32,  8, 32*32*8),
        ("Z-C_32x32x16", 32, 32, 16, 32*32*16),
        ("Z-D_8x8x64",    8,  8, 64,  8*8*64),
    ]

    results = {}

    for cfg_name, z_h, z_w, k, total_bits in configs:
        print(f"\n{'='*100}")
        print(f"CONFIG: {cfg_name} ({total_bits} bits)")
        print("="*100)

        torch.manual_seed(args.seed)
        enc = FlexEncoder(z_h, z_w, k).to(device)
        dec = FlexDecoder(z_h, z_w, k).to(device)

        enc_p = sum(p.numel() for p in enc.parameters())
        dec_p = sum(p.numel() for p in dec.parameters())
        print(f"    Encoder params: {enc_p:,}  Decoder params: {dec_p:,}")

        # Train ADC
        print("  Training ADC...")
        adc_loss = train_adc(enc, dec, train_x, device, epochs=args.adc_epochs, bs=128)
        print(f"    ADC loss: {adc_loss:.4f}")

        # Encode
        print("  Encoding...")
        z_train = encode_all(enc, train_x, device)
        z_test = encode_all(enc, test_x, device)
        print(f"    z shape: {z_train.shape}")

        # Z stats
        stats = z_stats(z_train)
        print(f"    mean_act={stats['mean_act']:.3f}  entropy={stats['entropy']:.3f}  "
              f"ch_util={stats['ch_util']:.3f}  spatial_corr={stats['spatial_corr']:.3f}")

        # Reconstruction quality
        with torch.no_grad():
            sample = train_x[:100].to(device)
            z_s = enc(sample)
            recon = dec(z_s)
            mse = F.mse_loss(recon, sample).item()
            bce = F.binary_cross_entropy(recon, sample).item()
        print(f"    Recon MSE={mse:.4f}  BCE={bce:.4f}")

        # Conv probe
        print("  Training conv probe...")
        probe_conv = ConvProbe(k, z_h).to(device)
        train_probe(probe_conv, z_train, train_y, device, epochs=args.probe_epochs)
        acc_conv = eval_probe(probe_conv, z_test, test_y, device)
        print(f"    Conv probe: {acc_conv:.3f}")

        # Linear probe
        print("  Training linear probe...")
        probe_lin = LinearProbe(k, z_h, z_w).to(device)
        train_probe(probe_lin, z_train, train_y, device, epochs=args.probe_epochs)
        acc_lin = eval_probe(probe_lin, z_test, test_y, device)
        print(f"    Linear probe: {acc_lin:.3f}")

        results[cfg_name] = {
            'total_bits': total_bits, 'z_shape': f"{z_h}x{z_w}x{k}",
            'enc_params': enc_p, 'dec_params': dec_p,
            'adc_loss': adc_loss, 'recon_mse': mse, 'recon_bce': bce,
            'conv_acc': acc_conv, 'linear_acc': acc_lin,
            **stats,
        }

        del enc, dec, probe_conv, probe_lin
        torch.cuda.empty_cache()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*100}")
    print("BANDWIDTH SWEEP SUMMARY")
    print("="*100)
    print(f"\n  {'config':<20} {'bits':>7} {'conv':>7} {'linear':>7} {'MSE':>7} {'entropy':>8} {'ch_util':>8}")
    print(f"  {'-'*66}")
    for name, r in results.items():
        print(f"  {name:<20} {r['total_bits']:>7} {r['conv_acc']:>7.3f} {r['linear_acc']:>7.3f} "
              f"{r['recon_mse']:>7.4f} {r['entropy']:>8.3f} {r['ch_util']:>8.3f}")

    # Diagnosis
    best = max(results.items(), key=lambda x: x[1]['conv_acc'])
    worst = min(results.items(), key=lambda x: x[1]['conv_acc'])
    print(f"\n  Best: {best[0]} conv={best[1]['conv_acc']:.3f}")
    print(f"  Worst: {worst[0]} conv={worst[1]['conv_acc']:.3f}")
    print(f"  Spread: {best[1]['conv_acc'] - worst[1]['conv_acc']:.3f}")

    if best[1]['conv_acc'] >= 0.55:
        print("\n  DIAGNOSIS: Bandwidth IS the bottleneck — larger z helps significantly")
        print("  → Proceed with best z config for C2/C3")
    elif best[1]['conv_acc'] <= 0.50:
        print("\n  DIAGNOSIS: Bandwidth is NOT the main bottleneck — encoder/training matters more")
        print("  → C2 (staged encoder + self-supervised) is essential")
    else:
        print("\n  DIAGNOSIS: Mixed — some bandwidth gain but encoder upgrade also needed")

    print(f"\n{'='*100}")
    print("EXPERIMENT COMPLETE")
    print("="*100)

    # Save results
    os.makedirs('outputs/exp_cifar10_bandwidth', exist_ok=True)
    with open('outputs/exp_cifar10_bandwidth/results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
