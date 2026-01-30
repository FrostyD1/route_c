#!/usr/bin/env python3
"""
Phase 10A: Feature Translator v2 (Fixed: Function Preservation, Not Compression)
==================================================================================
Phase 10 failed because: 256ch → 8 bits = 32× compression, impossible.

Fix #1: Bit-width sweep from easy to hard: k=64,32,16,8
Fix #2: 3×3 conv Q/R with skip connections (more capacity, spatial context)
Fix #3: KL-dominant loss (preserve function, not reconstruct h pointwise)

Goal: Find the "information capacity boundary" — at what k does agreement
drop below 90%? That boundary itself is a paper-grade result.

Backbone: ResNet18 frozen, layer3 (14×14×256).
rest = layer4 + avgpool + fc (frozen, for function evaluation).

Success: agreement ≥ 90% at some k. Then sweep down to find boundary.

Usage:
    python3 -u benchmarks/exp_phase10a_bridge_v2.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, csv, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))


# ============================================================================
# QUANTIZER
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


# ============================================================================
# FEATURE TRANSLATOR v2: 3×3 conv + skip, more capacity
# ============================================================================

class FeatureQuantizerV2(nn.Module):
    """Q: h (256×14×14) → z (k×14×14) with 3×3 conv + skip."""
    def __init__(self, in_channels=256, n_bits=64):
        super().__init__()
        mid = max(n_bits, in_channels // 4)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(mid, n_bits, 3, padding=1))
        self.skip = nn.Conv2d(in_channels, n_bits, 1)
        self.quantizer = GumbelSigmoid()

    def forward(self, h):
        logits = self.conv(h) + self.skip(h)
        return self.quantizer(logits)

    def set_temperature(self, tau):
        self.quantizer.set_temperature(tau)


class FeatureReconstructorV2(nn.Module):
    """R: z (k×14×14) → ĥ (256×14×14) with 3×3 conv + skip."""
    def __init__(self, out_channels=256, n_bits=64):
        super().__init__()
        mid = max(n_bits, out_channels // 4)
        self.conv = nn.Sequential(
            nn.Conv2d(n_bits, mid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(mid, out_channels, 3, padding=1))
        self.skip = nn.Conv2d(n_bits, out_channels, 1)

    def forward(self, z):
        return self.conv(z) + self.skip(z)


class FeatureTranslatorV2(nn.Module):
    def __init__(self, feature_channels=256, n_bits=64):
        super().__init__()
        self.n_bits = n_bits
        self.Q = FeatureQuantizerV2(feature_channels, n_bits)
        self.R = FeatureReconstructorV2(feature_channels, n_bits)

    def forward(self, h):
        z = self.Q(h)
        h_hat = self.R(z)
        return z, h_hat

    def set_temperature(self, tau):
        self.Q.set_temperature(tau)


# ============================================================================
# BACKBONE
# ============================================================================

def build_backbone(device):
    from torchvision import models
    resnet = models.resnet18(weights='DEFAULT')
    resnet.eval()
    feat_extractor = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3)
    rest = nn.Sequential(
        resnet.layer4, resnet.avgpool, nn.Flatten(), resnet.fc)
    feat_extractor = feat_extractor.to(device).eval()
    rest = rest.to(device).eval()
    for p in feat_extractor.parameters(): p.requires_grad = False
    for p in rest.parameters(): p.requires_grad = False
    return feat_extractor, rest


def load_fmnist_for_resnet(train_n=1000, test_n=300, seed=42):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    tr = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    te = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(tr), train_n, replace=False)
    si = rng.choice(len(te), test_n, replace=False)
    return (torch.stack([tr[i][0] for i in ti]), torch.tensor([tr[i][1] for i in ti]),
            torch.stack([te[i][0] for i in si]), torch.tensor([te[i][1] for i in si]))


def extract_features(feat_extractor, data_x, device, batch_size=16):
    feat_extractor.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(data_x), batch_size):
            x = data_x[i:i+batch_size].to(device)
            all_feats.append(feat_extractor(x).cpu())
    return torch.cat(all_feats, dim=0)


# ============================================================================
# TRAINING: KL-dominant loss
# ============================================================================

def train_translator(translator, train_h, rest, device, epochs=25, batch_size=32,
                     alpha_kl=1.0, beta_mse=0.1):
    """
    Train with KL-dominant loss:
      L = α * KL(softmax(logits_orig) || softmax(logits_recon)) + β * MSE(h, ĥ)

    KL is the PRIMARY objective (function preservation).
    MSE is a REGULARIZER (prevents degenerate solutions).
    """
    opt = torch.optim.Adam(translator.parameters(), lr=1e-3)
    N = len(train_h)

    for epoch in range(epochs):
        translator.train()
        # Temperature annealing: 1.0 → 0.3
        tau = 1.0 + (0.3 - 1.0) * epoch / max(1, epochs - 1)
        translator.set_temperature(tau)
        perm = torch.randperm(N)
        tl_kl, tl_mse, nb = 0., 0., 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            h = train_h[idx].to(device)
            opt.zero_grad()

            z, h_hat = translator(h)

            # Function preservation (KL on logits) — PRIMARY
            with torch.no_grad():
                logits_orig = rest(h)
                probs_orig = F.softmax(logits_orig, dim=1)
            logits_recon = rest(h_hat)
            loss_kl = F.kl_div(
                F.log_softmax(logits_recon, dim=1),
                probs_orig, reduction='batchmean')

            # Feature regularizer — SECONDARY
            loss_mse = F.mse_loss(h_hat, h)

            loss = alpha_kl * loss_kl + beta_mse * loss_mse
            loss.backward(); opt.step()
            tl_kl += loss_kl.item(); tl_mse += loss_mse.item(); nb += 1

        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{epochs}: KL={tl_kl/nb:.4f} MSE={tl_mse/nb:.4f} τ={tau:.2f}")

    return translator


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_translator(translator, test_h, rest, device, batch_size=16):
    translator.eval()
    cosines, agreements, kl_divs = [], [], []
    delta_logits = []

    with torch.no_grad():
        for i in range(0, len(test_h), batch_size):
            h = test_h[i:i+batch_size].to(device)
            z, h_hat = translator(h)

            # Cosine similarity
            h_flat = h.reshape(h.shape[0], -1)
            hh_flat = h_hat.reshape(h_hat.shape[0], -1)
            cos = F.cosine_similarity(h_flat, hh_flat, dim=1)
            cosines.extend(cos.cpu().tolist())

            # Function preservation
            logits_orig = rest(h)
            logits_recon = rest(h_hat)
            preds_orig = logits_orig.argmax(1)
            preds_recon = logits_recon.argmax(1)
            agreements.extend((preds_orig == preds_recon).cpu().tolist())

            # KL divergence
            kl = F.kl_div(
                F.log_softmax(logits_recon, dim=1),
                F.softmax(logits_orig, dim=1),
                reduction='none').sum(1)
            kl_divs.extend(kl.cpu().tolist())

            # Δlogit
            dl = (logits_orig - logits_recon).abs().mean(1)
            delta_logits.extend(dl.cpu().tolist())

    return {
        'cosine_mean': np.mean(cosines),
        'agreement': np.mean(agreements),
        'kl_div': np.mean(kl_divs),
        'delta_logit': np.mean(delta_logits),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_phase10a')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 10A: FEATURE TRANSLATOR v2 (FUNCTION PRESERVATION, NOT COMPRESSION)")
    print("=" * 100)

    print("[1] Loading FMNIST for ResNet18...")
    train_x, _, test_x, _ = load_fmnist_for_resnet(1000, 300, args.seed)

    print("[2] Loading pretrained ResNet18...")
    feat_extractor, rest = build_backbone(device)

    print("[3] Extracting layer3 features (14×14×256)...")
    train_h = extract_features(feat_extractor, train_x, device)
    test_h = extract_features(feat_extractor, test_x, device)
    feat_channels = train_h.shape[1]
    print(f"    Feature shape: {tuple(train_h.shape[1:])}")

    del feat_extractor; torch.cuda.empty_cache()

    # Bit-width sweep: from easy to hard
    bit_widths = [64, 32, 16, 8]
    all_results = []

    print(f"\n{'k':>4} {'ratio':>6} {'cosine':>8} {'agree%':>8} {'KL':>8} {'Δlogit':>8} "
          f"{'Q_params':>9} {'verdict'}")
    print("-" * 70)

    for n_bits in bit_widths:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        ratio = feat_channels / n_bits

        translator = FeatureTranslatorV2(feat_channels, n_bits).to(device)

        print(f"\n  Training k={n_bits} (ratio={ratio:.0f}×)...")
        train_translator(translator, train_h, rest, device, epochs=25)

        r = evaluate_translator(translator, test_h, rest, device)
        r['n_bits'] = n_bits
        r['compression_ratio'] = ratio
        r['q_params'] = sum(p.numel() for p in translator.Q.parameters())
        r['r_params'] = sum(p.numel() for p in translator.R.parameters())
        r['total_z_bits'] = n_bits * 14 * 14  # total discrete state size

        viable = r['agreement'] >= 0.90
        r['viable'] = viable
        all_results.append(r)

        print(f"{n_bits:>4} {ratio:>5.0f}× {r['cosine_mean']:>8.4f} {r['agreement']:>7.1%} "
              f"{r['kl_div']:>8.4f} {r['delta_logit']:>8.4f} "
              f"{r['q_params']:>9,} {'✓ VIABLE' if viable else '✗ DEGRADED'}")

        del translator; torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 100)
    print("INFORMATION CAPACITY BOUNDARY")
    print("=" * 100)

    boundary_found = False
    for i, r in enumerate(all_results):
        status = "VIABLE" if r['viable'] else "DEGRADED"
        print(f"  k={r['n_bits']:>3} ({r['compression_ratio']:.0f}×): "
              f"agreement={r['agreement']:.1%}, cosine={r['cosine_mean']:.4f}, "
              f"z_size={r['total_z_bits']} bits → {status}")
        if not r['viable'] and not boundary_found:
            boundary_found = True
            prev = all_results[i-1] if i > 0 else None
            if prev and prev['viable']:
                print(f"\n  ⮕ BOUNDARY: between k={prev['n_bits']} (viable) "
                      f"and k={r['n_bits']} (degraded)")
                print(f"    Information capacity boundary ≈ {prev['total_z_bits']}-"
                      f"{r['total_z_bits']} bits for 14×14 grid")

    if not boundary_found:
        if all(r['viable'] for r in all_results):
            print(f"\n  All viable! Even k={all_results[-1]['n_bits']} "
                  f"({all_results[-1]['compression_ratio']:.0f}×) maintains agreement.")
        else:
            print(f"\n  No viable configuration found. Need even more bits or better Q/R.")

    # Key comparison with Phase 10
    print(f"\n  Phase 10 (1×1 conv, MSE-dominant):  k=8 → agreement=10.3%, cosine=0.497")
    r8 = [r for r in all_results if r['n_bits'] == 8]
    if r8:
        r8 = r8[0]
        print(f"  Phase 10A (3×3 conv, KL-dominant):  k=8 → agreement={r8['agreement']:.1%}, "
              f"cosine={r8['cosine_mean']:.4f}")
        improvement = r8['agreement'] - 0.103
        print(f"  Improvement from architecture+loss fix: {improvement:+.1%}")

    csv_path = os.path.join(args.output_dir, "phase10a_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(set().union(*(r.keys() for r in all_results)))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")
    print("\n" + "=" * 100)
    print("Phase 10A experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
