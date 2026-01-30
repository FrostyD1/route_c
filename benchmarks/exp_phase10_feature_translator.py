#!/usr/bin/env python3
"""
Phase 10: Feature Translator (Inverse Mapping Bridge)
======================================================
Core question: Can we take any pretrained network's intermediate features h,
discretize them to z ∈ {0,1}^{k×H×W}, reconstruct ĥ ≈ h, and preserve
the network's downstream computation?

This is NOT another autoencoder. This is "protocol-izing" any feature space:
  h → Q(h) → z (discrete, repairable, hardware-friendly) → R(z) → ĥ

We measure:
  1. Feature reconstruction: cosine(h, ĥ)
  2. Function preservation: agreement(f_rest(h), f_rest(ĥ))
  3. Bit-width scaling: 8/4/2 bits

Backbone: ResNet18 pretrained (frozen), layer3 features (14×14×256).
Data: FMNIST resized to 224×224 (3-channel replicate).
GPU: Designed for 4GB — batch=16, frozen backbone, no_grad forward.

Usage:
    python3 -u benchmarks/exp_phase10_feature_translator.py --device cuda
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
# GUMBEL-SIGMOID QUANTIZER
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
# FEATURE TRANSLATOR: Q (encoder) and R (decoder) for feature space
# ============================================================================

class FeatureQuantizer(nn.Module):
    """Q: h (C×H×W) → z (k×H×W) discrete."""
    def __init__(self, in_channels=256, n_bits=8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1), nn.ReLU(),
            nn.Conv2d(in_channels // 2, n_bits, 1))
        self.quantizer = GumbelSigmoid()

    def forward(self, h):
        logits = self.proj(h)
        return self.quantizer(logits)

    def set_temperature(self, tau):
        self.quantizer.set_temperature(tau)


class FeatureReconstructor(nn.Module):
    """R: z (k×H×W) → ĥ (C×H×W) continuous."""
    def __init__(self, out_channels=256, n_bits=8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(n_bits, out_channels // 2, 1), nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, 1))

    def forward(self, z):
        return self.proj(z)


class FeatureTranslator(nn.Module):
    """Complete h → z → ĥ pipeline."""
    def __init__(self, feature_channels=256, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.Q = FeatureQuantizer(feature_channels, n_bits)
        self.R = FeatureReconstructor(feature_channels, n_bits)

    def forward(self, h):
        z = self.Q(h)
        h_hat = self.R(z)
        return z, h_hat

    def set_temperature(self, tau):
        self.Q.set_temperature(tau)


# ============================================================================
# BACKBONE: ResNet18 split into feature extractor + rest
# ============================================================================

def build_backbone(device):
    """Load pretrained ResNet18, split at layer3."""
    from torchvision import models
    resnet = models.resnet18(weights='DEFAULT')
    resnet.eval()

    # Feature extractor: everything up to and including layer3
    # Output: 14×14×256 for 224×224 input
    feat_extractor = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3)

    # Rest: layer4 + avgpool + fc
    rest = nn.Sequential(
        resnet.layer4,
        resnet.avgpool,
        nn.Flatten(),
        resnet.fc)

    feat_extractor = feat_extractor.to(device).eval()
    rest = rest.to(device).eval()

    # Freeze everything
    for p in feat_extractor.parameters(): p.requires_grad = False
    for p in rest.parameters(): p.requires_grad = False

    return feat_extractor, rest


# ============================================================================
# DATA
# ============================================================================

def load_fmnist_for_resnet(train_n=1000, test_n=300, seed=42):
    """Load FMNIST, resize to 224×224, replicate to 3 channels."""
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
    train_x = torch.stack([tr[i][0] for i in ti])
    train_y = torch.tensor([tr[i][1] for i in ti])
    test_x = torch.stack([te[i][0] for i in si])
    test_y = torch.tensor([te[i][1] for i in si])
    return train_x, train_y, test_x, test_y


# ============================================================================
# EXTRACT FEATURES (once, cached)
# ============================================================================

def extract_features(feat_extractor, data_x, device, batch_size=16):
    """Extract features from frozen backbone. Returns CPU tensor."""
    feat_extractor.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(data_x), batch_size):
            x = data_x[i:i+batch_size].to(device)
            h = feat_extractor(x)
            all_feats.append(h.cpu())
    return torch.cat(all_feats, dim=0)


# ============================================================================
# TRAINING
# ============================================================================

def train_translator(translator, train_h, rest, device, epochs=15, batch_size=32):
    """Train Q and R to minimize feature reconstruction + function preservation."""
    opt = torch.optim.Adam(translator.parameters(), lr=1e-3)
    N = len(train_h)

    for epoch in range(epochs):
        translator.train()
        translator.set_temperature(1.0 + (0.3 - 1.0) * epoch / max(1, epochs - 1))
        perm = torch.randperm(N)
        tl, tl_feat, tl_func, nb = 0., 0., 0., 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            h = train_h[idx].to(device)
            opt.zero_grad()

            z, h_hat = translator(h)

            # Feature reconstruction loss
            loss_feat = F.mse_loss(h_hat, h)

            # Function preservation: rest(h) ≈ rest(ĥ)
            with torch.no_grad():
                logits_orig = rest(h)
            logits_recon = rest(h_hat)
            loss_func = F.kl_div(
                F.log_softmax(logits_recon, dim=1),
                F.softmax(logits_orig, dim=1),
                reduction='batchmean')

            loss = loss_feat + 0.1 * loss_func
            loss.backward(); opt.step()
            tl += loss.item(); tl_feat += loss_feat.item()
            tl_func += loss_func.item(); nb += 1

        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{epochs}: loss={tl/nb:.4f} "
                  f"feat={tl_feat/nb:.4f} func={tl_func/nb:.4f}")

    return translator


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_translator(translator, test_h, rest, device, batch_size=16):
    """Evaluate feature reconstruction and function preservation."""
    translator.eval()
    cosines, agreements, kl_divs = [], [], []

    with torch.no_grad():
        for i in range(0, len(test_h), batch_size):
            h = test_h[i:i+batch_size].to(device)
            z, h_hat = translator(h)

            # Cosine similarity (per sample, averaged over spatial dims)
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

    return {
        'cosine_mean': np.mean(cosines),
        'cosine_std': np.std(cosines),
        'agreement': np.mean(agreements),
        'kl_div_mean': np.mean(kl_divs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_phase10')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 10: FEATURE TRANSLATOR (INVERSE MAPPING BRIDGE)")
    print("=" * 100)

    print("[1] Loading FMNIST (resized to 224×224 for ResNet18)...")
    train_x, train_y, test_x, test_y = load_fmnist_for_resnet(
        train_n=1000, test_n=300, seed=args.seed)

    print("[2] Loading pretrained ResNet18 (frozen)...")
    feat_extractor, rest = build_backbone(device)

    print("[3] Extracting features (layer3: 14×14×256)...")
    train_h = extract_features(feat_extractor, train_x, device)
    test_h = extract_features(feat_extractor, test_x, device)
    feat_channels = train_h.shape[1]
    print(f"    Feature shape: {train_h.shape[1:]} ({feat_channels} channels)")

    # Free backbone feature extractor from GPU (keep rest for eval)
    del feat_extractor
    torch.cuda.empty_cache()

    # Bit-width sweep
    bit_widths = [8, 4, 2]
    all_results = []

    print(f"\n{'bits':>5} {'cosine':>8} {'agreement':>11} {'KL_div':>8} {'Q_params':>9} {'R_params':>9}")
    print("-" * 55)

    for n_bits in bit_widths:
        torch.manual_seed(args.seed)
        translator = FeatureTranslator(feat_channels, n_bits).to(device)

        print(f"\n[4] Training translator (k={n_bits} bits)...")
        train_translator(translator, train_h, rest, device)

        results = evaluate_translator(translator, test_h, rest, device)
        results['n_bits'] = n_bits
        results['q_params'] = sum(p.numel() for p in translator.Q.parameters())
        results['r_params'] = sum(p.numel() for p in translator.R.parameters())
        all_results.append(results)

        print(f"{n_bits:>5} {results['cosine_mean']:>8.4f} {results['agreement']:>11.1%} "
              f"{results['kl_div_mean']:>8.4f} {results['q_params']:>9,} {results['r_params']:>9,}")

        del translator
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("FEATURE TRANSLATOR VERDICT")
    print("=" * 80)

    for r in all_results:
        viable = r['cosine_mean'] > 0.9 and r['agreement'] > 0.8
        print(f"  k={r['n_bits']}: cosine={r['cosine_mean']:.4f}, agreement={r['agreement']:.1%}, "
              f"KL={r['kl_div_mean']:.4f} → {'VIABLE' if viable else 'DEGRADED'}")

    r8 = [r for r in all_results if r['n_bits'] == 8][0]
    r2 = [r for r in all_results if r['n_bits'] == 2][0]
    print(f"\n  8→2 bit degradation: cosine {r8['cosine_mean']:.4f}→{r2['cosine_mean']:.4f}, "
          f"agreement {r8['agreement']:.1%}→{r2['agreement']:.1%}")
    print(f"  → {'Graceful' if r2['cosine_mean'] > 0.8 else 'Steep'} degradation curve")

    csv_path = os.path.join(args.output_dir, "phase10_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(set().union(*(r.keys() for r in all_results)))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")
    print("\n" + "=" * 100)
    print("Phase 10 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
