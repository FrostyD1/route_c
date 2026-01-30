#!/usr/bin/env python3
"""
Phase 11: Feature Repairability (Discrete Core Provides New Capability)
========================================================================
Core question: If features h are corrupted (occlusion, noise, dropout),
can we repair them in z-space using evidence repair + amortized inference,
and recover downstream function?

This demonstrates the UNIQUE VALUE of the discrete core:
  - Traditional continuous features: corruption → degraded output, no repair path
  - Discrete core z: corruption → evidence repair → recovered function

Pipeline:
  1. h_corrupt = corrupt(h)  (spatial dropout, block mask, channel noise)
  2. z_corrupt = Q(h_corrupt)
  3. z_repaired = InpaintNet(z_corrupt, mask)
  4. h_repaired = R(z_repaired)
  5. Compare: f_rest(h_repaired) vs f_rest(h_clean)

Success criterion: repair recovers function (agreement > no-repair baseline).

Usage:
    python3 -u benchmarks/exp_phase11_repairability.py --device cuda
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
# COMPONENTS (reuse from Phase 10)
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

class FeatureQuantizer(nn.Module):
    def __init__(self, in_channels=256, n_bits=8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1), nn.ReLU(),
            nn.Conv2d(in_channels // 2, n_bits, 1))
        self.quantizer = GumbelSigmoid()
    def forward(self, h): return self.quantizer(self.proj(h))
    def set_temperature(self, tau): self.quantizer.set_temperature(tau)

class FeatureReconstructor(nn.Module):
    def __init__(self, out_channels=256, n_bits=8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(n_bits, out_channels // 2, 1), nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, 1))
    def forward(self, z): return self.proj(z)

class FeatureTranslator(nn.Module):
    def __init__(self, feature_channels=256, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.Q = FeatureQuantizer(feature_channels, n_bits)
        self.R = FeatureReconstructor(feature_channels, n_bits)
    def forward(self, h):
        z = self.Q(h); h_hat = self.R(z)
        return z, h_hat
    def set_temperature(self, tau): self.Q.set_temperature(tau)

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
# CORRUPTION FUNCTIONS
# ============================================================================

def corrupt_block(h, rng, block_ratio=0.25):
    """Drop a spatial block from features."""
    B, C, H, W = h.shape
    h_out = h.clone()
    bh, bw = max(1, int(H * block_ratio**0.5)), max(1, int(W * block_ratio**0.5))
    masks = torch.ones(B, 1, H, W, device=h.device)
    for b in range(B):
        y = rng.integers(0, max(1, H - bh + 1))
        x = rng.integers(0, max(1, W - bw + 1))
        h_out[b, :, y:y+bh, x:x+bw] = 0
        masks[b, 0, y:y+bh, x:x+bw] = 0
    return h_out, masks

def corrupt_center(h):
    """Drop center 50% of features."""
    B, C, H, W = h.shape
    h_out = h.clone()
    h4, w4 = H // 4, W // 4
    h_out[:, :, h4:H-h4, w4:W-w4] = 0
    mask = torch.ones(B, 1, H, W, device=h.device)
    mask[:, :, h4:H-h4, w4:W-w4] = 0
    return h_out, mask

def corrupt_channels(h, rng, drop_ratio=0.5):
    """Randomly zero out channels (simulating feature corruption)."""
    B, C, H, W = h.shape
    h_out = h.clone()
    mask_spatial = torch.ones(B, 1, H, W, device=h.device)
    n_drop = int(C * drop_ratio)
    for b in range(B):
        drop_idx = rng.choice(C, n_drop, replace=False)
        h_out[b, drop_idx] = 0
    # For z-space repair, we need spatial mask
    # Channel corruption → all positions partially corrupted → mask everything
    mask_spatial[:] = 0  # all positions need repair
    return h_out, mask_spatial


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
# TRAINING
# ============================================================================

def make_spatial_mask(H, W, rng):
    """Random spatial mask for z-space InpaintNet training."""
    p = rng.random()
    mask = np.ones((H, W), dtype=np.float32)
    if p < 0.4:
        # Block
        bh, bw = rng.integers(3, max(4, H//2+1)), rng.integers(3, max(4, W//2+1))
        y, x = rng.integers(0, max(1, H-bh+1)), rng.integers(0, max(1, W-bw+1))
        mask[y:y+bh, x:x+bw] = 0
    elif p < 0.7:
        # Center
        h4, w4 = H//4, W//4
        mask[h4:H-h4, w4:W-w4] = 0
    else:
        # Scattered
        for _ in range(rng.integers(2, 5)):
            s = rng.integers(1, 3)
            y, x = rng.integers(0, max(1, H-s+1)), rng.integers(0, max(1, W-s+1))
            mask[y:y+s, x:x+s] = 0
    return mask

def train_translator_and_inpaint(train_h, rest, device, n_bits=8, epochs_trans=15, epochs_inp=15):
    """Train Feature Translator + InpaintNet for z-space repair."""
    feat_channels = train_h.shape[1]

    # Step 1: Train translator
    translator = FeatureTranslator(feat_channels, n_bits).to(device)
    opt = torch.optim.Adam(translator.parameters(), lr=1e-3)
    N = len(train_h)

    for epoch in range(epochs_trans):
        translator.train()
        translator.set_temperature(1.0 + (0.3 - 1.0) * epoch / max(1, epochs_trans - 1))
        perm = torch.randperm(N)
        for i in range(0, N, 32):
            idx = perm[i:i+32]; h = train_h[idx].to(device)
            opt.zero_grad()
            z, h_hat = translator(h)
            loss_feat = F.mse_loss(h_hat, h)
            with torch.no_grad(): lo = rest(h)
            loss_func = F.kl_div(F.log_softmax(rest(h_hat), dim=1),
                                 F.softmax(lo, dim=1), reduction='batchmean')
            (loss_feat + 0.1 * loss_func).backward(); opt.step()

    # Step 2: Train InpaintNet on z-space
    translator.eval()
    inpaint = InpaintNet(k=n_bits, hidden=64).to(device)
    iopt = torch.optim.Adam(inpaint.parameters(), lr=1e-3)
    rng = np.random.default_rng(42)
    _, _, H_z, W_z = translator.Q(train_h[:1].to(device)).shape

    for epoch in range(epochs_inp):
        inpaint.train(); perm = torch.randperm(N)
        for i in range(0, N, 32):
            idx = perm[i:i+32]; h = train_h[idx].to(device)
            with torch.no_grad():
                z = translator.Q(h)
                z_hard = (z > 0.5).float()
            B = z_hard.shape[0]
            masks = []
            for _ in range(B):
                m = make_spatial_mask(H_z, W_z, rng)
                masks.append(torch.from_numpy(m).float())
            bit_mask = torch.stack(masks).unsqueeze(1).to(device)  # (B, 1, H, W)
            bit_mask_full = bit_mask.expand(-1, n_bits, -1, -1)
            z_masked = z_hard * (1 - bit_mask_full)
            logits = inpaint(z_masked, bit_mask)
            loss = F.binary_cross_entropy_with_logits(
                logits[bit_mask_full.bool()], z_hard[bit_mask_full.bool()])
            iopt.zero_grad(); loss.backward(); iopt.step()

    return translator, inpaint


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_repairability(translator, inpaint, test_h, rest, device,
                           corruption_type='center', n_bits=8):
    """Compare: corrupted (no repair) vs repaired vs clean."""
    translator.eval(); inpaint.eval()
    rng = np.random.default_rng(123)

    agree_clean, agree_corrupt, agree_repaired = [], [], []
    cosine_corrupt, cosine_repaired = [], []

    with torch.no_grad():
        for i in range(0, len(test_h), 16):
            h_clean = test_h[i:i+16].to(device)

            # Get clean reference
            logits_clean = rest(h_clean)

            # Corrupt
            if corruption_type == 'center':
                h_corrupt, spatial_mask = corrupt_center(h_clean)
            elif corruption_type == 'block':
                h_corrupt, spatial_mask = corrupt_block(h_clean, rng)
            elif corruption_type == 'channels':
                h_corrupt, spatial_mask = corrupt_channels(h_clean, rng)

            # No repair path: just reconstruct from corrupted features
            z_corrupt = translator.Q(h_corrupt)
            z_corrupt_hard = (z_corrupt > 0.5).float()
            h_no_repair = translator.R(z_corrupt_hard)

            # Repair path: fix z in occluded regions
            bit_mask = (1 - spatial_mask).expand(-1, n_bits, -1, -1)  # 1=masked
            z_masked = z_corrupt_hard * (1 - bit_mask)
            logits_repair = inpaint(z_masked, (1 - spatial_mask))
            z_repaired = z_corrupt_hard.clone()
            z_repaired[bit_mask.bool()] = (torch.sigmoid(logits_repair) > 0.5).float()[bit_mask.bool()]
            h_repaired = translator.R(z_repaired)

            # Evaluate
            logits_no_repair = rest(h_no_repair)
            logits_repaired = rest(h_repaired)

            # Agreement with clean
            pred_clean = logits_clean.argmax(1)
            agree_corrupt.extend(
                (logits_no_repair.argmax(1) == pred_clean).cpu().tolist())
            agree_repaired.extend(
                (logits_repaired.argmax(1) == pred_clean).cpu().tolist())

            # Cosine with clean h
            h_flat = h_clean.reshape(h_clean.shape[0], -1)
            cosine_corrupt.extend(
                F.cosine_similarity(h_no_repair.reshape_as(h_flat), h_flat, dim=1).cpu().tolist())
            cosine_repaired.extend(
                F.cosine_similarity(h_repaired.reshape_as(h_flat), h_flat, dim=1).cpu().tolist())

    return {
        'corruption': corruption_type,
        'agree_no_repair': np.mean(agree_corrupt),
        'agree_repaired': np.mean(agree_repaired),
        'cosine_no_repair': np.mean(cosine_corrupt),
        'cosine_repaired': np.mean(cosine_repaired),
        'repair_gain_agree': np.mean(agree_repaired) - np.mean(agree_corrupt),
        'repair_gain_cosine': np.mean(cosine_repaired) - np.mean(cosine_corrupt),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_phase11')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 11: FEATURE REPAIRABILITY (DISCRETE CORE → NEW CAPABILITY)")
    print("=" * 100)

    print("[1] Loading data and backbone...")
    train_x, train_y, test_x, test_y = load_fmnist_for_resnet(1000, 300, args.seed)
    feat_extractor, rest = build_backbone(device)

    print("[2] Extracting features...")
    train_h = extract_features(feat_extractor, train_x, device)
    test_h = extract_features(feat_extractor, test_x, device)
    del feat_extractor; torch.cuda.empty_cache()

    n_bits = 8
    print(f"[3] Training Feature Translator (k={n_bits}) + InpaintNet...")
    translator, inpaint = train_translator_and_inpaint(
        train_h, rest, device, n_bits=n_bits)

    corruption_types = ['center', 'block', 'channels']
    all_results = []

    print(f"\n{'corruption':<12} {'agree_no_rep':>13} {'agree_repair':>13} "
          f"{'cos_no_rep':>11} {'cos_repair':>11} {'repair_gain':>12}")
    print("-" * 80)

    for ct in corruption_types:
        r = evaluate_repairability(translator, inpaint, test_h, rest, device,
                                   corruption_type=ct, n_bits=n_bits)
        all_results.append(r)
        print(f"{ct:<12} {r['agree_no_repair']:>13.1%} {r['agree_repaired']:>13.1%} "
              f"{r['cosine_no_repair']:>11.4f} {r['cosine_repaired']:>11.4f} "
              f"{r['repair_gain_agree']:>+12.1%}")

    print("\n" + "=" * 80)
    print("REPAIRABILITY VERDICT")
    print("=" * 80)
    for r in all_results:
        gain = r['repair_gain_agree']
        print(f"  {r['corruption']:<12}: agreement {r['agree_no_repair']:.1%} → {r['agree_repaired']:.1%} "
              f"(gain={gain:+.1%}) → {'REPAIR HELPS' if gain > 0 else 'NO GAIN'}")

    avg_gain = np.mean([r['repair_gain_agree'] for r in all_results])
    print(f"\n  Average repair gain: {avg_gain:+.1%}")
    print(f"  → Discrete core {'PROVIDES' if avg_gain > 0 else 'DOES NOT PROVIDE'} "
          f"repairability on external features")

    csv_path = os.path.join(args.output_dir, "phase11_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(set().union(*(r.keys() for r in all_results)))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")
    print("\n" + "=" * 100)
    print("Phase 11 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
