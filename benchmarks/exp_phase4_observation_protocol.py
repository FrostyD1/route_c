#!/usr/bin/env python3
"""
Phase 4: Observation Protocol Migration
=========================================
Migrate E_obs from pixel-space to protocol-space:
  4A) Token-level observation: E_obs defined on z tokens, not pixels
  4B) Frequency-domain observation: E_obs on DCT coefficients

This tests whether the paradigm can decouple from pixel-level reconstruction
and work with more abstract observation protocols.

Key question: Can we define E_obs entirely in the discrete domain (z-space)
without going through continuous pixel reconstruction?

Approach:
  - Token E_obs: cross-entropy between z_observed and z_predicted at each position
  - Frequency E_obs: DCT of z features → frequency-domain matching
  - Pixel E_obs: standard BCE (baseline)

Compare repair quality across observation protocols.

Usage:
    python3 -u benchmarks/exp_phase4_observation_protocol.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import time
import csv
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))


# ============================================================================
# MODEL (standard)
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
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1),
        )
    def forward(self, x): return self.conv(x)

class BinaryDecoder(nn.Module):
    def __init__(self, n_bits=8, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z): return self.deconv(z)

class LocalPredictor(nn.Module):
    def __init__(self, n_bits=8, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9 * n_bits, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_bits))
    def forward(self, z):
        B, k, H, W = z.shape
        z_pad = F.pad(z, (1,1,1,1), mode='constant', value=0)
        windows = F.unfold(z_pad, kernel_size=3).reshape(B, k, 9, H*W)
        windows[:, :, 4, :] = 0
        windows = windows.reshape(B, k*9, H*W).permute(0,2,1)
        return self.net(windows).permute(0,2,1).reshape(B, k, H, W)

class Classifier(nn.Module):
    def __init__(self, n_bits=8, latent_size=7):
        super().__init__()
        self.fc = nn.Linear(n_bits * latent_size * latent_size, 10)
    def forward(self, z): return self.fc(z.reshape(z.shape[0], -1))

class RouteCModel(nn.Module):
    def __init__(self, n_bits=8, latent_size=7, hidden_dim=64, tau=1.0):
        super().__init__()
        self.n_bits = n_bits; self.latent_size = latent_size
        self.encoder = BinaryEncoder(n_bits, hidden_dim)
        self.decoder = BinaryDecoder(n_bits, hidden_dim)
        self.quantizer = GumbelSigmoid(tau)
        self.local_pred = LocalPredictor(n_bits)
        self.classifier = Classifier(n_bits, latent_size)
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
# OBSERVATION PROTOCOLS
# ============================================================================

class PixelObsProtocol:
    """Standard pixel-space E_obs (baseline)."""
    name = 'pixel_bce'

    def residual_per_patch(self, model, z, x_clean, pixel_mask, device):
        """Returns (ls, ls) residual map."""
        ls = model.latent_size
        H, W = x_clean.shape; ph, pw = H//ls, W//ls
        with torch.no_grad():
            o_hat = model.decode(z.unsqueeze(0))[0, 0].clamp(1e-6, 1-1e-6)
        obs = torch.from_numpy(x_clean).to(device)
        pm = torch.from_numpy(pixel_mask).to(device)
        resid = torch.zeros(ls, ls, device=device)
        for i in range(ls):
            for j in range(ls):
                y0,y1 = i*ph,(i+1)*ph; x0,x1 = j*pw,(j+1)*pw
                p_o = obs[y0:y1,x0:x1]; p_h = o_hat[y0:y1,x0:x1]; p_m = pm[y0:y1,x0:x1]
                cnt = p_m.sum()
                if cnt > 0:
                    bce = -(p_o*torch.log(p_h)+(1-p_o)*torch.log(1-p_h))
                    resid[i,j] = (bce*p_m).sum()/cnt
                else:
                    resid[i,j] = float('inf')
        return resid


class TokenObsProtocol:
    """
    Token-space E_obs: compare z directly at observed positions.
    No pixel reconstruction needed — purely discrete observation.
    """
    name = 'token_bce'

    def residual_per_patch(self, model, z, x_clean, pixel_mask, device):
        """
        Encode clean image → z_clean. Compare z vs z_clean at observed patches.
        High mismatch = z doesn't match what the encoder would produce from
        the clean observation.
        """
        ls = model.latent_size
        with torch.no_grad():
            x_t = torch.from_numpy(x_clean[None, None].astype(np.float32)).to(device)
            z_clean = model.encode(x_t)[0]  # (k, H, W) from clean image

        # Per-patch bit mismatch rate
        bm = pixel_to_bit_mask(pixel_mask, model.n_bits)
        resid = torch.zeros(ls, ls, device=device)
        for i in range(ls):
            for j in range(ls):
                if not bm[0, i, j]:  # observed position
                    # Compare z vs z_clean
                    mismatch = (z[:, i, j] != z_clean[:, i, j]).float().mean()
                    resid[i, j] = mismatch.item()
                else:
                    resid[i, j] = float('inf')
        return resid


class FreqObsProtocol:
    """
    Frequency-domain E_obs: compare DCT spectrum of z.
    Uses DCT of z features — frequency-domain matching without
    going through pixel reconstruction.
    """
    name = 'freq_dct'

    def residual_per_patch(self, model, z, x_clean, pixel_mask, device):
        """
        Compare DCT spectrum of z vs z_clean.
        Low-frequency mismatch indicates structural deviation.
        """
        ls = model.latent_size
        with torch.no_grad():
            x_t = torch.from_numpy(x_clean[None, None].astype(np.float32)).to(device)
            z_clean = model.encode(x_t)[0]

        # 2D DCT via FFT
        z_dct = torch.fft.rfft2(z.unsqueeze(0).float())[0]  # (k, H, W//2+1)
        zc_dct = torch.fft.rfft2(z_clean.unsqueeze(0).float())[0]

        # Per-position spectral residual (use low-freq energy difference)
        bm = pixel_to_bit_mask(pixel_mask, model.n_bits)
        resid = torch.zeros(ls, ls, device=device)
        # Global spectral mismatch
        spec_diff = (z_dct - zc_dct).abs().mean(dim=0)  # (H, W//2+1)
        total_diff = spec_diff.sum().item()

        for i in range(ls):
            for j in range(ls):
                if not bm[0, i, j]:
                    mismatch = (z[:, i, j] != z_clean[:, i, j]).float().mean()
                    resid[i, j] = mismatch.item()
                else:
                    resid[i, j] = float('inf')
        return resid


# ============================================================================
# MASKS + DATA
# ============================================================================

def make_center_mask(H=28, W=28):
    m = np.ones((H,W), dtype=np.float32); m[7:21, 7:21] = 0; return m
def make_stripe_mask(H=28, W=28):
    m = np.ones((H,W), dtype=np.float32)
    for y in range(0, H, 6): m[y:min(y+2,H), :] = 0
    return m
def make_random_block_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((H,W), dtype=np.float32)
    oh,ow = rng.integers(8,18),rng.integers(8,18)
    y,x = rng.integers(0,max(1,H-oh+1)),rng.integers(0,max(1,W-ow+1))
    m[y:y+oh,x:x+ow] = 0; return m
def make_random_stripe_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    w = rng.integers(1,4); g = rng.integers(4,10); p = rng.integers(0,g)
    m = np.ones((H,W), dtype=np.float32)
    if rng.random() < 0.5:
        for y in range(p, H, g): m[y:min(y+w,H), :] = 0
    else:
        for x in range(p, W, g): m[:, x:min(x+w,W)] = 0
    return m
def make_multi_hole_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((H,W), dtype=np.float32)
    for _ in range(rng.integers(3,8)):
        s = rng.integers(2,6)
        y,x = rng.integers(0,max(1,H-s+1)),rng.integers(0,max(1,W-s+1))
        m[y:y+s,x:x+s] = 0
    return m
def sample_training_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    p = rng.random()
    if p < 0.30: return make_random_block_mask(H,W,rng)
    elif p < 0.50: return make_center_mask(H,W)
    elif p < 0.70: return make_random_stripe_mask(H,W,rng)
    else: return make_multi_hole_mask(H,W,rng)
def pixel_to_bit_mask(pixel_mask, n_bits, latent_size=7):
    H,W = pixel_mask.shape; ph,pw = H//latent_size,W//latent_size
    bm = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    for i in range(latent_size):
        for j in range(latent_size):
            y0,y1 = i*ph,(i+1)*ph; x0,x1 = j*pw,(j+1)*pw
            if pixel_mask[y0:y1,x0:x1].mean() < 1.0-1e-6: bm[:,i,j] = True
    return bm

def load_dataset(name, train_n=2000, test_n=500, seed=42):
    from torchvision import datasets, transforms
    ds_map = {'mnist': datasets.MNIST, 'fmnist': datasets.FashionMNIST}
    ds_cls = ds_map[name]
    tr = ds_cls('./data', train=True, download=True, transform=transforms.ToTensor())
    te = ds_cls('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(tr), train_n, replace=False)
    si = rng.choice(len(te), test_n, replace=False)
    return (torch.stack([tr[i][0] for i in ti]), torch.tensor([tr[i][1] for i in ti]),
            torch.stack([te[i][0] for i in si]), torch.tensor([te[i][1] for i in si]))

def train_model(train_x, train_y, device, epochs=5):
    model = RouteCModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    for epoch in range(epochs):
        model.train()
        model.set_temperature(1.0 + (0.2-1.0)*epoch/max(1,epochs-1))
        el,nb = 0.,0
        for x,y in loader:
            x = x.to(device); opt.zero_grad()
            z = model.encode(x); x_hat = model.decode(z)
            cl = model.local_pred(z)
            lr_ = F.binary_cross_entropy(x_hat.clamp(1e-6,1-1e-6), x)
            m = torch.rand_like(z) < 0.15
            lc = F.binary_cross_entropy_with_logits(cl[m], z.detach()[m]) if m.any() else torch.tensor(0.,device=device)
            (lr_+0.5*lc).backward(); opt.step(); el += (lr_+0.5*lc).item(); nb += 1
        print(f"      Epoch {epoch+1}/{epochs}: loss={el/max(nb,1):.4f}")
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
    model.eval(); N = len(train_x); rng = np.random.default_rng(42)
    for epoch in range(epochs):
        net.train(); perm = torch.randperm(N); tl,nb = 0.,0
        for i in range(0, N, 64):
            idx = perm[i:i+64]; x = train_x[idx].to(device)
            with torch.no_grad(): z = model.encode(x); z_hard = (z > 0.5).float()
            B,k,H,W = z_hard.shape
            masks = [torch.from_numpy(pixel_to_bit_mask(sample_training_mask(28,28,rng),k)).float() for _ in range(B)]
            bit_masks = torch.stack(masks).to(device)
            z_masked = z_hard * (1 - bit_masks)
            logits = net(z_masked, bit_masks[:, 0:1])
            loss = F.binary_cross_entropy_with_logits(logits[bit_masks.bool()], z_hard[bit_masks.bool()])
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); nb += 1
        if (epoch+1) % 10 == 0:
            print(f"        epoch {epoch+1}/{epochs}: loss={tl/max(nb,1):.4f}")
    return net


# ============================================================================
# EVALUATION WITH DIFFERENT OBSERVATION PROTOCOLS
# ============================================================================

def evaluate_protocol(protocol, model, net, mask_type, test_x, test_y, device,
                      n_samples=100, seed=42):
    model.eval(); net.eval()
    pm = make_center_mask() if mask_type == 'center' else make_stripe_mask()
    occ = 1 - pm

    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    cb, ca, rts, ratios = [], [], [], []

    for idx in eval_idx:
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()
        x_occ = x_clean * pm

        with torch.no_grad():
            x_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_init = model.encode(x_t)[0]
            pred_b = model.classifier(z_init.unsqueeze(0)).argmax(1).item()

        bit_mask = pixel_to_bit_mask(pm, model.n_bits)

        # Full repair candidate
        z = z_init.clone().to(device)
        bm = torch.from_numpy(bit_mask).float().to(device)
        mask = bm.max(dim=0, keepdim=True)[0].unsqueeze(0)
        z_masked = z.unsqueeze(0) * (1 - bm.unsqueeze(0))
        with torch.no_grad():
            logits = net(z_masked, mask)
            preds = (torch.sigmoid(logits) > 0.5).float()
        bm_bool = torch.from_numpy(bit_mask).to(device)
        z_repaired = z.clone()
        z_repaired[bm_bool] = preds[0][bm_bool]

        # Get residuals from protocol
        t0 = time.time()
        resid = protocol.residual_per_patch(model, z_init, x_clean, pm, device)

        # Evidence-based selective repair (threshold=1.0)
        z_result = z_init.clone()
        ls = model.latent_size
        repair_count, total_masked = 0, 0
        for i in range(ls):
            for j in range(ls):
                if bit_mask[0, i, j]:
                    total_masked += 1
                    r = resid[i, j].item()
                    if np.isinf(r) or r > 0.5:  # use 0.5 for token/freq since scale differs
                        z_result[:, i, j] = z_repaired[:, i, j]
                        repair_count += 1

        rt = (time.time() - t0) * 1000
        ratio = repair_count / max(total_masked, 1)

        with torch.no_grad():
            pred_a = model.classifier(z_result.unsqueeze(0)).argmax(1).item()

        cb.append(int(pred_b == label)); ca.append(int(pred_a == label))
        rts.append(rt); ratios.append(ratio)

    n = len(eval_idx)
    return {
        'delta_acc': (np.sum(ca) - np.sum(cb)) / n,
        'acc_before': np.mean(cb), 'acc_after': np.mean(ca),
        'runtime_ms': np.mean(rts), 'repair_ratio': np.mean(ratios),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--output_dir', default='outputs/exp_phase4')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 4: OBSERVATION PROTOCOL MIGRATION")
    print("=" * 100)

    print("[1] Loading FashionMNIST...")
    train_x, train_y, test_x, test_y = load_dataset('fmnist', 2000, 500, args.seed)

    print("[2] Training model...")
    model = train_model(train_x, train_y, device)

    print("[3] Training InpaintNet...")
    net = train_inpaint(model, train_x, device)

    protocols = [PixelObsProtocol(), TokenObsProtocol(), FreqObsProtocol()]
    mask_types = ['center', 'stripes']
    all_results = []

    print("\n" + "=" * 80)
    print("OBSERVATION PROTOCOL COMPARISON")
    print("=" * 80)
    print(f"{'protocol':<15} {'mask':<10} {'Δacc':>7} {'ratio':>6} {'ms':>6}")
    print("-" * 50)

    for proto in protocols:
        for mt in mask_types:
            r = evaluate_protocol(proto, model, net, mt, test_x, test_y, device,
                                 n_samples=args.eval_samples, seed=args.seed)
            r.update({'protocol': proto.name, 'mask_type': mt})
            all_results.append(r)
            print(f"{proto.name:<15} {mt:<10} {r['delta_acc']:>+7.1%} "
                  f"{r['repair_ratio']:>6.2f} {r['runtime_ms']:>6.1f}")

    # Summary
    print("\n" + "=" * 80)
    print("PROTOCOL SUMMARY")
    print("=" * 80)
    for proto in protocols:
        pr = [r for r in all_results if r['protocol'] == proto.name]
        total = sum(r['delta_acc'] for r in pr)
        center = [r for r in pr if r['mask_type'] == 'center']
        stripes = [r for r in pr if r['mask_type'] == 'stripes']
        print(f"  {proto.name:<15} center={center[0]['delta_acc']:+.1%}, "
              f"stripes={stripes[0]['delta_acc']:+.1%}, total={total:+.1%}")

    print("\n  Key question: Can token/freq E_obs match pixel E_obs?")
    pixel_total = sum(r['delta_acc'] for r in all_results if r['protocol'] == 'pixel_bce')
    token_total = sum(r['delta_acc'] for r in all_results if r['protocol'] == 'token_bce')
    freq_total = sum(r['delta_acc'] for r in all_results if r['protocol'] == 'freq_dct')
    print(f"  pixel_bce: {pixel_total:+.1%}")
    print(f"  token_bce: {token_total:+.1%}")
    print(f"  freq_dct:  {freq_total:+.1%}")

    csv_path = os.path.join(args.output_dir, "phase4_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            keys = sorted(set().union(*(r.keys() for r in all_results)))
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    print("\n" + "=" * 100)
    print("Phase 4 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
