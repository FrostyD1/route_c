#!/usr/bin/env python3
"""
Scale-Up Experiment: 7×7 → 14×14
==================================
Critical Experiment B: Does the global relation operator (GDA) become
necessary at larger grid scale?

At 7×7 (N=49), mask mixture dominated over GDA (+5% vs ~0%).
At 14×14 (N=196), there are 4× more tokens → more retrieval value.

Paradigm predictions:
  - If Hopfield hypothesis holds: GDA gap grows with grid size
  - If wrong: GDA gap stays flat or shrinks

Tests: FMNIST (strongest signal from cross-dataset experiment)
Configs: 7×7 vs 14×14, amortized vs amortized+GDA

Usage:
    python3 -u benchmarks/exp_scale14.py --device cuda --seed 42
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
# MODEL — Parameterized for different grid sizes
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


class Encoder7(nn.Module):
    """28×28 → 7×7 (stride 2, 2)"""
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1),
        )
    def forward(self, x): return self.conv(x)


class Decoder7(nn.Module):
    """7×7 → 28×28"""
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z): return self.deconv(z)


class Encoder14(nn.Module):
    """28×28 → 14×14 (single stride-2 layer)"""
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1),
        )
    def forward(self, x): return self.conv(x)


class Decoder14(nn.Module):
    """14×14 → 28×28"""
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z): return self.deconv(z)


class LocalPredictor(nn.Module):
    def __init__(self, n_bits, hidden_dim=32):
        super().__init__()
        self.n_bits = n_bits
        self.net = nn.Sequential(
            nn.Linear(9 * n_bits, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_bits),
        )
    def forward(self, z):
        B, k, H, W = z.shape
        z_pad = F.pad(z, (1,1,1,1), mode='constant', value=0)
        windows = F.unfold(z_pad, kernel_size=3)
        windows = windows.reshape(B, k, 9, H*W)
        windows[:, :, 4, :] = 0
        windows = windows.reshape(B, k*9, H*W).permute(0,2,1)
        logits = self.net(windows)
        return logits.permute(0,2,1).reshape(B, k, H, W)


class Classifier(nn.Module):
    def __init__(self, n_bits, latent_size, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(n_bits * latent_size * latent_size, n_classes)
    def forward(self, z): return self.fc(z.reshape(z.shape[0], -1))


class RouteCModel(nn.Module):
    def __init__(self, n_bits=8, latent_size=7, hidden_dim=64, energy_hidden=32, tau=1.0):
        super().__init__()
        self.n_bits = n_bits
        self.latent_size = latent_size
        if latent_size == 7:
            self.encoder = Encoder7(n_bits, hidden_dim)
            self.decoder = Decoder7(n_bits, hidden_dim)
        elif latent_size == 14:
            self.encoder = Encoder14(n_bits, hidden_dim)
            self.decoder = Decoder14(n_bits, hidden_dim)
        else:
            raise ValueError(f"Unsupported latent_size={latent_size}")
        self.quantizer = GumbelSigmoid(tau)
        self.local_pred = LocalPredictor(n_bits, energy_hidden)
        self.classifier = Classifier(n_bits, latent_size)
    def encode(self, x): return self.quantizer(self.encoder(x))
    def decode(self, z): return self.decoder(z)
    def set_temperature(self, tau): self.quantizer.set_temperature(tau)
    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z), self.classifier(z), self.local_pred(z)


# ============================================================================
# GDA (Global Discrete Attention)
# ============================================================================

class GlobalDiscreteAttention(nn.Module):
    """GDA with evidence gating for 14×14 grids."""
    def __init__(self, k=8, d_value=32, temperature=1.0):
        super().__init__()
        self.k = k
        self.d_value = d_value
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.value_embed = nn.Embedding(2**k, d_value)
        self.mask_value = nn.Parameter(torch.randn(d_value) * 0.01)
        # 256-entry popcount LUT
        pop_lut = torch.zeros(256, dtype=torch.float32)
        for i in range(256):
            pop_lut[i] = bin(i).count('1')
        self.register_buffer('pop_lut', pop_lut)

    def forward(self, z, evidence_mask):
        """
        Args:
            z: (B, k, H, W) binary
            evidence_mask: (B, N) bool — True where evidence exists
        Returns:
            out: (B, d_value, H, W)
        """
        B, k, H, W = z.shape
        N = H * W
        # Bitpack
        bits = (z > 0.5).long()
        shifts = (2 ** torch.arange(k, device=z.device)).view(1, k, 1, 1)
        codes = (bits * shifts).sum(dim=1).reshape(B, N)  # (B, N) uint8 codes
        # Hamming via popcount LUT
        xor = codes.unsqueeze(2) ^ codes.unsqueeze(1)  # (B, N, N)
        dist = self.pop_lut[xor.clamp(0, 255)]  # (B, N, N)
        # Attention scores with evidence gating
        scores = -dist / self.temperature.clamp(min=0.1)
        gate = evidence_mask.unsqueeze(1).expand(-1, N, -1)  # (B, N, N)
        scores = scores.masked_fill(~gate, float('-inf'))
        attn = F.softmax(scores, dim=-1)  # (B, N, N)
        # NaN guard (rows where all K/V are masked → uniform)
        nan_mask = attn.isnan()
        if nan_mask.any():
            attn = attn.masked_fill(nan_mask, 0.0)
        # Values with learned mask token
        values = self.value_embed(codes.clamp(0, 255))  # (B, N, d_value)
        no_evidence = ~evidence_mask  # (B, N)
        mask_val = self.mask_value.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        values = torch.where(no_evidence.unsqueeze(-1), mask_val, values)
        # Aggregate
        out = torch.bmm(attn, values)  # (B, N, d_value)
        return out.permute(0, 2, 1).reshape(B, self.d_value, H, W)


class InpaintNet(nn.Module):
    """Local-only InpaintNet."""
    def __init__(self, k=8, hidden=64):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Conv2d(k+1, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(hidden, k, 3, padding=1, padding_mode='circular'),
        )
        self.skip = nn.Conv2d(k+1, k, 1)
    def forward(self, z_masked, mask):
        x = torch.cat([z_masked, mask], dim=1)
        return self.net(x) + self.skip(x)


class InpaintNetGDA(nn.Module):
    """InpaintNet with GDA branch (local + global)."""
    def __init__(self, k=8, hidden=64, d_value=32):
        super().__init__()
        self.k = k
        self.local_net = nn.Sequential(
            nn.Conv2d(k+1, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(hidden, k, 3, padding=1, padding_mode='circular'),
        )
        self.local_skip = nn.Conv2d(k+1, k, 1)
        self.gda = GlobalDiscreteAttention(k=k, d_value=d_value)
        self.fuse = nn.Conv2d(k + d_value, k, 1)

    def forward(self, z_masked, mask, evidence_mask=None):
        x = torch.cat([z_masked, mask], dim=1)
        local_out = self.local_net(x) + self.local_skip(x)
        if evidence_mask is None:
            # evidence = unmasked positions
            evidence_mask = (mask[:, 0] < 0.5).reshape(mask.shape[0], -1)
        gda_out = self.gda(z_masked, evidence_mask)
        fused = self.fuse(torch.cat([local_out, gda_out], dim=1))
        return fused


# ============================================================================
# MASKS + DATA
# ============================================================================

def make_center_mask(H=28, W=28):
    m = np.ones((H,W), dtype=np.float32); m[7:21, 7:21] = 0; return m

def make_stripe_mask(H=28, W=28):
    m = np.ones((H,W), dtype=np.float32)
    for y in range(0, H, 6): m[y:min(y+2,H), :] = 0
    return m

def make_random_stripe_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    w = rng.integers(1, 4); g = rng.integers(4, 10); p = rng.integers(0, g)
    m = np.ones((H,W), dtype=np.float32)
    if rng.random() < 0.5:
        for y in range(p, H, g): m[y:min(y+w,H), :] = 0
    else:
        for x in range(p, W, g): m[:, x:min(x+w,W)] = 0
    return m

def make_random_block_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((H,W), dtype=np.float32)
    oh, ow = rng.integers(8,18), rng.integers(8,18)
    y, x = rng.integers(0, max(1,H-oh+1)), rng.integers(0, max(1,W-ow+1))
    m[y:y+oh, x:x+ow] = 0; return m

def make_multi_hole_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((H,W), dtype=np.float32)
    for _ in range(rng.integers(3,8)):
        s = rng.integers(2,6)
        y, x = rng.integers(0, max(1,H-s+1)), rng.integers(0, max(1,W-s+1))
        m[y:y+s, x:x+s] = 0
    return m

def sample_training_mask(H=28, W=28, rng=None):
    if rng is None: rng = np.random.default_rng()
    p = rng.random()
    if p < 0.4: return make_random_block_mask(H,W,rng)
    elif p < 0.6: return make_center_mask(H,W)
    elif p < 0.8: return make_random_stripe_mask(H,W,rng)
    else: return make_multi_hole_mask(H,W,rng)

def pixel_to_bit_mask(pixel_mask, n_bits, latent_size):
    """Map pixel mask to bit mask for arbitrary grid size."""
    H_img, W_img = pixel_mask.shape
    patch_h = H_img // latent_size
    patch_w = W_img // latent_size
    bm = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    for i in range(latent_size):
        for j in range(latent_size):
            y0, y1 = i * patch_h, (i+1) * patch_h
            x0, x1 = j * patch_w, (j+1) * patch_w
            if pixel_mask[y0:y1, x0:x1].mean() < 1.0 - 1e-6:
                bm[:, i, j] = True
    return bm

def apply_noise(image, noise_type, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    if noise_type == 'noise':
        return np.clip(image + rng.normal(0, 0.1, image.shape).astype(np.float32), 0, 1)
    return image.copy()

def load_fmnist(train_n=2000, test_n=1000, seed=42):
    from torchvision import datasets, transforms
    tr = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    te = datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(tr), train_n, replace=False)
    si = rng.choice(len(te), test_n, replace=False)
    return (torch.stack([tr[i][0] for i in ti]), torch.tensor([tr[i][1] for i in ti]),
            torch.stack([te[i][0] for i in si]), torch.tensor([te[i][1] for i in si]))


# ============================================================================
# TRAINING
# ============================================================================

def train_model(train_x, train_y, device, latent_size=7, epochs=5, lr=1e-3, batch_size=64):
    model = RouteCModel(latent_size=latent_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        tau = 1.0 + (0.2-1.0) * epoch / max(1, epochs-1)
        model.set_temperature(tau)
        el, nb = 0., 0
        for x, y in loader:
            x = x.to(device); opt.zero_grad()
            z, x_hat, _, cl = model(x)
            lr_ = F.binary_cross_entropy(x_hat.clamp(1e-6,1-1e-6), x)
            m = torch.rand_like(z) < 0.15
            lc = F.binary_cross_entropy_with_logits(cl[m], z.detach()[m]) if m.any() else torch.tensor(0., device=device)
            (lr_ + 0.5*lc).backward(); opt.step()
            el += (lr_+0.5*lc).item(); nb += 1
        print(f"      Epoch {epoch+1}/{epochs}: loss={el/max(nb,1):.4f}")
    # Frozen probe
    for p in model.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True
    co = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    for _ in range(3):
        model.classifier.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): z = model.encode(x)
            F.cross_entropy(model.classifier(z), y).backward(); co.step(); co.zero_grad()
    for p in model.parameters(): p.requires_grad = True
    return model


def train_inpaint(model, train_x, device, net, epochs=20, batch_size=64, lr=1e-3, use_gda=False):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    model.eval()
    N = len(train_x)
    rng = np.random.default_rng(42)
    ls = model.latent_size

    for epoch in range(epochs):
        net.train(); perm = torch.randperm(N); tl, nb = 0., 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]; x = train_x[idx].to(device)
            with torch.no_grad():
                z = model.encode(x); z_hard = (z > 0.5).float()
            B, k, H, W = z_hard.shape
            masks = []
            for b in range(B):
                pm = sample_training_mask(28, 28, rng)
                bm = pixel_to_bit_mask(pm, k, ls)
                masks.append(torch.from_numpy(bm).float())
            bit_masks = torch.stack(masks).to(device)
            pos_masks = bit_masks[:, 0:1, :, :]
            z_masked = z_hard * (1 - bit_masks)

            if use_gda:
                evidence = (pos_masks[:, 0] < 0.5).reshape(B, -1)
                logits = net(z_masked, pos_masks, evidence)
            else:
                logits = net(z_masked, pos_masks)

            loss = F.binary_cross_entropy_with_logits(logits[bit_masks.bool()], z_hard[bit_masks.bool()])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        if (epoch+1) % 5 == 0:
            print(f"        epoch {epoch+1}/{epochs}: loss={tl/max(nb,1):.4f}")
    return net


# ============================================================================
# EVALUATION
# ============================================================================

def amortized_inpaint(net, z_init, bit_mask, device, use_gda=False):
    net.eval()
    z = z_init.clone().to(device)
    bm = torch.from_numpy(bit_mask).float().to(device)
    mask = bm.max(dim=0, keepdim=True)[0].unsqueeze(0)
    z_masked = z.unsqueeze(0) * (1 - bm.unsqueeze(0))
    with torch.no_grad():
        if use_gda:
            evidence = (mask[:, 0] < 0.5).reshape(1, -1)
            logits = net(z_masked, mask, evidence)
        else:
            logits = net(z_masked, mask)
        preds = (torch.sigmoid(logits) > 0.5).float()
    bm_bool = torch.from_numpy(bit_mask).to(device)
    z_result = z.clone()
    z_result[bm_bool] = preds[0][bm_bool]
    return z_result


def evaluate(model, net, mask_type, noise_type, test_x, test_y, device,
             n_samples=100, seed=42, use_gda=False):
    model.eval(); net.eval()
    pm = make_center_mask() if mask_type == 'center' else make_stripe_mask()
    occ = 1 - pm
    ls = model.latent_size

    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    cb, ca, bce_b, bce_a, rts = [], [], [], [], []

    for idx in eval_idx:
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()
        x_noisy = apply_noise(x_clean, noise_type, rng)
        x_occ = x_noisy * pm

        with torch.no_grad():
            x_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_init = model.encode(x_t)[0]
            pred_b = model.classifier(z_init.unsqueeze(0)).argmax(1).item()
            o_hat_b = model.decode(z_init.unsqueeze(0))[0, 0]

        bit_mask = pixel_to_bit_mask(pm, model.n_bits, ls)

        t0 = time.time()
        z_final = amortized_inpaint(net, z_init, bit_mask, device, use_gda)
        rt = (time.time() - t0) * 1000

        with torch.no_grad():
            pred_a = model.classifier(z_final.unsqueeze(0)).argmax(1).item()
            o_hat_a = model.decode(z_final.unsqueeze(0))[0, 0]

        xt = torch.from_numpy(x_clean).to(device)
        ot = torch.from_numpy(occ).to(device)
        os_ = ot.sum().clamp(min=1.0).item()

        def ob(oh):
            l = oh.clamp(1e-6,1-1e-6)
            return (-(xt*torch.log(l)+(1-xt)*torch.log(1-l))*ot).sum().item()/os_

        cb.append(int(pred_b == label)); ca.append(int(pred_a == label))
        bce_b.append(ob(o_hat_b)); bce_a.append(ob(o_hat_a))
        rts.append(rt)

    n = len(eval_idx)
    return {
        'acc_before': np.mean(cb),
        'acc_after': np.mean(ca),
        'delta_acc': (np.sum(ca) - np.sum(cb)) / n,
        'bce_before': np.mean(bce_b),
        'bce_after': np.mean(bce_a),
        'runtime_ms': np.mean(rts),
        'n_samples': n,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--inpaint_epochs', type=int, default=20)
    parser.add_argument('--output_dir', default='outputs/exp_scale14')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("SCALE-UP EXPERIMENT: 7×7 vs 14×14")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Dataset: FashionMNIST (strongest cross-dataset signal)")
    print()

    print("[1] Loading FashionMNIST...")
    train_x, train_y, test_x, test_y = load_fmnist(2000, 1000, args.seed)

    configs = [('center', 'clean'), ('center', 'noise')]
    all_results = []

    for latent_size in [7, 14]:
        N_tokens = latent_size * latent_size
        print(f"\n{'='*80}")
        print(f"  GRID: {latent_size}×{latent_size} (N={N_tokens} tokens)")
        print(f"{'='*80}")

        torch.manual_seed(args.seed); np.random.seed(args.seed)

        print(f"\n  [2] Training Route C model (latent={latent_size})...")
        model = train_model(train_x, train_y, device,
                          latent_size=latent_size, epochs=args.epochs)
        model.eval()
        with torch.no_grad():
            z = model.encode(test_x[:500].to(device))
            acc = (model.classifier(z).argmax(1).cpu() == test_y[:500]).float().mean().item()
        print(f"      Clean probe accuracy: {acc:.1%}")
        print(f"      z shape: {z.shape}")

        # Train local-only InpaintNet
        print(f"\n  [3a] Training InpaintNet (local only)...")
        net_local = InpaintNet(k=model.n_bits).to(device)
        train_inpaint(model, train_x, device, net_local, epochs=args.inpaint_epochs)

        # Train InpaintNet + GDA
        print(f"\n  [3b] Training InpaintNet+GDA...")
        net_gda = InpaintNetGDA(k=model.n_bits).to(device)
        train_inpaint(model, train_x, device, net_gda, epochs=args.inpaint_epochs, use_gda=True)

        for mt, nt in configs:
            # Local only
            r = evaluate(model, net_local, mt, nt, test_x, test_y, device,
                        n_samples=args.eval_samples, seed=args.seed, use_gda=False)
            r.update({'grid': f'{latent_size}x{latent_size}', 'method': 'local',
                      'mask_type': mt, 'noise_type': nt, 'N_tokens': N_tokens})
            all_results.append(r)
            print(f"    {mt}+{nt} local:  Δacc={r['delta_acc']:+.1%}, "
                  f"bce={r['bce_before']:.2f}→{r['bce_after']:.2f}, t={r['runtime_ms']:.1f}ms")

            # GDA
            r = evaluate(model, net_gda, mt, nt, test_x, test_y, device,
                        n_samples=args.eval_samples, seed=args.seed, use_gda=True)
            r.update({'grid': f'{latent_size}x{latent_size}', 'method': 'local+GDA',
                      'mask_type': mt, 'noise_type': nt, 'N_tokens': N_tokens})
            all_results.append(r)
            print(f"    {mt}+{nt} +GDA:   Δacc={r['delta_acc']:+.1%}, "
                  f"bce={r['bce_before']:.2f}→{r['bce_after']:.2f}, t={r['runtime_ms']:.1f}ms")

    # Summary
    print("\n" + "=" * 100)
    print("SCALE-UP SUMMARY")
    print("=" * 100)
    print(f"{'grid':<8} {'method':<12} {'mask':<8} {'noise':<6} "
          f"{'acc_bef':>7} {'acc_aft':>7} {'Δacc':>7} "
          f"{'bce_aft':>8} {'ms':>8}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['grid']:<8} {r['method']:<12} {r['mask_type']:<8} {r['noise_type']:<6} "
              f"{r['acc_before']:>7.1%} {r['acc_after']:>7.1%} {r['delta_acc']:>+7.1%} "
              f"{r['bce_after']:>8.2f} {r['runtime_ms']:>8.1f}")

    # GDA gap analysis
    print("\n" + "=" * 80)
    print("KEY: GDA GAP (does it grow with scale?)")
    print("=" * 80)
    for mt, nt in configs:
        print(f"\n  {mt}+{nt}:")
        for ls in [7, 14]:
            grid = f'{ls}x{ls}'
            local = [r for r in all_results if r['grid']==grid and r['method']=='local'
                     and r['mask_type']==mt and r['noise_type']==nt]
            gda = [r for r in all_results if r['grid']==grid and r['method']=='local+GDA'
                   and r['mask_type']==mt and r['noise_type']==nt]
            if local and gda:
                gap = gda[0]['delta_acc'] - local[0]['delta_acc']
                print(f"    {grid}: local={local[0]['delta_acc']:+.1%}, "
                      f"+GDA={gda[0]['delta_acc']:+.1%}, gap={gap:+.1%}")

    # Paradigm verdict
    print("\n" + "=" * 80)
    print("HOPFIELD HYPOTHESIS: GDA gap grows with grid size?")
    print("=" * 80)
    for mt, nt in configs:
        gap_7 = None; gap_14 = None
        for ls in [7, 14]:
            grid = f'{ls}x{ls}'
            local = [r for r in all_results if r['grid']==grid and r['method']=='local'
                     and r['mask_type']==mt and r['noise_type']==nt]
            gda = [r for r in all_results if r['grid']==grid and r['method']=='local+GDA'
                   and r['mask_type']==mt and r['noise_type']==nt]
            if local and gda:
                gap = gda[0]['delta_acc'] - local[0]['delta_acc']
                if ls == 7: gap_7 = gap
                else: gap_14 = gap
        if gap_7 is not None and gap_14 is not None:
            grows = gap_14 > gap_7
            verdict = "CONFIRMED" if grows else "NOT CONFIRMED"
            print(f"  {mt}+{nt}: gap_7={gap_7:+.1%}, gap_14={gap_14:+.1%} → {verdict}")

    # Save
    csv_path = os.path.join(args.output_dir, "scale14_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")

    print("\n" + "=" * 100)
    print("Scale-up experiment complete.")
    print("=" * 100)

    return all_results


if __name__ == "__main__":
    main()
