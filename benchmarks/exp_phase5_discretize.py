#!/usr/bin/env python3
"""
Phase 5: Boundary Translation Layer Discretization
====================================================
Test: How much can we quantize the encoder/decoder without breaking
the paradigm?

Approach:
  1. Train full-precision model (FP32)
  2. Post-training quantize encoder/decoder weights to INT8
  3. Compare: FP32 vs INT8 on key metrics (probe acc, repair Δacc)
  4. Also test replacing conv layers with fixed filters (Sobel/DCT)

Usage:
    python3 -u benchmarks/exp_phase5_discretize.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os, sys, time, csv, argparse

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

class FixedFilterEncoder(nn.Module):
    """Encoder using fixed Sobel/edge filters + learnable gate only."""
    def __init__(self, n_bits=8, hidden_dim=32):
        super().__init__()
        # Fixed Sobel-like edge detectors (not learned)
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        avg = torch.ones(3,3, dtype=torch.float32) / 9.0
        laplacian = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)

        fixed = torch.stack([sobel_x, sobel_y, avg, laplacian])  # (4, 3, 3)
        self.register_buffer('fixed_filters', fixed.unsqueeze(1))  # (4, 1, 3, 3)

        # Learnable gate + projection (minimal learnable params)
        self.gate = nn.Sequential(
            nn.Conv2d(4, hidden_dim, 1), nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 1))
        self.pool = nn.AvgPool2d(4, stride=4)  # 28→7

    def forward(self, x):
        # Fixed feature extraction
        feat = F.conv2d(x, self.fixed_filters, padding=1)  # (B, 4, 28, 28)
        # Learnable gate + downscale
        return self.gate(self.pool(feat))  # (B, n_bits, 7, 7)

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

class Classifier(nn.Module):
    def __init__(self, n_bits=8, latent_size=7):
        super().__init__()
        self.fc = nn.Linear(n_bits*latent_size*latent_size, 10)
    def forward(self, z): return self.fc(z.reshape(z.shape[0],-1))

class RouteCModel(nn.Module):
    def __init__(self, n_bits=8, hidden_dim=64, encoder_type='learned', tau=1.0):
        super().__init__()
        self.n_bits = n_bits; self.latent_size = 7
        if encoder_type == 'learned':
            self.encoder = BinaryEncoder(n_bits, hidden_dim)
        elif encoder_type == 'fixed_filter':
            self.encoder = FixedFilterEncoder(n_bits, hidden_dim//2)
        self.decoder = BinaryDecoder(n_bits, hidden_dim)
        self.quantizer = GumbelSigmoid(tau)
        self.local_pred = LocalPredictor(n_bits)
        self.classifier = Classifier(n_bits)
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


def make_center_mask():
    m = np.ones((28,28), dtype=np.float32); m[7:21,7:21] = 0; return m
def make_random_block_mask(rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((28,28), dtype=np.float32)
    oh,ow = rng.integers(8,18),rng.integers(8,18)
    y,x = rng.integers(0,max(1,28-oh+1)),rng.integers(0,max(1,28-ow+1))
    m[y:y+oh,x:x+ow] = 0; return m
def make_random_stripe_mask(rng=None):
    if rng is None: rng = np.random.default_rng()
    w = rng.integers(1,4); g = rng.integers(4,10); p = rng.integers(0,g)
    m = np.ones((28,28), dtype=np.float32)
    if rng.random() < 0.5:
        for y in range(p,28,g): m[y:min(y+w,28),:] = 0
    else:
        for x in range(p,28,g): m[:,x:min(x+w,28)] = 0
    return m
def make_multi_hole_mask(rng=None):
    if rng is None: rng = np.random.default_rng()
    m = np.ones((28,28), dtype=np.float32)
    for _ in range(rng.integers(3,8)):
        s = rng.integers(2,6)
        y,x = rng.integers(0,max(1,28-s+1)),rng.integers(0,max(1,28-s+1))
        m[y:y+s,x:x+s] = 0
    return m
def sample_training_mask(rng=None):
    if rng is None: rng = np.random.default_rng()
    p = rng.random()
    if p < 0.30: return make_random_block_mask(rng)
    elif p < 0.50: return make_center_mask()
    elif p < 0.70: return make_random_stripe_mask(rng)
    else: return make_multi_hole_mask(rng)
def pixel_to_bit_mask(pixel_mask, n_bits, ls=7):
    bm = np.zeros((n_bits,ls,ls), dtype=bool)
    ph,pw = 28//ls, 28//ls
    for i in range(ls):
        for j in range(ls):
            if pixel_mask[i*ph:(i+1)*ph,j*pw:(j+1)*pw].mean() < 1.0-1e-6:
                bm[:,i,j] = True
    return bm

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


def quantize_model_int8(model):
    """Post-training INT8 quantization of encoder/decoder weights."""
    model_q = type(model)(n_bits=model.n_bits)
    model_q.load_state_dict(model.state_dict())
    # Quantize all conv/linear weights to INT8 scale
    with torch.no_grad():
        for name, param in model_q.named_parameters():
            if 'encoder' in name or 'decoder' in name:
                if param.dim() >= 2:
                    scale = param.abs().max() / 127.0
                    if scale > 0:
                        param.copy_(torch.round(param / scale) * scale)
    return model_q


def train_and_evaluate(encoder_type, train_x, train_y, test_x, test_y, device,
                       quantize=False, seed=42, n_samples=100):
    torch.manual_seed(seed); np.random.seed(seed)
    hidden = 64 if encoder_type == 'learned' else 32
    model = RouteCModel(encoder_type=encoder_type, hidden_dim=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)

    for epoch in range(5):
        model.train()
        model.set_temperature(1.0 + (0.2-1.0)*epoch/4)
        el,nb = 0.,0
        for x,y in loader:
            x = x.to(device); opt.zero_grad()
            z = model.encode(x); x_hat = model.decode(z)
            cl = model.local_pred(z)
            lr_ = F.binary_cross_entropy(x_hat.clamp(1e-6,1-1e-6), x)
            m = torch.rand_like(z) < 0.15
            lc = F.binary_cross_entropy_with_logits(cl[m], z.detach()[m]) if m.any() else torch.tensor(0.,device=device)
            (lr_+0.5*lc).backward(); opt.step(); el += (lr_+0.5*lc).item(); nb += 1

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

    if quantize:
        model = quantize_model_int8(model).to(device)

    model.eval()
    with torch.no_grad():
        z = model.encode(test_x[:500].to(device))
        probe_acc = (model.classifier(z).argmax(1).cpu() == test_y[:500]).float().mean().item()

    # Train inpaint
    net = InpaintNet(k=model.n_bits).to(device)
    iopt = torch.optim.Adam(net.parameters(), lr=1e-3)
    model.eval(); rng = np.random.default_rng(42)
    for epoch in range(20):
        net.train(); perm = torch.randperm(len(train_x)); tl,nb = 0.,0
        for i in range(0, len(train_x), 64):
            idx = perm[i:i+64]; x = train_x[idx].to(device)
            with torch.no_grad(): z = model.encode(x); z_hard = (z > 0.5).float()
            B,k,H,W = z_hard.shape
            masks = [torch.from_numpy(pixel_to_bit_mask(sample_training_mask(rng),k)).float() for _ in range(B)]
            bm = torch.stack(masks).to(device)
            z_masked = z_hard * (1 - bm)
            logits = net(z_masked, bm[:, 0:1])
            loss = F.binary_cross_entropy_with_logits(logits[bm.bool()], z_hard[bm.bool()])
            iopt.zero_grad(); loss.backward(); iopt.step(); tl += loss.item(); nb += 1

    # Evaluate center Δacc
    pm = make_center_mask(); occ = 1-pm
    rng2 = np.random.default_rng(seed+100)
    eval_idx = rng2.choice(len(test_x), min(n_samples, len(test_x)), replace=False)
    cb,ca = [],[]
    net.eval()
    for idx in eval_idx:
        x_clean = test_x[idx].numpy()[0]; label = test_y[idx].item()
        x_occ = x_clean * pm
        with torch.no_grad():
            xt = torch.from_numpy(x_occ[None,None].astype(np.float32)).to(device)
            z_init = model.encode(xt)[0]
            pred_b = model.classifier(z_init.unsqueeze(0)).argmax(1).item()
        bit_mask = pixel_to_bit_mask(pm, model.n_bits)
        z = z_init.clone(); bm = torch.from_numpy(bit_mask).float().to(device)
        mask = bm.max(dim=0,keepdim=True)[0].unsqueeze(0)
        z_masked = z.unsqueeze(0) * (1-bm.unsqueeze(0))
        with torch.no_grad():
            logits = net(z_masked, mask)
            preds = (torch.sigmoid(logits) > 0.5).float()
        z_result = z.clone()
        z_result[torch.from_numpy(bit_mask).to(device)] = preds[0][torch.from_numpy(bit_mask).to(device)]
        with torch.no_grad():
            pred_a = model.classifier(z_result.unsqueeze(0)).argmax(1).item()
        cb.append(int(pred_b==label)); ca.append(int(pred_a==label))

    n = len(eval_idx)
    # Count params
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())

    return {
        'probe_acc': probe_acc,
        'delta_acc': (sum(ca)-sum(cb))/n,
        'enc_params': enc_params,
        'dec_params': dec_params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--output_dir', default='outputs/exp_phase5')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 100)
    print("PHASE 5: BOUNDARY TRANSLATION DISCRETIZATION")
    print("=" * 100)

    train_x, train_y, test_x, test_y = load_dataset('fmnist', 2000, 500, args.seed)

    configs = [
        ('learned_fp32', 'learned', False),
        ('learned_int8', 'learned', True),
        ('fixed_filter_fp32', 'fixed_filter', False),
    ]

    all_results = []
    print(f"\n{'config':<22} {'probe_acc':>10} {'center_Δacc':>12} "
          f"{'enc_params':>11} {'dec_params':>11}")
    print("-" * 70)

    for name, enc_type, quant in configs:
        r = train_and_evaluate(enc_type, train_x, train_y, test_x, test_y,
                               device, quantize=quant, seed=args.seed,
                               n_samples=args.eval_samples)
        r['config'] = name
        all_results.append(r)
        print(f"{name:<22} {r['probe_acc']:>10.1%} {r['delta_acc']:>+12.1%} "
              f"{r['enc_params']:>11,} {r['dec_params']:>11,}")

    print("\n" + "=" * 80)
    print("DISCRETIZATION VERDICT")
    print("=" * 80)
    fp32 = [r for r in all_results if r['config'] == 'learned_fp32'][0]
    int8 = [r for r in all_results if r['config'] == 'learned_int8'][0]
    fixed = [r for r in all_results if r['config'] == 'fixed_filter_fp32'][0]

    probe_drop_int8 = fp32['probe_acc'] - int8['probe_acc']
    dacc_drop_int8 = fp32['delta_acc'] - int8['delta_acc']
    print(f"  INT8 quantization: probe drop={probe_drop_int8:+.1%}, Δacc drop={dacc_drop_int8:+.1%}")
    print(f"    → {'VIABLE' if abs(probe_drop_int8) < 0.05 and abs(dacc_drop_int8) < 0.05 else 'DEGRADED'}")

    probe_drop_fixed = fp32['probe_acc'] - fixed['probe_acc']
    param_ratio = fixed['enc_params'] / fp32['enc_params']
    print(f"  Fixed filters: probe drop={probe_drop_fixed:+.1%}, "
          f"params={param_ratio:.1%} of learned")
    print(f"    → {'VIABLE' if abs(probe_drop_fixed) < 0.10 else 'DEGRADED'}")

    csv_path = os.path.join(args.output_dir, "phase5_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(set().union(*(r.keys() for r in all_results)))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in all_results: w.writerow(r)

    print(f"\nCSV saved to {csv_path}")
    print("\n" + "=" * 100)
    print("Phase 5 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
