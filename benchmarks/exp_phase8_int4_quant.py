#!/usr/bin/env python3
"""
Phase 8: Hardware-Friendly Encoder Discretization (INT4 + Gabor/DCT Bank)
==========================================================================
GPT recommendation: INT8 works → push to INT4/shift-add.
Fixed filters failed with Sobel/LoG → try richer Gabor/DCT filter bank.

Configs:
  1. learned_fp32 (baseline)
  2. learned_int8 (from Phase 5, sanity check)
  3. learned_int4_ptq (post-training INT4, 16 levels)
  4. learned_int4_qat (quantization-aware training INT4)
  5. gabor_dct_bank (fixed Gabor+DCT bank + learnable 1×1 mixing)

Success criterion:
  - INT4 PTQ: probe drop < 5%, Δacc drop < 10%
  - Gabor/DCT bank: probe > 50% (vs Sobel/LoG 39.8%)

Usage:
    python3 -u benchmarks/exp_phase8_int4_quant.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import os, sys, csv, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))


# ============================================================================
# MODEL COMPONENTS
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

class FakeQuantize(torch.autograd.Function):
    """STE-based fake quantization for QAT."""
    @staticmethod
    def forward(ctx, x, n_levels=16):
        scale = x.abs().max() / (n_levels // 2)
        if scale == 0:
            return x
        x_q = torch.round(x / scale).clamp(-(n_levels//2), n_levels//2 - 1) * scale
        return x_q
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def fake_quantize_int4(x):
    return FakeQuantize.apply(x, 16)

class BinaryEncoderQAT(nn.Module):
    """Encoder with fake INT4 quantization on weights during forward."""
    def __init__(self, n_bits=8, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, n_bits, 3, padding=1)

    def forward(self, x):
        # Quantize weights on-the-fly (STE)
        w1 = fake_quantize_int4(self.conv1.weight)
        w2 = fake_quantize_int4(self.conv2.weight)
        w3 = fake_quantize_int4(self.conv3.weight)
        x = F.relu(F.conv2d(x, w1, self.conv1.bias, stride=2, padding=1))
        x = F.relu(F.conv2d(x, w2, self.conv2.bias, stride=2, padding=1))
        x = F.conv2d(x, w3, self.conv3.bias, padding=1)
        return x

class BinaryDecoderQAT(nn.Module):
    """Decoder with fake INT4 quantization on weights during forward."""
    def __init__(self, n_bits=8, hidden_dim=64):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 1, 3, padding=1)

    def forward(self, z):
        w1 = fake_quantize_int4(self.deconv1.weight)
        w2 = fake_quantize_int4(self.deconv2.weight)
        w3 = fake_quantize_int4(self.conv3.weight)
        x = F.relu(F.conv_transpose2d(z, w1, self.deconv1.bias, stride=2, padding=1))
        x = F.relu(F.conv_transpose2d(x, w2, self.deconv2.bias, stride=2, padding=1))
        x = torch.sigmoid(F.conv2d(x, w3, self.conv3.bias, padding=1))
        return x


def make_gabor_dct_bank(n_filters=16):
    """Create a diverse fixed filter bank: Gabor (8 orientations) + DCT (4) + edge (4)."""
    filters = []

    # 8 Gabor filters at different orientations
    for theta_idx in range(8):
        theta = theta_idx * math.pi / 8
        sigma = 1.5; lambd = 3.0; gamma = 0.5
        kernel = np.zeros((3, 3), dtype=np.float32)
        for y in range(-1, 2):
            for x in range(-1, 2):
                xp = x * math.cos(theta) + y * math.sin(theta)
                yp = -x * math.sin(theta) + y * math.cos(theta)
                kernel[y+1, x+1] = math.exp(-(xp**2 + gamma**2 * yp**2) / (2*sigma**2)) * \
                                   math.cos(2 * math.pi * xp / lambd)
        filters.append(kernel)

    # 4 DCT-like basis functions (2D)
    for u in range(2):
        for v in range(2):
            kernel = np.zeros((3, 3), dtype=np.float32)
            for y in range(3):
                for x in range(3):
                    kernel[y, x] = math.cos(math.pi * (2*x+1) * u / 6) * \
                                   math.cos(math.pi * (2*y+1) * v / 6)
            filters.append(kernel)

    # 4 edge/structure detectors
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
    laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    avg = np.ones((3,3), dtype=np.float32) / 9.0
    filters.extend([sobel_x, sobel_y, laplacian, avg])

    bank = np.stack(filters[:n_filters])  # (n_filters, 3, 3)
    return torch.from_numpy(bank).unsqueeze(1)  # (n_filters, 1, 3, 3)


class GaborDCTEncoder(nn.Module):
    """Fixed Gabor/DCT filter bank + learnable 1×1 mixing."""
    def __init__(self, n_bits=8, n_filters=16, hidden_dim=32):
        super().__init__()
        bank = make_gabor_dct_bank(n_filters)
        self.register_buffer('fixed_bank', bank)
        self.n_filters = n_filters
        # Learnable 1×1 mixing (very few params, easy to netlist)
        self.mix = nn.Sequential(
            nn.Conv2d(n_filters, hidden_dim, 1), nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 1))
        self.pool = nn.AvgPool2d(4, stride=4)  # 28→7

    def forward(self, x):
        feat = F.conv2d(x, self.fixed_bank, padding=1)  # (B, n_filters, 28, 28)
        return self.mix(self.pool(feat))  # (B, n_bits, 7, 7)


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
            self.decoder = BinaryDecoder(n_bits, hidden_dim)
        elif encoder_type == 'qat_int4':
            self.encoder = BinaryEncoderQAT(n_bits, hidden_dim)
            self.decoder = BinaryDecoderQAT(n_bits, hidden_dim)
        elif encoder_type == 'gabor_dct':
            self.encoder = GaborDCTEncoder(n_bits, n_filters=16, hidden_dim=hidden_dim//2)
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


# ============================================================================
# MASKS + DATA
# ============================================================================

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


# ============================================================================
# QUANTIZATION UTILITIES
# ============================================================================

def quantize_model_int8(model):
    """Post-training INT8 quantization."""
    state = model.state_dict()
    model_q = RouteCModel(encoder_type='learned')
    model_q.load_state_dict(state)
    with torch.no_grad():
        for name, param in model_q.named_parameters():
            if 'encoder' in name or 'decoder' in name:
                if param.dim() >= 2:
                    scale = param.abs().max() / 127.0
                    if scale > 0:
                        param.copy_(torch.round(param / scale) * scale)
    return model_q

def quantize_model_int4(model):
    """Post-training INT4 quantization (16 levels)."""
    state = model.state_dict()
    model_q = RouteCModel(encoder_type='learned')
    model_q.load_state_dict(state)
    with torch.no_grad():
        for name, param in model_q.named_parameters():
            if 'encoder' in name or 'decoder' in name:
                if param.dim() >= 2:
                    scale = param.abs().max() / 7.0  # 4-bit: -7 to +7
                    if scale > 0:
                        param.copy_(torch.round(param / scale).clamp(-7, 7) * scale)
    return model_q


# ============================================================================
# TRAIN + EVALUATE
# ============================================================================

def train_and_evaluate(config_name, encoder_type, train_x, train_y, test_x, test_y,
                       device, quantize_fn=None, seed=42, n_samples=100):
    torch.manual_seed(seed); np.random.seed(seed)
    hidden = 64
    model = RouteCModel(encoder_type=encoder_type, hidden_dim=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)

    for epoch in range(5):
        model.train()
        model.set_temperature(1.0 + (0.2-1.0)*epoch/4)
        for x,y in loader:
            x = x.to(device); opt.zero_grad()
            z = model.encode(x); x_hat = model.decode(z)
            cl = model.local_pred(z)
            lr_ = F.binary_cross_entropy(x_hat.clamp(1e-6,1-1e-6), x)
            m = torch.rand_like(z) < 0.15
            lc = F.binary_cross_entropy_with_logits(cl[m], z.detach()[m]) if m.any() else torch.tensor(0.,device=device)
            (lr_+0.5*lc).backward(); opt.step()

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

    # Apply post-training quantization if specified
    if quantize_fn is not None:
        model = quantize_fn(model).to(device)

    model.eval()
    with torch.no_grad():
        z = model.encode(test_x[:500].to(device))
        probe_acc = (model.classifier(z).argmax(1).cpu() == test_y[:500]).float().mean().item()

    # Train inpaint
    net = InpaintNet(k=model.n_bits).to(device)
    iopt = torch.optim.Adam(net.parameters(), lr=1e-3)
    model.eval(); rng = np.random.default_rng(42)
    for epoch in range(20):
        net.train(); perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), 64):
            idx = perm[i:i+64]; x = train_x[idx].to(device)
            with torch.no_grad(): z = model.encode(x); z_hard = (z > 0.5).float()
            B,k,H,W = z_hard.shape
            masks = [torch.from_numpy(pixel_to_bit_mask(sample_training_mask(rng),k)).float() for _ in range(B)]
            bm = torch.stack(masks).to(device)
            z_masked = z_hard * (1 - bm)
            logits = net(z_masked, bm[:, 0:1])
            loss = F.binary_cross_entropy_with_logits(logits[bm.bool()], z_hard[bm.bool()])
            iopt.zero_grad(); loss.backward(); iopt.step()

    # Evaluate center Δacc
    pm = make_center_mask()
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
        bm = torch.from_numpy(bit_mask).float().to(device)
        mask = bm.max(dim=0,keepdim=True)[0].unsqueeze(0)
        z_masked = z_init.unsqueeze(0) * (1-bm.unsqueeze(0))
        with torch.no_grad():
            logits = net(z_masked, mask)
            preds = (torch.sigmoid(logits) > 0.5).float()
        z_result = z_init.clone()
        z_result[torch.from_numpy(bit_mask).to(device)] = preds[0][torch.from_numpy(bit_mask).to(device)]
        with torch.no_grad():
            pred_a = model.classifier(z_result.unsqueeze(0)).argmax(1).item()
        cb.append(int(pred_b==label)); ca.append(int(pred_a==label))

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())

    return {
        'config': config_name,
        'probe_acc': probe_acc,
        'delta_acc': (sum(ca)-sum(cb))/len(eval_idx),
        'enc_params': enc_params,
        'dec_params': dec_params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--output_dir', default='outputs/exp_phase8')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 100)
    print("PHASE 8: INT4 QUANTIZATION + GABOR/DCT FILTER BANK")
    print("=" * 100)

    train_x, train_y, test_x, test_y = load_dataset('fmnist', 2000, 500, args.seed)

    configs = [
        ('learned_fp32',  'learned',   None),
        ('learned_int8',  'learned',   quantize_model_int8),
        ('learned_int4_ptq', 'learned', quantize_model_int4),
        ('qat_int4',      'qat_int4',  None),
        ('gabor_dct_bank', 'gabor_dct', None),
    ]

    all_results = []
    print(f"\n{'config':<22} {'probe_acc':>10} {'center_Δacc':>12} "
          f"{'enc_params':>11} {'dec_params':>11}")
    print("-" * 70)

    for name, enc_type, quant_fn in configs:
        r = train_and_evaluate(name, enc_type, train_x, train_y, test_x, test_y,
                               device, quantize_fn=quant_fn, seed=args.seed,
                               n_samples=args.eval_samples)
        all_results.append(r)
        print(f"{name:<22} {r['probe_acc']:>10.1%} {r['delta_acc']:>+12.1%} "
              f"{r['enc_params']:>11,} {r['dec_params']:>11,}")

    print("\n" + "=" * 80)
    print("PHASE 8 VERDICT")
    print("=" * 80)

    fp32 = [r for r in all_results if r['config'] == 'learned_fp32'][0]

    for name in ['learned_int8', 'learned_int4_ptq', 'qat_int4', 'gabor_dct_bank']:
        r = [x for x in all_results if x['config'] == name][0]
        probe_drop = fp32['probe_acc'] - r['probe_acc']
        dacc_drop = fp32['delta_acc'] - r['delta_acc']
        viable = abs(probe_drop) < 0.05 and abs(dacc_drop) < 0.10
        param_ratio = r['enc_params'] / fp32['enc_params']
        print(f"  {name:<22}: probe_drop={probe_drop:+.1%}, Δacc_drop={dacc_drop:+.1%}, "
              f"enc_params={param_ratio:.1%}  → {'VIABLE' if viable else 'DEGRADED'}")

    csv_path = os.path.join(args.output_dir, "phase8_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(set().union(*(r.keys() for r in all_results)))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")
    print("\n" + "=" * 100)
    print("Phase 8 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
