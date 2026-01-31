#!/usr/bin/env python3
"""
CIFAR-10 Classification Probe — Paradigm Validation
=====================================================
Classification as a PROBE ONLY — never enters world model loss.
Tests whether the discrete core protocol carries semantic information.

Experiment matrix (minimal but high-info):
  Z configs: 32×32×8 (Z1), 16×16×16 (Z2)
  Freq:      baseline denoise, +freq_full_ms
  Repair:    no repair, evidence repair
  = 2×2×2 = 8 configs

For each config:
  1. Train ADC/DAC (protocol layer)
  2. Train E_core (local energy)
  3. Train denoiser (amortized inference)
  4. Freeze everything above
  5. Train linear probe on z → labels (frozen encoder)
  6. Evaluate: probe acc, Δacc from repair, protocol metrics

Baselines: ResNet18 (small CIFAR version), tiny CNN matching encoder size.

4GB GPU constraint.

Usage:
    python3 -u benchmarks/exp_cifar10_classify.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, csv, argparse
from collections import OrderedDict
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# DCT + FREQ UTILITIES
# ============================================================================

def dct2d(x):
    B, C, H, W = x.shape
    def dct_matrix(N):
        n = torch.arange(N, dtype=x.dtype, device=x.device)
        k = n.unsqueeze(1)
        D = torch.cos(np.pi * (2*n+1) * k / (2*N))
        D[0] *= 1.0/np.sqrt(N); D[1:] *= np.sqrt(2.0/N)
        return D
    DH = dct_matrix(H); DW = dct_matrix(W)
    return torch.einsum('hH,bcHW,wW->bchw', DH, x, DW)

def idct2d(X):
    B, C, H, W = X.shape
    def dct_matrix(N):
        n = torch.arange(N, dtype=X.dtype, device=X.device)
        k = n.unsqueeze(1)
        D = torch.cos(np.pi * (2*n+1) * k / (2*N))
        D[0] *= 1.0/np.sqrt(N); D[1:] *= np.sqrt(2.0/N)
        return D
    DH = dct_matrix(H).T; DW = dct_matrix(W).T
    return torch.einsum('Hh,bchw,Ww->bcHW', DH, X, DW)

def get_freq_masks(H, W, device='cpu'):
    fy = torch.arange(H, device=device).float()
    fx = torch.arange(W, device=device).float()
    fg = fy.unsqueeze(1) + fx.unsqueeze(0)
    mf = H+W-2; t1=mf/3.0; t2=2*mf/3.0
    return (fg<=t1).float(), ((fg>t1)&(fg<=t2)).float(), (fg>t2).float()

def decompose_bands(x):
    B,C,H,W = x.shape
    lm,mm,hm = get_freq_masks(H,W,x.device)
    d = dct2d(x)
    return idct2d(d*lm), idct2d(d*mm), idct2d(d*hm)

def freq_scheduled_loss(x_pred, x_target, progress):
    pl,pm,ph = decompose_bands(x_pred)
    tl,tm,th = decompose_bands(x_target)
    wl=3.0; wm=min(1.0,progress*2); wh=0.5*max(0,(progress-0.3)/0.7)
    return wl*F.mse_loss(pl,tl)+wm*F.mse_loss(pm,tm)+wh*F.mse_loss(ph,th)

def hf_coherence_loss(pred_h, tgt_h):
    B,C,H,W = pred_h.shape
    def ac(x):
        xf=x.reshape(B*C,1,H,W)
        le=F.avg_pool2d(xf**2,3,1,1); lm_=F.avg_pool2d(xf,3,1,1)
        xs=F.pad(xf[:,:,:,:-1],(1,0))
        lc=F.avg_pool2d(xf*xs,3,1,1)
        v=(le-lm_**2).clamp(min=1e-8)
        cv=lc-lm_*F.avg_pool2d(xs,3,1,1)
        return (cv/v).reshape(B,C,H,W)
    return F.mse_loss(ac(pred_h), ac(tgt_h))


# ============================================================================
# GUMBEL SIGMOID
# ============================================================================

class GumbelSigmoid(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__(); self.temperature=temperature
    def forward(self, logits):
        if self.training:
            u=torch.rand_like(logits).clamp(1e-8,1-1e-8)
            noisy=(logits-torch.log(-torch.log(u)))/self.temperature
        else: noisy=logits/self.temperature
        soft=torch.sigmoid(noisy); hard=(soft>0.5).float()
        return hard-soft.detach()+soft
    def set_temperature(self,tau): self.temperature=tau


# ============================================================================
# ENCODER / DECODER (resolution-adaptive)
# ============================================================================

class Encoder16(nn.Module):
    """32×32×3 → 16×16×k via stride-2."""
    def __init__(self, n_bits):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, n_bits, 3, padding=1),
        )
        self.q = GumbelSigmoid()
    def forward(self, x):
        logits = self.net(x); return self.q(logits), logits
    def set_temperature(self, tau): self.q.set_temperature(tau)

class Decoder16(nn.Module):
    """16×16×k → 32×32×3 via stride-2 transpose."""
    def __init__(self, n_bits):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(n_bits, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z): return self.net(z)

class Encoder32(nn.Module):
    """32×32×3 → 32×32×k (1:1 spatial)."""
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
    """32×32×k → 32×32×3 (1:1 spatial)."""
    def __init__(self, n_bits):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z): return self.net(z)

def make_encoder_decoder(z_h, n_bits, device):
    if z_h == 16:
        return Encoder16(n_bits).to(device), Decoder16(n_bits).to(device)
    elif z_h == 32:
        return Encoder32(n_bits).to(device), Decoder32(n_bits).to(device)
    raise ValueError(f"Unsupported z_h={z_h}")


# ============================================================================
# E_CORE
# ============================================================================

class LocalEnergyCore(nn.Module):
    def __init__(self, n_bits, hidden=64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(9*n_bits-1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def get_context(self, z, bi, i, j):
        B,K,H,W=z.shape; ctx=[]
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                ni,nj=(i+di)%H,(j+dj)%W
                for b in range(K):
                    if di==0 and dj==0 and b==bi: continue
                    ctx.append(z[:,b,ni,nj])
        return torch.stack(ctx,dim=1)
    def violation_rate(self, z, n_samples=50):
        B,K,H,W=z.shape; v=[]
        for _ in range(min(n_samples,H*W*K)):
            b=torch.randint(K,(1,)).item(); i=torch.randint(H,(1,)).item(); j=torch.randint(W,(1,)).item()
            ctx=self.get_context(z,b,i,j)
            pred=(self.predictor(ctx).squeeze(1)>0).float()
            v.append((pred!=z[:,b,i,j]).float().mean().item())
        return np.mean(v)


# ============================================================================
# DENOISER
# ============================================================================

class FreqDenoiser(nn.Module):
    def __init__(self, n_bits):
        super().__init__(); self.n_bits=n_bits
        hid = min(128, max(64, n_bits * 4))
        self.net = nn.Sequential(
            nn.Conv2d(n_bits+1, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, n_bits, 3, padding=1))
        self.skip = nn.Conv2d(n_bits, n_bits, 1)

    def forward(self, z_noisy, noise_level):
        B=z_noisy.shape[0]
        nl=noise_level.view(B,1,1,1).expand(-1,1,z_noisy.shape[2],z_noisy.shape[3])
        return self.net(torch.cat([z_noisy,nl],dim=1)) + self.skip(z_noisy)

    @torch.no_grad()
    def repair(self, z, mask, n_steps=5, temperature=0.5):
        """Repair masked positions using iterative denoising.
        mask: 1=keep (evidence), 0=repair (missing)."""
        B,K,H,W = z.shape
        z_rep = z.clone()
        for step in range(n_steps):
            nl = torch.tensor([1.0 - step/n_steps], device=z.device).expand(B)
            logits = self(z_rep, nl)
            probs = torch.sigmoid(logits / temperature)
            z_new = (torch.rand_like(z_rep) < probs).float()
            # Only update masked (missing) positions
            z_rep = mask * z + (1-mask) * z_new
        # Final step
        logits = self(z_rep, torch.zeros(B, device=z.device))
        z_final = (torch.sigmoid(logits) > 0.5).float()
        return mask * z + (1-mask) * z_final


# ============================================================================
# CLASSIFICATION PROBES
# ============================================================================

class LinearProbe(nn.Module):
    """Flatten z and classify with 1 linear layer."""
    def __init__(self, z_dim, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(z_dim, n_classes)
    def forward(self, z):
        return self.fc(z.reshape(z.shape[0], -1))


class ConvProbe(nn.Module):
    """Small conv head preserving spatial structure."""
    def __init__(self, n_bits, z_h, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32*4*4, n_classes),
        )
    def forward(self, z):
        return self.net(z)


# ============================================================================
# BASELINES
# ============================================================================

class TinyCNN(nn.Module):
    """Small CNN baseline matching ~encoder param count."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128*4*4, n_classes),
        )
    def forward(self, x): return self.net(x)


class ResNet18Cifar(nn.Module):
    """ResNet18 adapted for CIFAR-10 (no pretrained, small input)."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.fc = nn.Linear(512, n_classes)

    def _make_layer(self, in_c, out_c, n_blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_c, out_c, stride))
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def train_adc(encoder, decoder, train_x, device, epochs=40, bs=256):
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
    with torch.no_grad():
        tb = train_x[:32].to(device); z, _ = encoder(tb)
        mse = F.mse_loss(decoder(z), tb).item()
    return mse


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
                b=torch.randint(K,(1,)).item(); ii=torch.randint(H,(1,)).item(); jj=torch.randint(W,(1,)).item()
                ctx=e_core.get_context(z,b,ii,jj)
                tl_+=F.binary_cross_entropy_with_logits(e_core.predictor(ctx).squeeze(1),z[:,b,ii,jj])
            (tl_/20).backward(); eopt.step()
    e_core.eval()


def train_denoiser(denoiser, decoder, z_data, device, epochs=30, bs=256, use_freq=False):
    dopt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)
    for epoch in tqdm(range(epochs), desc="Denoiser"):
        denoiser.train(); perm = torch.randperm(N)
        tl, nb = 0., 0
        progress = epoch / max(epochs-1, 1)
        for i in range(0, N, bs):
            z_clean = z_data[perm[i:i+bs]].to(device); B_ = z_clean.shape[0]
            noise_level = torch.rand(B_, device=device)
            flip = (torch.rand_like(z_clean) < noise_level.view(B_,1,1,1)).float()
            z_noisy = z_clean*(1-flip) + (1-z_clean)*flip
            dopt.zero_grad()
            logits = denoiser(z_noisy, noise_level)
            loss = F.binary_cross_entropy_with_logits(logits, z_clean)

            if use_freq:
                s = torch.sigmoid(logits); h = (s>0.5).float()
                z_pred = h - s.detach() + s
                with torch.no_grad(): x_clean = decoder(z_clean)
                x_pred = decoder(z_pred)
                loss_freq = freq_scheduled_loss(x_pred, x_clean, progress)
                _,_,pred_h = decompose_bands(x_pred); _,_,tgt_h = decompose_bands(x_clean)
                loss_coh = hf_coherence_loss(pred_h, tgt_h)
                loss = loss + 0.3*loss_freq + 0.1*loss_coh

            loss.backward(); dopt.step(); tl+=loss.item(); nb+=1
    print(f"      Denoiser done: loss={tl/nb:.4f}")
    denoiser.eval()


def train_probe(probe, z_data, labels, device, epochs=50, bs=256, lr=1e-3):
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for epoch in tqdm(range(epochs), desc="Probe", leave=False):
        probe.train(); perm = torch.randperm(len(z_data))
        tl, nc, nb = 0., 0, 0
        for i in range(0, len(z_data), bs):
            idx = perm[i:i+bs]
            z = z_data[idx].to(device); y = labels[idx].to(device)
            opt.zero_grad(); logits = probe(z)
            loss = F.cross_entropy(logits, y)
            loss.backward(); opt.step()
            tl += loss.item(); nc += (logits.argmax(1)==y).sum().item(); nb += len(y)
    return nc / nb


def eval_probe(probe, z_data, labels, device, bs=256):
    probe.eval()
    nc, nb = 0, 0
    with torch.no_grad():
        for i in range(0, len(z_data), bs):
            z = z_data[i:i+bs].to(device); y = labels[i:i+bs].to(device)
            logits = probe(z)
            nc += (logits.argmax(1)==y).sum().item(); nb += len(y)
    return nc / nb


def train_baseline(model, train_x, train_y, test_x, test_y, device, epochs=30, bs=256):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in tqdm(range(epochs), desc="Baseline"):
        model.train(); perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), bs):
            idx = perm[i:i+bs]
            x = train_x[idx].to(device); y = train_y[idx].to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward(); opt.step()
    model.eval()
    nc = 0
    with torch.no_grad():
        for i in range(0, len(test_x), bs):
            x = test_x[i:i+bs].to(device); y = test_y[i:i+bs].to(device)
            nc += (model(x).argmax(1)==y).sum().item()
    return nc / len(test_y)


# ============================================================================
# EVIDENCE REPAIR MASK
# ============================================================================

def make_evidence_mask(z, decoder, encoder, x_orig, device, threshold=1.0):
    """Create repair mask based on E_obs residual.
    Returns mask: 1=keep (good evidence), 0=repair (poor evidence)."""
    B, K, H, W = z.shape
    with torch.no_grad():
        x_recon = decoder(z.to(device))
        # Per-pixel residual
        residual = (x_recon - x_orig.to(device)).pow(2).mean(dim=1)  # B×H_img×W_img

        # Downsample residual to z resolution
        res_z = F.adaptive_avg_pool2d(residual.unsqueeze(1), (H, W)).squeeze(1)  # B×H×W

        # Threshold: high residual = poor evidence = should repair
        mask = (res_z < threshold).float()  # 1=keep, 0=repair
        # Expand to all bit channels
        mask = mask.unsqueeze(1).expand(-1, K, -1, -1)
    return mask


def make_center_mask(z):
    """Center occlusion mask for testing repair."""
    B, K, H, W = z.shape
    mask = torch.ones(B, K, H, W)
    h4, w4 = H//4, W//4
    mask[:, :, h4:3*h4, w4:3*w4] = 0
    return mask


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_cifar10_classify')
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("CIFAR-10 CLASSIFICATION PROBE — PARADIGM VALIDATION")
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
    print(f"    Labels: {train_y.shape}, {test_y.shape}")

    # ========================================================================
    # Phase C0: Baselines
    # ========================================================================
    print("\n" + "=" * 100)
    print("PHASE C0: SUPERVISED BASELINES (upper bounds)")
    print("=" * 100)

    # TinyCNN baseline
    print("\n  [C0.1] TinyCNN baseline...")
    tiny = TinyCNN().to(device)
    tiny_acc = train_baseline(tiny, train_x, train_y, test_x, test_y, device, epochs=30)
    tiny_params = sum(p.numel() for p in tiny.parameters())
    print(f"    TinyCNN: acc={tiny_acc:.3f}, params={tiny_params:,}")
    del tiny; torch.cuda.empty_cache()

    # ResNet18 baseline
    print("\n  [C0.2] ResNet18-CIFAR baseline...")
    rn18 = ResNet18Cifar().to(device)
    rn18_acc = train_baseline(rn18, train_x, train_y, test_x, test_y, device, epochs=30)
    rn18_params = sum(p.numel() for p in rn18.parameters())
    print(f"    ResNet18: acc={rn18_acc:.3f}, params={rn18_params:,}")
    del rn18; torch.cuda.empty_cache()

    # ========================================================================
    # Phase C1-C2: RouteC protocol + probe
    # ========================================================================
    Z_CONFIGS = OrderedDict([
        ("Z1_32x32x8",  {"z_h": 32, "n_bits": 8}),
        ("Z2_16x16x16", {"z_h": 16, "n_bits": 16}),
    ])
    FREQ_MODES = [False, True]  # baseline, +freq_full_ms
    REPAIR_MODES = [False, True]  # no repair, evidence repair

    all_results = []

    for z_name, z_cfg in Z_CONFIGS.items():
        z_h = z_cfg['z_h']; n_bits = z_cfg['n_bits']
        total_bits = z_h * z_h * n_bits
        z_dim = total_bits

        for use_freq in FREQ_MODES:
            freq_tag = "freq" if use_freq else "base"
            config_name = f"{z_name}_{freq_tag}"

            print(f"\n{'='*100}")
            print(f"CONFIG: {config_name} | z={z_h}×{z_h}×{n_bits} = {total_bits} bits | freq={use_freq}")
            print("=" * 100)

            torch.manual_seed(args.seed)

            # Build models
            encoder, decoder = make_encoder_decoder(z_h, n_bits, device)
            denoiser = FreqDenoiser(n_bits).to(device)
            e_core = LocalEnergyCore(n_bits).to(device)

            enc_p = sum(p.numel() for p in encoder.parameters())
            dec_p = sum(p.numel() for p in decoder.parameters())
            print(f"    Params: enc={enc_p:,} dec={dec_p:,}")

            # Train ADC/DAC
            print("  [A] Training ADC/DAC...")
            oracle_mse = train_adc(encoder, decoder, train_x, device, epochs=40, bs=64)
            print(f"    Oracle MSE: {oracle_mse:.4f}")

            # Encode datasets
            print("  [B] Encoding datasets...")
            z_train = encode_dataset(encoder, train_x, device)
            z_test = encode_dataset(encoder, test_x, device)
            print(f"    z_train: {z_train.shape}, usage={z_train.mean():.3f}")

            # Train E_core
            print("  [C] Training E_core...")
            train_ecore(e_core, z_train, device, epochs=10)
            viol = e_core.violation_rate(z_test[:100].to(device))
            print(f"    Violation: {viol:.4f}")

            # Train denoiser
            print("  [D] Training denoiser...")
            train_denoiser(denoiser, decoder, z_train, device,
                          epochs=30, bs=64, use_freq=use_freq)

            # Cycle metric
            with torch.no_grad():
                zc = z_test[:100].to(device)
                xc = decoder(zc); zcy, _ = encoder(xc)
                cycle = (zc != zcy).float().mean().item()
            print(f"    Cycle: {cycle:.4f}")

            # ---- PROBES ----
            for use_repair in REPAIR_MODES:
                repair_tag = "repair" if use_repair else "norepair"
                full_name = f"{config_name}_{repair_tag}"
                print(f"\n    --- PROBE: {full_name} ---")

                if use_repair:
                    # Apply center mask repair to test set
                    print("      Applying evidence repair...")
                    z_test_input = []
                    for i in range(0, len(z_test), 64):
                        z_batch = z_test[i:i+64]
                        x_batch = test_x[i:i+64]
                        # Center mask
                        mask = make_center_mask(z_batch).to(device)
                        z_masked = z_batch.to(device) * mask
                        z_repaired = denoiser.repair(z_masked, mask)
                        z_test_input.append(z_repaired.cpu())
                    z_test_eval = torch.cat(z_test_input)
                    # Also check if repair helps on clean (full evidence)
                    z_train_eval = z_train  # Don't repair training set
                else:
                    z_test_eval = z_test
                    z_train_eval = z_train

                # Linear probe
                probe_lin = LinearProbe(z_dim).to(device)
                train_probe(probe_lin, z_train_eval, train_y, device, epochs=50)
                acc_lin = eval_probe(probe_lin, z_test_eval, test_y, device)
                print(f"      Linear probe: {acc_lin:.3f}")

                # Conv probe
                probe_conv = ConvProbe(n_bits, z_h).to(device)
                train_probe(probe_conv, z_train_eval, train_y, device, epochs=50)
                acc_conv = eval_probe(probe_conv, z_test_eval, test_y, device)
                print(f"      Conv probe:   {acc_conv:.3f}")

                r = {
                    'config': full_name,
                    'z_config': z_name, 'z_h': z_h, 'n_bits': n_bits,
                    'total_bits': total_bits, 'use_freq': use_freq,
                    'use_repair': use_repair,
                    'oracle_mse': oracle_mse, 'violation': viol, 'cycle': cycle,
                    'acc_linear': acc_lin, 'acc_conv': acc_conv,
                    'enc_params': enc_p, 'dec_params': dec_p,
                }
                all_results.append(r)

                del probe_lin, probe_conv
                torch.cuda.empty_cache()

            del encoder, decoder, denoiser, e_core, z_train, z_test
            torch.cuda.empty_cache()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 100)
    print("CIFAR-10 CLASSIFICATION PROBE — SUMMARY")
    print("=" * 100)
    print(f"\n  Baselines: TinyCNN={tiny_acc:.3f}  ResNet18={rn18_acc:.3f}")
    print()

    h = f"{'config':<35} {'bits':>6} {'freq':>5} {'repair':>7} {'mse':>6} {'viol':>6} {'cycle':>6} {'lin':>6} {'conv':>6}"
    print(h); print("-"*len(h))
    for r in all_results:
        print(f"{r['config']:<35} {r['total_bits']:>6} "
              f"{'Y' if r['use_freq'] else 'N':>5} "
              f"{'Y' if r['use_repair'] else 'N':>7} "
              f"{r['oracle_mse']:>6.4f} {r['violation']:>6.4f} "
              f"{r['cycle']:>6.4f} {r['acc_linear']:>6.3f} {r['acc_conv']:>6.3f}")

    # Repair Δacc
    print(f"\n  REPAIR EFFECT (Δacc = repair - norepair):")
    for z_name in Z_CONFIGS:
        for freq in [False, True]:
            ft = "freq" if freq else "base"
            nr = [r for r in all_results if r['z_config']==z_name and r['use_freq']==freq and not r['use_repair']]
            yr = [r for r in all_results if r['z_config']==z_name and r['use_freq']==freq and r['use_repair']]
            if nr and yr:
                d_lin = yr[0]['acc_linear'] - nr[0]['acc_linear']
                d_conv = yr[0]['acc_conv'] - nr[0]['acc_conv']
                print(f"    {z_name}_{ft}: Δlin={d_lin:+.3f}  Δconv={d_conv:+.3f}")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "classify_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(all_results[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\n  CSV: {csv_path}")

    # Paradigm conclusions
    print(f"\n{'='*100}")
    print("PARADIGM DIAGNOSIS:")
    best_conv = max(r['acc_conv'] for r in all_results)
    print(f"  Best conv probe: {best_conv:.3f} (vs TinyCNN={tiny_acc:.3f}, ResNet18={rn18_acc:.3f})")
    ratio = best_conv / rn18_acc if rn18_acc > 0 else 0
    print(f"  Protocol retention: {ratio:.1%} of ResNet18 performance")

    any_repair_gain = any(
        r['use_repair'] and r['acc_conv'] >
        next((nr['acc_conv'] for nr in all_results
              if nr['z_config']==r['z_config'] and nr['use_freq']==r['use_freq']
              and not nr['use_repair']), 0)
        for r in all_results if r['use_repair']
    )
    print(f"  Repair provides classification gain: {'YES' if any_repair_gain else 'NO'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
