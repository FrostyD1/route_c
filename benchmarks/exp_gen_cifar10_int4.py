#!/usr/bin/env python3
"""
CIFAR-10 Generation with INT4 Tokens (16 levels per position)
==============================================================
Key change: z ∈ {0,...,15}^{k×H×W} instead of {0,1}^{k×H×W}

Each position carries 4 bits of amplitude information.
Binary 32×32×16 = 16384 bits → INT4 32×32×8 = 32768 effective bits (same channels, 4× info per channel)

Configs:
  1. int4_32x32x4  = 16384 eff bits (matches binary 32×32×16)
  2. int4_32x32x8  = 32768 eff bits (double, recommended)
  3. int4_32x32x16 = 65536 eff bits (max)

Denoiser: categorical diffusion — noise = random token replacement,
          denoise = predict categorical distribution over 16 levels.

4GB GPU constraint, BS=256, tqdm.

Usage:
    python3 -u benchmarks/exp_gen_cifar10_int4.py --device cuda
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

def per_band_energy_distance(ig, ir, device='cpu'):
    def t(x):
        if isinstance(x,np.ndarray): x=torch.tensor(x,dtype=torch.float32)
        if x.dim()==3: x=x.unsqueeze(1)
        return x.to(device)
    g=t(ig[:200]); r=t(ir[:200])
    dg=dct2d(g); dr=dct2d(r); H,W=g.shape[2],g.shape[3]
    res={}
    for nm,m in zip(['low','mid','high'], get_freq_masks(H,W,device)):
        eg=(dg**2*m).mean(dim=(0,1)).sum().item()
        er=(dr**2*m).mean(dim=(0,1)).sum().item()
        res[f'energy_gap_{nm}']=abs(eg-er)/(er+1e-12)
    return res

def hf_coherence_metric(images, device='cpu'):
    if isinstance(images,np.ndarray): images=torch.tensor(images,dtype=torch.float32)
    if images.dim()==3: images=images.unsqueeze(1)
    images=images[:200].to(device)
    _,_,xh=decompose_bands(images)
    xf=xh.reshape(-1,xh.shape[2],xh.shape[3])
    xm=xf.mean(dim=(1,2),keepdim=True); xs=xf.std(dim=(1,2),keepdim=True).clamp(min=1e-8)
    xn=(xf-xm)/xs
    corrs=[]
    for dy,dx in [(0,1),(1,0),(1,1)]:
        a,b=xn,xn
        if dy>0: a,b=a[:,dy:,:],b[:,:-dy,:]
        if dx>0: a,b=a[:,:,dx:],b[:,:,:-dx]
        corrs.append((a*b).mean(dim=(1,2)).mean().item())
    return np.mean(corrs)

def hf_noise_index(images, device='cpu'):
    if isinstance(images,np.ndarray): images=torch.tensor(images,dtype=torch.float32)
    if images.dim()==3: images=images.unsqueeze(1)
    images=images[:200].to(device)
    gx=images[:,:,:,1:]-images[:,:,:,:-1]; gy=images[:,:,1:,:]-images[:,:,:-1,:]
    ge=(gx**2).mean().item()+(gy**2).mean().item()
    _,_,xh=decompose_bands(images)
    return ge/((xh**2).mean().item()+1e-12)

def connectedness_proxy(images, threshold=0.3):
    if isinstance(images,torch.Tensor): images=images.cpu().numpy()
    if images.ndim==4 and images.shape[1]==3:
        images=0.299*images[:,0]+0.587*images[:,1]+0.114*images[:,2]
    elif images.ndim==4: images=images[:,0]
    scores=[]
    for img in images[:100]:
        b=(img>threshold).astype(np.int32); tf=b.sum()
        if tf<5: scores.append(0.0); continue
        H,W=b.shape; vis=np.zeros_like(b,dtype=bool); mc=0
        for i in range(H):
            for j in range(W):
                if b[i,j]==1 and not vis[i,j]:
                    st=[(i,j)]; vis[i,j]=True; sz=0
                    while st:
                        ci,cj=st.pop(); sz+=1
                        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ni,nj=ci+di,cj+dj
                            if 0<=ni<H and 0<=nj<W and b[ni,nj]==1 and not vis[ni,nj]:
                                vis[ni,nj]=True; st.append((ni,nj))
                    mc=max(mc,sz)
        scores.append(mc/tf)
    return float(np.mean(scores))

def compute_diversity_int4(z, n_pairs=500):
    """Diversity for INT4 tokens: fraction of differing positions."""
    N=len(z); zf=z.reshape(N,-1); d=[]
    for _ in range(n_pairs):
        i,j=np.random.choice(N,2,replace=False)
        d.append((zf[i]!=zf[j]).float().mean().item())
    return np.mean(d)

def compute_1nn(gn, tn, nc=100):
    gf=gn[:nc].reshape(nc,-1); tf=tn.reshape(len(tn),-1); d=[]
    for i in range(nc): d.append(((tf-gf[i:i+1])**2).sum(1).min().item())
    return np.mean(d)

def save_grid(images, path, nrow=8):
    try:
        from torchvision.utils import save_image
        if isinstance(images,np.ndarray): images=torch.tensor(images)
        if images.dim()==3: images=images.unsqueeze(1)
        save_image(images[:64],path,nrow=nrow,normalize=False)
        print(f"    Grid: {path}")
    except Exception as e:
        print(f"    Grid fail: {e}")

def vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


# ============================================================================
# INT4 QUANTIZER (16 levels, STE + EMA scale calibration)
# ============================================================================

class INT4Quantizer(nn.Module):
    """Quantize continuous values to 16 levels {0,...,15}.
    Uses learned scale with EMA calibration + STE for gradients.
    Outputs integer tokens AND normalized float representation."""

    def __init__(self, n_levels=16):
        super().__init__()
        self.n_levels = n_levels
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('bias', torch.tensor(0.0))
        self.register_buffer('ema_min', torch.tensor(0.0))
        self.register_buffer('ema_max', torch.tensor(1.0))
        self.ema_decay = 0.99

    @torch.no_grad()
    def update_scale(self, x):
        """EMA calibration of quantization range."""
        xmin = x.min().item()
        xmax = x.max().item()
        self.ema_min.mul_(self.ema_decay).add_((1-self.ema_decay) * xmin)
        self.ema_max.mul_(self.ema_decay).add_((1-self.ema_decay) * xmax)
        rng = (self.ema_max - self.ema_min).clamp(min=1e-6)
        self.scale.copy_(rng / (self.n_levels - 1))
        self.bias.copy_(self.ema_min)

    def forward(self, x):
        """Returns (tokens_int, tokens_float_normalized).
        tokens_int: LongTensor {0,...,15}
        tokens_float: FloatTensor [0,1] (STE-enabled for backward)"""
        if self.training:
            self.update_scale(x)

        # Normalize to [0, n_levels-1]
        x_norm = (x - self.bias) / self.scale.clamp(min=1e-6)
        x_clamp = x_norm.clamp(0, self.n_levels - 1)

        # Quantize
        x_round = x_clamp.round()
        tokens_int = x_round.long()

        # STE: forward uses rounded, backward uses continuous
        x_ste = x_round - x_clamp.detach() + x_clamp

        # Normalize to [0, 1] for decoder input
        tokens_float = x_ste / (self.n_levels - 1)

        return tokens_int, tokens_float

    def tokens_to_float(self, tokens_int):
        """Convert integer tokens back to normalized [0,1] float."""
        return tokens_int.float() / (self.n_levels - 1)


# ============================================================================
# ENCODER / DECODER for INT4
# ============================================================================

class Encoder32INT4(nn.Module):
    """32×32×3 → 32×32×k continuous → INT4 quantize → tokens."""
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, n_channels, 3, padding=1),
        )
        self.quantizer = INT4Quantizer(n_levels=16)

    def forward(self, x):
        logits = self.net(x)
        tokens_int, tokens_float = self.quantizer(logits)
        return tokens_int, tokens_float, logits


class Decoder32INT4(nn.Module):
    """32×32×k float [0,1] → 32×32×3 image."""
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z_float):
        return self.net(z_float)


# ============================================================================
# CATEGORICAL DENOISER (predicts distribution over 16 levels)
# ============================================================================

class CategoricalDenoiser(nn.Module):
    """Denoiser for INT4 tokens. Predicts 16-way categorical per position.
    Input: token embeddings + noise level.
    Output: logits over 16 levels per channel per position."""

    def __init__(self, n_channels, n_levels=16):
        super().__init__()
        self.n_channels = n_channels
        self.n_levels = n_levels

        # Direct float input: normalize tokens to [0,1], no embedding overhead
        # Input: n_channels (token float) + 1 (noise level)
        in_ch = n_channels + 1
        hid = 128

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, n_channels * n_levels, 3, padding=1),
        )

        # Skip connection
        self.skip = nn.Conv2d(in_ch, n_channels * n_levels, 1)

    def forward(self, z_int, noise_level):
        """
        z_int: LongTensor [B, K, H, W] with values in {0,...,15}
        noise_level: FloatTensor [B]
        Returns: logits [B, K, 16, H, W]
        """
        B, K, H, W = z_int.shape

        # Direct float input: normalize to [0,1]
        z_float = z_int.float() / (self.n_levels - 1)  # [B, K, H, W]

        # Add noise level channel
        nl = noise_level.view(B, 1, 1, 1).expand(-1, 1, H, W)
        inp = torch.cat([z_float, nl], dim=1)

        # Predict
        out = self.net(inp) + self.skip(inp)

        # Reshape to [B, K, n_levels, H, W]
        out = out.reshape(B, K, self.n_levels, H, W)
        return out

    @torch.no_grad()
    def sample(self, n, K, H, W, device, n_steps=20, temperature=0.8):
        """Categorical denoising sampling."""
        # Start from uniform random tokens
        z = torch.randint(0, self.n_levels, (n, K, H, W), device=device)

        for step in range(n_steps):
            progress = step / n_steps
            nl = torch.tensor([1.0 - progress], device=device).expand(n)

            logits = self(z, nl)  # [B, K, 16, H, W]

            # Temperature-scaled sampling
            if progress < 0.3:
                t = temperature * 0.5  # More random early
                conf = 0.2 + 0.3 * (progress / 0.3)
            elif progress < 0.7:
                t = temperature
                conf = 0.5 + 0.3 * ((progress - 0.3) / 0.4)
            else:
                t = temperature * 1.5  # Sharper late
                conf = 0.8 + 0.2 * ((progress - 0.7) / 0.3)

            # Sample from categorical
            probs = F.softmax(logits / t, dim=2)  # [B, K, 16, H, W]
            probs_flat = probs.permute(0, 1, 3, 4, 2).reshape(-1, self.n_levels)
            sampled = torch.multinomial(probs_flat, 1).reshape(B if 'B' in dir() else n, K, H, W)

            # Confidence mask: only update fraction of positions
            mask = (torch.rand(n, K, H, W, device=device) < conf).long()
            z = mask * sampled + (1 - mask) * z

        # Final greedy step
        logits = self(z, torch.zeros(n, device=device))
        z_final = logits.argmax(dim=2)
        return z_final


# ============================================================================
# E_CORE for INT4
# ============================================================================

class LocalEnergyCoreINT4(nn.Module):
    """E_core for multi-value tokens. Predicts token value from context."""
    def __init__(self, n_channels, n_levels=16, hidden=128):
        super().__init__()
        self.n_levels = n_levels
        self.n_channels = n_channels
        # Context: 3×3 neighborhood × n_channels values (excluding target) = 9*n_ch - 1
        ctx_dim = 9 * n_channels - 1
        self.predictor = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_levels))

    def get_context(self, z_float, bi, i, j):
        """Get 3×3 neighborhood context as float values."""
        B, K, H, W = z_float.shape
        ctx = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = (i+di) % H, (j+dj) % W
                for b in range(K):
                    if di == 0 and dj == 0 and b == bi:
                        continue
                    ctx.append(z_float[:, b, ni, nj])
        return torch.stack(ctx, dim=1)

    def violation_rate(self, z_int, n_samples=50):
        """Fraction of positions where predicted token != actual."""
        B, K, H, W = z_int.shape
        z_float = z_int.float() / (self.n_levels - 1)
        v = []
        for _ in range(min(n_samples, H*W*K)):
            b = torch.randint(K, (1,)).item()
            i = torch.randint(H, (1,)).item()
            j = torch.randint(W, (1,)).item()
            ctx = self.get_context(z_float, b, i, j)
            pred = self.predictor(ctx).argmax(dim=1)
            v.append((pred != z_int[:, b, i, j]).float().mean().item())
        return np.mean(v)


# ============================================================================
# TRAINING
# ============================================================================

def train_pipeline(encoder, decoder, denoiser, e_core, train_x, device,
                   n_channels, adc_epochs=40, den_epochs=40, bs=256):
    """Full training pipeline for INT4 generation."""

    # [A] Train ADC/DAC
    print("  [A] Training ADC/DAC (INT4)...")
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    for epoch in tqdm(range(adc_epochs), desc="ADC"):
        encoder.train(); decoder.train()
        perm = torch.randperm(len(train_x)); tl, nb = 0., 0
        for i in range(0, len(train_x), bs):
            x = train_x[perm[i:i+bs]].to(device)
            opt.zero_grad()
            tokens_int, tokens_float, logits = encoder(x)
            xh = decoder(tokens_float)
            loss_recon = F.mse_loss(xh, x) + 0.5 * F.binary_cross_entropy(xh, x)
            # Boundary margin loss: push away from rounding boundaries
            frac = logits - logits.floor()
            loss_margin = -(frac * (1 - frac)).mean() * 0.1
            loss = loss_recon + loss_margin
            loss.backward(); opt.step()
            tl += loss_recon.item(); nb += 1
    print(f"    ADC done: loss={tl/nb:.4f} VRAM={vram_mb():.0f}MB")
    encoder.eval(); decoder.eval()

    # Oracle recon
    with torch.no_grad():
        tb = train_x[:64].to(device)
        _, tf, _ = encoder(tb)
        oracle_mse = F.mse_loss(decoder(tf), tb).item()
    print(f"    Oracle MSE: {oracle_mse:.4f}")

    # [B] Encode training set
    print("  [B] Encoding training set...")
    z_int_data = []
    with torch.no_grad():
        for i in range(0, len(train_x), bs):
            ti, _, _ = encoder(train_x[i:i+bs].to(device))
            z_int_data.append(ti.cpu())
    z_int_data = torch.cat(z_int_data)
    K, H, W = z_int_data.shape[1:]
    print(f"    z_data: {z_int_data.shape}, token range=[{z_int_data.min()},{z_int_data.max()}], "
          f"mean={z_int_data.float().mean():.2f}")

    # [C] Train E_core
    print("  [C] Training E_core...")
    eopt = torch.optim.Adam(e_core.parameters(), lr=1e-3)
    for ep in tqdm(range(15), desc="E_core"):
        e_core.train(); perm = torch.randperm(len(z_int_data))
        for i in range(0, len(z_int_data), bs):
            z = z_int_data[perm[i:i+bs]].to(device)
            z_float = z.float() / 15.0
            eopt.zero_grad(); tl_ = 0.
            for _ in range(20):
                b = torch.randint(K, (1,)).item()
                ii = torch.randint(H, (1,)).item()
                jj = torch.randint(W, (1,)).item()
                ctx = e_core.get_context(z_float, b, ii, jj)
                tl_ += F.cross_entropy(e_core.predictor(ctx), z[:, b, ii, jj])
            (tl_ / 20).backward(); eopt.step()
    e_core.eval()

    # [D] Train categorical denoiser
    print("  [D] Training categorical denoiser...")
    dopt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_int_data)
    for epoch in tqdm(range(den_epochs), desc="Denoiser"):
        denoiser.train(); perm = torch.randperm(N)
        tl, fl, nb = 0., 0., 0
        progress = epoch / max(den_epochs - 1, 1)
        for i in range(0, N, bs):
            z_clean = z_int_data[perm[i:i+bs]].to(device)
            B_ = z_clean.shape[0]

            # Categorical noise: replace random fraction with uniform random tokens
            noise_level = torch.rand(B_, device=device)
            noise_mask = (torch.rand(B_, K, H, W, device=device)
                         < noise_level.view(B_, 1, 1, 1)).long()
            z_noise = torch.randint(0, 16, (B_, K, H, W), device=device)
            z_noisy = noise_mask * z_noise + (1 - noise_mask) * z_clean

            dopt.zero_grad()
            logits = denoiser(z_noisy, noise_level)  # [B, K, 16, H, W]

            # Cross-entropy loss per position
            loss_ce = F.cross_entropy(
                logits.permute(0, 1, 3, 4, 2).reshape(-1, 16),
                z_clean.reshape(-1))

            loss_ce.backward(); dopt.step()
            tl += loss_ce.item(); nb += 1
    print(f"    Denoiser done: CE={tl/nb:.4f} VRAM={vram_mb():.0f}MB")
    denoiser.eval()

    return z_int_data, oracle_mse


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_gen_cifar10_int4')
    parser.add_argument('--n_samples', type=int, default=256)
    parser.add_argument('--bs', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("CIFAR-10 INT4 TOKEN GENERATION")
    print("=" * 100)

    # Load CIFAR-10
    print("\n[1] Loading CIFAR-10...")
    from torchvision import datasets, transforms
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(args.seed)
    train_x = torch.stack([train_ds[i][0] for i in rng.choice(len(train_ds), 5000, replace=False)])
    test_x = torch.stack([test_ds[i][0] for i in rng.choice(len(test_ds), 500, replace=False)])
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")

    # Reference metrics
    print("\n[2] Reference metrics...")
    real_hf_coh = hf_coherence_metric(test_x[:200], device)
    real_hf_noi = hf_noise_index(test_x[:200], device)
    real_conn = connectedness_proxy(test_x[:100].numpy())
    print(f"    HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  conn={real_conn:.4f}")
    save_grid(test_x[:64], os.path.join(args.output_dir, 'real_samples.png'))
    train_x_np = train_x.numpy().reshape(len(train_x), -1)

    # Configs
    CONFIGS = OrderedDict([
        ("int4_32x32x4",  {"n_ch": 4}),    # 16384 eff bits
        ("int4_32x32x8",  {"n_ch": 8}),    # 32768 eff bits
        ("int4_32x32x16", {"n_ch": 16}),   # 65536 eff bits
    ])

    all_results = []
    for name, cfg in CONFIGS.items():
        n_ch = cfg['n_ch']
        eff_bits = 32 * 32 * n_ch * 4  # 4 bits per INT4 position
        print(f"\n{'='*100}")
        print(f"CONFIG: {name} | z=32×32×{n_ch} INT4 = {eff_bits} effective bits")
        print("=" * 100)

        torch.manual_seed(args.seed)

        encoder = Encoder32INT4(n_ch).to(device)
        decoder = Decoder32INT4(n_ch).to(device)
        denoiser = CategoricalDenoiser(n_ch).to(device)
        e_core = LocalEnergyCoreINT4(n_ch).to(device)

        enc_p = sum(p.numel() for p in encoder.parameters())
        dec_p = sum(p.numel() for p in decoder.parameters())
        den_p = sum(p.numel() for p in denoiser.parameters())
        print(f"    Params: enc={enc_p:,} dec={dec_p:,} den={den_p:,}")

        z_int_data, oracle_mse = train_pipeline(
            encoder, decoder, denoiser, e_core, train_x, device,
            n_ch, adc_epochs=40, den_epochs=40, bs=args.bs)

        K, H, W = z_int_data.shape[1:]

        # Generate
        print(f"\n  [E] Generating {args.n_samples} samples...")
        torch.manual_seed(args.seed)
        gen_bs = 64
        z_gen_list = []
        for gi in tqdm(range(0, args.n_samples, gen_bs), desc="Sampling"):
            n = min(gen_bs, args.n_samples - gi)
            z_gen_list.append(denoiser.sample(n, K, H, W, device).cpu())
        z_gen = torch.cat(z_gen_list)

        # Decode
        with torch.no_grad():
            x_gen_list = []
            for gi in range(0, len(z_gen), gen_bs):
                z_float = z_gen[gi:gi+gen_bs].float().to(device) / 15.0
                x_gen_list.append(decoder(z_float).cpu())
            x_gen = torch.cat(x_gen_list)
        save_grid(x_gen, os.path.join(args.output_dir, f'gen_{name}.png'))

        # Evaluate
        print("  [F] Evaluating...")
        viol = e_core.violation_rate(z_gen[:100].to(device))
        div = compute_diversity_int4(z_gen)

        # Cycle
        with torch.no_grad():
            zc = z_gen[:100].to(device)
            zc_float = zc.float() / 15.0
            xc = decoder(zc_float)
            zcy, _, _ = encoder(xc)
            cycle = (zc != zcy).float().mean().item()

        x_np = x_gen.numpy().reshape(len(x_gen), -1)
        nn_d = compute_1nn(x_np, train_x_np)
        conn = connectedness_proxy(x_gen[:100])
        band = per_band_energy_distance(x_gen[:200], test_x[:200], device)
        hfc = hf_coherence_metric(x_gen[:200], device)
        hfn = hf_noise_index(x_gen[:200], device)

        r = {
            'config': name, 'n_channels': n_ch, 'eff_bits': eff_bits,
            'oracle_mse': oracle_mse, 'violation': viol, 'diversity': div,
            'cycle': cycle, 'nn_dist': nn_d, 'connectedness': conn,
            'hf_coherence': hfc, 'hf_noise_index': hfn,
            'energy_gap_low': band['energy_gap_low'],
            'energy_gap_mid': band['energy_gap_mid'],
            'energy_gap_high': band['energy_gap_high'],
            'enc_params': enc_p, 'dec_params': dec_p, 'den_params': den_p,
        }
        all_results.append(r)

        print(f"\n  RESULTS:")
        print(f"    oracle_mse={oracle_mse:.4f} viol={viol:.4f} div={div:.4f} "
              f"cycle={cycle:.4f} conn={conn:.4f}")
        print(f"    HF_coh={hfc:.4f}(real={real_hf_coh:.4f}) "
              f"HF_noise={hfn:.2f}(real={real_hf_noi:.2f})")
        print(f"    E_gap: L={band['energy_gap_low']:.4f} "
              f"M={band['energy_gap_mid']:.4f} H={band['energy_gap_high']:.4f}")

        del encoder, decoder, denoiser, e_core, z_int_data, z_gen, x_gen
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 100)
    print("INT4 TOKEN GENERATION SUMMARY")
    print(f"Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  conn={real_conn:.4f}")
    print(f"Compare: binary 32×32×16 = 16384 bits had E_gap_mid=0.299, viol=0.433")
    print("=" * 100)

    h = (f"{'config':<20} {'bits':>7} {'mse':>6} {'viol':>6} {'div':>6} "
         f"{'cycle':>6} {'HFcoh':>6} {'HFnoi':>7} {'Eg_L':>6} {'Eg_M':>6} {'Eg_H':>6}")
    print(h); print("-" * len(h))
    for r in all_results:
        print(f"{r['config']:<20} {r['eff_bits']:>7} "
              f"{r['oracle_mse']:>6.4f} {r['violation']:>6.4f} {r['diversity']:>6.4f} "
              f"{r['cycle']:>6.4f} {r['hf_coherence']:>6.4f} {r['hf_noise_index']:>7.2f} "
              f"{r['energy_gap_low']:>6.4f} {r['energy_gap_mid']:>6.4f} "
              f"{r['energy_gap_high']:>6.4f}")

    # Binary vs INT4 comparison
    print(f"\n{'='*60}")
    print("INT4 vs BINARY COMPARISON:")
    print(f"  Binary 32×32×16:  16384 bits, viol=0.433, div=0.459, Eg_M=0.299")
    for r in all_results:
        status = []
        if r['violation'] < 0.433: status.append("viol_BETTER")
        if r['energy_gap_mid'] < 0.299: status.append("mid_BETTER")
        if r['diversity'] > 0.15: status.append("diverse")
        print(f"  {r['config']}: {r['eff_bits']}bits, viol={r['violation']:.3f}, "
              f"div={r['diversity']:.3f}, Eg_M={r['energy_gap_mid']:.3f} → {' + '.join(status) if status else 'WORSE'}")
    print("=" * 60)

    csv_path = os.path.join(args.output_dir, "int4_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(all_results[0].keys())
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
