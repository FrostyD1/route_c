#!/usr/bin/env python3
"""
Stage S1: Protocol Bandwidth Sweep for CIFAR-10 Generation
===========================================================
Diagnosis: 16×16×8 = 2048 bits for 32×32×3 RGB = 24576 bits → 12× compression.
Mid/high energy gap >0.93 is a direct bandwidth bottleneck symptom.

This experiment sweeps z resolution and bits/position to find
where the bandwidth threshold is for structured high-frequency generation.

Configs (6 runs):
  1. 16×16×8  =  2048 bits (current baseline)
  2. 16×16×16 =  4096 bits (double bits, same resolution)
  3. 16×16×32 =  8192 bits (4× bits)
  4. 32×32×8  =  8192 bits (4× resolution, same bits)
  5. 32×32×16 = 16384 bits (high res + high bits)
  6. 32×32×32 = 32768 bits (max — slightly exceeds original 24576)

Each config: train ADC/DAC, train denoiser (with freq sched+coh),
generate, evaluate full metric suite.

Architecture scales with z resolution:
  - 16×16: stride-2 encoder/decoder (current)
  - 32×32: no spatial downsampling (1:1 pixel-to-token mapping)

4GB GPU constraint: batch_size adapts to model size.

Usage:
    python3 -u benchmarks/exp_gen_cifar10_bandwidth.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, csv, argparse
from collections import OrderedDict

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
        le=F.avg_pool2d(xf**2,3,1,1); lm=F.avg_pool2d(xf,3,1,1)
        xs=F.pad(xf[:,:,:,:-1],(1,0))
        lc=F.avg_pool2d(xf*xs,3,1,1)
        v=(le-lm**2).clamp(min=1e-8)
        cv=lc-lm*F.avg_pool2d(xs,3,1,1)
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

def compute_diversity(z, n_pairs=500):
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
# ENCODER/DECODER FACTORIES (resolution-adaptive)
# ============================================================================

def make_encoder_decoder(z_h, z_w, n_bits, device):
    """Create encoder/decoder pair adapted to z resolution.
    For 16×16: stride-2 from 32×32 (1 downsample)
    For 32×32: no spatial downsampling (1:1 mapping)
    """
    if z_h == 16:
        enc = Encoder16(n_bits).to(device)
        dec = Decoder16(n_bits).to(device)
    elif z_h == 32:
        enc = Encoder32(n_bits).to(device)
        dec = Decoder32(n_bits).to(device)
    else:
        raise ValueError(f"Unsupported z resolution: {z_h}")
    return enc, dec


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
        logits = self.net(x)
        return self.q(logits), logits
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
    """32×32×3 → 32×32×k (1:1 spatial, no downsampling)."""
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
        logits = self.net(x)
        return self.q(logits), logits
    def set_temperature(self, tau): self.q.set_temperature(tau)


class Decoder32(nn.Module):
    """32×32×k → 32×32×3 (1:1 spatial, no upsampling)."""
    def __init__(self, n_bits):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z): return self.net(z)


# ============================================================================
# E_CORE (adapts to z resolution)
# ============================================================================

class LocalEnergyCore(nn.Module):
    def __init__(self, n_bits):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(9*n_bits-1, 64), nn.ReLU(), nn.Linear(64, 1))
    def get_context(self, z, bi, i, j):
        B,K,H,W=z.shape; ctx=[]
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                ni,nj=(i+di)%H,(j+dj)%W
                for b in range(K):
                    if di==0 and dj==0 and b==bi: continue
                    ctx.append(z[:,b,ni,nj])
        return torch.stack(ctx,dim=1)
    def violation_rate(self, z):
        B,K,H,W=z.shape; v=[]
        for _ in range(min(50,H*W*K)):
            b=torch.randint(K,(1,)).item(); i=torch.randint(H,(1,)).item(); j=torch.randint(W,(1,)).item()
            ctx=self.get_context(z,b,i,j)
            pred=(self.predictor(ctx).squeeze(1)>0).float()
            v.append((pred!=z[:,b,i,j]).float().mean().item())
        return np.mean(v)


# ============================================================================
# DENOISER (adapts to n_bits)
# ============================================================================

class FreqDenoiser(nn.Module):
    def __init__(self, n_bits):
        super().__init__(); self.n_bits=n_bits
        # Scale hidden dim with n_bits to handle more info
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
    def sample_multiscale(self, n, H, W, device, n_steps=15, temperature=0.7):
        K=self.n_bits; z=(torch.rand(n,K,H,W,device=device)>0.5).float()
        for step in range(n_steps):
            nl=torch.tensor([1.0-step/n_steps],device=device).expand(n)
            logits=self(z,nl); p=step/n_steps
            if p<0.5: t=temperature*0.6; c=0.3+0.4*(p/0.5)
            else: t=temperature*1.2; c=0.7+0.3*((p-0.5)/0.5)
            probs=torch.sigmoid(logits/t)
            mask=(torch.rand_like(z)<c).float()
            z=mask*(torch.rand_like(z)<probs).float()+(1-mask)*z
        return (torch.sigmoid(self(z,torch.zeros(n,device=device)))>0.5).float()


def train_pipeline(encoder, decoder, denoiser, e_core, train_x, z_h,
                    n_bits, device, adc_epochs=40, den_epochs=30):
    """Train full pipeline: ADC/DAC → encode → E_core → denoiser."""
    batch_size = 16 if (z_h >= 32 and n_bits >= 16) else 32

    # [A] Train ADC/DAC
    print(f"  [A] Training ADC/DAC ({z_h}×{z_h}×{n_bits})...")
    opt = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=1e-3)
    for epoch in range(adc_epochs):
        encoder.train(); decoder.train()
        encoder.set_temperature(1.0 + (0.3-1.0)*epoch/(adc_epochs-1))
        perm = torch.randperm(len(train_x)); tl, nb = 0., 0
        for i in range(0, len(train_x), batch_size):
            idx = perm[i:i+batch_size]; x = train_x[idx].to(device)
            opt.zero_grad(); z, _ = encoder(x); xh = decoder(z)
            loss = F.mse_loss(xh, x) + 0.5*F.binary_cross_entropy(xh, x)
            loss.backward(); opt.step(); tl += loss.item(); nb += 1
        if (epoch+1) % 10 == 0:
            print(f"    ADC epoch {epoch+1}/{adc_epochs}: loss={tl/nb:.4f}")
    encoder.eval(); decoder.eval()

    # Oracle recon
    with torch.no_grad():
        tb = train_x[:32].to(device); zo, _ = encoder(tb)
        mse = F.mse_loss(decoder(zo), tb).item()
    print(f"    Oracle MSE: {mse:.4f}")

    # [B] Encode training set
    print("  [B] Encoding training set...")
    z_data = []
    with torch.no_grad():
        for i in range(0, len(train_x), batch_size):
            z, _ = encoder(train_x[i:i+batch_size].to(device)); z_data.append(z.cpu())
    z_data = torch.cat(z_data)
    K, H, W = z_data.shape[1:]
    print(f"    z_data: {z_data.shape}, bits={K*H*W}, usage={z_data.mean():.3f}")

    # [C] Train E_core
    print("  [C] Training E_core...")
    eopt = torch.optim.Adam(e_core.parameters(), lr=1e-3)
    for ep in range(10):
        e_core.train(); perm = torch.randperm(len(z_data))
        for i in range(0, len(z_data), 64):
            z = z_data[perm[i:i+64]].to(device); eopt.zero_grad(); tl_ = 0.
            for _ in range(20):
                b=torch.randint(K,(1,)).item(); ii=torch.randint(H,(1,)).item(); jj=torch.randint(W,(1,)).item()
                ctx=e_core.get_context(z,b,ii,jj)
                tl_+=F.binary_cross_entropy_with_logits(e_core.predictor(ctx).squeeze(1),z[:,b,ii,jj])
            (tl_/20).backward(); eopt.step()
    e_core.eval()

    # [D] Train denoiser with freq scheduled + coherence
    print("  [D] Training freq denoiser...")
    dopt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)
    for epoch in range(den_epochs):
        denoiser.train(); perm = torch.randperm(N)
        tl, fl, cl, nb = 0., 0., 0., 0
        progress = epoch / max(den_epochs-1, 1)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]; z_clean = z_data[idx].to(device); B = z_clean.shape[0]
            noise_level = torch.rand(B, device=device)
            flip = (torch.rand_like(z_clean) < noise_level.view(B,1,1,1)).float()
            z_noisy = z_clean*(1-flip) + (1-z_clean)*flip
            dopt.zero_grad()
            logits = denoiser(z_noisy, noise_level)
            loss_bce = F.binary_cross_entropy_with_logits(logits, z_clean)
            # Freq loss
            s = torch.sigmoid(logits); h = (s>0.5).float()
            z_pred = h - s.detach() + s
            with torch.no_grad(): x_clean = decoder(z_clean)
            x_pred = decoder(z_pred)
            loss_freq = freq_scheduled_loss(x_pred, x_clean, progress)
            _,_,pred_h = decompose_bands(x_pred); _,_,tgt_h = decompose_bands(x_clean)
            loss_coh = hf_coherence_loss(pred_h, tgt_h)
            loss = loss_bce + 0.3*loss_freq + 0.1*loss_coh
            loss.backward(); dopt.step()
            tl+=loss_bce.item(); fl+=loss_freq.item(); cl+=loss_coh.item(); nb+=1
        if (epoch+1) % 10 == 0:
            print(f"    Den epoch {epoch+1}/{den_epochs}: BCE={tl/nb:.4f} freq={fl/nb:.4f} coh={cl/nb:.4f}")
    denoiser.eval()

    return z_data, mse


# ============================================================================
# BANDWIDTH CONFIGS
# ============================================================================

CONFIGS = OrderedDict([
    ("16x16_8bit",  {"z_h": 16, "n_bits": 8,  "total_bits": 2048}),
    ("16x16_16bit", {"z_h": 16, "n_bits": 16, "total_bits": 4096}),
    ("16x16_32bit", {"z_h": 16, "n_bits": 32, "total_bits": 8192}),
    ("32x32_8bit",  {"z_h": 32, "n_bits": 8,  "total_bits": 8192}),
    ("32x32_16bit", {"z_h": 32, "n_bits": 16, "total_bits": 16384}),
    ("32x32_32bit", {"z_h": 32, "n_bits": 32, "total_bits": 32768}),
])


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_gen_cifar10_bw')
    parser.add_argument('--n_samples', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("STAGE S1: CIFAR-10 PROTOCOL BANDWIDTH SWEEP")
    print("=" * 100)

    # Load CIFAR-10
    print("\n[1] Loading CIFAR-10...")
    from torchvision import datasets, transforms
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(args.seed)
    train_x = torch.stack([train_ds[i][0] for i in rng.choice(len(train_ds), 3000, replace=False)])
    test_x = torch.stack([test_ds[i][0] for i in rng.choice(len(test_ds), 500, replace=False)])
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")

    # Reference metrics
    print("\n[2] Reference metrics...")
    real_hf_coh = hf_coherence_metric(test_x[:200], device)
    real_hf_noi = hf_noise_index(test_x[:200], device)
    real_conn = connectedness_proxy(test_x[:100].numpy())
    real_band = per_band_energy_distance(test_x[:200], test_x[:200], device)
    print(f"    HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  conn={real_conn:.4f}")

    save_grid(test_x[:64], os.path.join(args.output_dir, 'real_samples.png'))
    train_x_np = train_x.numpy().reshape(len(train_x), -1)

    # Run all configs
    all_results = []
    for name, cfg in CONFIGS.items():
        z_h = cfg['z_h']; n_bits = cfg['n_bits']; total = cfg['total_bits']
        print(f"\n{'='*100}")
        print(f"CONFIG: {name} | z={z_h}×{z_h}×{n_bits} = {total} bits "
              f"(compression: {24576/total:.1f}×)")
        print("=" * 100)

        torch.manual_seed(args.seed)

        encoder, decoder = make_encoder_decoder(z_h, z_h, n_bits, device)
        denoiser = FreqDenoiser(n_bits).to(device)
        e_core = LocalEnergyCore(n_bits).to(device)

        # Count params
        enc_p = sum(p.numel() for p in encoder.parameters())
        dec_p = sum(p.numel() for p in decoder.parameters())
        den_p = sum(p.numel() for p in denoiser.parameters())
        print(f"    Params: enc={enc_p:,} dec={dec_p:,} den={den_p:,}")

        z_data, oracle_mse = train_pipeline(
            encoder, decoder, denoiser, e_core, train_x,
            z_h, n_bits, device)

        K, H, W = z_data.shape[1:]

        # Generate
        print(f"  [E] Generating {args.n_samples} samples...")
        torch.manual_seed(args.seed)
        z_gen = denoiser.sample_multiscale(args.n_samples, H, W, device)

        with torch.no_grad():
            x_gen = decoder(z_gen.to(device)).cpu()
        save_grid(x_gen, os.path.join(args.output_dir, f'gen_{name}.png'))

        # Evaluate
        z_cpu = z_gen.cpu()
        viol = e_core.violation_rate(z_cpu[:100].to(device))
        div = compute_diversity(z_cpu)
        with torch.no_grad():
            zc=z_cpu[:100].to(device); xc=decoder(zc); zcy,_=encoder(xc)
            cycle=(zc!=zcy).float().mean().item()
        x_np=x_gen.numpy().reshape(len(x_gen),-1)
        nn_d=compute_1nn(x_np, train_x_np)
        conn=connectedness_proxy(x_gen[:100])
        band=per_band_energy_distance(x_gen[:200], test_x[:200], device)
        hfc=hf_coherence_metric(x_gen[:200], device)
        hfn=hf_noise_index(x_gen[:200], device)

        r = {
            'config': name, 'z_h': z_h, 'n_bits': n_bits, 'total_bits': total,
            'compression': 24576/total,
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

        # Free memory
        del encoder, decoder, denoiser, e_core, z_data, z_gen, x_gen
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 100)
    print("BANDWIDTH SWEEP SUMMARY")
    print(f"Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  conn={real_conn:.4f}")
    print(f"Original pixel info: 24576 bits")
    print("=" * 100)

    h = (f"{'config':<16} {'bits':>6} {'comp':>5} {'mse':>6} {'viol':>6} "
         f"{'div':>6} {'cycle':>6} {'HFcoh':>6} {'HFnoi':>7} "
         f"{'Eg_L':>6} {'Eg_M':>6} {'Eg_H':>6}")
    print(h); print("-"*len(h))
    for r in all_results:
        print(f"{r['config']:<16} {r['total_bits']:>6} {r['compression']:>5.1f}× "
              f"{r['oracle_mse']:>6.4f} {r['violation']:>6.4f} "
              f"{r['diversity']:>6.4f} {r['cycle']:>6.4f} "
              f"{r['hf_coherence']:>6.4f} {r['hf_noise_index']:>7.2f} "
              f"{r['energy_gap_low']:>6.4f} {r['energy_gap_mid']:>6.4f} "
              f"{r['energy_gap_high']:>6.4f}")

    # Diagnosis
    print("\n" + "-" * 60)
    print("BANDWIDTH DIAGNOSIS:")
    for r in all_results:
        diag = []
        if r['energy_gap_mid'] < 0.5: diag.append("mid_freq✓")
        if r['energy_gap_high'] < 0.5: diag.append("high_freq✓")
        if abs(r['hf_noise_index'] - real_hf_noi) < real_hf_noi * 0.5:
            diag.append("HF_noise✓")
        if r['diversity'] > 0.15: diag.append("diverse✓")
        if r['violation'] < 0.3: diag.append("valid✓")
        status = " ".join(diag) if diag else "bottlenecked"
        print(f"  {r['config']:<16} {r['total_bits']:>6}bits → {status}")

    csv_path = os.path.join(args.output_dir, "bandwidth_sweep.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(all_results[0].keys())
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV: {csv_path}")

    print("\n" + "=" * 100)
    print("Stage S1 bandwidth sweep complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
