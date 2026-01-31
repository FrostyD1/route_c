#!/usr/bin/env python3
"""
CIFAR-10 INT4 Generation v2 — Denoiser Speed/Quality Comparison
================================================================
Compare 4 denoiser architectures for INT4 token generation:

1. Cat-16:       Categorical (16-way softmax per position) — baseline, slow
2. Reg-R1:       Regression (predict clean value directly) — fast
3. Reg-R2:       Bounded residual (predict Δ ∈ [-2,+2] levels) — recommended
4. Bitplane-Gray: 4 binary planes via Gray code — discrete but fast

All share same: Encoder/Decoder (INT4 quantize), E_core, z config.
Fixed config: 32×32×8 INT4 = 32768 effective bits.

Metrics: runtime, z_cycle, drift5, violation, diversity, HF_noise, HF_coh, E_gap.

4GB GPU, BS=256, tqdm.

Usage:
    python3 -u benchmarks/exp_gen_cifar10_int4_v2.py --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, csv, argparse, time
from collections import OrderedDict
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

N_LEVELS = 16  # INT4

# ============================================================================
# DCT + FREQ UTILITIES
# ============================================================================

def dct2d(x):
    B, C, H, W = x.shape
    def dm(N):
        n = torch.arange(N, dtype=x.dtype, device=x.device); k = n.unsqueeze(1)
        D = torch.cos(np.pi * (2*n+1) * k / (2*N))
        D[0] *= 1.0/np.sqrt(N); D[1:] *= np.sqrt(2.0/N); return D
    return torch.einsum('hH,bcHW,wW->bchw', dm(H), x, dm(W))

def idct2d(X):
    B, C, H, W = X.shape
    def dm(N):
        n = torch.arange(N, dtype=X.dtype, device=X.device); k = n.unsqueeze(1)
        D = torch.cos(np.pi * (2*n+1) * k / (2*N))
        D[0] *= 1.0/np.sqrt(N); D[1:] *= np.sqrt(2.0/N); return D
    return torch.einsum('Hh,bchw,Ww->bcHW', dm(H).T, X, dm(W).T)

def get_freq_masks(H, W, device='cpu'):
    fy = torch.arange(H, device=device).float(); fx = torch.arange(W, device=device).float()
    fg = fy.unsqueeze(1) + fx.unsqueeze(0); mf = H+W-2
    return (fg<=mf/3).float(), ((fg>mf/3)&(fg<=2*mf/3)).float(), (fg>2*mf/3).float()

def decompose_bands(x):
    lm,mm,hm = get_freq_masks(x.shape[2],x.shape[3],x.device)
    d = dct2d(x); return idct2d(d*lm), idct2d(d*mm), idct2d(d*hm)

def per_band_energy_distance(ig, ir, device='cpu'):
    def t(x):
        if isinstance(x,np.ndarray): x=torch.tensor(x,dtype=torch.float32)
        if x.dim()==3: x=x.unsqueeze(1)
        return x.to(device)
    g=t(ig[:200]); r=t(ir[:200]); dg=dct2d(g); dr=dct2d(r)
    res={}
    for nm,m in zip(['low','mid','high'], get_freq_masks(g.shape[2],g.shape[3],device)):
        eg=(dg**2*m).mean(dim=(0,1)).sum().item(); er=(dr**2*m).mean(dim=(0,1)).sum().item()
        res[f'E_{nm}']=abs(eg-er)/(er+1e-12)
    return res

def hf_coherence_metric(images, device='cpu'):
    if isinstance(images,np.ndarray): images=torch.tensor(images,dtype=torch.float32)
    if images.dim()==3: images=images.unsqueeze(1)
    images=images[:200].to(device); _,_,xh=decompose_bands(images)
    xf=xh.reshape(-1,xh.shape[2],xh.shape[3])
    xn=(xf-xf.mean(dim=(1,2),keepdim=True))/(xf.std(dim=(1,2),keepdim=True).clamp(min=1e-8))
    cs=[]
    for dy,dx in [(0,1),(1,0),(1,1)]:
        a,b=xn,xn
        if dy>0: a,b=a[:,dy:,:],b[:,:-dy,:]
        if dx>0: a,b=a[:,:,dx:],b[:,:,:-dx]
        cs.append((a*b).mean().item())
    return np.mean(cs)

def hf_noise_index(images, device='cpu'):
    if isinstance(images,np.ndarray): images=torch.tensor(images,dtype=torch.float32)
    if images.dim()==3: images=images.unsqueeze(1)
    images=images[:200].to(device)
    gx=images[:,:,:,1:]-images[:,:,:,:-1]; gy=images[:,:,1:,:]-images[:,:,:-1,:]
    ge=(gx**2).mean().item()+(gy**2).mean().item()
    _,_,xh=decompose_bands(images); return ge/((xh**2).mean().item()+1e-12)

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
    N=len(z); zf=z.reshape(N,-1).float(); d=[]
    for _ in range(n_pairs):
        i,j=np.random.choice(N,2,replace=False)
        d.append((zf[i]!=zf[j]).float().mean().item())
    return np.mean(d)

def save_grid(images, path, nrow=8):
    try:
        from torchvision.utils import save_image
        if isinstance(images,np.ndarray): images=torch.tensor(images)
        if images.dim()==3: images=images.unsqueeze(1)
        save_image(images[:64],path,nrow=nrow,normalize=False)
    except: pass

def vram_mb():
    return torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0.0


# ============================================================================
# INT4 QUANTIZER
# ============================================================================

class INT4Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('bias', torch.tensor(0.0))
        self.register_buffer('ema_min', torch.tensor(0.0))
        self.register_buffer('ema_max', torch.tensor(1.0))

    @torch.no_grad()
    def update_scale(self, x):
        self.ema_min.mul_(0.99).add_(0.01 * x.min())
        self.ema_max.mul_(0.99).add_(0.01 * x.max())
        rng = (self.ema_max - self.ema_min).clamp(min=1e-6)
        self.scale.copy_(rng / 15.0)
        self.bias.copy_(self.ema_min)

    def forward(self, x):
        if self.training: self.update_scale(x)
        x_norm = ((x - self.bias) / self.scale.clamp(min=1e-6)).clamp(0, 15)
        x_round = x_norm.round()
        tokens = x_round.long()
        x_ste = x_round - x_norm.detach() + x_norm  # STE
        return tokens, x_ste / 15.0  # int, float[0,1]

    def tokens_to_float(self, t):
        return t.float() / 15.0


# ============================================================================
# GRAY CODE UTILS
# ============================================================================

def int_to_gray(n):
    return n ^ (n >> 1)

def gray_to_int(g):
    n = g
    mask = n >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n

# Precompute lookup tables
GRAY_ENC = torch.tensor([int_to_gray(i) for i in range(16)])  # int→gray
GRAY_DEC = torch.tensor([gray_to_int(i) for i in range(16)])  # gray→int

def tokens_to_bitplanes(tokens, device):
    """INT4 tokens [B,K,H,W] → 4 binary planes [B,K*4,H,W] via Gray code."""
    gray = GRAY_ENC.to(device)[tokens]  # [B,K,H,W] gray-coded
    bits = []
    for b in range(4):
        bits.append(((gray >> b) & 1).float())
    return torch.cat(bits, dim=1)  # [B, K*4, H, W]

def bitplanes_to_tokens(planes, K, device):
    """4 binary planes [B,K*4,H,W] → INT4 tokens [B,K,H,W] via Gray decode."""
    B, _, H, W = planes.shape
    planes = planes.reshape(B, 4, K, H, W)
    gray = torch.zeros(B, K, H, W, dtype=torch.long, device=device)
    for b in range(4):
        gray += ((planes[:, b] > 0.5).long() << b)
    return GRAY_DEC.to(device)[gray]


# ============================================================================
# ENCODER / DECODER
# ============================================================================

class Encoder32(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, n_ch, 3, padding=1))
        self.q = INT4Quantizer()
    def forward(self, x):
        logits = self.net(x); tokens, z_float = self.q(logits)
        return tokens, z_float, logits

class Decoder32(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid())
    def forward(self, z): return self.net(z)


# ============================================================================
# E_CORE
# ============================================================================

class LocalEnergyCore(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(9*n_ch-1, 128), nn.ReLU(), nn.Linear(128, 16))
    def get_context(self, z_float, bi, i, j):
        B,K,H,W=z_float.shape; ctx=[]
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                ni,nj=(i+di)%H,(j+dj)%W
                for b in range(K):
                    if di==0 and dj==0 and b==bi: continue
                    ctx.append(z_float[:,b,ni,nj])
        return torch.stack(ctx,dim=1)
    def violation_rate(self, z_int, n=50):
        B,K,H,W=z_int.shape; z_f=z_int.float()/15.0; v=[]
        for _ in range(min(n,H*W*K)):
            b=torch.randint(K,(1,)).item(); i=torch.randint(H,(1,)).item(); j=torch.randint(W,(1,)).item()
            ctx=self.get_context(z_f,b,i,j); pred=self.predictor(ctx).argmax(1)
            v.append((pred!=z_int[:,b,i,j]).float().mean().item())
        return np.mean(v)


# ============================================================================
# 4 DENOISER VARIANTS
# ============================================================================

class DenoiserBase(nn.Module):
    """Shared CNN backbone. Subclasses define head + loss + sampling."""
    def __init__(self, in_ch, out_ch, hid=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
    def backbone_forward(self, inp):
        return self.backbone(inp) + self.skip(inp)


# --- 1. Categorical (slow baseline) ---
class DenoiserCat16(DenoiserBase):
    def __init__(self, K):
        super().__init__(K+1, K*16); self.K=K
    def forward(self, z_int, nl):
        B,K,H,W=z_int.shape
        z_f=z_int.float()/15.0
        inp=torch.cat([z_f, nl.view(B,1,1,1).expand(-1,1,H,W)], dim=1)
        return self.backbone_forward(inp).reshape(B,K,16,H,W)
    def loss(self, z_noisy, z_clean, nl):
        logits=self(z_noisy, nl)
        return F.cross_entropy(logits.permute(0,1,3,4,2).reshape(-1,16), z_clean.reshape(-1))
    @torch.no_grad()
    def sample(self, n, K, H, W, device, steps=20, temp=0.8):
        z=torch.randint(0,16,(n,K,H,W),device=device)
        for s in range(steps):
            p=s/steps; nl=torch.tensor([1-p],device=device).expand(n)
            logits=self(z,nl); t=temp*(0.5 if p<0.3 else 1.2)
            c=min(0.9, 0.2+0.7*p)
            sampled=torch.multinomial(F.softmax(logits/t,dim=2).permute(0,1,3,4,2).reshape(-1,16),1).reshape(n,K,H,W)
            mask=(torch.rand(n,K,H,W,device=device)<c).long()
            z=mask*sampled+(1-mask)*z
        return self(z,torch.zeros(n,device=device)).argmax(dim=2)


# --- 2. Regression R1 (predict clean value) ---
class DenoiserRegR1(DenoiserBase):
    def __init__(self, K):
        super().__init__(K+1, K); self.K=K
    def forward(self, z_int, nl):
        B,K,H,W=z_int.shape
        z_f=z_int.float()/15.0
        inp=torch.cat([z_f, nl.view(B,1,1,1).expand(-1,1,H,W)], dim=1)
        return torch.sigmoid(self.backbone_forward(inp))  # output [0,1]
    def loss(self, z_noisy, z_clean, nl):
        pred=self(z_noisy, nl)
        target=z_clean.float()/15.0
        # Huber loss + margin loss (push away from rounding boundaries)
        loss_h = F.smooth_l1_loss(pred, target)
        # Margin: penalize being near x.5 boundaries in [0,15] space
        pred_scaled = pred * 15.0
        frac = pred_scaled - pred_scaled.floor()
        loss_margin = (frac * (1 - frac)).mean() * 0.3
        return loss_h + loss_margin
    @torch.no_grad()
    def sample(self, n, K, H, W, device, steps=20, temp=0.8):
        z=torch.randint(0,16,(n,K,H,W),device=device)
        for s in range(steps):
            p=s/steps; nl=torch.tensor([1-p],device=device).expand(n)
            pred=self(z,nl)  # [0,1]
            # Mix: interpolate toward predicted clean
            alpha=min(0.9, 0.3+0.6*p)
            z_float = z.float()/15.0*(1-alpha) + pred*alpha
            # Add small noise early
            if p<0.5: z_float += torch.randn_like(z_float)*0.02*(1-p)
            z=(z_float*15.0).clamp(0,15).round().long()
        pred=self(z,torch.zeros(n,device=device))
        return (pred*15.0).clamp(0,15).round().long()


# --- 3. Regression R2 (bounded residual, Δ ∈ [-2,+2]) ---
class DenoiserRegR2(DenoiserBase):
    def __init__(self, K, delta=2.0):
        super().__init__(K+1, K); self.K=K; self.delta=delta
    def forward(self, z_int, nl):
        B,K,H,W=z_int.shape
        z_f=z_int.float()/15.0
        inp=torch.cat([z_f, nl.view(B,1,1,1).expand(-1,1,H,W)], dim=1)
        raw=self.backbone_forward(inp)
        # Bounded residual: tanh → [-delta, +delta] in level space
        return torch.tanh(raw) * self.delta
    def loss(self, z_noisy, z_clean, nl):
        delta_pred=self(z_noisy, nl)  # [-2,+2] level units
        z_noisy_f = z_noisy.float()
        z_clean_f = z_clean.float()
        delta_target = z_clean_f - z_noisy_f  # actual difference in levels
        delta_target = delta_target.clamp(-self.delta, self.delta)  # clip target too
        loss_h = F.smooth_l1_loss(delta_pred, delta_target / self.delta)  # normalized
        # Margin on resulting value
        z_new = (z_noisy_f + delta_pred).clamp(0,15)
        frac = z_new - z_new.floor()
        loss_margin = (frac * (1-frac)).mean() * 0.3
        return loss_h + loss_margin
    @torch.no_grad()
    def sample(self, n, K, H, W, device, steps=20, temp=0.8):
        z=torch.randint(0,16,(n,K,H,W),device=device)
        for s in range(steps):
            p=s/steps; nl=torch.tensor([1-p],device=device).expand(n)
            delta=self(z,nl)  # [-2,+2]
            # Step size schedule: bigger steps early, smaller late
            step_scale = max(0.3, 1.0 - p*0.7)
            z_new = z.float() + delta * step_scale
            if p<0.4: z_new += torch.randn_like(z_new)*0.3*(1-p)
            z = z_new.clamp(0,15).round().long()
        # Final step
        delta=self(z,torch.zeros(n,device=device))
        return (z.float()+delta*0.5).clamp(0,15).round().long()


# --- 4. Bitplane Gray (4 binary planes) ---
class DenoiserBitplaneGray(DenoiserBase):
    def __init__(self, K):
        super().__init__(K*4+1, K*4); self.K=K
    def forward(self, z_int, nl):
        B,K,H,W=z_int.shape
        planes=tokens_to_bitplanes(z_int, z_int.device)  # [B,K*4,H,W]
        inp=torch.cat([planes, nl.view(B,1,1,1).expand(-1,1,H,W)], dim=1)
        return self.backbone_forward(inp)  # [B,K*4,H,W] logits
    def loss(self, z_noisy, z_clean, nl):
        logits=self(z_noisy, nl)
        target=tokens_to_bitplanes(z_clean, z_clean.device)
        return F.binary_cross_entropy_with_logits(logits, target)
    @torch.no_grad()
    def sample(self, n, K, H, W, device, steps=20, temp=0.8):
        z=torch.randint(0,16,(n,K,H,W),device=device)
        for s in range(steps):
            p=s/steps; nl=torch.tensor([1-p],device=device).expand(n)
            logits=self(z,nl)
            t=temp*(0.5 if p<0.3 else 1.2); c=min(0.9, 0.2+0.7*p)
            probs=torch.sigmoid(logits/t)
            sampled=(torch.rand_like(probs)<probs).float()
            mask=(torch.rand(n,K*4,H,W,device=device)<c).float()
            planes=mask*sampled+(1-mask)*tokens_to_bitplanes(z,device)
            z=bitplanes_to_tokens(planes, K, device)
        logits=self(z,torch.zeros(n,device=device))
        planes=(torch.sigmoid(logits)>0.5).float()
        return bitplanes_to_tokens(planes, K, device)


# ============================================================================
# TRAINING
# ============================================================================

def train_adc(enc, dec, train_x, device, epochs=40, bs=256):
    opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=1e-3)
    for ep in tqdm(range(epochs),desc="ADC"):
        enc.train(); dec.train()
        perm=torch.randperm(len(train_x)); tl,nb=0.,0
        for i in range(0,len(train_x),bs):
            x=train_x[perm[i:i+bs]].to(device); opt.zero_grad()
            _,zf,logits=enc(x); xh=dec(zf)
            loss=F.mse_loss(xh,x)+0.5*F.binary_cross_entropy(xh,x)
            frac=logits-logits.floor(); loss+=0.1*(frac*(1-frac)).mean()
            loss.backward(); opt.step(); tl+=loss.item(); nb+=1
    enc.eval(); dec.eval()
    with torch.no_grad():
        _,zf,_=enc(train_x[:64].to(device)); mse=F.mse_loss(dec(zf),train_x[:64].to(device)).item()
    print(f"    ADC done: loss={tl/nb:.4f} oracle_mse={mse:.4f}")
    return mse

def encode_all(enc, data, device, bs=256):
    zs=[]
    with torch.no_grad():
        for i in range(0,len(data),bs):
            t,_,_=enc(data[i:i+bs].to(device)); zs.append(t.cpu())
    return torch.cat(zs)

def train_ecore(ec, z_data, device, epochs=10, bs=256):
    opt=torch.optim.Adam(ec.parameters(),lr=1e-3); K,H,W=z_data.shape[1:]
    for ep in tqdm(range(epochs),desc="E_core"):
        ec.train(); perm=torch.randperm(len(z_data))
        for i in range(0,len(z_data),bs):
            z=z_data[perm[i:i+bs]].to(device); zf=z.float()/15.0; opt.zero_grad(); tl=0.
            for _ in range(20):
                b=torch.randint(K,(1,)).item();ii=torch.randint(H,(1,)).item();jj=torch.randint(W,(1,)).item()
                ctx=ec.get_context(zf,b,ii,jj)
                tl+=F.cross_entropy(ec.predictor(ctx),z[:,b,ii,jj])
            (tl/20).backward(); opt.step()
    ec.eval()

def train_denoiser(den, z_data, device, epochs=30, bs=256):
    opt=torch.optim.Adam(den.parameters(),lr=1e-3); N=len(z_data); K,H,W=z_data.shape[1:]
    t0=time.time()
    for ep in tqdm(range(epochs),desc="Denoiser"):
        den.train(); perm=torch.randperm(N); tl,nb=0.,0
        for i in range(0,N,bs):
            z_clean=z_data[perm[i:i+bs]].to(device); B_=z_clean.shape[0]
            nl=torch.rand(B_,device=device)
            noise_mask=(torch.rand(B_,K,H,W,device=device)<nl.view(B_,1,1,1)).long()
            z_noise=torch.randint(0,16,(B_,K,H,W),device=device)
            z_noisy=noise_mask*z_noise+(1-noise_mask)*z_clean
            opt.zero_grad(); loss=den.loss(z_noisy,z_clean,nl)
            loss.backward(); opt.step(); tl+=loss.item(); nb+=1
    elapsed=time.time()-t0
    den.eval()
    print(f"    Denoiser done: loss={tl/nb:.4f} time={elapsed:.1f}s ({elapsed/epochs:.1f}s/ep)")
    return elapsed


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output_dir',default='outputs/exp_gen_cifar10_int4v2')
    parser.add_argument('--n_samples',type=int,default=256)
    parser.add_argument('--bs',type=int,default=256)
    args=parser.parse_args()

    device=torch.device(args.device)
    os.makedirs(args.output_dir,exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    N_CH = 8  # fixed: 32×32×8 INT4 = 32768 eff bits

    print("="*100)
    print(f"CIFAR-10 INT4 DENOISER COMPARISON | z=32×32×{N_CH} INT4 = {32*32*N_CH*4} eff bits")
    print("="*100)

    # Load
    print("\n[1] Loading CIFAR-10...")
    from torchvision import datasets, transforms
    tds=datasets.CIFAR10('./data',train=True,download=True,transform=transforms.ToTensor())
    vds=datasets.CIFAR10('./data',train=False,download=True,transform=transforms.ToTensor())
    rng=np.random.default_rng(args.seed)
    train_x=torch.stack([tds[i][0] for i in rng.choice(len(tds),5000,replace=False)])
    test_x=torch.stack([vds[i][0] for i in rng.choice(len(vds),500,replace=False)])
    print(f"    Train: {train_x.shape}")

    # Reference
    print("\n[2] Reference metrics...")
    ref_hfc=hf_coherence_metric(test_x[:200],device)
    ref_hfn=hf_noise_index(test_x[:200],device)
    ref_conn=connectedness_proxy(test_x[:100].numpy())
    print(f"    HF_coh={ref_hfc:.4f} HF_noise={ref_hfn:.2f} conn={ref_conn:.4f}")
    save_grid(test_x[:64],os.path.join(args.output_dir,'real.png'))
    train_np=train_x.numpy().reshape(len(train_x),-1)

    # Shared: train ADC + E_core once
    print("\n[3] Training shared ADC/DAC + E_core...")
    enc=Encoder32(N_CH).to(device); dec=Decoder32(N_CH).to(device)
    oracle_mse=train_adc(enc,dec,train_x,device,epochs=40,bs=args.bs)
    z_data=encode_all(enc,train_x,device,bs=args.bs)
    K,H,W=z_data.shape[1:]
    print(f"    z: {z_data.shape}, range=[{z_data.min()},{z_data.max()}], mean={z_data.float().mean():.2f}")

    ec=LocalEnergyCore(N_CH).to(device)
    train_ecore(ec,z_data,device,epochs=10,bs=args.bs)

    # 4 denoisers
    DENOISERS = OrderedDict([
        ("Cat16", DenoiserCat16(N_CH)),
        ("RegR1", DenoiserRegR1(N_CH)),
        ("RegR2", DenoiserRegR2(N_CH)),
        ("Bitplane", DenoiserBitplaneGray(N_CH)),
    ])

    all_results=[]
    for dname, den in DENOISERS.items():
        print(f"\n{'='*80}")
        den_p=sum(p.numel() for p in den.parameters())
        print(f"DENOISER: {dname} | params={den_p:,}")
        print("="*80)

        den=den.to(device)
        torch.manual_seed(args.seed)
        elapsed=train_denoiser(den,z_data,device,epochs=30,bs=args.bs)

        # Generate
        print(f"  Generating {args.n_samples} samples...")
        torch.manual_seed(args.seed); t0=time.time()
        zg=[]
        for gi in range(0,args.n_samples,64):
            n=min(64,args.n_samples-gi)
            zg.append(den.sample(n,K,H,W,device).cpu())
        z_gen=torch.cat(zg)
        gen_time=time.time()-t0

        with torch.no_grad():
            xg=[]
            for gi in range(0,len(z_gen),64):
                xg.append(dec(z_gen[gi:gi+64].float().to(device)/15.0).cpu())
            x_gen=torch.cat(xg)
        save_grid(x_gen,os.path.join(args.output_dir,f'gen_{dname}.png'))

        # Eval
        viol=ec.violation_rate(z_gen[:100].to(device))
        div=compute_diversity(z_gen)
        with torch.no_grad():
            zc=z_gen[:100].to(device); xc=dec(zc.float()/15.0)
            zcy,_,_=enc(xc); cycle=(zc!=zcy).float().mean().item()
        # Multi-cycle drift
        with torch.no_grad():
            z0=z_gen[:50].to(device); zt=z0.clone()
            drifts=[]
            for cyc in range(5):
                xt=dec(zt.float()/15.0); zt,_,_=enc(xt)
                drifts.append((z0!=zt).float().mean().item())
        drift5=drifts[-1]

        xnp=x_gen.numpy().reshape(len(x_gen),-1)
        nn_d=float(np.mean([((train_np-xnp[i:i+1])**2).sum(1).min() for i in range(min(100,len(xnp)))]))
        conn=connectedness_proxy(x_gen[:100])
        band=per_band_energy_distance(x_gen[:200],test_x[:200],device)
        hfc=hf_coherence_metric(x_gen[:200],device)
        hfn=hf_noise_index(x_gen[:200],device)

        r={'denoiser':dname,'params':den_p,'train_time':elapsed,'gen_time':gen_time,
           'oracle_mse':oracle_mse,'viol':viol,'div':div,'cycle':cycle,'drift5':drift5,
           'nn_dist':nn_d,'conn':conn,'hf_coh':hfc,'hf_noise':hfn,
           'E_low':band['E_low'],'E_mid':band['E_mid'],'E_high':band['E_high']}
        all_results.append(r)

        print(f"  RESULTS: viol={viol:.4f} div={div:.4f} cycle={cycle:.4f} drift5={drift5:.4f}")
        print(f"    HF_coh={hfc:.4f}(real={ref_hfc:.4f}) HF_noise={hfn:.2f}(real={ref_hfn:.2f})")
        print(f"    E_gap: L={band['E_low']:.4f} M={band['E_mid']:.4f} H={band['E_high']:.4f}")
        print(f"    Time: train={elapsed:.0f}s gen={gen_time:.1f}s")

        den=den.cpu(); torch.cuda.empty_cache()

    # Summary
    print("\n"+"="*100)
    print("DENOISER COMPARISON SUMMARY")
    print("="*100)
    h=f"{'method':<12} {'params':>8} {'train':>7} {'gen':>5} {'viol':>6} {'div':>6} {'cycle':>6} {'drift5':>6} {'HFcoh':>6} {'HFnoi':>7} {'E_M':>6} {'E_H':>6}"
    print(h); print("-"*len(h))
    for r in all_results:
        print(f"{r['denoiser']:<12} {r['params']:>8,} {r['train_time']:>6.0f}s {r['gen_time']:>4.1f}s "
              f"{r['viol']:>6.4f} {r['div']:>6.4f} {r['cycle']:>6.4f} {r['drift5']:>6.4f} "
              f"{r['hf_coh']:>6.4f} {r['hf_noise']:>7.2f} {r['E_mid']:>6.4f} {r['E_high']:>6.4f}")

    csv_path=os.path.join(args.output_dir,"int4v2_results.csv")
    with open(csv_path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(all_results[0].keys())); w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV: {csv_path}")


if __name__=="__main__":
    main()
