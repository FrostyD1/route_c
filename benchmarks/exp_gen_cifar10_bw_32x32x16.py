#!/usr/bin/env python3
"""
Direct 32×32×16 bandwidth test for CIFAR-10 generation.
z = 32×32×16 = 16384 bits (1.5× compression from 24576 pixel bits).

Reports VRAM usage at each stage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, argparse
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# DCT + FREQ UTILITIES (copied from bandwidth sweep)
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

def vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


# ============================================================================
# MODELS
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_gen_cifar10_bw')
    parser.add_argument('--n_samples', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    Z_H, Z_W, N_BITS = 32, 32, 16
    TOTAL_BITS = Z_H * Z_W * N_BITS  # 16384
    BS = args.batch_size

    print("=" * 80)
    print(f"DIRECT TEST: 32×32×16 = {TOTAL_BITS} bits (compression: {24576/TOTAL_BITS:.1f}×)")
    print(f"Batch size: {BS}")
    print("=" * 80)

    # Load CIFAR-10
    print("\n[1] Loading CIFAR-10...")
    from torchvision import datasets, transforms
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(args.seed)
    train_x = torch.stack([train_ds[i][0] for i in rng.choice(len(train_ds), 3000, replace=False)])
    test_x = torch.stack([test_ds[i][0] for i in rng.choice(len(test_ds), 500, replace=False)])
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")

    # Reference
    print("\n[2] Reference metrics...")
    real_hf_coh = hf_coherence_metric(test_x[:200], device)
    real_hf_noi = hf_noise_index(test_x[:200], device)
    real_conn = connectedness_proxy(test_x[:100].numpy())
    print(f"    HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  conn={real_conn:.4f}")
    save_grid(test_x[:64], os.path.join(args.output_dir, 'real_samples.png'))
    train_x_np = train_x.numpy().reshape(len(train_x), -1)

    # Create models
    print(f"\n[3] Creating models... (VRAM before: {vram_mb():.0f}MB)")
    encoder = Encoder32(N_BITS).to(device)
    decoder = Decoder32(N_BITS).to(device)
    denoiser = FreqDenoiser(N_BITS).to(device)
    e_core = LocalEnergyCore(N_BITS).to(device)

    enc_p = sum(p.numel() for p in encoder.parameters())
    dec_p = sum(p.numel() for p in decoder.parameters())
    den_p = sum(p.numel() for p in denoiser.parameters())
    print(f"    Params: enc={enc_p:,} dec={dec_p:,} den={den_p:,}")
    print(f"    VRAM after model creation: {vram_mb():.0f}MB")

    # [A] Train ADC/DAC
    print(f"\n[A] Training ADC/DAC (32×32×{N_BITS})...")
    opt = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=1e-3)
    for epoch in tqdm(range(40), desc="ADC"):
        encoder.train(); decoder.train()
        encoder.set_temperature(1.0 + (0.3-1.0)*epoch/39)
        perm = torch.randperm(len(train_x)); tl, nb = 0., 0
        for i in range(0, len(train_x), BS):
            idx = perm[i:i+BS]; x = train_x[idx].to(device)
            opt.zero_grad(); z, _ = encoder(x); xh = decoder(z)
            loss = F.mse_loss(xh, x) + 0.5*F.binary_cross_entropy(xh, x)
            loss.backward(); opt.step(); tl += loss.item(); nb += 1
    print(f"    ADC done: loss={tl/nb:.4f} VRAM={vram_mb():.0f}MB")
    encoder.eval(); decoder.eval()

    with torch.no_grad():
        tb = train_x[:32].to(device); zo, _ = encoder(tb)
        oracle_mse = F.mse_loss(decoder(zo), tb).item()
    print(f"    Oracle MSE: {oracle_mse:.4f}")

    # [B] Encode
    print("\n[B] Encoding training set...")
    z_data = []
    with torch.no_grad():
        for i in range(0, len(train_x), BS):
            z, _ = encoder(train_x[i:i+BS].to(device)); z_data.append(z.cpu())
    z_data = torch.cat(z_data)
    K, H, W = z_data.shape[1:]
    print(f"    z_data: {z_data.shape}, bits={K*H*W}, usage={z_data.mean():.3f}")
    print(f"    VRAM after encode: {vram_mb():.0f}MB")

    # [C] Train E_core
    print("\n[C] Training E_core...")
    eopt = torch.optim.Adam(e_core.parameters(), lr=1e-3)
    for ep in tqdm(range(10), desc="E_core"):
        e_core.train(); perm = torch.randperm(len(z_data))
        for i in range(0, len(z_data), 256):
            z = z_data[perm[i:i+256]].to(device); eopt.zero_grad(); tl_ = 0.
            for _ in range(20):
                b=torch.randint(K,(1,)).item(); ii=torch.randint(H,(1,)).item(); jj=torch.randint(W,(1,)).item()
                ctx=e_core.get_context(z,b,ii,jj)
                tl_+=F.binary_cross_entropy_with_logits(e_core.predictor(ctx).squeeze(1),z[:,b,ii,jj])
            (tl_/20).backward(); eopt.step()
    e_core.eval()
    print(f"    VRAM after E_core: {vram_mb():.0f}MB")

    # [D] Train denoiser with freq scheduled + coherence
    print("\n[D] Training freq denoiser...")
    dopt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)
    for epoch in tqdm(range(30), desc="Denoiser"):
        denoiser.train(); perm = torch.randperm(N)
        tl, fl, cl, nb = 0., 0., 0., 0
        progress = epoch / 29
        for i in range(0, N, BS):
            idx = perm[i:i+BS]; z_clean = z_data[idx].to(device); B_ = z_clean.shape[0]
            noise_level = torch.rand(B_, device=device)
            flip = (torch.rand_like(z_clean) < noise_level.view(B_,1,1,1)).float()
            z_noisy = z_clean*(1-flip) + (1-z_clean)*flip
            dopt.zero_grad()
            logits = denoiser(z_noisy, noise_level)
            loss_bce = F.binary_cross_entropy_with_logits(logits, z_clean)
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
    print(f"    Denoiser done: BCE={tl/nb:.4f} freq={fl/nb:.4f} coh={cl/nb:.4f} VRAM={vram_mb():.0f}MB")
    denoiser.eval()

    # [E] Generate
    print(f"\n[E] Generating {args.n_samples} samples...")
    print(f"    VRAM before generation: {vram_mb():.0f}MB")
    torch.manual_seed(args.seed)
    # Generate in batches to control VRAM
    gen_bs = 32
    z_gen_list = []
    for gi in range(0, args.n_samples, gen_bs):
        n = min(gen_bs, args.n_samples - gi)
        z_gen_list.append(denoiser.sample_multiscale(n, H, W, device).cpu())
    z_gen = torch.cat(z_gen_list)

    with torch.no_grad():
        x_gen_list = []
        for gi in range(0, len(z_gen), gen_bs):
            x_gen_list.append(decoder(z_gen[gi:gi+gen_bs].to(device)).cpu())
        x_gen = torch.cat(x_gen_list)
    save_grid(x_gen, os.path.join(args.output_dir, 'gen_32x32_16bit_direct.png'))

    # [F] Evaluate
    print("\n[F] Evaluating...")
    z_cpu = z_gen
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

    print(f"\n{'='*80}")
    print(f"RESULTS: 32×32×16 = {TOTAL_BITS} bits (1.5× compression)")
    print(f"{'='*80}")
    print(f"  oracle_mse  = {oracle_mse:.4f}")
    print(f"  violation   = {viol:.4f}")
    print(f"  diversity   = {div:.4f}")
    print(f"  cycle       = {cycle:.4f}")
    print(f"  nn_dist     = {nn_d:.2f}")
    print(f"  connectedness = {conn:.4f}")
    print(f"  HF_coh      = {hfc:.4f}  (real={real_hf_coh:.4f})")
    print(f"  HF_noise    = {hfn:.2f}  (real={real_hf_noi:.2f})")
    print(f"  E_gap_low   = {band['energy_gap_low']:.4f}")
    print(f"  E_gap_mid   = {band['energy_gap_mid']:.4f}")
    print(f"  E_gap_high  = {band['energy_gap_high']:.4f}")
    print(f"  Peak VRAM   = {vram_mb():.0f}MB / 4096MB")

    # Diagnosis
    diag = []
    if band['energy_gap_mid'] < 0.5: diag.append("mid_freq_RESOLVED")
    if band['energy_gap_high'] < 0.5: diag.append("high_freq_RESOLVED")
    if abs(hfn - real_hf_noi) < real_hf_noi * 0.5: diag.append("HF_noise_OK")
    if div > 0.15: diag.append("diverse")
    if viol < 0.3: diag.append("valid")
    print(f"\n  DIAGNOSIS: {' + '.join(diag) if diag else 'BOTTLENECKED'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
