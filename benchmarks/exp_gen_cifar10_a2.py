#!/usr/bin/env python3
"""
CIFAR-10 A2: Structured HF Generation with Freq-Aware Energy Geometry
======================================================================
Applies the best configs from FMNIST A2 to CIFAR-10 RGB generation.

Configs (5 runs):
  1. baseline:        Standard denoising (no freq)
  2. freq_amp:        Band energy matching (λ_freq=0.3)
  3. freq_sched_coh:  Scheduled + coherence (λ_freq=0.3, λ_coh=0.1)
  4. freq_full:       Scheduled + coherence + guided sampling
  5. freq_full_ms:    Scheduled + coherence + multiscale sampling

Architecture: 32×32×3 → 16×16×8 binary (ResNet encoder/decoder)
4GB GPU: 3000 train, 500 test, batch_size=32

Usage:
    python3 -u benchmarks/exp_gen_cifar10_a2.py --device cuda
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
# FREQ UTILITIES (from exp_gen_freq_a2.py)
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
    freq_grid = fy.unsqueeze(1) + fx.unsqueeze(0)
    max_freq = H + W - 2
    t1, t2 = max_freq/3.0, 2*max_freq/3.0
    return ((freq_grid <= t1).float(),
            ((freq_grid > t1) & (freq_grid <= t2)).float(),
            (freq_grid > t2).float())

def decompose_bands(x):
    B, C, H, W = x.shape
    low_m, mid_m, high_m = get_freq_masks(H, W, x.device)
    dct_x = dct2d(x)
    return idct2d(dct_x * low_m), idct2d(dct_x * mid_m), idct2d(dct_x * high_m)

def freq_scheduled_loss(x_pred, x_target, progress):
    pred_l, pred_m, pred_h = decompose_bands(x_pred)
    tgt_l, tgt_m, tgt_h = decompose_bands(x_target)
    w_low = 3.0
    w_mid = 1.0 * min(1.0, progress * 2)
    w_high = 0.5 * max(0.0, (progress - 0.3) / 0.7)
    return (w_low * F.mse_loss(pred_l, tgt_l) +
            w_mid * F.mse_loss(pred_m, tgt_m) +
            w_high * F.mse_loss(pred_h, tgt_h))

def hf_coherence_loss(x_pred_high, x_target_high):
    B, C, H, W = x_pred_high.shape
    def autocorr_map(x):
        x_flat = x.reshape(B*C, 1, H, W)
        local_energy = F.avg_pool2d(x_flat**2, 3, stride=1, padding=1)
        local_mean = F.avg_pool2d(x_flat, 3, stride=1, padding=1)
        x_shift = F.pad(x_flat[:, :, :, :-1], (1, 0))
        local_cross = F.avg_pool2d(x_flat * x_shift, 3, stride=1, padding=1)
        var = (local_energy - local_mean**2).clamp(min=1e-8)
        cov = local_cross - local_mean * F.avg_pool2d(x_shift, 3, stride=1, padding=1)
        return (cov / var).reshape(B, C, H, W)
    return F.mse_loss(autocorr_map(x_pred_high), autocorr_map(x_target_high))

def hf_local_autocorrelation(x_high, shifts=[(0,1),(1,0),(1,1)]):
    B, C, H, W = x_high.shape
    x_flat = x_high.reshape(B*C, H, W)
    x_mean = x_flat.mean(dim=(1,2), keepdim=True)
    x_std = x_flat.std(dim=(1,2), keepdim=True).clamp(min=1e-8)
    x_norm = (x_flat - x_mean) / x_std
    correlations = []
    for dy, dx in shifts:
        a, b = x_norm, x_norm
        if dy > 0: a, b = a[:, dy:, :], b[:, :-dy, :]
        if dx > 0: a, b = a[:, :, dx:], b[:, :, :-dx]
        correlations.append((a*b).mean(dim=(1,2)).mean().item())
    return np.mean(correlations)


# ============================================================================
# METRICS
# ============================================================================

def per_band_energy_distance(images_gen, images_real, device='cpu'):
    def to_t(x):
        if isinstance(x, np.ndarray): x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 3: x = x.unsqueeze(1)
        return x.to(device)
    gen = to_t(images_gen[:200]); real = to_t(images_real[:200])
    dct_g = dct2d(gen); dct_r = dct2d(real)
    H, W = gen.shape[2], gen.shape[3]
    results = {}
    for name, mask in zip(['low','mid','high'], get_freq_masks(H, W, device)):
        e_g = (dct_g**2 * mask).mean(dim=(0,1)).sum().item()
        e_r = (dct_r**2 * mask).mean(dim=(0,1)).sum().item()
        results[f'energy_gap_{name}'] = abs(e_g - e_r) / (e_r + 1e-12)
    return results

def hf_coherence_metric(images, device='cpu'):
    if isinstance(images, np.ndarray): images = torch.tensor(images, dtype=torch.float32)
    if images.dim() == 3: images = images.unsqueeze(1)
    images = images[:200].to(device)
    _, _, x_high = decompose_bands(images)
    return hf_local_autocorrelation(x_high)

def hf_noise_index(images, device='cpu'):
    if isinstance(images, np.ndarray): images = torch.tensor(images, dtype=torch.float32)
    if images.dim() == 3: images = images.unsqueeze(1)
    images = images[:200].to(device)
    gx = images[:,:,:,1:] - images[:,:,:,:-1]
    gy = images[:,:,1:,:] - images[:,:,:-1,:]
    grad_energy = (gx**2).mean().item() + (gy**2).mean().item()
    _, _, x_high = decompose_bands(images)
    hf_energy = (x_high**2).mean().item() + 1e-12
    return grad_energy / hf_energy

def connectedness_proxy(images, threshold=0.3):
    if isinstance(images, torch.Tensor): images = images.cpu().numpy()
    if images.ndim == 4 and images.shape[1] == 3:
        images = 0.299*images[:,0] + 0.587*images[:,1] + 0.114*images[:,2]
    elif images.ndim == 4:
        images = images[:, 0]
    scores = []
    for img in images[:100]:
        binary = (img > threshold).astype(np.int32)
        total_fg = binary.sum()
        if total_fg < 5: scores.append(0.0); continue
        H, W = binary.shape
        visited = np.zeros_like(binary, dtype=bool); max_comp = 0
        for i in range(H):
            for j in range(W):
                if binary[i,j]==1 and not visited[i,j]:
                    stack=[(i,j)]; visited[i,j]=True; sz=0
                    while stack:
                        ci,cj=stack.pop(); sz+=1
                        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ni,nj=ci+di,cj+dj
                            if 0<=ni<H and 0<=nj<W and binary[ni,nj]==1 and not visited[ni,nj]:
                                visited[ni,nj]=True; stack.append((ni,nj))
                    max_comp = max(max_comp, sz)
        scores.append(max_comp / total_fg)
    return float(np.mean(scores))

def compute_diversity(z_samples, n_pairs=500):
    N = len(z_samples); z_flat = z_samples.reshape(N, -1); dists = []
    for _ in range(n_pairs):
        i,j = np.random.choice(N, 2, replace=False)
        dists.append((z_flat[i] != z_flat[j]).float().mean().item())
    return np.mean(dists), np.std(dists)

def compute_1nn_distance(gen_samples, train_samples, n_check=100):
    gen_flat = gen_samples[:n_check].reshape(n_check, -1)
    train_flat = train_samples.reshape(len(train_samples), -1); dists = []
    for i in range(n_check):
        d = ((train_flat - gen_flat[i:i+1])**2).sum(1); dists.append(d.min().item())
    return np.mean(dists), np.std(dists)

def token_histogram_kl(z_real, z_gen, n_bits=8):
    N_r, K, H, W = z_real.shape; N_g = z_gen.shape[0]; n_tokens = 2**K
    positions = [(i,j) for i in range(H) for j in range(W)]
    np.random.shuffle(positions); kls = []
    for i,j in positions[:50]:
        idx_r = torch.zeros(N_r, dtype=torch.long)
        idx_g = torch.zeros(N_g, dtype=torch.long)
        for b in range(K):
            idx_r += (z_real[:,b,i,j].long() << b)
            idx_g += (z_gen[:,b,i,j].long() << b)
        p = torch.bincount(idx_r, minlength=n_tokens).float()+1
        q = torch.bincount(idx_g, minlength=n_tokens).float()+1
        p /= p.sum(); q /= q.sum()
        kls.append((p*(p/q).log()).sum().item())
    return np.mean(kls)

def save_grid(images, path, nrow=8):
    try:
        from torchvision.utils import save_image
        if isinstance(images, np.ndarray): images = torch.tensor(images)
        if images.dim() == 3: images = images.unsqueeze(1)
        save_image(images[:64], path, nrow=nrow, normalize=False)
        print(f"    Grid: {path}")
    except Exception as e:
        print(f"    Grid save failed: {e}")


# ============================================================================
# CIFAR-10 ADC/DAC
# ============================================================================

class GumbelSigmoid(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__(); self.temperature = temperature
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

class CifarEncoder(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = self._res(64, 64)
        self.res2 = self._res(64, 64)
        self.head = nn.Conv2d(64, n_bits, 3, padding=1)
        self.quantizer = GumbelSigmoid()
    def _res(self, ic, oc):
        return nn.Sequential(nn.Conv2d(ic,oc,3,padding=1), nn.BatchNorm2d(oc), nn.ReLU(),
                             nn.Conv2d(oc,oc,3,padding=1), nn.BatchNorm2d(oc))
    def forward(self, x):
        h = self.stem(x); h = F.relu(h + self.res1(h)); h = F.relu(h + self.res2(h))
        logits = self.head(h)
        return self.quantizer(logits), logits
    def set_temperature(self, tau): self.quantizer.set_temperature(tau)

class CifarDecoder(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ConvTranspose2d(n_bits, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = self._res(64, 64)
        self.res2 = self._res(64, 64)
        self.head = nn.Sequential(nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid())
    def _res(self, ic, oc):
        return nn.Sequential(nn.Conv2d(ic,oc,3,padding=1), nn.BatchNorm2d(oc), nn.ReLU(),
                             nn.Conv2d(oc,oc,3,padding=1), nn.BatchNorm2d(oc))
    def forward(self, z):
        h = self.stem(z); h = F.relu(h + self.res1(h)); h = F.relu(h + self.res2(h))
        return self.head(h)


# ============================================================================
# E_CORE
# ============================================================================

class LocalEnergyCore(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__(); self.n_bits = n_bits
        self.predictor = nn.Sequential(
            nn.Linear(9*n_bits-1, 64), nn.ReLU(), nn.Linear(64, 1))
    def get_context(self, z, bi, i, j):
        B,K,H,W = z.shape; ctx = []
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                ni,nj = (i+di)%H, (j+dj)%W
                for b in range(K):
                    if di==0 and dj==0 and b==bi: continue
                    ctx.append(z[:,b,ni,nj])
        return torch.stack(ctx, dim=1)
    def violation_rate(self, z):
        B,K,H,W = z.shape; violations = []
        for _ in range(min(50, H*W*K)):
            b=torch.randint(K,(1,)).item()
            i=torch.randint(H,(1,)).item()
            j=torch.randint(W,(1,)).item()
            ctx = self.get_context(z, b, i, j)
            pred = (self.predictor(ctx).squeeze(1) > 0).float()
            violations.append((pred != z[:,b,i,j]).float().mean().item())
        return np.mean(violations)


# ============================================================================
# DENOISER
# ============================================================================

class FreqDenoiser(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__(); self.n_bits = n_bits
        self.net = nn.Sequential(
            nn.Conv2d(n_bits+1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, n_bits, 3, padding=1))
        self.skip = nn.Conv2d(n_bits, n_bits, 1)

    def forward(self, z_noisy, noise_level):
        B = z_noisy.shape[0]
        nl = noise_level.view(B,1,1,1).expand(-1,1,z_noisy.shape[2],z_noisy.shape[3])
        return self.net(torch.cat([z_noisy, nl], dim=1)) + self.skip(z_noisy)

    @torch.no_grad()
    def sample_standard(self, n, H, W, device, n_steps=15, temperature=0.7):
        K = self.n_bits; z = (torch.rand(n,K,H,W,device=device)>0.5).float()
        for step in range(n_steps):
            nl = torch.tensor([1.0-step/n_steps], device=device).expand(n)
            probs = torch.sigmoid(self(z, nl) / temperature)
            conf = (step+1)/n_steps
            mask = (torch.rand_like(z) < conf).float()
            z = mask * (torch.rand_like(z) < probs).float() + (1-mask) * z
        return (torch.sigmoid(self(z, torch.zeros(n, device=device))) > 0.5).float()

    @torch.no_grad()
    def sample_multiscale(self, n, H, W, device, n_steps=15, temperature=0.7):
        K = self.n_bits; z = (torch.rand(n,K,H,W,device=device)>0.5).float()
        for step in range(n_steps):
            nl = torch.tensor([1.0-step/n_steps], device=device).expand(n)
            logits = self(z, nl)
            progress = step/n_steps
            if progress < 0.5:
                t = temperature * 0.6; conf = 0.3 + 0.4*(progress/0.5)
            else:
                t = temperature * 1.2; conf = 0.7 + 0.3*((progress-0.5)/0.5)
            probs = torch.sigmoid(logits / t)
            mask = (torch.rand_like(z) < conf).float()
            z = mask * (torch.rand_like(z) < probs).float() + (1-mask) * z
        return (torch.sigmoid(self(z, torch.zeros(n, device=device))) > 0.5).float()

    @torch.no_grad()
    def sample_guided(self, n, H, W, decoder, device, n_steps=15, temperature=0.7):
        K = self.n_bits; z = (torch.rand(n,K,H,W,device=device)>0.5).float()
        for step in range(n_steps):
            nl = torch.tensor([1.0-step/n_steps], device=device).expand(n)
            progress = (step+1)/n_steps
            t = temperature * (0.6 if progress < 0.5 else 1.0)
            probs = torch.sigmoid(self(z, nl) / t)
            z_proposed = (torch.rand_like(z) < probs).float()
            if 0 < step < n_steps-1:
                x_cur = decoder(z); x_prop = decoder(z_proposed)
                _, _, h_cur = decompose_bands(x_cur)
                _, _, h_prop = decompose_bands(x_prop)
                hf_e_prop = F.avg_pool2d(h_prop.abs().mean(dim=1,keepdim=True), 3, 1, 1)
                hf_e_cur = F.avg_pool2d(h_cur.abs().mean(dim=1,keepdim=True), 3, 1, 1)
                hf_adv = (hf_e_prop - hf_e_cur).clamp(min=0)
                hf_max = hf_adv.amax(dim=(2,3), keepdim=True).clamp(min=1e-8)
                guidance = F.interpolate(hf_adv/hf_max, size=(H,W), mode='bilinear',
                                         align_corners=False).expand(-1,K,-1,-1)
                boost = 0.3 * max(0.0, (progress-0.3)/0.7)
                update_prob = (progress + boost * guidance).clamp(0, 1)
            else:
                update_prob = torch.full_like(z, progress)
            mask = (torch.rand_like(z) < update_prob).float()
            z = mask * z_proposed + (1-mask) * z
        return (torch.sigmoid(self(z, torch.zeros(n, device=device))) > 0.5).float()


def train_denoiser(denoiser, z_data, decoder, device, cfg, epochs=30, batch_size=32):
    opt = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    N = len(z_data)
    for epoch in range(epochs):
        denoiser.train(); perm = torch.randperm(N)
        tl, fl, cl, nb = 0., 0., 0., 0
        progress = epoch / max(epochs-1, 1)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]; z_clean = z_data[idx].to(device); B = z_clean.shape[0]
            noise_level = torch.rand(B, device=device)
            flip = (torch.rand_like(z_clean) < noise_level.view(B,1,1,1)).float()
            z_noisy = z_clean*(1-flip) + (1-z_clean)*flip
            opt.zero_grad()
            logits = denoiser(z_noisy, noise_level)
            loss_bce = F.binary_cross_entropy_with_logits(logits, z_clean)
            loss_freq = loss_coh = torch.tensor(0.0, device=device)
            if cfg.get('use_freq_amp') or cfg.get('use_freq_coh'):
                s = torch.sigmoid(logits); h = (s>0.5).float()
                z_pred = h - s.detach() + s
                with torch.no_grad(): x_clean = decoder(z_clean)
                x_pred = decoder(z_pred)
                if cfg.get('use_freq_amp'):
                    if cfg.get('use_schedule'):
                        loss_freq = freq_scheduled_loss(x_pred, x_clean, progress)
                    else:
                        pl,pm,ph = decompose_bands(x_pred); tl_,tm_,th_ = decompose_bands(x_clean)
                        loss_freq = 3*F.mse_loss(pl,tl_)+F.mse_loss(pm,tm_)+0.3*F.mse_loss(ph,th_)
                if cfg.get('use_freq_coh'):
                    _,_,pred_h = decompose_bands(x_pred); _,_,tgt_h = decompose_bands(x_clean)
                    loss_coh = hf_coherence_loss(pred_h, tgt_h)
            loss = loss_bce + cfg.get('lam_freq',0.3)*loss_freq + cfg.get('lam_coh',0.1)*loss_coh
            loss.backward(); opt.step()
            tl += loss_bce.item(); fl += loss_freq.item(); cl += loss_coh.item(); nb += 1
        if (epoch+1) % 10 == 0:
            msg = f"    epoch {epoch+1}/{epochs}: BCE={tl/nb:.4f}"
            if cfg.get('use_freq_amp'): msg += f" freq={fl/nb:.4f}"
            if cfg.get('use_freq_coh'): msg += f" coh={cl/nb:.4f}"
            print(msg)


# ============================================================================
# CONFIGS
# ============================================================================

CONFIGS = OrderedDict([
    ("baseline", {"use_freq_amp":False, "use_freq_coh":False, "use_schedule":False,
                  "lam_freq":0, "lam_coh":0, "sampler":"standard"}),
    ("freq_amp", {"use_freq_amp":True, "use_freq_coh":False, "use_schedule":False,
                  "lam_freq":0.3, "lam_coh":0, "sampler":"standard"}),
    ("freq_sched_coh", {"use_freq_amp":True, "use_freq_coh":True, "use_schedule":True,
                        "lam_freq":0.3, "lam_coh":0.1, "sampler":"standard"}),
    ("freq_full", {"use_freq_amp":True, "use_freq_coh":True, "use_schedule":True,
                   "lam_freq":0.3, "lam_coh":0.1, "sampler":"guided"}),
    ("freq_full_ms", {"use_freq_amp":True, "use_freq_coh":True, "use_schedule":True,
                      "lam_freq":0.3, "lam_coh":0.1, "sampler":"multiscale"}),
])


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_gen_cifar10_a2')
    parser.add_argument('--n_bits', type=int, default=8)
    parser.add_argument('--n_samples', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print(f"CIFAR-10 A2: STRUCTURED HF GENERATION — 5-run matrix")
    print("=" * 100)

    # [1] Load
    print("\n[1] Loading CIFAR-10...")
    from torchvision import datasets, transforms
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(args.seed)
    train_x = torch.stack([train_ds[i][0] for i in rng.choice(len(train_ds), 3000, replace=False)])
    test_x = torch.stack([test_ds[i][0] for i in rng.choice(len(test_ds), 500, replace=False)])
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")

    # [2] Shared ADC/DAC
    print(f"\n[2] Training ADC/DAC (32×32×3 → 16×16×{args.n_bits})...")
    encoder = CifarEncoder(args.n_bits).to(device)
    decoder = CifarDecoder(args.n_bits).to(device)
    opt = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=1e-3)
    for epoch in range(40):
        encoder.train(); decoder.train()
        encoder.set_temperature(1.0 + (0.3-1.0)*epoch/39)
        perm = torch.randperm(len(train_x)); tl, nb = 0., 0
        for i in range(0, len(train_x), 32):
            idx = perm[i:i+32]; x = train_x[idx].to(device)
            opt.zero_grad(); z, _ = encoder(x); x_hat = decoder(z)
            loss = F.mse_loss(x_hat, x) + 0.5*F.binary_cross_entropy(x_hat, x)
            loss.backward(); opt.step(); tl += loss.item(); nb += 1
        if (epoch+1) % 10 == 0: print(f"    epoch {epoch+1}/40: loss={tl/nb:.4f}")
    encoder.eval(); decoder.eval()

    with torch.no_grad():
        tb = test_x[:64].to(device); zo, _ = encoder(tb)
        print(f"    Oracle MSE: {F.mse_loss(decoder(zo), tb).item():.4f}")
    save_grid(decoder(zo).cpu(), os.path.join(args.output_dir, 'oracle_recon.png'))
    save_grid(test_x[:64], os.path.join(args.output_dir, 'real_samples.png'))

    # [3] Encode
    print("\n[3] Encoding training set...")
    z_data = []
    with torch.no_grad():
        for i in range(0, len(train_x), 32):
            z, _ = encoder(train_x[i:i+32].to(device)); z_data.append(z.cpu())
    z_data = torch.cat(z_data); K, H, W = z_data.shape[1:]
    print(f"    z_data: {z_data.shape}, bit usage: {z_data.mean():.3f}")

    # [4] E_core
    print("\n[4] Training E_core...")
    e_core = LocalEnergyCore(args.n_bits).to(device)
    eopt = torch.optim.Adam(e_core.parameters(), lr=1e-3)
    for ep in range(10):
        e_core.train(); perm = torch.randperm(len(z_data))
        for i in range(0, len(z_data), 64):
            z = z_data[perm[i:i+64]].to(device); eopt.zero_grad(); tl_ = 0.
            for _ in range(20):
                b=torch.randint(K,(1,)).item(); ii=torch.randint(H,(1,)).item(); jj=torch.randint(W,(1,)).item()
                ctx = e_core.get_context(z,b,ii,jj)
                tl_ += F.binary_cross_entropy_with_logits(e_core.predictor(ctx).squeeze(1), z[:,b,ii,jj])
            (tl_/20).backward(); eopt.step()
    e_core.eval()

    # [5] Reference
    print("\n[5] Reference metrics...")
    real_hf_coh = hf_coherence_metric(test_x[:200], device)
    real_hf_noi = hf_noise_index(test_x[:200], device)
    real_conn = connectedness_proxy(test_x[:100].numpy())
    print(f"    HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  conn={real_conn:.4f}")

    # [6] Run configs
    print("\n" + "=" * 100)
    print("RUNNING 5-CONFIG EXPERIMENT MATRIX")
    print("=" * 100)

    all_results = []
    train_x_np = train_x.numpy().reshape(len(train_x), -1)

    for name, cfg in CONFIGS.items():
        print(f"\n{'='*80}\nCONFIG: {name} | sampler={cfg['sampler']}\n{'='*80}")
        torch.manual_seed(args.seed + hash(name) % 10000)

        denoiser = FreqDenoiser(args.n_bits).to(device)
        print("  Training denoiser...")
        train_denoiser(denoiser, z_data, decoder, device, cfg, epochs=30, batch_size=32)
        denoiser.eval()

        print(f"  Sampling {args.n_samples}...")
        torch.manual_seed(args.seed)
        if cfg['sampler'] == 'standard':
            z_gen = denoiser.sample_standard(args.n_samples, H, W, device)
        elif cfg['sampler'] == 'multiscale':
            z_gen = denoiser.sample_multiscale(args.n_samples, H, W, device)
        elif cfg['sampler'] == 'guided':
            z_gen = denoiser.sample_guided(args.n_samples, H, W, decoder, device)

        with torch.no_grad(): x_gen = decoder(z_gen.to(device)).cpu()
        save_grid(x_gen, os.path.join(args.output_dir, f'gen_{name}.png'))

        # Evaluate
        z_cpu = z_gen.cpu()
        viol = e_core.violation_rate(z_cpu[:100].to(device))
        tok_kl = token_histogram_kl(z_data[:500], z_cpu[:min(500,len(z_cpu))], args.n_bits)
        div_m, _ = compute_diversity(z_cpu)
        with torch.no_grad():
            zc = z_cpu[:100].to(device); xc = decoder(zc); zcy, _ = encoder(xc)
            cycle = (zc != zcy).float().mean().item()
        x_np = x_gen.numpy().reshape(len(x_gen), -1)
        nn_m, _ = compute_1nn_distance(x_np, train_x_np)
        conn = connectedness_proxy(x_gen[:100])
        band = per_band_energy_distance(x_gen[:200], test_x[:200], device)
        hfc = hf_coherence_metric(x_gen[:200], device)
        hfn = hf_noise_index(x_gen[:200], device)

        r = {'config':name, 'violation':viol, 'token_kl':tok_kl, 'diversity':div_m,
             'cycle':cycle, 'nn_dist':nn_m, 'connectedness':conn,
             'hf_coherence':hfc, 'hf_noise_index':hfn,
             'energy_gap_low':band['energy_gap_low'],
             'energy_gap_mid':band['energy_gap_mid'],
             'energy_gap_high':band['energy_gap_high']}
        all_results.append(r)

        print(f"    viol={viol:.4f} div={div_m:.4f} cycle={cycle:.4f} conn={conn:.4f}")
        print(f"    HF_coh={hfc:.4f}(real={real_hf_coh:.4f}) HF_noi={hfn:.2f}(real={real_hf_noi:.2f})")
        print(f"    E_gap: L={band['energy_gap_low']:.4f} M={band['energy_gap_mid']:.4f} H={band['energy_gap_high']:.4f}")

    # Summary
    print("\n" + "=" * 100)
    print(f"CIFAR-10 A2 SUMMARY")
    print(f"Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  conn={real_conn:.4f}")
    print("=" * 100)
    h = f"{'config':<18} {'viol':>7} {'div':>7} {'cycle':>7} {'conn':>7} {'HF_coh':>7} {'HF_noi':>7} {'Eg_L':>7} {'Eg_M':>7} {'Eg_H':>7}"
    print(h); print("-"*len(h))
    for r in all_results:
        print(f"{r['config']:<18} {r['violation']:>7.4f} {r['diversity']:>7.4f} "
              f"{r['cycle']:>7.4f} {r['connectedness']:>7.4f} {r['hf_coherence']:>7.4f} "
              f"{r['hf_noise_index']:>7.2f} {r['energy_gap_low']:>7.4f} "
              f"{r['energy_gap_mid']:>7.4f} {r['energy_gap_high']:>7.4f}")

    # Gate
    print("\nGATE CHECK (vs baseline):")
    bl = all_results[0]
    for r in all_results[1:]:
        vd = (r['violation']-bl['violation'])/(bl['violation']+1e-8)*100
        g1 = "PASS" if vd < 20 else "FAIL"
        g2 = "PASS" if r['cycle']-bl['cycle'] <= 0.01 else "FAIL"
        ci = abs(r['hf_coherence']-real_hf_coh) < abs(bl['hf_coherence']-real_hf_coh)
        ni = abs(r['hf_noise_index']-real_hf_noi) < abs(bl['hf_noise_index']-real_hf_noi)
        fv = "BETTER" if ci and ni else ("MIXED" if ci or ni else "WORSE")
        print(f"  {r['config']:<18} viol[{g1}] cycle[{g2}] "
              f"div={r['diversity']-bl['diversity']:+.4f} freq:{fv}")

    csv_path = os.path.join(args.output_dir, "cifar10_a2_results.csv")
    with open(csv_path, 'w', newline='') as f:
        keys = sorted(all_results[0].keys())
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nCSV: {csv_path}")
    print("\n" + "=" * 100)
    print("CIFAR-10 A2 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
