#!/usr/bin/env python3
"""
Phase 7: High-Resolution Protocol Observation
===============================================
GPT diagnosis: Phase 4's token/freq failure was due to resolution mismatch.
At 7×7 latent (4×4 pixel patches), partial occlusion info lost in z-space.

Fix: Use 14×14 latent grid (2×2 pixel patches) so token/freq E_obs
can distinguish "partially occluded" vs "fully occluded" patches.

Compare pixel_bce vs token_bce vs freq_dct at both 7×7 and 14×14.

Success criterion: token/freq stripes total no longer negative at 14×14.

Usage:
    python3 -u benchmarks/exp_phase7_hires_protocol.py --device cuda
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


# ============================================================================
# MODEL COMPONENTS (parameterized by latent_size)
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

class BinaryEncoder7(nn.Module):
    """28→7: stride=2 twice."""
    def __init__(self, n_bits=8, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1))
    def forward(self, x): return self.conv(x)

class BinaryDecoder7(nn.Module):
    """7→28: stride=2 transpose twice."""
    def __init__(self, n_bits=8, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1), nn.Sigmoid())
    def forward(self, z): return self.deconv(z)

class BinaryEncoder14(nn.Module):
    """28→14: stride=2 once, then stride=1."""
    def __init__(self, n_bits=8, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1))
    def forward(self, x): return self.conv(x)

class BinaryDecoder14(nn.Module):
    """14→28: stride=2 transpose once."""
    def __init__(self, n_bits=8, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1), nn.Sigmoid())
    def forward(self, z): return self.deconv(z)

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
    def __init__(self, n_bits=8, latent_size=7, hidden_dim=64, tau=1.0):
        super().__init__()
        self.n_bits = n_bits; self.latent_size = latent_size
        if latent_size == 7:
            self.encoder = BinaryEncoder7(n_bits, hidden_dim)
            self.decoder = BinaryDecoder7(n_bits, hidden_dim)
        elif latent_size == 14:
            self.encoder = BinaryEncoder14(n_bits, hidden_dim)
            self.decoder = BinaryDecoder14(n_bits, hidden_dim)
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
    name = 'pixel_bce'
    def residual_per_patch(self, model, z, x_clean, pixel_mask, device):
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
    name = 'token_bce'
    def residual_per_patch(self, model, z, x_clean, pixel_mask, device):
        ls = model.latent_size
        with torch.no_grad():
            x_t = torch.from_numpy(x_clean[None, None].astype(np.float32)).to(device)
            z_clean = model.encode(x_t)[0]
        bm = pixel_to_bit_mask(pixel_mask, model.n_bits, ls)
        resid = torch.zeros(ls, ls, device=device)
        for i in range(ls):
            for j in range(ls):
                if not bm[0, i, j]:
                    mismatch = (z[:, i, j] != z_clean[:, i, j]).float().mean()
                    resid[i, j] = mismatch.item()
                else:
                    resid[i, j] = float('inf')
        return resid

class FreqObsProtocol:
    name = 'freq_dct'
    def residual_per_patch(self, model, z, x_clean, pixel_mask, device):
        ls = model.latent_size
        with torch.no_grad():
            x_t = torch.from_numpy(x_clean[None, None].astype(np.float32)).to(device)
            z_clean = model.encode(x_t)[0]
        # Per-channel DCT comparison
        z_dct = torch.fft.rfft2(z.unsqueeze(0).float())[0]
        zc_dct = torch.fft.rfft2(z_clean.unsqueeze(0).float())[0]
        # Spectral difference per channel
        spec_diff = (z_dct - zc_dct).abs()  # (k, H, W//2+1)

        bm = pixel_to_bit_mask(pixel_mask, model.n_bits, ls)
        resid = torch.zeros(ls, ls, device=device)
        for i in range(ls):
            for j in range(ls):
                if not bm[0, i, j]:
                    # Local spectral contribution: use spatial mismatch weighted by freq energy
                    mismatch = (z[:, i, j] != z_clean[:, i, j]).float().mean()
                    # Add spectral context: average spectral diff at this row/col
                    freq_i = spec_diff[:, i, :].mean().item() if j < spec_diff.shape[2] else 0
                    resid[i, j] = mismatch.item() + 0.1 * freq_i
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
# TRAINING
# ============================================================================

def train_model(train_x, train_y, device, latent_size=7, epochs=5, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = RouteCModel(latent_size=latent_size).to(device)
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
    return model

def train_inpaint(model, train_x, device, epochs=20):
    net = InpaintNet(k=model.n_bits).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    model.eval(); N = len(train_x); rng = np.random.default_rng(42)
    ls = model.latent_size
    for epoch in range(epochs):
        net.train(); perm = torch.randperm(N); tl,nb = 0.,0
        for i in range(0, N, 64):
            idx = perm[i:i+64]; x = train_x[idx].to(device)
            with torch.no_grad(): z = model.encode(x); z_hard = (z > 0.5).float()
            B,k,H,W = z_hard.shape
            masks = [torch.from_numpy(pixel_to_bit_mask(
                sample_training_mask(28,28,rng), k, ls)).float() for _ in range(B)]
            bit_masks = torch.stack(masks).to(device)
            z_masked = z_hard * (1 - bit_masks)
            logits = net(z_masked, bit_masks[:, 0:1])
            loss = F.binary_cross_entropy_with_logits(logits[bit_masks.bool()], z_hard[bit_masks.bool()])
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); nb += 1
        if (epoch+1) % 10 == 0:
            print(f"        epoch {epoch+1}/{epochs}: loss={tl/max(nb,1):.4f}")
    return net


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_protocol(protocol, model, net, mask_type, test_x, test_y, device,
                      n_samples=100, seed=42):
    model.eval(); net.eval()
    ls = model.latent_size
    pm = make_center_mask() if mask_type == 'center' else make_stripe_mask()

    rng = np.random.default_rng(seed + 100)
    eval_idx = rng.choice(len(test_x), min(n_samples, len(test_x)), replace=False)

    cb, ca, rts = [], [], []
    for idx in eval_idx:
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()
        x_occ = x_clean * pm

        with torch.no_grad():
            x_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_init = model.encode(x_t)[0]
            pred_b = model.classifier(z_init.unsqueeze(0)).argmax(1).item()

        bit_mask = pixel_to_bit_mask(pm, model.n_bits, ls)

        # Full repair
        bm = torch.from_numpy(bit_mask).float().to(device)
        mask = bm.max(dim=0, keepdim=True)[0].unsqueeze(0)
        z_masked = z_init.unsqueeze(0) * (1 - bm.unsqueeze(0))
        with torch.no_grad():
            logits = net(z_masked, mask)
            preds = (torch.sigmoid(logits) > 0.5).float()
        bm_bool = torch.from_numpy(bit_mask).to(device)
        z_repaired = z_init.clone()
        z_repaired[bm_bool] = preds[0][bm_bool]

        # Protocol-based selective repair
        t0 = time.time()
        resid = protocol.residual_per_patch(model, z_init, x_clean, pm, device)

        z_result = z_init.clone()
        for i in range(ls):
            for j in range(ls):
                if bit_mask[0, i, j]:
                    r = resid[i, j].item()
                    if np.isinf(r) or r > 0.5:
                        z_result[:, i, j] = z_repaired[:, i, j]
        rt = (time.time() - t0) * 1000

        with torch.no_grad():
            pred_a = model.classifier(z_result.unsqueeze(0)).argmax(1).item()
        cb.append(int(pred_b == label)); ca.append(int(pred_a == label))
        rts.append(rt)

    n = len(eval_idx)
    return {
        'delta_acc': (np.sum(ca) - np.sum(cb)) / n,
        'acc_before': np.mean(cb), 'acc_after': np.mean(ca),
        'runtime_ms': np.mean(rts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--output_dir', default='outputs/exp_phase7')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 100)
    print("PHASE 7: HIGH-RESOLUTION PROTOCOL OBSERVATION")
    print("=" * 100)

    print("[1] Loading FashionMNIST...")
    train_x, train_y, test_x, test_y = load_dataset('fmnist', 2000, 500, args.seed)

    protocols = [PixelObsProtocol(), TokenObsProtocol(), FreqObsProtocol()]
    mask_types = ['center', 'stripes']
    all_results = []

    for ls in [7, 14]:
        print(f"\n{'='*80}")
        print(f"GRID SIZE: {ls}×{ls} (patch = {28//ls}×{28//ls} pixels)")
        print(f"{'='*80}")

        print(f"[2] Training model (latent_size={ls})...")
        model = train_model(train_x, train_y, device, latent_size=ls, seed=args.seed)

        print(f"[3] Training InpaintNet...")
        net = train_inpaint(model, train_x, device)

        for proto in protocols:
            for mt in mask_types:
                r = evaluate_protocol(proto, model, net, mt, test_x, test_y, device,
                                     n_samples=args.eval_samples, seed=args.seed)
                r.update({'protocol': proto.name, 'mask_type': mt, 'grid': f'{ls}x{ls}'})
                all_results.append(r)
                print(f"  {proto.name:<12} {mt:<10} Δacc={r['delta_acc']:>+7.1%} "
                      f"  {r['runtime_ms']:>6.1f}ms")

    # Summary comparison
    print("\n" + "=" * 100)
    print("RESOLUTION COMPARISON: 7×7 vs 14×14")
    print("=" * 100)
    print(f"{'grid':<8} {'protocol':<12} {'center':>8} {'stripes':>8} {'total':>8}")
    print("-" * 50)

    for ls in ['7x7', '14x14']:
        for proto in protocols:
            pr = [r for r in all_results if r['grid'] == ls and r['protocol'] == proto.name]
            center = [r for r in pr if r['mask_type'] == 'center']
            stripes = [r for r in pr if r['mask_type'] == 'stripes']
            total = center[0]['delta_acc'] + stripes[0]['delta_acc']
            print(f"{ls:<8} {proto.name:<12} {center[0]['delta_acc']:>+8.1%} "
                  f"{stripes[0]['delta_acc']:>+8.1%} {total:>+8.1%}")

    # Key verdict
    print("\n" + "=" * 80)
    print("PHASE 7 VERDICT: Does higher resolution fix token/freq?")
    print("=" * 80)

    for proto in protocols:
        r7 = [r for r in all_results if r['grid'] == '7x7' and r['protocol'] == proto.name]
        r14 = [r for r in all_results if r['grid'] == '14x14' and r['protocol'] == proto.name]
        t7 = sum(r['delta_acc'] for r in r7)
        t14 = sum(r['delta_acc'] for r in r14)
        stripe7 = [r for r in r7 if r['mask_type'] == 'stripes'][0]['delta_acc']
        stripe14 = [r for r in r14 if r['mask_type'] == 'stripes'][0]['delta_acc']
        improved = stripe14 > stripe7
        print(f"  {proto.name:<12}: 7×7 total={t7:+.1%} → 14×14 total={t14:+.1%} "
              f"  stripes: {stripe7:+.1%}→{stripe14:+.1%} "
              f"{'✓ IMPROVED' if improved else '✗ NO CHANGE'}")

    csv_path = os.path.join(args.output_dir, "phase7_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            keys = sorted(set().union(*(r.keys() for r in all_results)))
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in all_results: w.writerow(r)
    print(f"\nCSV saved to {csv_path}")
    print("\n" + "=" * 100)
    print("Phase 7 experiment complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
