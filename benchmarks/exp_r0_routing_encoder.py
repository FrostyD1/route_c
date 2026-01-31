#!/usr/bin/env python3
"""
R0: Routing vs Encoder Capacity — Causal Separation
=====================================================
Core question: Is the performance ceiling due to (a) missing content-dependent
routing in the flow operator, or (b) encoder capacity bottleneck, or (c) both?

Prior evidence against routing:
  - GDA (content-dependent Hamming routing) showed 0% gap at 7×7 and 14×14
  - FGO only triggers at 28×28 with distributed occlusion
  - Same-color-tone was fixed by statistical priors (E2a/E2b), not routing
  - Classification ceiling was explained by encoder capacity (C2: ResBlock +6.4%)

Why test again: Current regime differs from GDA era:
  - Flow-based iterative operator (not one-shot amortized inference)
  - 16×16×16 grid (256 tokens, larger than 7×7=49 or 14×14=196 in z-space)
  - Energy hinge training allows routing to be amplified by energy descent

Design: 5 configs in 2 encoder groups

  Group 1 (shared Encoder16 + E_core):
    A: FlatStepFn_Norm (baseline)
    B: FlatStepFn_Norm + content-sparse routing (K=4)
    C: FlatStepFn_Norm + random-sparse routing (K=4, control)

  Group 2 (shared EncoderDeep16 + E_core):
    D: FlatStepFn_Norm (stronger encoder only)
    E: FlatStepFn_Norm + content-sparse routing (K=4, interaction)

Causal interpretation:
  B > C > A → content routing matters
  D > A → encoder capacity matters
  E > D and B ≈ A → routing only useful with strong encoder (interaction)
  D >> B → encoder is the bottleneck, not routing

Three-mode evaluation (same z, same operator):
  Repair: ham_unmasked (hard gate), ham_masked, cycle_repair
  Generation: violation, diversity, HF_noise, conn, cycle, energy descent
  Classification: conv probe, mixed training (clean + repair)

4GB GPU: 3000 train, 500 test, batch_size=32, 16×16×16 z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, argparse, json
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from exp_flow_f0 import (
    dct2d, idct2d, get_freq_masks, decompose_bands, freq_scheduled_loss,
    per_band_energy_distance, hf_coherence_metric, hf_noise_index,
    connectedness_proxy, compute_diversity, save_grid,
    GumbelSigmoid, DiffEnergyCore,
    quantize, compute_e_core_grad,
    evaluate
)

from exp_flow_f0c_fixes import FlatStepFn_Norm, get_sigma

from exp_g2_protocol_density import (
    Encoder16, Decoder16,
    train_adc, encode_all, train_ecore, train_step_fn
)

from exp_e2a_global_prior import compute_hue_var


# ============================================================================
# SPARSE ROUTING MODULE
# ============================================================================

class SparseRouting(nn.Module):
    """Content-dependent or random sparse routing between spatial positions.

    For N=256 (16×16), computes full pairwise similarity and takes top-K.
    At scale, would use LSH for O(N) approximate top-K.

    Modes:
      'content': top-K neighbors by learned cosine similarity (the hypothesis)
      'random':  K random neighbors, uniform weight (control for "just adding edges")
      'identity': no routing, pure passthrough (control for "just adding parameters")
    """
    def __init__(self, channels, K=4, mode='content'):
        super().__init__()
        self.K = K
        self.mode = mode
        # Init scale at 0 → starts as identity, routing gradually activates
        self.scale = nn.Parameter(torch.zeros(1))

        if mode == 'content':
            d = max(16, channels // 4)
            self.proj_q = nn.Conv2d(channels, d, 1)
            self.proj_k = nn.Conv2d(channels, d, 1)
            self.proj_v = nn.Conv2d(channels, channels, 1)
        elif mode == 'random':
            self.proj_v = nn.Conv2d(channels, channels, 1)
        # 'identity' has no parameters beyond scale

    def forward(self, x):
        if self.mode == 'identity':
            return x

        B, C, H, W = x.shape
        N = H * W

        if self.mode == 'content':
            q = self.proj_q(x).reshape(B, -1, N)  # [B, d, N]
            k = self.proj_k(x).reshape(B, -1, N)  # [B, d, N]
            v = self.proj_v(x).reshape(B, C, N)    # [B, C, N]

            d = q.shape[1]
            sim = torch.bmm(q.transpose(1, 2), k) / (d ** 0.5)  # [B, N, N]

            # Mask self-attention (don't attend to self)
            sim = sim - 1e9 * torch.eye(N, device=x.device).unsqueeze(0)

            # Top-K selection
            topk_val, topk_idx = sim.topk(self.K, dim=-1)  # [B, N, K]
            weights = F.softmax(topk_val, dim=-1)           # [B, N, K]

            # Memory-efficient gather: flatten K into N dimension
            v_t = v.transpose(1, 2)  # [B, N, C]
            idx_flat = topk_idx.reshape(B, N * self.K)                    # [B, N*K]
            idx_gather = idx_flat.unsqueeze(-1).expand(-1, -1, C)         # [B, N*K, C]
            gathered = torch.gather(v_t, 1, idx_gather)                   # [B, N*K, C]
            gathered = gathered.reshape(B, N, self.K, C)                  # [B, N, K, C]

            out = (weights.unsqueeze(-1) * gathered).sum(dim=2)  # [B, N, C]
            out = out.transpose(1, 2).reshape(B, C, H, W)

        elif self.mode == 'random':
            v = self.proj_v(x).reshape(B, C, N)
            v_t = v.transpose(1, 2)  # [B, N, C]

            rand_idx = torch.randint(0, N, (B, N, self.K), device=x.device)
            idx_flat = rand_idx.reshape(B, N * self.K)
            idx_gather = idx_flat.unsqueeze(-1).expand(-1, -1, C)
            gathered = torch.gather(v_t, 1, idx_gather)
            gathered = gathered.reshape(B, N, self.K, C)

            out = gathered.mean(dim=2)  # [B, N, C]
            out = out.transpose(1, 2).reshape(B, C, H, W)

        return x + self.scale * out


# ============================================================================
# STEP FUNCTION WITH ROUTING
# ============================================================================

class FlatStepFn_Routing(nn.Module):
    """FlatStepFn_Norm + SparseRouting inserted between conv2 and conv3.

    Architecture: conv1 → conv2 → [routing] → conv3 → GroupNorm → out + skip
    """
    def __init__(self, n_bits, routing_mode='content', K=4):
        super().__init__()
        self.n_bits = n_bits
        hid = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_bits * 2 + 1, hid, 3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU())
        self.routing = SparseRouting(hid, K=K, mode=routing_mode)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU())
        self.norm = nn.GroupNorm(8, hid)
        self.out = nn.Conv2d(hid, n_bits, 3, padding=1)
        self.skip = nn.Conv2d(n_bits, n_bits, 1)

    def forward(self, u, e_grad, t_scalar):
        B = u.shape[0]
        t_map = t_scalar.view(B, 1, 1, 1).expand(-1, 1, u.shape[2], u.shape[3])
        inp = torch.cat([u, e_grad, t_map], dim=1)
        h = self.conv1(inp)
        h = self.conv2(h)
        h = self.routing(h)
        h = self.conv3(h)
        h = self.norm(h)
        return self.out(h) + self.skip(u)


# ============================================================================
# DEEPER ENCODER (more ResBlocks + wider channels)
# ============================================================================

class EncoderDeep16(nn.Module):
    """Deeper 16×16 encoder: 4 ResBlocks (vs 2) with channel expansion to 128.

    32×32×3 → stride-2 → 16×16×64 → ResBlock×2 → expand 128 → ResBlock×2 → head
    """
    def __init__(self, n_bits=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = self._res(64, 64)
        self.res2 = self._res(64, 64)
        self.expand = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.res3 = self._res(128, 128)
        self.res4 = self._res(128, 128)
        self.head = nn.Conv2d(128, n_bits, 3, padding=1)
        self.q = GumbelSigmoid()

    def _res(self, ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(),
            nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc))

    def forward(self, x):
        h = self.stem(x)
        h = F.relu(h + self.res1(h))
        h = F.relu(h + self.res2(h))
        h = self.expand(h)
        h = F.relu(h + self.res3(h))
        h = F.relu(h + self.res4(h))
        return self.q(self.head(h)), self.head(h)

    def set_temperature(self, tau):
        self.q.set_temperature(tau)


# ============================================================================
# MASKS
# ============================================================================

def make_center_mask(B, K, H, W, device='cpu'):
    mask = torch.ones(B, K, H, W, device=device)
    h4, w4 = H // 4, W // 4
    mask[:, :, h4:3*h4, w4:3*w4] = 0
    return mask


# ============================================================================
# CONV PROBE (mixed training)
# ============================================================================

class ConvProbe(nn.Module):
    def __init__(self, n_bits, H, W, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bits, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, n_classes)
        )

    def forward(self, z):
        return self.net(z.float())


def train_probe_mixed(probe, z_clean, z_repair, labels, device, epochs=30, bs=128):
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    N = len(z_clean)
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            use_repair = torch.rand(len(idx)) < 0.5
            z_batch = torch.where(
                use_repair.view(-1, 1, 1, 1),
                z_repair[idx], z_clean[idx]
            ).to(device)
            y = labels[idx].to(device)
            loss = F.cross_entropy(probe(z_batch), y)
            opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def eval_probe(probe, z, labels, device, bs=128):
    correct = total = 0
    for i in range(0, len(z), bs):
        z_b = z[i:i+bs].to(device)
        y_b = labels[i:i+bs].to(device)
        correct += (probe(z_b).argmax(1) == y_b).sum().item()
        total += len(y_b)
    return correct / total


# ============================================================================
# REPAIR WITH FLOW
# ============================================================================

@torch.no_grad()
def repair_flow(step_fn, e_core, z_clean, mask, device, T=10, dt=0.5):
    B, K, H, W = z_clean.shape
    u_evidence = z_clean * 2.0 - 1.0
    u = u_evidence * mask + torch.randn_like(u_evidence) * 0.5 * (1 - mask)

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((B,), t_frac, device=device)
        e_grad = compute_e_core_grad(e_core, u)
        delta_u = step_fn(u, e_grad, t_tensor)
        u = u + dt * delta_u
        u = u_evidence * mask + u * (1 - mask)  # clamp evidence

    return mask * z_clean + (1 - mask) * quantize(u)


# ============================================================================
# GENERATION WITH FLOW
# ============================================================================

@torch.no_grad()
def sample_flow_gen(step_fn, e_core, n, K, H, W, device,
                    T=20, dt=0.5, sigma_schedule='cosine'):
    u = torch.randn(n, K, H, W, device=device) * 0.5
    trajectory = {'e_core': [], 'delta_u_norm': []}

    for step in range(T):
        t_frac = 1.0 - step / T
        t_tensor = torch.full((n,), t_frac, device=device)
        e_grad = compute_e_core_grad(e_core, u)
        delta_u = step_fn(u, e_grad, t_tensor)
        trajectory['delta_u_norm'].append(delta_u.abs().mean().item())
        u = u + dt * delta_u

        sigma = get_sigma(sigma_schedule, step, T)
        if sigma > 0.01:
            u = u + sigma * torch.randn_like(u)

        z_cur = quantize(u)
        trajectory['e_core'].append(e_core.energy(z_cur).item())

    return quantize(u), trajectory


# ============================================================================
# EVALUATION: ALL THREE MODES
# ============================================================================

def eval_all_modes(step_fn, e_core, enc, dec, z_train, z_test,
                   train_x, test_x, train_y, test_y,
                   real_hf_coh, real_hf_noi, device,
                   T_repair=10, T_gen=20, n_gen=256, bs=32):
    """Evaluate one operator across repair, generation, classification."""
    K, H, W = z_train.shape[1:]
    result = {}

    # ===== REPAIR =====
    z_test_rep_list = []
    with torch.no_grad():
        for i in range(0, len(z_test), bs):
            z_b = z_test[i:i+bs].to(device)
            B = z_b.shape[0]
            mask = make_center_mask(B, K, H, W, device)
            z_rep = repair_flow(step_fn, e_core, z_b, mask, device, T=T_repair)
            z_test_rep_list.append(z_rep.cpu())
    z_test_repaired = torch.cat(z_test_rep_list)

    # Contract metrics
    mask_full = make_center_mask(len(z_test), K, H, W)
    diff = (z_test != z_test_repaired).float()
    ham_unmasked = (diff * mask_full).sum() / max(mask_full.sum().item(), 1)
    ham_masked = (diff * (1 - mask_full)).sum() / max((1 - mask_full).sum().item(), 1)

    # Cycle on repaired
    cycle_list = []
    with torch.no_grad():
        for i in range(0, min(200, len(z_test_repaired)), bs):
            zr = z_test_repaired[i:i+bs].to(device)
            xr = dec(zr); zcy, _ = enc(xr)
            cycle_list.append((zr != zcy).float().mean().item())
    cycle_repair = np.mean(cycle_list)

    result['repair'] = {
        'ham_unmasked': ham_unmasked.item(),
        'ham_masked': ham_masked.item(),
        'cycle_repair': cycle_repair,
    }

    # ===== GENERATION =====
    torch.manual_seed(142)
    z_gen_list, all_traj = [], []
    for gi in range(0, n_gen, bs):
        nb = min(bs, n_gen - gi)
        z_batch, traj = sample_flow_gen(
            step_fn, e_core, nb, K, H, W, device,
            T=T_gen, dt=0.5, sigma_schedule='cosine')
        z_gen_list.append(z_batch.cpu())
        all_traj.append(traj)
    z_gen = torch.cat(z_gen_list)

    agg_traj = {
        key: [np.mean([t[key][s] for t in all_traj])
              for s in range(len(all_traj[0][key]))]
        for key in all_traj[0].keys()
    }

    gen_r, x_gen = evaluate(z_gen, dec, enc, e_core, z_train, test_x,
                            real_hf_coh, real_hf_noi, device, trajectory=agg_traj)
    gen_r['hue_var'] = compute_hue_var(x_gen)['hue_var']

    # Color KL
    gen_means = x_gen.mean(dim=(2, 3))
    real_means = test_x[:len(x_gen)].mean(dim=(2, 3))
    ckl = 0.0
    for c in range(gen_means.shape[1]):
        gh = torch.histc(gen_means[:, c], bins=50, min=0, max=1) + 1e-8
        rh = torch.histc(real_means[:, c], bins=50, min=0, max=1) + 1e-8
        gh /= gh.sum(); rh /= rh.sum()
        ckl += (gh * (gh / rh).log()).sum().item()
    gen_r['color_kl'] = ckl / gen_means.shape[1]

    # Energy descent efficiency
    e_traj = agg_traj['e_core']
    deltas = [e_traj[i+1] - e_traj[i] for i in range(len(e_traj)-1)]
    mono_rate = sum(1 for d in deltas if d < 0) / max(len(deltas), 1)
    gen_r['mono_rate'] = mono_rate
    gen_r['delta_u_final'] = agg_traj['delta_u_norm'][-1]
    gen_r['e_per_step'] = np.mean(deltas) if deltas else 0.0

    result['gen'] = {k: v for k, v in gen_r.items()
                     if not isinstance(v, (list, np.ndarray))}

    # ===== CLASSIFICATION =====
    # Repair train set for mixed probe
    z_train_rep_list = []
    with torch.no_grad():
        for i in range(0, len(z_train), bs):
            z_b = z_train[i:i+bs].to(device)
            B = z_b.shape[0]
            mask = make_center_mask(B, K, H, W, device)
            z_rep = repair_flow(step_fn, e_core, z_b, mask, device, T=T_repair)
            z_train_rep_list.append(z_rep.cpu())
    z_train_repaired = torch.cat(z_train_rep_list)

    probe = ConvProbe(K, H, W).to(device)
    train_probe_mixed(probe, z_train, z_train_repaired, train_y, device,
                      epochs=30, bs=128)
    acc_clean = eval_probe(probe, z_test, test_y, device)
    acc_repair = eval_probe(probe, z_test_repaired, test_y, device)
    gap = acc_clean - acc_repair
    result['classify'] = {
        'acc_clean': acc_clean,
        'acc_repair': acc_repair,
        'gap': gap,
    }

    del probe
    torch.cuda.empty_cache()

    return result, x_gen, z_test_repaired


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs/exp_r0_routing_encoder')
    parser.add_argument('--n_gen', type=int, default=256)
    parser.add_argument('--T_gen', type=int, default=20)
    parser.add_argument('--T_repair', type=int, default=10)
    parser.add_argument('--K', type=int, default=4, help='Number of routing neighbors')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    N_BITS = 16

    print("=" * 100)
    print("R0: ROUTING vs ENCODER CAPACITY — CAUSAL SEPARATION")
    print("=" * 100)

    # ========== DATA ==========
    print("\n[1] Loading CIFAR-10...")
    from torchvision import datasets, transforms
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())

    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(len(train_ds), 3000, replace=False)
    test_idx = rng.choice(len(test_ds), 500, replace=False)
    train_x = torch.stack([train_ds[i][0] for i in train_idx])
    test_x = torch.stack([test_ds[i][0] for i in test_idx])
    train_y = torch.tensor([train_ds[i][1] for i in train_idx])
    test_y = torch.tensor([test_ds[i][1] for i in test_idx])
    print(f"    Train: {train_x.shape}, Test: {test_x.shape}")

    # Reference metrics
    print("\n[2] Reference metrics...")
    real_hf_coh = hf_coherence_metric(test_x[:200], device)
    real_hf_noi = hf_noise_index(test_x[:200], device)
    real_hue_var = compute_hue_var(test_x[:200])['hue_var']
    print(f"    Real: HF_coh={real_hf_coh:.4f}  HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")

    all_results = {}

    # ==================================================================
    # GROUP 1: Encoder16 (shared ADC/E_core for configs A, B, C)
    # ==================================================================
    print(f"\n{'='*80}")
    print("GROUP 1: Encoder16 (configs A, B, C)")
    print("=" * 80)

    torch.manual_seed(args.seed)
    enc1 = Encoder16(N_BITS).to(device)
    dec1 = Decoder16(N_BITS).to(device)
    adc_loss1 = train_adc(enc1, dec1, train_x, device, epochs=40, bs=32)
    print(f"    ADC loss: {adc_loss1:.4f}")

    z_train1 = encode_all(enc1, train_x, device, bs=32)
    z_test1 = encode_all(enc1, test_x, device, bs=32)
    K, H, W = z_train1.shape[1:]
    n_enc1 = sum(p.numel() for p in enc1.parameters())
    print(f"    z: {z_train1.shape}, usage={z_train1.float().mean():.3f}, enc_params={n_enc1:,}")

    e_core1 = DiffEnergyCore(N_BITS).to(device)
    train_ecore(e_core1, z_train1, device, epochs=15, bs=128)

    # Config A: baseline (no routing)
    print(f"\n--- Config A: FlatStepFn_Norm (baseline) ---")
    torch.manual_seed(args.seed)
    step_a = FlatStepFn_Norm(N_BITS).to(device)
    n_params_a = sum(p.numel() for p in step_a.parameters())
    print(f"    StepFn params: {n_params_a:,}")
    train_step_fn(step_a, e_core1, z_train1, dec1, device, epochs=30, bs=32)
    result_a, x_gen_a, _ = eval_all_modes(
        step_a, e_core1, enc1, dec1, z_train1, z_test1,
        train_x, test_x, train_y, test_y, real_hf_coh, real_hf_noi, device,
        T_repair=args.T_repair, T_gen=args.T_gen, n_gen=args.n_gen)
    save_grid(x_gen_a, os.path.join(args.output_dir, 'gen_A_baseline.png'))
    all_results['A_baseline'] = result_a
    print_result('A_baseline', result_a)
    del step_a; torch.cuda.empty_cache()

    # Config B: content-sparse routing
    print(f"\n--- Config B: FlatStepFn + content-sparse routing (K={args.K}) ---")
    torch.manual_seed(args.seed)
    step_b = FlatStepFn_Routing(N_BITS, routing_mode='content', K=args.K).to(device)
    n_params_b = sum(p.numel() for p in step_b.parameters())
    print(f"    StepFn params: {n_params_b:,} (routing adds {n_params_b - n_params_a:,})")
    train_step_fn(step_b, e_core1, z_train1, dec1, device, epochs=30, bs=32)
    result_b, x_gen_b, _ = eval_all_modes(
        step_b, e_core1, enc1, dec1, z_train1, z_test1,
        train_x, test_x, train_y, test_y, real_hf_coh, real_hf_noi, device,
        T_repair=args.T_repair, T_gen=args.T_gen, n_gen=args.n_gen)
    save_grid(x_gen_b, os.path.join(args.output_dir, 'gen_B_content_route.png'))
    all_results['B_content_route'] = result_b
    print_result('B_content_route', result_b)
    del step_b; torch.cuda.empty_cache()

    # Config C: random-sparse routing (control)
    print(f"\n--- Config C: FlatStepFn + random-sparse routing (K={args.K}) ---")
    torch.manual_seed(args.seed)
    step_c = FlatStepFn_Routing(N_BITS, routing_mode='random', K=args.K).to(device)
    n_params_c = sum(p.numel() for p in step_c.parameters())
    print(f"    StepFn params: {n_params_c:,}")
    train_step_fn(step_c, e_core1, z_train1, dec1, device, epochs=30, bs=32)
    result_c, x_gen_c, _ = eval_all_modes(
        step_c, e_core1, enc1, dec1, z_train1, z_test1,
        train_x, test_x, train_y, test_y, real_hf_coh, real_hf_noi, device,
        T_repair=args.T_repair, T_gen=args.T_gen, n_gen=args.n_gen)
    save_grid(x_gen_c, os.path.join(args.output_dir, 'gen_C_random_route.png'))
    all_results['C_random_route'] = result_c
    print_result('C_random_route', result_c)
    del step_c; torch.cuda.empty_cache()

    # Free group 1 shared resources
    del enc1, dec1, e_core1, z_train1, z_test1
    torch.cuda.empty_cache()

    # ==================================================================
    # GROUP 2: EncoderDeep16 (shared ADC/E_core for configs D, E)
    # ==================================================================
    print(f"\n{'='*80}")
    print("GROUP 2: EncoderDeep16 (configs D, E)")
    print("=" * 80)

    torch.manual_seed(args.seed)
    enc2 = EncoderDeep16(N_BITS).to(device)
    dec2 = Decoder16(N_BITS).to(device)
    adc_loss2 = train_adc(enc2, dec2, train_x, device, epochs=40, bs=32)
    print(f"    ADC loss: {adc_loss2:.4f}")

    z_train2 = encode_all(enc2, train_x, device, bs=32)
    z_test2 = encode_all(enc2, test_x, device, bs=32)
    n_enc2 = sum(p.numel() for p in enc2.parameters())
    print(f"    z: {z_train2.shape}, usage={z_train2.float().mean():.3f}, enc_params={n_enc2:,}")

    e_core2 = DiffEnergyCore(N_BITS).to(device)
    train_ecore(e_core2, z_train2, device, epochs=15, bs=128)

    # Config D: deep encoder, no routing
    print(f"\n--- Config D: EncoderDeep16 + FlatStepFn_Norm ---")
    torch.manual_seed(args.seed)
    step_d = FlatStepFn_Norm(N_BITS).to(device)
    train_step_fn(step_d, e_core2, z_train2, dec2, device, epochs=30, bs=32)
    result_d, x_gen_d, _ = eval_all_modes(
        step_d, e_core2, enc2, dec2, z_train2, z_test2,
        train_x, test_x, train_y, test_y, real_hf_coh, real_hf_noi, device,
        T_repair=args.T_repair, T_gen=args.T_gen, n_gen=args.n_gen)
    save_grid(x_gen_d, os.path.join(args.output_dir, 'gen_D_deep_enc.png'))
    all_results['D_deep_enc'] = result_d
    print_result('D_deep_enc', result_d)
    del step_d; torch.cuda.empty_cache()

    # Config E: deep encoder + content routing (interaction test)
    print(f"\n--- Config E: EncoderDeep16 + content-sparse routing (K={args.K}) ---")
    torch.manual_seed(args.seed)
    step_e = FlatStepFn_Routing(N_BITS, routing_mode='content', K=args.K).to(device)
    train_step_fn(step_e, e_core2, z_train2, dec2, device, epochs=30, bs=32)
    result_e, x_gen_e, _ = eval_all_modes(
        step_e, e_core2, enc2, dec2, z_train2, z_test2,
        train_x, test_x, train_y, test_y, real_hf_coh, real_hf_noi, device,
        T_repair=args.T_repair, T_gen=args.T_gen, n_gen=args.n_gen)
    save_grid(x_gen_e, os.path.join(args.output_dir, 'gen_E_deep_enc_route.png'))
    all_results['E_deep_route'] = result_e
    print_result('E_deep_route', result_e)
    del step_e, enc2, dec2, e_core2
    torch.cuda.empty_cache()

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'='*100}")
    print("R0 SUMMARY: ROUTING vs ENCODER CAPACITY")
    print(f"Real: HF_noise={real_hf_noi:.2f}  HueVar={real_hue_var:.4f}")
    print("=" * 100)

    # Repair table
    print(f"\n--- REPAIR (center mask) ---")
    hdr = f"{'Config':<20} {'ham_unmask':>10} {'ham_mask':>10} {'cycle_rep':>10} {'Gate0':>6}"
    print(hdr); print("-" * len(hdr))
    for name, r in all_results.items():
        rep = r['repair']
        gate0 = "PASS" if rep['ham_unmasked'] < 0.001 else "FAIL"
        print(f"{name:<20} {rep['ham_unmasked']:>10.4f} {rep['ham_masked']:>10.4f} "
              f"{rep['cycle_repair']:>10.4f} {gate0:>6}")

    # Generation table
    print(f"\n--- GENERATION ---")
    hdr = f"{'Config':<20} {'viol':>7} {'div':>7} {'HFnoi':>7} {'HueV':>7} {'ColKL':>7} {'conn':>7} {'mono':>5} {'dE/step':>8}"
    print(hdr); print("-" * len(hdr))
    for name, r in all_results.items():
        g = r['gen']
        print(f"{name:<20} {g['violation']:>7.4f} {g['diversity']:>7.4f} "
              f"{g['hf_noise_index']:>7.1f} {g.get('hue_var', 0):>7.4f} "
              f"{g.get('color_kl', 0):>7.3f} {g['connectedness']:>7.4f} "
              f"{g.get('mono_rate', 0):>5.2f} {g.get('e_per_step', 0):>8.4f}")

    # Classification table
    print(f"\n--- CLASSIFICATION (conv probe, mixed) ---")
    hdr = f"{'Config':<20} {'acc_clean':>10} {'acc_repair':>10} {'gap':>8}"
    print(hdr); print("-" * len(hdr))
    for name, r in all_results.items():
        c = r['classify']
        print(f"{name:<20} {c['acc_clean']:>10.3f} {c['acc_repair']:>10.3f} {c['gap']:>+8.3f}")

    # ===== CAUSAL ANALYSIS =====
    print(f"\n{'='*100}")
    print("CAUSAL ANALYSIS")
    print("=" * 100)

    a = all_results['A_baseline']
    b = all_results['B_content_route']
    c = all_results['C_random_route']
    d = all_results['D_deep_enc']
    e = all_results['E_deep_route']

    # Routing effect (within Encoder16)
    route_gen = b['gen']['diversity'] - a['gen']['diversity']
    route_cls = b['classify']['acc_clean'] - a['classify']['acc_clean']
    random_gen = c['gen']['diversity'] - a['gen']['diversity']
    random_cls = c['classify']['acc_clean'] - a['classify']['acc_clean']

    print(f"\n  Routing effect (Encoder16, B vs A):")
    print(f"    gen diversity: {route_gen:+.4f}")
    print(f"    classify acc:  {route_cls:+.3f}")
    print(f"  Random control (C vs A):")
    print(f"    gen diversity: {random_gen:+.4f}")
    print(f"    classify acc:  {random_cls:+.3f}")
    print(f"  Content > Random? {b['gen']['diversity'] > c['gen']['diversity'] and b['classify']['acc_clean'] > c['classify']['acc_clean']}")

    # Encoder effect
    enc_gen = d['gen']['diversity'] - a['gen']['diversity']
    enc_cls = d['classify']['acc_clean'] - a['classify']['acc_clean']
    print(f"\n  Encoder effect (D vs A):")
    print(f"    gen diversity: {enc_gen:+.4f}")
    print(f"    classify acc:  {enc_cls:+.3f}")

    # Interaction effect
    interaction_cls = (e['classify']['acc_clean'] - d['classify']['acc_clean']) - \
                      (b['classify']['acc_clean'] - a['classify']['acc_clean'])
    interaction_gen = (e['gen']['diversity'] - d['gen']['diversity']) - \
                      (b['gen']['diversity'] - a['gen']['diversity'])
    print(f"\n  Interaction effect (E-D vs B-A):")
    print(f"    gen diversity: {interaction_gen:+.4f}")
    print(f"    classify acc:  {interaction_cls:+.3f}")

    # Verdict
    print(f"\n  --- VERDICT ---")
    if enc_cls > 0.02 and abs(route_cls) < 0.01:
        print("  → Encoder capacity is the bottleneck. Routing not useful.")
        print("  → Consistent with GDA=0% prior evidence.")
    elif route_cls > 0.02 and route_cls > random_cls + 0.01:
        print("  → Content routing provides genuine benefit!")
        print("  → New regime (flow + 16×16) activates routing.")
    elif abs(route_cls) < 0.01 and abs(enc_cls) < 0.01:
        print("  → Neither routing nor encoder helps at this scale.")
        print("  → Bottleneck is elsewhere (training protocol, E_core, decoder).")
    elif interaction_cls > 0.02:
        print("  → Routing only works with strong encoder (interaction effect).")
        print("  → GDA=0% was because z didn't have routable structure.")
    else:
        print("  → Mixed results. Check tables for details.")

    # Save
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*100}")
    print("R0 COMPLETE")
    print("=" * 100)


def print_result(name, r):
    rep = r['repair']
    gen = r['gen']
    cls = r['classify']
    gate0 = "PASS" if rep['ham_unmasked'] < 0.001 else "FAIL"
    print(f"    Repair: ham_un={rep['ham_unmasked']:.4f} [{gate0}]  "
          f"ham_m={rep['ham_masked']:.4f}  cyc={rep['cycle_repair']:.4f}")
    print(f"    Gen:    viol={gen['violation']:.4f}  div={gen['diversity']:.4f}  "
          f"HF_noi={gen['hf_noise_index']:.1f}  mono={gen.get('mono_rate', 0):.2f}")
    print(f"    Class:  clean={cls['acc_clean']:.3f}  repair={cls['acc_repair']:.3f}  "
          f"gap={cls['gap']:+.3f}")


if __name__ == "__main__":
    main()
