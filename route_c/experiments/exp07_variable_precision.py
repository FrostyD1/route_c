#!/usr/bin/env python3
"""
Experiment 07: Variable Precision + Solver Scaling + Compilation Tradeoff
==========================================================================

Goal: Show Route C is a *discrete state-space system* whose capability scales
with precision k and whose solver/compilation properties are measurable.

Run: python exp07_variable_precision.py

Measures for each k ∈ {2, 4, 8, 16}:
  M1) Clean classification accuracy
  M2) Occluded accuracy (random & center) before/after discrete inference
  M3) Core closure strength (masked-bit prediction)
  M4) Compilability (DNF clauses, literals, DAG nodes, fidelity)

Expected narrative:
  - Increasing k increases capacity (up to saturation)
  - Inference benefit grows when core closure is strong
  - Compilation cost increases with k, but sharing also increases
  - Reveals tradeoff: precision ↔ robustness ↔ hardware complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Set, FrozenSet, Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw): return x


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    # Data
    train_samples: int = 10000
    test_samples: int = 2000
    batch_size: int = 64
    
    # Architecture (H=W=7 fixed, k varies)
    latent_size: int = 7
    hidden_dim: int = 64
    energy_hidden: int = 32
    
    # Training (FIXED for all k)
    epochs: int = 8
    lr: float = 1e-3
    tau_start: float = 1.0
    tau_end: float = 0.2
    
    # Loss weights (FIXED)
    alpha_recon: float = 1.0
    beta_core: float = 0.5
    gamma_cls: float = 1.0
    lambda_sparsity: float = 1e-4
    mask_ratio: float = 0.15
    
    # Inference
    inf_steps: int = 30
    inf_block_size: Tuple[int, int] = (2, 2)
    occlusion_size: Tuple[int, int] = (14, 14)
    occlusion_eval: int = 500
    
    # Precision values to test
    k_values: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    
    seed: int = 42
    device: str = "cpu"


# ============================================================================
# MODEL COMPONENTS (Parameterized by k)
# ============================================================================

class GumbelSigmoid(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits):
        if self.training:
            u = torch.rand_like(logits).clamp(1e-8, 1-1e-8)
            gumbel = -torch.log(-torch.log(u))
            noisy = (logits + gumbel) / self.temperature
        else:
            noisy = logits / self.temperature
        soft = torch.sigmoid(noisy)
        hard = (soft > 0.5).float()
        return hard - soft.detach() + soft
    
    def set_temperature(self, tau):
        self.temperature = tau


class Encoder(nn.Module):
    """CNN encoder: 28×28 → k×7×7 logits"""
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1),  # 28→14
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),  # 14→7
            nn.ReLU(),
            nn.Conv2d(hidden_dim, n_bits, 3, padding=1),  # 7×7, k channels
        )
    
    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    """Decoder: k×7×7 → 1×28×28"""
    def __init__(self, n_bits, hidden_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_bits, hidden_dim, 4, stride=2, padding=1),  # 7→14
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # 14→28
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, z):
        return self.deconv(z)


class LocalPredictor(nn.Module):
    """Predict masked bits from 3×3 neighborhood"""
    def __init__(self, n_bits, hidden_dim=32):
        super().__init__()
        self.n_bits = n_bits
        self.net = nn.Sequential(
            nn.Linear(9 * n_bits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_bits),
        )
    
    def forward(self, z):
        B, k, H, W = z.shape
        z_pad = F.pad(z, (1, 1, 1, 1), mode='constant', value=0)
        windows = F.unfold(z_pad, kernel_size=3)  # (B, k*9, H*W)
        windows = windows.reshape(B, k, 9, H * W)
        windows[:, :, 4, :] = 0  # Mask center
        windows = windows.reshape(B, k * 9, H * W)
        windows = windows.permute(0, 2, 1)  # (B, H*W, k*9)
        logits = self.net(windows)  # (B, H*W, k)
        return logits.permute(0, 2, 1).reshape(B, k, H, W)


class Classifier(nn.Module):
    """Linear classifier on flattened z"""
    def __init__(self, n_bits, latent_size=7, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(n_bits * latent_size * latent_size, n_classes)
    
    def forward(self, z):
        return self.fc(z.reshape(z.shape[0], -1))


class RouteCModel(nn.Module):
    """Complete Route C model parameterized by k (n_bits)"""
    def __init__(self, n_bits: int, cfg: Config):
        super().__init__()
        self.n_bits = n_bits
        self.encoder = Encoder(n_bits, cfg.hidden_dim)
        self.quantizer = GumbelSigmoid(cfg.tau_start)
        self.decoder = Decoder(n_bits, cfg.hidden_dim)
        self.local_pred = LocalPredictor(n_bits, cfg.energy_hidden)
        self.classifier = Classifier(n_bits, cfg.latent_size)
    
    def encode(self, x):
        return self.quantizer(self.encoder(x))
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        cls_logits = self.classifier(z)
        core_logits = self.local_pred(z)
        return z, x_hat, cls_logits, core_logits
    
    def set_temperature(self, tau):
        self.quantizer.set_temperature(tau)


# ============================================================================
# TRUNC / PAD UTILITIES
# ============================================================================

def trunc_k(z: torch.Tensor, k: int) -> torch.Tensor:
    """Truncate z to first k bits: z ∈ {0,1}^{K×H×W} → z ∈ {0,1}^{k×H×W}"""
    return z[:, :k, :, :] if z.dim() == 4 else z[:k, :, :]


def pad_k(z: torch.Tensor, k_target: int) -> torch.Tensor:
    """Pad z to k_target bits with zeros"""
    if z.dim() == 4:
        B, k, H, W = z.shape
        if k >= k_target:
            return z
        padding = torch.zeros(B, k_target - k, H, W, device=z.device, dtype=z.dtype)
        return torch.cat([z, padding], dim=1)
    else:
        k, H, W = z.shape
        if k >= k_target:
            return z
        padding = torch.zeros(k_target - k, H, W, device=z.device, dtype=z.dtype)
        return torch.cat([z, padding], dim=0)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(cfg: Config):
    """Load MNIST with fixed seed"""
    try:
        from torchvision import datasets, transforms
        transform = transforms.ToTensor()
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    except:
        raise RuntimeError("Please install torchvision: pip install torchvision")
    
    rng = np.random.default_rng(cfg.seed)
    train_idx = rng.choice(len(train_ds), cfg.train_samples, replace=False)
    test_idx = rng.choice(len(test_ds), cfg.test_samples, replace=False)
    
    train_x = torch.stack([train_ds[i][0] for i in train_idx])
    train_y = torch.tensor([train_ds[i][1] for i in train_idx])
    test_x = torch.stack([test_ds[i][0] for i in test_idx])
    test_y = torch.tensor([test_ds[i][1] for i in test_idx])
    
    return train_x, train_y, test_x, test_y


# ============================================================================
# OCCLUSION UTILITIES
# ============================================================================

def create_random_occlusion(image: np.ndarray, occ_size: Tuple[int, int], rng) -> Tuple[np.ndarray, np.ndarray]:
    """Random position occlusion"""
    H, W = image.shape
    oh, ow = occ_size
    y = rng.integers(0, max(1, H - oh + 1))
    x = rng.integers(0, max(1, W - ow + 1))
    
    occluded = image.copy()
    occluded[y:y+oh, x:x+ow] = 0
    
    mask = np.ones((H, W), dtype=np.float32)
    mask[y:y+oh, x:x+ow] = 0
    return occluded, mask


def create_center_occlusion(image: np.ndarray, occ_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Center position occlusion (fixed location for higher SNR)"""
    H, W = image.shape
    oh, ow = occ_size
    y = (H - oh) // 2
    x = (W - ow) // 2
    
    occluded = image.copy()
    occluded[y:y+oh, x:x+ow] = 0
    
    mask = np.ones((H, W), dtype=np.float32)
    mask[y:y+oh, x:x+ow] = 0
    return occluded, mask


def create_bit_mask(pixel_mask: np.ndarray, n_bits: int, latent_size: int = 7) -> np.ndarray:
    """Create bit mask from pixel mask"""
    patch_size = 28 // latent_size
    bit_mask = np.zeros((n_bits, latent_size, latent_size), dtype=bool)
    
    for i in range(latent_size):
        for j in range(latent_size):
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            obs_ratio = pixel_mask[y0:y1, x0:x1].mean()
            if obs_ratio < 0.5:
                bit_mask[:, i, j] = True
    return bit_mask


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model: RouteCModel, train_x, train_y, cfg: Config, k: int):
    """Train model with fixed protocol"""
    device = cfg.device
    model = model.to(device)
    
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    for epoch in range(cfg.epochs):
        model.train()
        tau = cfg.tau_start + (cfg.tau_end - cfg.tau_start) * epoch / max(1, cfg.epochs - 1)
        model.set_temperature(tau)
        
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            z, x_hat, cls_logits, core_logits = model(x)
            
            loss_recon = F.mse_loss(x_hat, x)
            
            mask = torch.rand_like(z) < cfg.mask_ratio
            z_target = z.detach()
            loss_core = F.binary_cross_entropy_with_logits(
                core_logits[mask], z_target[mask]
            ) if mask.any() else torch.tensor(0.0)
            
            loss_cls = F.cross_entropy(cls_logits, y)
            loss_sparse = (z.mean() - 0.3).pow(2)
            
            loss = (cfg.alpha_recon * loss_recon +
                    cfg.beta_core * loss_core +
                    cfg.gamma_cls * loss_cls +
                    cfg.lambda_sparsity * loss_sparse)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return model


@torch.no_grad()
def evaluate_clean(model: RouteCModel, test_x, test_y, cfg: Config) -> Dict:
    """Evaluate clean accuracy and core closure"""
    device = cfg.device
    model.eval()
    
    loader = DataLoader(TensorDataset(test_x, test_y), batch_size=cfg.batch_size)
    
    correct = 0
    total = 0
    core_correct = 0
    core_total = 0
    total_mse = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z, x_hat, cls_logits, core_logits = model(x)
        
        pred = cls_logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        total_mse += F.mse_loss(x_hat, x).item() * y.size(0)
        
        mask = torch.rand_like(z) < 0.2
        if mask.any():
            core_pred = (torch.sigmoid(core_logits) > 0.5).float()
            core_correct += (core_pred[mask] == z[mask]).sum().item()
            core_total += mask.sum().item()
    
    return {
        'accuracy': correct / total,
        'mse': total_mse / total,
        'core_acc': core_correct / max(1, core_total),
    }


# ============================================================================
# DISCRETE INFERENCE (BLOCK GIBBS)
# ============================================================================

class DiscreteInference:
    """Block Gibbs inference in z-space"""
    
    def __init__(self, model: RouteCModel, device: str, sigma_sq: float, cfg: Config):
        self.model = model
        self.device = device
        self.sigma_sq = sigma_sq
        self.cfg = cfg
        self.n_bits = model.n_bits
    
    def decode_np(self, z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z_t = torch.from_numpy(z.astype(np.float32))[None].to(self.device)
            return self.model.decode(z_t)[0, 0].cpu().numpy()
    
    def energy_core(self, z: np.ndarray) -> float:
        with torch.no_grad():
            z_t = torch.from_numpy(z.astype(np.float32))[None].to(self.device)
            logits = self.model.local_pred(z_t)
            return F.binary_cross_entropy_with_logits(logits, z_t, reduction='sum').item()
    
    def energy_obs(self, z: np.ndarray, o_obs: np.ndarray, mask: np.ndarray) -> float:
        o_hat = self.decode_np(z)
        diff = (o_hat - o_obs) * mask
        mse = (diff ** 2).sum() / max(1, mask.sum())
        return mse / (2 * self.sigma_sq)
    
    def total_energy(self, z: np.ndarray, o_obs: np.ndarray, mask: np.ndarray) -> float:
        return self.energy_core(z) + self.energy_obs(z, o_obs, mask)
    
    def run(self, z_init: np.ndarray, o_obs: np.ndarray, pixel_mask: np.ndarray, bit_mask: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.cfg.seed)
        z = z_init.copy()
        k, H, W = z.shape
        bh, bw = self.cfg.inf_block_size
        
        z[bit_mask] = rng.integers(0, 2, size=bit_mask.sum())
        
        for step in range(self.cfg.inf_steps):
            for bi in range(0, H, bh):
                for bj in range(0, W, bw):
                    i_end, j_end = min(bi + bh, H), min(bj + bw, W)
                    block_mask = bit_mask[:, bi:i_end, bj:j_end]
                    masked_pos = np.argwhere(block_mask)
                    
                    if len(masked_pos) == 0:
                        continue
                    
                    n = len(masked_pos)
                    if n <= 6:
                        energies = []
                        for config in range(2 ** n):
                            for idx, (b, i, j) in enumerate(masked_pos):
                                z[b, bi + i, bj + j] = (config >> idx) & 1
                            energies.append(self.total_energy(z, o_obs, pixel_mask))
                        
                        energies = np.array(energies)
                        energies = energies - energies.min()
                        probs = np.exp(-energies)
                        probs = probs / probs.sum()
                        chosen = rng.choice(2 ** n, p=probs)
                        
                        for idx, (b, i, j) in enumerate(masked_pos):
                            z[b, bi + i, bj + j] = (chosen >> idx) & 1
                    else:
                        E_curr = self.total_energy(z, o_obs, pixel_mask)
                        for _ in range(3):
                            n_flip = rng.integers(1, min(4, n) + 1)
                            flip_idx = rng.choice(n, size=n_flip, replace=False)
                            z_prop = z.copy()
                            for idx in flip_idx:
                                b, i, j = masked_pos[idx]
                                z_prop[b, bi + i, bj + j] = 1 - z_prop[b, bi + i, bj + j]
                            E_prop = self.total_energy(z_prop, o_obs, pixel_mask)
                            if E_prop < E_curr or rng.random() < np.exp(-(E_prop - E_curr)):
                                z, E_curr = z_prop, E_prop
        return z


def estimate_sigma_sq(model: RouteCModel, test_x, device: str, n_samples: int = 500) -> float:
    """Estimate σ² from reconstruction residuals"""
    model.eval()
    residuals = []
    with torch.no_grad():
        for i in range(min(n_samples, len(test_x))):
            x = test_x[i:i+1].to(device)
            z = model.encode(x)
            x_hat = model.decode(z)
            residuals.append(((x_hat - x) ** 2).mean().item())
    return np.mean(residuals)


def evaluate_occlusion(model: RouteCModel, test_x, test_y, cfg: Config, occlusion_type: str = 'random') -> Dict:
    """Evaluate occlusion recovery with discrete inference"""
    device = cfg.device
    model.eval()
    
    sigma_sq = estimate_sigma_sq(model, test_x, device)
    engine = DiscreteInference(model, device, sigma_sq, cfg)
    
    rng = np.random.default_rng(cfg.seed + 1000)
    eval_idx = rng.choice(len(test_x), min(cfg.occlusion_eval, len(test_x)), replace=False)
    
    baseline_correct = 0
    inferred_correct = 0
    n = len(eval_idx)
    
    for idx in eval_idx:
        x_clean = test_x[idx].numpy()[0]
        label = test_y[idx].item()
        
        if occlusion_type == 'center':
            x_occ, pixel_mask = create_center_occlusion(x_clean, cfg.occlusion_size)
        else:
            x_occ, pixel_mask = create_random_occlusion(x_clean, cfg.occlusion_size, rng)
        
        with torch.no_grad():
            x_occ_t = torch.from_numpy(x_occ[None, None].astype(np.float32)).to(device)
            z_occ = model.encode(x_occ_t)[0].cpu().numpy()
            
            logits = model.classifier(torch.from_numpy(z_occ[None]).to(device))
            baseline_correct += int(logits.argmax(dim=1).item() == label)
        
        bit_mask = create_bit_mask(pixel_mask, model.n_bits, cfg.latent_size)
        z_inf = engine.run(z_occ, x_occ, pixel_mask, bit_mask)
        
        with torch.no_grad():
            logits = model.classifier(torch.from_numpy(z_inf[None]).to(device))
            inferred_correct += int(logits.argmax(dim=1).item() == label)
    
    return {
        'baseline': baseline_correct / n,
        'inferred': inferred_correct / n,
        'delta': (inferred_correct - baseline_correct) / n,
    }


# ============================================================================
# LOGIC DISTILLATION
# ============================================================================

Literal = Tuple[int, bool]
Clause = FrozenSet[Literal]
DNF = Set[Clause]


class DecisionTreeBinary:
    """Simple decision tree for binary features"""
    
    def __init__(self, max_depth=10, min_samples_leaf=10):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self._build(X, y, 0)
    
    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        p = y.mean()
        return 2 * p * (1 - p)
    
    def _build(self, X, y, depth):
        n = len(y)
        n_pos = y.sum()
        
        if depth >= self.max_depth or n < self.min_samples_leaf * 2 or n_pos == 0 or n_pos == n:
            return {'leaf': True, 'pred': 1 if n_pos > n / 2 else 0, 'proba': [1 - n_pos/n, n_pos/n]}
        
        best_gain, best_feat = -1, 0
        curr_gini = self._gini(y)
        
        for f in range(X.shape[1]):
            left = X[:, f] < 0.5
            if left.sum() < self.min_samples_leaf or (~left).sum() < self.min_samples_leaf:
                continue
            gain = curr_gini - (left.sum() * self._gini(y[left]) + (~left).sum() * self._gini(y[~left])) / n
            if gain > best_gain:
                best_gain, best_feat = gain, f
        
        if best_gain <= 0:
            return {'leaf': True, 'pred': 1 if n_pos > n / 2 else 0, 'proba': [1 - n_pos/n, n_pos/n]}
        
        left = X[:, best_feat] < 0.5
        return {
            'leaf': False,
            'feat': best_feat,
            'left': self._build(X[left], y[left], depth + 1),
            'right': self._build(X[~left], y[~left], depth + 1),
        }
    
    def predict(self, X):
        return np.array([self._pred_one(x) for x in X])
    
    def _pred_one(self, x):
        node = self.root
        while not node['leaf']:
            node = node['left'] if x[node['feat']] < 0.5 else node['right']
        return node['pred']
    
    def predict_proba(self, X):
        return np.array([self._proba_one(x) for x in X])
    
    def _proba_one(self, x):
        node = self.root
        while not node['leaf']:
            node = node['left'] if x[node['feat']] < 0.5 else node['right']
        return node['proba']


def tree_to_dnf(tree, target=1) -> DNF:
    """Convert tree to DNF"""
    dnf = set()
    
    def traverse(node, path):
        if node['leaf']:
            if node['pred'] == target:
                dnf.add(frozenset(path))
            return
        traverse(node['left'], path + [(node['feat'], False)])
        traverse(node['right'], path + [(node['feat'], True)])
    
    traverse(tree.root, [])
    return dnf


def optimize_dnf(dnf: DNF) -> DNF:
    """Remove contradictions and subsumptions"""
    # Remove contradictions
    result = set()
    for clause in dnf:
        pos = {l[0] for l in clause if l[1]}
        neg = {l[0] for l in clause if not l[1]}
        if not pos.intersection(neg):
            result.add(clause)
    
    # Subsumption elimination
    clauses = sorted(result, key=len)
    final = set()
    for clause in clauses:
        if not any(existing.issubset(clause) for existing in final):
            final.add(clause)
    
    return final


class LogicDistiller:
    """Distill classifier to DNF"""
    
    def __init__(self, n_classes=10, max_depth=10):
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.trees = []
        self.dnfs = []
    
    def fit(self, Z, labels):
        self.trees = []
        self.dnfs = []
        
        for c in range(self.n_classes):
            y = (labels == c).astype(np.int32)
            tree = DecisionTreeBinary(max_depth=self.max_depth)
            tree.fit(Z, y)
            self.trees.append(tree)
            
            dnf = tree_to_dnf(tree, target=1)
            dnf = optimize_dnf(dnf)
            self.dnfs.append(dnf)
    
    def predict(self, Z):
        scores = np.zeros((len(Z), self.n_classes))
        for c, tree in enumerate(self.trees):
            scores[:, c] = tree.predict_proba(Z)[:, 1]
        return scores.argmax(axis=1)
    
    def fidelity(self, Z, teacher_labels):
        return (self.predict(Z) == teacher_labels).mean()
    
    def get_stats(self):
        total_clauses = sum(len(dnf) for dnf in self.dnfs)
        total_literals = sum(sum(len(c) for c in dnf) for dnf in self.dnfs)
        
        # Estimate DAG nodes (shared literals across clauses)
        all_literals = set()
        for dnf in self.dnfs:
            for clause in dnf:
                all_literals.update(clause)
        
        return {
            'clauses': total_clauses,
            'literals': total_literals,
            'dag_nodes': len(all_literals) + total_clauses + self.n_classes,
            'shared': len(all_literals),
        }


def dnf_to_verilog(dnfs: List[DNF], n_features: int, module_name: str) -> str:
    """Generate Verilog from DNF"""
    lines = [
        f"// Route C Exp07: k-bit classifier",
        f"module {module_name} (",
        f"    input wire [{n_features-1}:0] z,",
        f"    output wire [9:0] class_scores",
        f");",
        "",
    ]
    
    for c, dnf in enumerate(dnfs):
        if not dnf:
            lines.append(f"    assign class_scores[{c}] = 1'b0;")
        else:
            clauses = []
            for clause in dnf:
                if not clause:
                    clauses.append("1'b1")
                else:
                    lits = [f"z[{v}]" if pos else f"~z[{v}]" for v, pos in sorted(clause)]
                    clauses.append("(" + " & ".join(lits) + ")")
            lines.append(f"    assign class_scores[{c}] = " + " | ".join(clauses) + ";")
    
    lines.extend(["", "endmodule"])
    return "\n".join(lines)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    print("="*80)
    print("EXPERIMENT 07: VARIABLE PRECISION + SOLVER SCALING + COMPILATION TRADEOFF")
    print("="*80)
    
    cfg = Config()
    if torch.cuda.is_available():
        cfg.device = "cuda"
        print(f"Device: CUDA")
    else:
        print(f"Device: CPU")
    
    # Load data once
    print("\n[1] Loading data...")
    train_x, train_y, test_x, test_y = load_data(cfg)
    print(f"    Train: {len(train_x)}, Test: {len(test_x)}")
    
    os.makedirs("outputs", exist_ok=True)
    
    # Results storage
    results = []
    
    # Run for each k
    for k in cfg.k_values:
        print(f"\n{'='*80}")
        print(f"[k={k}] Training and evaluating...")
        print("="*80)
        
        # Create and train model
        t0 = time.time()
        model = RouteCModel(n_bits=k, cfg=cfg)
        model = train_model(model, train_x, train_y, cfg, k)
        train_time = time.time() - t0
        print(f"    Training time: {train_time:.1f}s")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"outputs/exp07_k{k}_model.pt")
        
        # M1: Clean accuracy
        print(f"    Evaluating clean accuracy...")
        clean_stats = evaluate_clean(model, test_x, test_y, cfg)
        
        # M2: Occlusion (random)
        print(f"    Evaluating random occlusion...")
        occ_rand = evaluate_occlusion(model, test_x, test_y, cfg, 'random')
        
        # M2: Occlusion (center)
        print(f"    Evaluating center occlusion...")
        occ_center = evaluate_occlusion(model, test_x, test_y, cfg, 'center')
        
        # M3: Core closure (from clean_stats)
        core_acc = clean_stats['core_acc']
        
        # M4: Logic distillation
        print(f"    Distilling to DNF...")
        model.eval()
        with torch.no_grad():
            Z_all = []
            labels_all = []
            loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64)
            for x, y in loader:
                z = model.encode(x.to(cfg.device))
                Z_all.append(z.cpu().numpy())
                logits = model.classifier(z)
                labels_all.append(logits.argmax(dim=1).cpu().numpy())
            Z_flat = np.concatenate(Z_all).reshape(len(train_x), -1)
            teacher_labels = np.concatenate(labels_all)
        
        distiller = LogicDistiller(n_classes=10, max_depth=10)
        distiller.fit(Z_flat, teacher_labels)
        fidelity = distiller.fidelity(Z_flat, teacher_labels)
        dnf_stats = distiller.get_stats()
        
        # Generate Verilog
        verilog = dnf_to_verilog(distiller.dnfs, k * 7 * 7, f"classifier_k{k}")
        with open(f"outputs/exp07_k{k}_classifier.v", "w") as f:
            f.write(verilog)
        
        # Store results
        results.append({
            'k': k,
            'clean_acc': clean_stats['accuracy'],
            'occ_rand_base': occ_rand['baseline'],
            'occ_rand_after': occ_rand['inferred'],
            'occ_rand_delta': occ_rand['delta'],
            'occ_center_base': occ_center['baseline'],
            'occ_center_after': occ_center['inferred'],
            'occ_center_delta': occ_center['delta'],
            'core_acc': core_acc,
            'dnf_fidelity': fidelity,
            'clauses': dnf_stats['clauses'],
            'literals': dnf_stats['literals'],
            'dag_nodes': dnf_stats['dag_nodes'],
            'shared': dnf_stats['shared'],
        })
        
        print(f"    Clean: {clean_stats['accuracy']:.1%}, "
              f"Rand Δ: {occ_rand['delta']:+.1%}, "
              f"Center Δ: {occ_center['delta']:+.1%}, "
              f"Core: {core_acc:.1%}, "
              f"DNF fid: {fidelity:.1%}")
    
    # Print summary table
    print("\n" + "="*120)
    print("SUMMARY TABLE")
    print("="*120)
    
    header = "| k  | clean | rand_base | rand_after |   Δ   | center_base | center_after |   Δ   | core  | fidelity | clauses | literals | DAG | shared |"
    sep = "|" + "-"*4 + "|" + ("-"*7 + "|")*8 + ("-"*9 + "|")*2 + ("-"*5 + "|")*2
    
    print(header)
    print(sep)
    
    for r in results:
        print(f"| {r['k']:>2} | {r['clean_acc']:>5.1%} | {r['occ_rand_base']:>9.1%} | {r['occ_rand_after']:>10.1%} | {r['occ_rand_delta']:>+5.1%} | "
              f"{r['occ_center_base']:>11.1%} | {r['occ_center_after']:>12.1%} | {r['occ_center_delta']:>+5.1%} | "
              f"{r['core_acc']:>5.1%} | {r['dnf_fidelity']:>8.1%} | {r['clauses']:>7} | {r['literals']:>8} | {r['dag_nodes']:>3} | {r['shared']:>6} |")
    
    # Route C interpretation
    print("\n" + "="*120)
    print("ROUTE C INTERPRETATION")
    print("="*120)
    
    print("""
This experiment reveals the fundamental tradeoffs in Route C's discrete state-space paradigm:

1. PRECISION vs CAPACITY:
   - Increasing k (bits per position) increases representational capacity
   - Clean accuracy typically improves with k until saturation
   - Very low k (e.g., k=2) may lack capacity for complex patterns

2. CORE CLOSURE vs INFERENCE BENEFIT:
   - Masked-bit prediction accuracy measures how well z captures local structure
   - Stronger core closure → better discrete inference (larger Δ)
   - Center occlusion has higher SNR than random (more consistent evaluation)

3. COMPILABILITY SCALING:
   - DNF clauses/literals grow with k (more binary variables to describe)
   - However, DAG sharing also increases (common sub-expressions)
   - This is NOT just a bigger neural net - it's measurable logical complexity

4. UNIQUE ROUTE C PROPERTIES:
   - Solver-based inference: Block Gibbs in z-space, not gradient descent
   - Compile-to-netlist: Actual Verilog output with combinational logic
   - Variable precision: trunc/pad define nested representations
   - These distinguish Route C from feed-forward neural networks

The tradeoff is: higher precision k → better accuracy → more hardware complexity
But Route C makes this tradeoff EXPLICIT and CONTROLLABLE.
""")
    
    print("="*120)
    print(f"Outputs saved to: outputs/exp07_k*_model.pt, outputs/exp07_k*_classifier.v")
    print("="*120)
    
    return results


if __name__ == "__main__":
    results = run_experiment()
