"""
Route C Learning Module
=======================
- N-gram count-based conditional model for tokens
- Log-odds classifier head
- Rule export utilities
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import defaultdict


# ============================================================================
# TOKEN CONDITIONAL MODEL (COUNT-BASED)
# ============================================================================

class TokenConditionalModel:
    """
    Count-based conditional probability model for tokens.
    Models P(token | context) where context is the neighborhood tokens.
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_size: int = 8,  # 3x3 neighborhood excluding center
        smoothing: float = 1.0,  # Laplace smoothing
        boundary: str = "standard",
    ):
        """
        Args:
            vocab_size: number of token types (K)
            context_size: number of context tokens (8 for 3x3 Moore neighborhood)
            smoothing: Laplace smoothing parameter (add-alpha smoothing)
            boundary: "standard" or "torus"
        """
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.smoothing = smoothing
        self.boundary = boundary
        
        # Counts: context_tuple -> array of counts for each token
        self.counts: Dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(vocab_size, dtype=np.float64)
        )
        self.total_counts: Dict[tuple, float] = defaultdict(float)
        
        # Marginal counts (for backoff)
        self.marginal_counts = np.zeros(vocab_size, dtype=np.float64)
        self.total_marginal = 0.0
    
    def _extract_context(
        self,
        token_grid: np.ndarray,
        i: int,
        j: int,
    ) -> tuple:
        """Extract context tuple from position (i, j)."""
        H, W = token_grid.shape
        context = []
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # Skip center
                
                ni, nj = i + di, j + dj
                
                if self.boundary == "torus":
                    ni = ni % H
                    nj = nj % W
                    context.append(token_grid[ni, nj])
                elif self.boundary == "standard":
                    if 0 <= ni < H and 0 <= nj < W:
                        context.append(token_grid[ni, nj])
                    else:
                        context.append(-1)  # Padding token
        
        return tuple(context)
    
    def fit(self, token_grids: np.ndarray, verbose: bool = False):
        """
        Fit the model by counting token occurrences in context.
        
        Args:
            token_grids: (N, H, W) array of token grids
        """
        N, H, W = token_grids.shape
        
        if verbose:
            print(f"Fitting TokenConditionalModel on {N} token grids of shape ({H}, {W})")
        
        for n in range(N):
            if verbose and n % 1000 == 0:
                print(f"  Processing grid {n}/{N}")
            
            for i in range(H):
                for j in range(W):
                    context = self._extract_context(token_grids[n], i, j)
                    token = token_grids[n, i, j]
                    
                    self.counts[context][token] += 1
                    self.total_counts[context] += 1
                    
                    self.marginal_counts[token] += 1
                    self.total_marginal += 1
        
        if verbose:
            n_contexts = len(self.counts)
            print(f"  Found {n_contexts} unique contexts")
    
    def predict_proba(self, context: tuple) -> np.ndarray:
        """
        Return P(token | context) with smoothing.
        
        Args:
            context: tuple of context tokens
        
        Returns:
            probs: (K,) probability distribution over tokens
        """
        if context in self.counts:
            counts = self.counts[context]
            total = self.total_counts[context]
        else:
            # Backoff to marginal distribution
            counts = self.marginal_counts
            total = self.total_marginal
        
        # Laplace smoothing
        probs = (counts + self.smoothing) / (total + self.smoothing * self.vocab_size)
        return probs
    
    def log_likelihood(self, token_grid: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Compute log-likelihood of token grid (optionally only at masked positions).
        
        Args:
            token_grid: (H, W) array
            mask: (H, W) boolean array, True where to compute likelihood
        
        Returns:
            log_likelihood: sum of log P(token | context)
        """
        H, W = token_grid.shape
        ll = 0.0
        
        for i in range(H):
            for j in range(W):
                if mask is not None and not mask[i, j]:
                    continue
                
                context = self._extract_context(token_grid, i, j)
                token = token_grid[i, j]
                probs = self.predict_proba(context)
                ll += np.log(probs[token] + 1e-10)
        
        return ll
    
    def sample(self, context: tuple, rng: np.random.Generator) -> int:
        """Sample a token given context."""
        probs = self.predict_proba(context)
        return rng.choice(self.vocab_size, p=probs)


# ============================================================================
# SPATIAL HISTOGRAM FEATURES
# ============================================================================

def token_grid_to_histogram(
    token_grid: np.ndarray,
    vocab_size: int,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert token grid to global histogram feature.
    
    Args:
        token_grid: (H, W) array
        vocab_size: number of token types
        normalize: whether to normalize to sum to 1
    
    Returns:
        hist: (vocab_size,) histogram
    """
    hist = np.bincount(token_grid.flatten(), minlength=vocab_size).astype(np.float64)
    if normalize:
        hist = hist / (hist.sum() + 1e-10)
    return hist


def token_grid_to_spatial_histogram(
    token_grid: np.ndarray,
    vocab_size: int,
    n_bins: Tuple[int, int] = (4, 4),
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert token grid to spatial histogram feature (grid of local histograms).
    
    Args:
        token_grid: (H, W) array
        vocab_size: number of token types
        n_bins: (n_rows, n_cols) spatial bins
        normalize: whether to normalize each bin histogram
    
    Returns:
        features: (n_bins[0] * n_bins[1] * vocab_size,) flattened spatial histogram
    """
    H, W = token_grid.shape
    br, bc = n_bins
    
    # Compute bin boundaries
    row_edges = np.linspace(0, H, br + 1).astype(int)
    col_edges = np.linspace(0, W, bc + 1).astype(int)
    
    features = []
    
    for ri in range(br):
        for ci in range(bc):
            # Extract bin region
            r0, r1 = row_edges[ri], row_edges[ri + 1]
            c0, c1 = col_edges[ci], col_edges[ci + 1]
            
            if r1 <= r0 or c1 <= c0:
                # Empty bin
                hist = np.zeros(vocab_size)
            else:
                region = token_grid[r0:r1, c0:c1]
                hist = np.bincount(region.flatten(), minlength=vocab_size).astype(np.float64)
            
            if normalize:
                hist = hist / (hist.sum() + 1e-10)
            
            features.append(hist)
    
    return np.concatenate(features)


def extract_features(
    token_grid: np.ndarray,
    vocab_size: int,
    spatial_bins: Tuple[int, int] = (4, 4),
    include_global: bool = True,
) -> np.ndarray:
    """
    Extract combined feature vector from token grid.
    
    Returns:
        features: combined feature vector
    """
    features = []
    
    if include_global:
        global_hist = token_grid_to_histogram(token_grid, vocab_size)
        features.append(global_hist)
    
    spatial_hist = token_grid_to_spatial_histogram(token_grid, vocab_size, spatial_bins)
    features.append(spatial_hist)
    
    return np.concatenate(features)


def extract_features_batch(
    token_grids: np.ndarray,
    vocab_size: int,
    spatial_bins: Tuple[int, int] = (4, 4),
    include_global: bool = True,
) -> np.ndarray:
    """Extract features for batch of token grids."""
    N = token_grids.shape[0]
    features = [extract_features(token_grids[i], vocab_size, spatial_bins, include_global) for i in range(N)]
    return np.stack(features)


# ============================================================================
# LOG-ODDS CLASSIFIER
# ============================================================================

class LogOddsClassifier:
    """
    Simple log-odds (naive Bayes-style) classifier.
    Models P(y | x) ∝ exp(W[y] · x + b[y])
    """
    
    def __init__(self, n_features: int, n_classes: int, smoothing: float = 1.0):
        self.n_features = n_features
        self.n_classes = n_classes
        self.smoothing = smoothing
        
        self.W: Optional[np.ndarray] = None  # (n_classes, n_features)
        self.b: Optional[np.ndarray] = None  # (n_classes,)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """
        Fit log-odds weights from training data.
        
        Uses class-conditional mean features and log-ratio approach.
        
        Args:
            X: (N, D) feature matrix
            y: (N,) class labels
        """
        N, D = X.shape
        K = self.n_classes
        
        if verbose:
            print(f"Fitting LogOddsClassifier: {N} samples, {D} features, {K} classes")
        
        # Compute class-conditional statistics
        class_means = np.zeros((K, D))
        class_counts = np.zeros(K)
        
        for k in range(K):
            mask = y == k
            class_counts[k] = mask.sum()
            if class_counts[k] > 0:
                class_means[k] = X[mask].mean(axis=0)
        
        # Overall mean (for computing log-odds)
        overall_mean = X.mean(axis=0)
        
        # Log-odds weights: W[k] = log(P(feature | class k) / P(feature))
        # For histogram features, this is proportional to log(class_mean / overall_mean)
        # We use smoothed version
        eps = self.smoothing / (N + self.smoothing * K)
        
        self.W = np.log(class_means + eps) - np.log(overall_mean + eps)
        self.b = np.log(class_counts + self.smoothing) - np.log(N + self.smoothing * K)
        
        if verbose:
            print(f"  W shape: {self.W.shape}, b shape: {self.b.shape}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: (N, D) or (D,) feature matrix/vector
        
        Returns:
            probs: (N, K) or (K,) probability matrix/vector
        """
        if X.ndim == 1:
            X = X[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False
        
        # Compute logits: (N, K)
        logits = X @ self.W.T + self.b
        
        # Softmax
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        if squeeze:
            return probs[0]
        return probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        if probs.ndim == 1:
            return np.argmax(probs)
        return np.argmax(probs, axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        preds = self.predict(X)
        return (preds == y).mean()


# ============================================================================
# RULE EXPORT (LUT-like)
# ============================================================================

def export_conditional_rules(
    model: TokenConditionalModel,
    min_count: int = 10,
    top_k: int = 3,
) -> List[Dict]:
    """
    Export conditional model rules as interpretable table.
    
    Args:
        model: fitted TokenConditionalModel
        min_count: minimum count to include a context
        top_k: number of top tokens to include per context
    
    Returns:
        rules: list of dicts with context, top_tokens, and probabilities
    """
    rules = []
    
    for context, counts in model.counts.items():
        total = model.total_counts[context]
        if total < min_count:
            continue
        
        probs = model.predict_proba(context)
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        rule = {
            'context': context,
            'total_count': int(total),
            'top_tokens': top_indices.tolist(),
            'top_probs': [float(probs[i]) for i in top_indices],
        }
        rules.append(rule)
    
    # Sort by total count
    rules.sort(key=lambda r: r['total_count'], reverse=True)
    
    return rules


def print_rules(rules: List[Dict], max_rules: int = 20):
    """Pretty-print exported rules."""
    print(f"\n{'='*60}")
    print(f"Top {min(max_rules, len(rules))} rules by frequency:")
    print(f"{'='*60}")
    
    for i, rule in enumerate(rules[:max_rules]):
        context_str = ','.join(str(c) for c in rule['context'])
        tokens_str = ', '.join(f"{t}({p:.2f})" for t, p in zip(rule['top_tokens'], rule['top_probs']))
        print(f"{i+1:3d}. [{context_str}] -> {tokens_str}  (n={rule['total_count']})")
