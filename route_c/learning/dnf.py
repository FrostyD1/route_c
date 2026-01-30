"""
Route C DNF Learning
====================
DNF head implementations:
1. Teacher→Rules Distillation: Train linear/tree, export to DNF
2. Differentiable DNF: End-to-end learnable with STE/Gumbel

Key Insight:
DNF and log-odds are different parameterizations of the SAME "readout layer" R(z)->y.
DNF is valuable for:
- Interpretability (human-readable rules)
- Logic hardening (convert to fast lookup tables)
- Rule extraction from trained models
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


# ============================================================================
# DECISION TREE -> DNF CONVERSION
# ============================================================================

@dataclass
class TreeNode:
    """Simple decision tree node."""
    feature_idx: Optional[int] = None  # Split feature (None for leaf)
    threshold: float = 0.5  # Split threshold
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    class_label: Optional[int] = None  # Leaf class (None for internal)
    n_samples: int = 0
    

def build_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 10,
    min_samples_leaf: int = 5,
    min_samples_split: int = 10,
) -> TreeNode:
    """
    Build a simple decision tree (pure numpy implementation).
    
    Args:
        X: (N, D) feature matrix (should be binary/binarized)
        y: (N,) class labels
        max_depth: maximum tree depth
        min_samples_leaf: minimum samples per leaf
        min_samples_split: minimum samples to consider split
    
    Returns:
        root: TreeNode representing the decision tree
    """
    def gini_impurity(labels):
        if len(labels) == 0:
            return 0.0
        classes, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return 1.0 - np.sum(probs ** 2)
    
    def best_split(X_sub, y_sub):
        N, D = X_sub.shape
        best_gain = -1
        best_feature = None
        best_threshold = 0.5
        
        parent_gini = gini_impurity(y_sub)
        
        for d in range(D):
            # For binary features, threshold at 0.5
            left_mask = X_sub[:, d] <= 0.5
            right_mask = ~left_mask
            
            if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
                continue
            
            left_gini = gini_impurity(y_sub[left_mask])
            right_gini = gini_impurity(y_sub[right_mask])
            
            n_left = left_mask.sum()
            n_right = right_mask.sum()
            
            weighted_gini = (n_left * left_gini + n_right * right_gini) / N
            gain = parent_gini - weighted_gini
            
            if gain > best_gain:
                best_gain = gain
                best_feature = d
        
        return best_feature, best_threshold, best_gain
    
    def build_recursive(X_sub, y_sub, depth):
        node = TreeNode(n_samples=len(y_sub))
        
        # Check stopping conditions
        if depth >= max_depth or len(y_sub) < min_samples_split or len(np.unique(y_sub)) == 1:
            # Make leaf
            if len(y_sub) > 0:
                classes, counts = np.unique(y_sub, return_counts=True)
                node.class_label = classes[np.argmax(counts)]
            else:
                node.class_label = 0
            return node
        
        # Find best split
        feature, threshold, gain = best_split(X_sub, y_sub)
        
        if feature is None or gain <= 0:
            # No good split, make leaf
            classes, counts = np.unique(y_sub, return_counts=True)
            node.class_label = classes[np.argmax(counts)]
            return node
        
        # Split
        node.feature_idx = feature
        node.threshold = threshold
        
        left_mask = X_sub[:, feature] <= threshold
        right_mask = ~left_mask
        
        node.left = build_recursive(X_sub[left_mask], y_sub[left_mask], depth + 1)
        node.right = build_recursive(X_sub[right_mask], y_sub[right_mask], depth + 1)
        
        return node
    
    return build_recursive(X, y, 0)


def tree_to_dnf_rules(
    tree: TreeNode,
    target_class: int,
) -> List[List[Tuple[int, bool]]]:
    """
    Extract DNF rules for a specific class from decision tree.
    
    Each root-to-leaf path becomes a clause (AND of conditions).
    All paths to leaves with target_class are combined with OR.
    
    Args:
        tree: TreeNode root
        target_class: class to extract rules for
    
    Returns:
        rules: DNF as list of clauses, each clause is list of (feature_idx, is_positive)
    """
    rules = []
    
    def extract_paths(node, current_path):
        if node.class_label is not None:
            # Leaf node
            if node.class_label == target_class:
                rules.append(list(current_path))
            return
        
        # Internal node: split on feature
        feature = node.feature_idx
        
        # Left branch: feature <= threshold (feature is False for binary)
        current_path.append((feature, False))
        extract_paths(node.left, current_path)
        current_path.pop()
        
        # Right branch: feature > threshold (feature is True for binary)
        current_path.append((feature, True))
        extract_paths(node.right, current_path)
        current_path.pop()
    
    extract_paths(tree, [])
    return rules


def predict_tree(tree: TreeNode, x: np.ndarray) -> int:
    """Predict class for single sample."""
    node = tree
    while node.class_label is None:
        if x[node.feature_idx] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.class_label


def predict_tree_batch(tree: TreeNode, X: np.ndarray) -> np.ndarray:
    """Predict classes for batch."""
    return np.array([predict_tree(tree, x) for x in X])


# ============================================================================
# TEACHER -> RULES DISTILLATION
# ============================================================================

class TeacherDistillation:
    """
    Distill a trained classifier (teacher) to interpretable DNF rules.
    
    Pipeline:
    1. Use teacher's predictions as soft labels
    2. Train decision tree on (features, teacher_predictions)
    3. Extract DNF rules from tree paths
    4. Optionally optimize rules with logic simplification
    """
    
    def __init__(
        self,
        n_classes: int,
        max_depth: int = 8,
        min_samples_leaf: int = 10,
    ):
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        
        self.trees: List[TreeNode] = []
        self.rules: Dict[int, List[List[Tuple[int, bool]]]] = {}
    
    def distill(
        self,
        X: np.ndarray,
        teacher_predictions: np.ndarray,
        optimize: bool = True,
        verbose: bool = False,
    ) -> Dict[int, List[List[Tuple[int, bool]]]]:
        """
        Distill teacher predictions to DNF rules.
        
        Args:
            X: (N, D) binary feature matrix
            teacher_predictions: (N,) predicted class labels from teacher
            optimize: whether to apply logic optimization
            verbose: print progress
        
        Returns:
            rules: dict mapping class -> DNF rules
        """
        # Handle both package and direct imports
        try:
            from route_c.logic import optimize_dnf
        except ImportError:
            from ..logic import optimize_dnf
        
        self.rules = {}
        
        for c in range(self.n_classes):
            if verbose:
                print(f"  Building tree for class {c}...")
            
            # One-vs-rest: target class vs others
            y_binary = (teacher_predictions == c).astype(int)
            
            # Build tree
            tree = build_decision_tree(
                X, y_binary,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            self.trees.append(tree)
            
            # Extract rules for class 1 (target class)
            rules_c = tree_to_dnf_rules(tree, target_class=1)
            
            if verbose:
                print(f"    Extracted {len(rules_c)} clauses")
            
            # Optimize
            if optimize and rules_c:
                rules_c, stats = optimize_dnf(rules_c, verbose=False)
                if verbose:
                    print(f"    After optimization: {len(rules_c)} clauses")
            
            self.rules[c] = rules_c
        
        return self.rules
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using extracted DNF rules.
        
        Uses argmax over rule-based scores (number of satisfied clauses).
        """
        # Handle both package and direct imports
        try:
            from route_c.logic import evaluate_dnf_batch
        except ImportError:
            from ..logic import evaluate_dnf_batch
        
        N = X.shape[0]
        scores = np.zeros((N, self.n_classes))
        
        for c, rules in self.rules.items():
            if rules:
                # Score = fraction of clauses satisfied
                for clause in rules:
                    # Check if clause is satisfied
                    clause_satisfied = np.ones(N, dtype=bool)
                    for var_idx, is_pos in clause:
                        if var_idx < X.shape[1]:
                            val = X[:, var_idx] > 0.5
                            clause_satisfied &= (val == is_pos)
                    scores[:, c] += clause_satisfied
                scores[:, c] /= max(1, len(rules))
        
        return np.argmax(scores, axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        preds = self.predict(X)
        return (preds == y).mean()
    
    def get_rule_complexity(self) -> Dict[str, Any]:
        """Get rule complexity statistics."""
        total_clauses = sum(len(r) for r in self.rules.values())
        total_literals = sum(sum(len(c) for c in r) for r in self.rules.values())
        
        return {
            'n_classes': self.n_classes,
            'total_clauses': total_clauses,
            'total_literals': total_literals,
            'avg_clauses_per_class': total_clauses / max(1, self.n_classes),
            'avg_literals_per_clause': total_literals / max(1, total_clauses),
            'clauses_per_class': {c: len(r) for c, r in self.rules.items()},
        }


# ============================================================================
# DIFFERENTIABLE DNF (OPTION B - SIMPLIFIED)
# ============================================================================

class DifferentiableDNF:
    """
    Differentiable DNF with soft gates (numpy implementation with gradient approximation).
    
    Architecture:
        - Include gates α_m,i ∈ [0,1]: whether literal i is used in minterm m
        - Sign gates π_m,i ∈ [0,1]: whether literal is positive or negated
        - Soft AND (product): minterm value = ∏_i soft_lit_i
        - Soft OR: class score = 1 - ∏_m (1 - minterm_m)
    
    Training:
        - Use straight-through estimator for binary decisions
        - Temperature annealing: τ decreases to harden decisions
        - Sparsity regularization: encourage small α values
    """
    
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_minterms: int = 30,
        temperature: float = 1.0,
        sparsity_weight: float = 0.01,
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_minterms = n_minterms
        self.temperature = temperature
        self.sparsity_weight = sparsity_weight
        
        # Parameters: (n_classes, n_minterms, n_features)
        # Raw values before sigmoid
        self.alpha_raw = np.random.randn(n_classes, n_minterms, n_features) * 0.1
        self.pi_raw = np.zeros((n_classes, n_minterms, n_features))  # Start at 0.5 probability
        
        # Learning rate
        self.lr = 0.01
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)), 
                       np.exp(x) / (1 + np.exp(x)))
    
    def _soft_literal(self, x: np.ndarray, alpha: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """
        Compute soft literal values.
        
        l̃ = π*x + (1-π)*(1-x)  # literal value (pos or neg)
        l̃' = (1-α) + α*l̃      # ignore if not selected
        """
        # x: (batch, features)
        # alpha, pi: (features,)
        
        # Literal value
        pi_soft = self._sigmoid(pi / self.temperature)
        lit_val = pi_soft * x + (1 - pi_soft) * (1 - x)
        
        # Include gate
        alpha_soft = self._sigmoid(self.alpha_raw / self.temperature)
        return (1 - alpha_soft) + alpha_soft * lit_val
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            X: (N, D) binary features
        
        Returns:
            probs: (N, C) class probabilities
        """
        N = X.shape[0]
        scores = np.zeros((N, self.n_classes))
        
        alpha_soft = self._sigmoid(self.alpha_raw / self.temperature)
        pi_soft = self._sigmoid(self.pi_raw / self.temperature)
        
        for c in range(self.n_classes):
            minterm_scores = np.ones((N, self.n_minterms))
            
            for m in range(self.n_minterms):
                for d in range(self.n_features):
                    # Literal value
                    lit_val = pi_soft[c, m, d] * X[:, d] + (1 - pi_soft[c, m, d]) * (1 - X[:, d])
                    # Include gate
                    soft_lit = (1 - alpha_soft[c, m, d]) + alpha_soft[c, m, d] * lit_val
                    # Soft AND
                    minterm_scores[:, m] *= soft_lit
            
            # Soft OR: 1 - prod(1 - minterm)
            scores[:, c] = 1 - np.prod(1 - minterm_scores, axis=1)
        
        # Softmax
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        return probs
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 64,
        anneal_schedule: Optional[List[float]] = None,
        verbose: bool = False,
    ):
        """
        Train the differentiable DNF.
        
        Uses gradient descent with numerical gradients (simplified).
        """
        N = X.shape[0]
        
        if anneal_schedule is None:
            # Default: start warm, end cold
            anneal_schedule = [1.0] * (n_epochs // 3) + \
                             [0.5] * (n_epochs // 3) + \
                             [0.1] * (n_epochs - 2 * (n_epochs // 3))
        
        for epoch in range(n_epochs):
            # Temperature annealing
            if epoch < len(anneal_schedule):
                self.temperature = anneal_schedule[epoch]
            
            # Shuffle
            perm = np.random.permutation(N)
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, N, batch_size):
                batch_idx = perm[i:i+batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Forward
                probs = self.forward(X_batch)
                
                # Cross-entropy loss
                eps = 1e-10
                ce_loss = -np.mean(np.log(probs[np.arange(len(y_batch)), y_batch] + eps))
                
                # Sparsity loss
                alpha_soft = self._sigmoid(self.alpha_raw / self.temperature)
                sparsity_loss = self.sparsity_weight * np.mean(alpha_soft)
                
                loss = ce_loss + sparsity_loss
                epoch_loss += loss
                n_batches += 1
                
                # Numerical gradient (simplified - slow but works)
                delta = 0.001
                
                for c in range(self.n_classes):
                    for m in range(self.n_minterms):
                        for d in range(self.n_features):
                            # Alpha gradient
                            self.alpha_raw[c, m, d] += delta
                            probs_plus = self.forward(X_batch)
                            loss_plus = -np.mean(np.log(probs_plus[np.arange(len(y_batch)), y_batch] + eps))
                            self.alpha_raw[c, m, d] -= delta
                            
                            grad_alpha = (loss_plus - ce_loss) / delta
                            self.alpha_raw[c, m, d] -= self.lr * grad_alpha
                            
                            # Pi gradient (less frequent update)
                            if epoch % 5 == 0:
                                self.pi_raw[c, m, d] += delta
                                probs_plus = self.forward(X_batch)
                                loss_plus = -np.mean(np.log(probs_plus[np.arange(len(y_batch)), y_batch] + eps))
                                self.pi_raw[c, m, d] -= delta
                                
                                grad_pi = (loss_plus - ce_loss) / delta
                                self.pi_raw[c, m, d] -= self.lr * grad_pi
            
            if verbose and epoch % 10 == 0:
                acc = self.score(X, y)
                print(f"  Epoch {epoch}: loss={epoch_loss/n_batches:.4f}, acc={acc:.3f}, τ={self.temperature:.2f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        preds = self.predict(X)
        return (preds == y).mean()
    
    def harden(self, threshold: float = 0.5) -> Dict[int, List[List[Tuple[int, bool]]]]:
        """
        Harden soft gates to discrete DNF rules.
        
        α > threshold -> literal is selected
        π > 0.5 -> positive literal, else negated
        
        Returns:
            rules: dict mapping class -> DNF rules
        """
        alpha_soft = self._sigmoid(self.alpha_raw / self.temperature)
        pi_soft = self._sigmoid(self.pi_raw / self.temperature)
        
        rules = {}
        
        for c in range(self.n_classes):
            clauses = []
            
            for m in range(self.n_minterms):
                clause = []
                for d in range(self.n_features):
                    if alpha_soft[c, m, d] > threshold:
                        is_pos = pi_soft[c, m, d] > 0.5
                        clause.append((d, is_pos))
                
                if clause:  # Non-empty clause
                    clauses.append(clause)
            
            rules[c] = clauses
        
        return rules
    
    def get_rule_complexity(self) -> Dict[str, Any]:
        """Get complexity of hardened rules."""
        rules = self.harden()
        total_clauses = sum(len(r) for r in rules.values())
        total_literals = sum(sum(len(c) for c in r) for r in rules.values())
        
        return {
            'n_classes': self.n_classes,
            'n_minterms': self.n_minterms,
            'total_clauses': total_clauses,
            'total_literals': total_literals,
            'avg_literals_per_clause': total_literals / max(1, total_clauses),
        }


# ============================================================================
# FEATURE BINARIZATION FOR DNF
# ============================================================================

def binarize_features(
    features: np.ndarray,
    thresholds: Optional[List[float]] = None,
    n_bins: int = 3,
) -> np.ndarray:
    """
    Binarize continuous features for DNF input.
    
    Each feature f becomes n_bins binary features:
        f > t1, f > t2, ..., f > t_{n_bins}
    
    Args:
        features: (N, D) continuous features
        thresholds: list of thresholds; if None, use percentiles
        n_bins: number of binary features per original feature
    
    Returns:
        binary_features: (N, D * n_bins) binary matrix
    """
    N, D = features.shape
    
    if thresholds is None:
        # Use percentiles
        percentiles = np.linspace(100/(n_bins+1), 100*n_bins/(n_bins+1), n_bins)
        thresholds = [np.percentile(features, p, axis=0) for p in percentiles]
    
    binary = []
    for t in thresholds:
        binary.append((features > t).astype(np.float32))
    
    return np.concatenate(binary, axis=1)


def create_dnf_features_from_histogram(
    features: np.ndarray,
    top_k: int = 50,
) -> Tuple[np.ndarray, List[int]]:
    """
    Create binary DNF features from histogram features.
    
    Selects top-k most variable features and binarizes them.
    
    Returns:
        binary_features: (N, top_k * n_bins) binary matrix
        selected_indices: indices of selected original features
    """
    # Find most variable features
    variances = np.var(features, axis=0)
    selected_indices = np.argsort(variances)[-top_k:]
    
    selected_features = features[:, selected_indices]
    binary_features = binarize_features(selected_features, n_bins=2)
    
    return binary_features, selected_indices.tolist()
