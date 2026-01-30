"""
Route C: Logic Distillation
===========================

Distill trained classifier/predictor into explicit logic rules.

Pipeline:
1. Teacher model: pω(y|z) classifier on binary z
2. Train decision trees (one-vs-rest) to approximate teacher
3. Convert tree paths to DNF (Disjunctive Normal Form)
4. Apply logic simplification (subsumption, factoring)
5. Output: DNF rules + optional Verilog netlist

Key difference from previous approach:
- Features are z bits directly (k*H*W binary variables)
- NOT handcrafted pixel histograms
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional, FrozenSet
from dataclasses import dataclass


# ============================================================================
# DATA STRUCTURES
# ============================================================================

Literal = Tuple[int, bool]  # (variable_index, is_positive)
Clause = FrozenSet[Literal]  # AND of literals
DNF = Set[Clause]  # OR of ANDs


@dataclass
class DistillationResult:
    """Result of distilling one class."""
    class_idx: int
    dnf: DNF
    n_clauses: int
    n_literals: int
    fidelity: float  # Agreement with teacher


# ============================================================================
# DECISION TREE (Pure NumPy)
# ============================================================================

@dataclass
class TreeNode:
    """Binary decision tree node."""
    is_leaf: bool = False
    feature_idx: int = -1
    threshold: float = 0.5  # For binary: split at 0.5
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    prediction: int = 0  # For leaf nodes
    class_proba: Optional[np.ndarray] = None


class DecisionTreeBinary:
    """
    Decision tree for binary features (z bits).
    
    Since features are {0,1}, threshold is always 0.5:
    - feature < 0.5 → go left (feature = 0)
    - feature >= 0.5 → go right (feature = 1)
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        min_samples_split: int = 10,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_features = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit tree on binary features.
        
        Args:
            X: (N, D) binary features {0,1}
            y: (N,) binary labels {0,1}
        """
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)
    
    def _gini(self, y: np.ndarray) -> float:
        """Gini impurity."""
        if len(y) == 0:
            return 0.0
        p1 = y.mean()
        return 2 * p1 * (1 - p1)
    
    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[int, float]:
        """Find best feature to split on."""
        best_gain = -1
        best_feature = 0
        
        n_samples = len(y)
        current_gini = self._gini(y)
        
        for feat in range(X.shape[1]):
            # Binary split at 0.5
            left_mask = X[:, feat] < 0.5
            right_mask = ~left_mask
            
            n_left = left_mask.sum()
            n_right = right_mask.sum()
            
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            
            gini_left = self._gini(y[left_mask])
            gini_right = self._gini(y[right_mask])
            
            weighted_gini = (n_left * gini_left + n_right * gini_right) / n_samples
            gain = current_gini - weighted_gini
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feat
        
        return best_feature, best_gain
    
    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
    ) -> TreeNode:
        """Recursively build tree."""
        n_samples = len(y)
        n_pos = y.sum()
        
        # Stopping conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_pos == 0 or n_pos == n_samples):
            return TreeNode(
                is_leaf=True,
                prediction=1 if n_pos > n_samples / 2 else 0,
                class_proba=np.array([1 - n_pos/n_samples, n_pos/n_samples])
            )
        
        # Find best split
        best_feat, best_gain = self._find_best_split(X, y)
        
        if best_gain <= 0:
            return TreeNode(
                is_leaf=True,
                prediction=1 if n_pos > n_samples / 2 else 0,
                class_proba=np.array([1 - n_pos/n_samples, n_pos/n_samples])
            )
        
        # Split
        left_mask = X[:, best_feat] < 0.5
        right_mask = ~left_mask
        
        return TreeNode(
            is_leaf=False,
            feature_idx=best_feat,
            threshold=0.5,
            left=self._build_tree(X[left_mask], y[left_mask], depth + 1),
            right=self._build_tree(X[right_mask], y[right_mask], depth + 1),
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.array([self._predict_one(x) for x in X])
    
    def _predict_one(self, x: np.ndarray) -> int:
        node = self.root
        while not node.is_leaf:
            if x[node.feature_idx] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return np.array([self._predict_proba_one(x) for x in X])
    
    def _predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        node = self.root
        while not node.is_leaf:
            if x[node.feature_idx] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.class_proba


# ============================================================================
# TREE → DNF CONVERSION
# ============================================================================

def tree_to_dnf(tree: DecisionTreeBinary, target_class: int = 1) -> DNF:
    """
    Convert decision tree to DNF.
    
    Each path from root to a leaf predicting target_class
    becomes an AND clause (conjunction of literals).
    All such clauses are ORed together.
    
    Args:
        tree: trained decision tree
        target_class: which class to extract rules for
    
    Returns:
        DNF: set of clauses (OR of ANDs)
    """
    dnf: DNF = set()
    
    def traverse(node: TreeNode, path: List[Literal]):
        if node.is_leaf:
            if node.prediction == target_class:
                # This path leads to target class → add clause
                dnf.add(frozenset(path))
            return
        
        feat = node.feature_idx
        
        # Left branch: feature < 0.5 → feature = 0 → negative literal
        traverse(node.left, path + [(feat, False)])
        
        # Right branch: feature >= 0.5 → feature = 1 → positive literal
        traverse(node.right, path + [(feat, True)])
    
    traverse(tree.root, [])
    return dnf


# ============================================================================
# LOGIC SIMPLIFICATION
# ============================================================================

def remove_contradictions(dnf: DNF) -> DNF:
    """Remove clauses with x AND ¬x."""
    result = set()
    for clause in dnf:
        vars_pos = {lit[0] for lit in clause if lit[1]}
        vars_neg = {lit[0] for lit in clause if not lit[1]}
        if not vars_pos.intersection(vars_neg):
            result.add(clause)
    return result


def subsumption_elimination(dnf: DNF) -> DNF:
    """
    Remove subsumed clauses.
    
    If A ⊆ B (A has fewer literals), then A subsumes B.
    A ∨ (A ∧ x) = A, so we can remove B.
    """
    clauses = sorted(dnf, key=len)  # Shorter first
    result = set()
    
    for clause in clauses:
        # Check if any existing clause subsumes this one
        subsumed = False
        for existing in result:
            if existing.issubset(clause):
                subsumed = True
                break
        
        if not subsumed:
            result.add(clause)
    
    return result


def factor_common(dnf: DNF) -> Tuple[DNF, int]:
    """
    Simple factoring: find common literals.
    
    (A ∧ B) ∨ (A ∧ C) → A ∧ (B ∨ C)
    
    Returns simplified DNF and count of factored terms.
    """
    # Find most common literal
    literal_counts: Dict[Literal, int] = {}
    for clause in dnf:
        for lit in clause:
            literal_counts[lit] = literal_counts.get(lit, 0) + 1
    
    if not literal_counts:
        return dnf, 0
    
    # Find literal appearing in most clauses
    best_lit = max(literal_counts.keys(), key=lambda l: literal_counts[l])
    count = literal_counts[best_lit]
    
    if count < 2:
        return dnf, 0
    
    # Factor out this literal
    with_lit = {c for c in dnf if best_lit in c}
    without_lit = dnf - with_lit
    
    # Remove the literal from clauses that have it
    factored = {frozenset(c - {best_lit}) for c in with_lit}
    factored = {c for c in factored if c}  # Remove empty
    
    # Result: (best_lit AND (factored)) OR without_lit
    # For DNF representation, we can't directly represent this
    # But we can simplify the factored part
    
    return dnf, 0  # Simplified version: just return original


def optimize_dnf(dnf: DNF) -> Tuple[DNF, Dict[str, int]]:
    """
    Full optimization pipeline.
    
    Returns:
        optimized_dnf: simplified DNF
        stats: optimization statistics
    """
    original_clauses = len(dnf)
    original_literals = sum(len(c) for c in dnf)
    
    # Remove contradictions
    dnf = remove_contradictions(dnf)
    
    # Subsumption elimination
    dnf = subsumption_elimination(dnf)
    
    final_clauses = len(dnf)
    final_literals = sum(len(c) for c in dnf)
    
    stats = {
        'original_clauses': original_clauses,
        'final_clauses': final_clauses,
        'original_literals': original_literals,
        'final_literals': final_literals,
        'clauses_removed': original_clauses - final_clauses,
        'literals_removed': original_literals - final_literals,
    }
    
    return dnf, stats


# ============================================================================
# FULL DISTILLATION PIPELINE
# ============================================================================

class LogicDistiller:
    """
    Distill classifier into DNF logic rules.
    
    Pipeline:
    1. Get teacher predictions on training z
    2. Train decision trees (one per class)
    3. Convert to DNF
    4. Simplify
    """
    
    def __init__(
        self,
        n_classes: int = 10,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
    ):
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees: List[DecisionTreeBinary] = []
        self.dnfs: List[DNF] = []
        self.feature_names: List[str] = []
    
    def fit(
        self,
        Z: np.ndarray,
        teacher_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Distill teacher predictions into logic rules.
        
        Args:
            Z: (N, D) binary features (flattened z)
            teacher_labels: (N,) teacher's predictions
            feature_names: optional names for features (z bits)
        """
        N, D = Z.shape
        
        # Generate feature names if not provided
        if feature_names is None:
            self.feature_names = [f"z[{i}]" for i in range(D)]
        else:
            self.feature_names = feature_names
        
        self.trees = []
        self.dnfs = []
        
        for c in range(self.n_classes):
            # Binary target: is this class?
            y_binary = (teacher_labels == c).astype(np.int32)
            
            # Train tree
            tree = DecisionTreeBinary(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(Z, y_binary)
            self.trees.append(tree)
            
            # Convert to DNF
            dnf = tree_to_dnf(tree, target_class=1)
            dnf, _ = optimize_dnf(dnf)
            self.dnfs.append(dnf)
    
    def predict(self, Z: np.ndarray) -> np.ndarray:
        """
        Predict using distilled trees.
        
        Uses argmax over tree confidences.
        """
        N = Z.shape[0]
        scores = np.zeros((N, self.n_classes))
        
        for c, tree in enumerate(self.trees):
            proba = tree.predict_proba(Z)
            scores[:, c] = proba[:, 1]  # P(class=c)
        
        return scores.argmax(axis=1)
    
    def fidelity(
        self,
        Z: np.ndarray,
        teacher_labels: np.ndarray,
    ) -> float:
        """Compute agreement with teacher."""
        preds = self.predict(Z)
        return (preds == teacher_labels).mean()
    
    def get_stats(self) -> Dict[str, any]:
        """Get distillation statistics."""
        total_clauses = sum(len(dnf) for dnf in self.dnfs)
        total_literals = sum(sum(len(c) for c in dnf) for dnf in self.dnfs)
        
        return {
            'n_classes': self.n_classes,
            'total_clauses': total_clauses,
            'total_literals': total_literals,
            'avg_clauses_per_class': total_clauses / self.n_classes,
            'avg_literals_per_clause': total_literals / max(1, total_clauses),
        }
    
    def dnf_to_str(self, class_idx: int) -> str:
        """Get string representation of DNF for a class."""
        dnf = self.dnfs[class_idx]
        
        if not dnf:
            return "FALSE"
        
        clause_strs = []
        for clause in dnf:
            if not clause:
                clause_strs.append("TRUE")
            else:
                lits = []
                for var_idx, is_pos in sorted(clause):
                    name = self.feature_names[var_idx]
                    if is_pos:
                        lits.append(name)
                    else:
                        lits.append(f"~{name}")
                clause_strs.append(" & ".join(lits))
        
        return " | ".join(f"({c})" for c in clause_strs)


# ============================================================================
# VERILOG GENERATION
# ============================================================================

def dnf_to_verilog(
    dnfs: List[DNF],
    feature_names: List[str],
    module_name: str = "classifier",
) -> str:
    """
    Generate Verilog module implementing DNF classification.
    
    Args:
        dnfs: list of DNF per class
        feature_names: names of input bits
        module_name: name of Verilog module
    
    Returns:
        Verilog code string
    """
    n_classes = len(dnfs)
    n_features = len(feature_names)
    
    lines = []
    lines.append(f"// Auto-generated Route C classifier")
    lines.append(f"// {n_classes} classes, {n_features} input bits")
    lines.append(f"")
    lines.append(f"module {module_name} (")
    lines.append(f"    input wire [{n_features-1}:0] z,")
    lines.append(f"    output wire [{n_classes-1}:0] class_scores")
    lines.append(f");")
    lines.append(f"")
    
    # Generate logic for each class
    for c, dnf in enumerate(dnfs):
        if not dnf:
            lines.append(f"    assign class_scores[{c}] = 1'b0;")
        elif len(dnf) == 1 and not list(dnf)[0]:
            lines.append(f"    assign class_scores[{c}] = 1'b1;")
        else:
            clauses = []
            for clause in dnf:
                if not clause:
                    clauses.append("1'b1")
                else:
                    lits = []
                    for var_idx, is_pos in sorted(clause):
                        if is_pos:
                            lits.append(f"z[{var_idx}]")
                        else:
                            lits.append(f"~z[{var_idx}]")
                    clauses.append("(" + " & ".join(lits) + ")")
            
            expr = " | ".join(clauses)
            lines.append(f"    assign class_scores[{c}] = {expr};")
    
    lines.append(f"")
    lines.append(f"endmodule")
    
    return "\n".join(lines)


def generate_feature_names(n_bits: int, latent_size: int) -> List[str]:
    """Generate meaningful feature names for z bits."""
    names = []
    for b in range(n_bits):
        for i in range(latent_size):
            for j in range(latent_size):
                names.append(f"z_{b}_{i}_{j}")
    return names
