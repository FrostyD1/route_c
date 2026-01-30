"""
Route C Logic Optimization
==========================
DNF optimization utilities:
- Canonicalization
- Subsumption elimination
- Factoring
- CSE / DAG sharing

DNF Form: OR of ANDs (clauses)
    rule = [clause1, clause2, ...]
    clause = [(var_idx, sign), ...]  # sign: True=positive, False=negated
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional, FrozenSet
from collections import defaultdict
from dataclasses import dataclass


# ============================================================================
# DATA STRUCTURES
# ============================================================================

Literal = Tuple[int, bool]  # (variable_index, is_positive)
Clause = FrozenSet[Literal]  # AND of literals
DNF = Set[Clause]  # OR of clauses


@dataclass
class OptimizationStats:
    """Statistics from DNF optimization."""
    original_clauses: int
    final_clauses: int
    original_literals: int
    final_literals: int
    removed_redundant: int
    removed_subsumed: int
    factored: int
    shared_nodes: int  # For DAG representation
    
    @property
    def clause_reduction(self) -> float:
        if self.original_clauses == 0:
            return 0.0
        return 1.0 - self.final_clauses / self.original_clauses
    
    @property
    def literal_reduction(self) -> float:
        if self.original_literals == 0:
            return 0.0
        return 1.0 - self.final_literals / self.original_literals


def clause_from_list(literals: List[Tuple[int, bool]]) -> Clause:
    """Create clause from list of (var_idx, sign) tuples."""
    return frozenset(literals)


def dnf_from_list(clauses: List[List[Tuple[int, bool]]]) -> DNF:
    """Create DNF from nested list representation."""
    return {clause_from_list(c) for c in clauses}


def dnf_to_list(dnf: DNF) -> List[List[Tuple[int, bool]]]:
    """Convert DNF to nested list for serialization."""
    return [list(clause) for clause in dnf]


# ============================================================================
# CANONICALIZATION
# ============================================================================

def canonicalize_clause(clause: Clause) -> Clause:
    """
    Canonicalize a clause:
    1. Sort literals by variable index
    2. Remove duplicate literals
    
    Returns the clause in canonical form.
    """
    # FrozenSet already handles duplicates, just need to check for contradictions
    return clause


def detect_contradiction(clause: Clause) -> bool:
    """
    Check if clause contains both x and ¬x (always False).
    """
    vars_seen: Dict[int, bool] = {}
    for var_idx, is_pos in clause:
        if var_idx in vars_seen:
            if vars_seen[var_idx] != is_pos:
                return True  # Contradiction: x AND ¬x
        vars_seen[var_idx] = is_pos
    return False


def remove_contradictions(dnf: DNF) -> DNF:
    """Remove clauses that contain contradictions (x AND ¬x)."""
    return {c for c in dnf if not detect_contradiction(c)}


def remove_duplicates(dnf: DNF) -> DNF:
    """Remove duplicate clauses. (Set already handles this, but explicit for clarity)"""
    return dnf  # Already a set


# ============================================================================
# SUBSUMPTION
# ============================================================================

def clause_subsumes(c1: Clause, c2: Clause) -> bool:
    """
    Check if c1 subsumes c2 (c1 is more general).
    
    c1 subsumes c2 if c1 ⊆ c2 (c1 has fewer or equal literals, all in c2).
    In DNF, a subsuming clause makes the subsumed one redundant.
    
    Example: (A) subsumes (A ∧ B) because if A is true, (A ∧ B) ∨ (A) = (A)
    """
    return c1.issubset(c2) and c1 != c2


def subsumption_elimination(dnf: DNF) -> Tuple[DNF, int]:
    """
    Remove subsumed clauses.
    
    If clause C1 subsumes C2 (C1 ⊂ C2), remove C2.
    
    Returns:
        optimized_dnf: DNF with subsumed clauses removed
        n_removed: number of clauses removed
    """
    clauses = list(dnf)
    to_remove = set()
    
    for i, c1 in enumerate(clauses):
        if i in to_remove:
            continue
        for j, c2 in enumerate(clauses):
            if i == j or j in to_remove:
                continue
            if clause_subsumes(c1, c2):
                to_remove.add(j)
    
    n_removed = len(to_remove)
    result = {c for i, c in enumerate(clauses) if i not in to_remove}
    return result, n_removed


# ============================================================================
# ABSORPTION
# ============================================================================

def absorption(dnf: DNF) -> Tuple[DNF, int]:
    """
    Apply absorption law: A ∨ (A ∧ B) = A
    
    This is essentially subsumption elimination from the other direction.
    """
    return subsumption_elimination(dnf)


# ============================================================================
# FACTORING (BASIC)
# ============================================================================

def find_common_literal(clauses: List[Clause]) -> Optional[Literal]:
    """
    Find a literal that appears in all clauses.
    """
    if not clauses:
        return None
    
    common = set(clauses[0])
    for clause in clauses[1:]:
        common &= set(clause)
    
    if common:
        return min(common)  # Return one common literal
    return None


def factor_dnf(dnf: DNF) -> Tuple[DNF, int]:
    """
    Basic factoring: (A ∧ B) ∨ (A ∧ C) -> A ∧ (B ∨ C)
    
    This is a heuristic optimization that groups clauses with common prefixes.
    Full factoring is complex; this implements a greedy approximation.
    
    Returns:
        optimized_dnf: factored DNF (may have nested structure flattened)
        n_factored: number of factoring operations applied
    """
    clauses = list(dnf)
    n_factored = 0
    
    # Find groups of clauses that share a common literal
    changed = True
    while changed:
        changed = False
        
        # Group clauses by their literals
        literal_to_clauses: Dict[Literal, List[int]] = defaultdict(list)
        for i, clause in enumerate(clauses):
            for lit in clause:
                literal_to_clauses[lit].append(i)
        
        # Find literal appearing in multiple clauses
        best_lit = None
        best_count = 1
        for lit, indices in literal_to_clauses.items():
            if len(indices) > best_count:
                best_lit = lit
                best_count = len(indices)
        
        if best_lit is not None and best_count >= 2:
            # Factor out this literal
            indices = literal_to_clauses[best_lit]
            
            # Remove the common literal from these clauses
            new_clauses = []
            for i, clause in enumerate(clauses):
                if i in indices:
                    # Remove the factored literal
                    remaining = clause - {best_lit}
                    if remaining:
                        new_clauses.append(remaining)
                    else:
                        # Clause becomes just the literal
                        # A ∨ (A ∧ B) case - keep as is for now
                        new_clauses.append(clause)
                else:
                    new_clauses.append(clause)
            
            # Only apply if it reduces complexity
            old_lits = sum(len(c) for c in clauses)
            new_lits = sum(len(c) for c in new_clauses)
            
            if new_lits < old_lits:
                clauses = new_clauses
                n_factored += 1
                changed = True
    
    return set(clauses), n_factored


# ============================================================================
# CSE / DAG SHARING
# ============================================================================

@dataclass
class DAGNode:
    """Node in DAG representation of logic formula."""
    op: str  # 'var', 'not', 'and', 'or'
    var_idx: Optional[int] = None  # For 'var' nodes
    children: Tuple['DAGNode', ...] = ()
    
    def __hash__(self):
        return hash((self.op, self.var_idx, self.children))
    
    def __eq__(self, other):
        if not isinstance(other, DAGNode):
            return False
        return (self.op, self.var_idx, self.children) == (other.op, other.var_idx, other.children)


def dnf_to_dag(dnf: DNF) -> Tuple[DAGNode, Dict[DAGNode, int]]:
    """
    Convert DNF to DAG with Common Subexpression Elimination.
    
    Returns:
        root: root DAGNode representing the formula
        node_count: dict mapping nodes to their usage count (for sharing stats)
    """
    node_cache: Dict[Tuple, DAGNode] = {}
    node_count: Dict[DAGNode, int] = defaultdict(int)
    
    def get_or_create_var(idx: int) -> DAGNode:
        key = ('var', idx)
        if key not in node_cache:
            node_cache[key] = DAGNode(op='var', var_idx=idx)
        node = node_cache[key]
        node_count[node] += 1
        return node
    
    def get_or_create_not(child: DAGNode) -> DAGNode:
        key = ('not', id(child))
        if key not in node_cache:
            node_cache[key] = DAGNode(op='not', children=(child,))
        node = node_cache[key]
        node_count[node] += 1
        return node
    
    def get_or_create_and(children: Tuple[DAGNode, ...]) -> DAGNode:
        sorted_children = tuple(sorted(children, key=id))
        key = ('and', tuple(id(c) for c in sorted_children))
        if key not in node_cache:
            node_cache[key] = DAGNode(op='and', children=sorted_children)
        node = node_cache[key]
        node_count[node] += 1
        return node
    
    def get_or_create_or(children: Tuple[DAGNode, ...]) -> DAGNode:
        sorted_children = tuple(sorted(children, key=id))
        key = ('or', tuple(id(c) for c in sorted_children))
        if key not in node_cache:
            node_cache[key] = DAGNode(op='or', children=sorted_children)
        node = node_cache[key]
        node_count[node] += 1
        return node
    
    # Convert each clause to AND node
    clause_nodes = []
    for clause in dnf:
        lit_nodes = []
        for var_idx, is_pos in sorted(clause):
            var_node = get_or_create_var(var_idx)
            if is_pos:
                lit_nodes.append(var_node)
            else:
                lit_nodes.append(get_or_create_not(var_node))
        
        if len(lit_nodes) == 1:
            clause_nodes.append(lit_nodes[0])
        else:
            clause_nodes.append(get_or_create_and(tuple(lit_nodes)))
    
    # Combine clauses with OR
    if len(clause_nodes) == 1:
        root = clause_nodes[0]
    else:
        root = get_or_create_or(tuple(clause_nodes))
    
    return root, dict(node_count)


def count_shared_nodes(node_count: Dict[DAGNode, int]) -> int:
    """Count nodes that are shared (used more than once)."""
    return sum(1 for count in node_count.values() if count > 1)


# ============================================================================
# MAIN OPTIMIZATION PIPELINE
# ============================================================================

def optimize_dnf(
    rules: List[List[Tuple[int, bool]]],
    verbose: bool = False,
) -> Tuple[List[List[Tuple[int, bool]]], OptimizationStats]:
    """
    Apply all optimizations to DNF rules.
    
    Pipeline:
    1. Remove contradictions
    2. Remove duplicates
    3. Subsumption elimination
    4. Basic factoring
    5. CSE analysis
    
    Args:
        rules: DNF as nested list of (var_idx, sign) tuples
        verbose: print optimization progress
    
    Returns:
        optimized_rules: optimized DNF in same format
        stats: optimization statistics
    """
    # Convert to internal representation
    dnf = dnf_from_list(rules)
    
    original_clauses = len(dnf)
    original_literals = sum(len(c) for c in dnf)
    
    if verbose:
        print(f"Original: {original_clauses} clauses, {original_literals} literals")
    
    # 1. Remove contradictions
    dnf = remove_contradictions(dnf)
    if verbose:
        removed = original_clauses - len(dnf)
        if removed:
            print(f"  Removed {removed} contradictory clauses")
    
    # 2. Remove duplicates (implicit in set)
    dnf = remove_duplicates(dnf)
    
    # 3. Subsumption elimination
    dnf, n_subsumed = subsumption_elimination(dnf)
    if verbose and n_subsumed:
        print(f"  Removed {n_subsumed} subsumed clauses")
    
    # 4. Basic factoring
    dnf, n_factored = factor_dnf(dnf)
    if verbose and n_factored:
        print(f"  Applied {n_factored} factoring operations")
    
    # 5. CSE analysis
    _, node_count = dnf_to_dag(dnf)
    n_shared = count_shared_nodes(node_count)
    if verbose:
        print(f"  DAG has {n_shared} shared nodes")
    
    final_clauses = len(dnf)
    final_literals = sum(len(c) for c in dnf)
    
    stats = OptimizationStats(
        original_clauses=original_clauses,
        final_clauses=final_clauses,
        original_literals=original_literals,
        final_literals=final_literals,
        removed_redundant=original_clauses - final_clauses - n_subsumed,
        removed_subsumed=n_subsumed,
        factored=n_factored,
        shared_nodes=n_shared,
    )
    
    if verbose:
        print(f"Final: {final_clauses} clauses, {final_literals} literals")
        print(f"Reduction: {stats.clause_reduction:.1%} clauses, {stats.literal_reduction:.1%} literals")
    
    # Convert back to list format
    optimized = dnf_to_list(dnf)
    
    return optimized, stats


# ============================================================================
# DNF EVALUATION
# ============================================================================

def evaluate_clause(clause: List[Tuple[int, bool]], x: np.ndarray) -> bool:
    """Evaluate a clause (AND of literals) on binary input x."""
    for var_idx, is_pos in clause:
        val = x[var_idx] > 0.5 if isinstance(x[var_idx], float) else bool(x[var_idx])
        lit_val = val if is_pos else not val
        if not lit_val:
            return False
    return True


def evaluate_dnf(rules: List[List[Tuple[int, bool]]], x: np.ndarray) -> bool:
    """Evaluate DNF (OR of clauses) on binary input x."""
    for clause in rules:
        if evaluate_clause(clause, x):
            return True
    return False


def evaluate_dnf_batch(
    rules: List[List[Tuple[int, bool]]],
    X: np.ndarray,
) -> np.ndarray:
    """
    Evaluate DNF on batch of inputs.
    
    Args:
        rules: DNF rules
        X: (N, D) binary input matrix
    
    Returns:
        results: (N,) boolean array
    """
    N = X.shape[0]
    results = np.zeros(N, dtype=bool)
    for i in range(N):
        results[i] = evaluate_dnf(rules, X[i])
    return results


# ============================================================================
# PRETTY PRINTING
# ============================================================================

def literal_to_str(lit: Tuple[int, bool], var_names: Optional[List[str]] = None) -> str:
    """Convert literal to string."""
    var_idx, is_pos = lit
    if var_names and var_idx < len(var_names):
        name = var_names[var_idx]
    else:
        name = f"x{var_idx}"
    return name if is_pos else f"¬{name}"


def clause_to_str(clause: List[Tuple[int, bool]], var_names: Optional[List[str]] = None) -> str:
    """Convert clause to string."""
    if not clause:
        return "TRUE"
    return " ∧ ".join(literal_to_str(lit, var_names) for lit in sorted(clause))


def dnf_to_str(rules: List[List[Tuple[int, bool]]], var_names: Optional[List[str]] = None, max_clauses: int = 10) -> str:
    """Convert DNF to human-readable string."""
    if not rules:
        return "FALSE"
    
    clauses_str = []
    for i, clause in enumerate(rules):
        if i >= max_clauses:
            clauses_str.append(f"... ({len(rules) - max_clauses} more clauses)")
            break
        clauses_str.append(f"({clause_to_str(clause, var_names)})")
    
    return " ∨ ".join(clauses_str)
