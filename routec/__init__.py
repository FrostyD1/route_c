"""
Route C: Discrete-Core Inference Paradigm
==========================================
Frozen API contract (Phase 0).

Three contracts:
  1. Representation: z ∈ {0,1}^{k×H×W}
  2. Energy: E_core(z) + E_obs(z,o) + optional E_relation(z)
  3. Inference: q_φ(z|o,m) via amortized/iterative, MCMC diagnostic only

No task-specific loss enters the world model.
"""
