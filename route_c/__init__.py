"""
Route C: Boundary Translation + Discrete Core Closure
=====================================================

A unified discrete-world modeling framework that is:
(a) Scalable
(b) Supports inference-time dynamics (iterative solving)
(c) Supports interventions and structural priors (symmetry/topology)
(d) Can degrade to continuous-style computation as performance lower bound

Mathematical Model
------------------
- Continuous observations are boundary/interface signals only
- Reasoning/inference happens in a closed discrete domain (z-space)
- ADC: o(d) -> z_k(d) via encoder + quantizer
- DAC: z_k(d) -> ô(d) for rendering back to continuous
- Inference: iterative solving (Gibbs/Metropolis/ICM) in discrete space

Key Insight: Expressivity Lower Bound
------------------------------------
Discrete system is NOT inherently weaker than continuous. If discrete core
includes (or can simulate) add/mul, it can emulate ResNet etc.
Any observed weakness is due to effective capacity / learnability caused by
restricted latent family or weak learner, NOT fundamental expressivity limit.
"""

__version__ = "0.2.0"
