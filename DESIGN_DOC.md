# Route C: Learned Discrete Routing + Amortized Inpainting — Design Document

**Version 2 — incorporates reviewer feedback (2026-01-30)**

## 1. Problem Diagnosis: Why Fixed XOR/Hamming Gates Are Insufficient

**Exp09 finding:** D-RoPE with fixed `gate(i→j) = 1[popcount(z_i XOR z_j) < τ]` gives:
- center occlusion: Δacc = -1.6% (worse than baseline -0.4%), sweeps_95% = 22 > baseline 20
- random occlusion: Δacc = +2.2% (better than baseline -1.4%), sweeps_95% = 23 > baseline 20
- sparsity ablation: all densities yield negative Δacc on center (-2.0% to -2.5%)

**Root cause analysis:**

1. **Fixed Hamming = isotropic metric.** All k bits contribute equally to the gate. But not all bits carry equal semantic importance. Bit 0 (encoding coarse structure) should matter more for routing than bit 7 (encoding fine noise). The fixed Hamming distance treats `z_i XOR z_j = 10000000` identically to `z_i XOR z_j = 00000001`.

2. **No learned routing.** The Routing Transformer (Roy et al., 2021) showed that **learned** k-means centroids outperform fixed LSH (Reformer) because routing adapts to data distribution. Our XOR gate is analogous to fixed LSH — data-independent.

3. **Test-time MCMC is slow and fragile.** 20-30 sweeps of block Gibbs per sample at ~2-5s/sample is impractical. MaskGIT (Chang et al., 2022) showed that amortized parallel decoding (8-12 steps) replaces sequential MCMC.

4. **D-RoPE is NOT a universal improvement — it is a strong prior without evidence gating.** The data directly shows: sweeps_95% increases (22/23 vs 20), not decreases. D-RoPE changes the energy landscape and proposal distribution, which sometimes aligns with the task (random occlusion: periphery provides useful long-range signal) and sometimes introduces hallucination (center occlusion: distant tokens may route incorrect structure into the missing core). **The XOR gate has no mechanism to suppress bad matches — it treats all low-Hamming pairs equally regardless of semantic relevance.**

## 2. Literature Alignment

### 2.1 Routing Transformer (Roy et al., 2021) — Learned Clustering for Sparse Attention
- **Mechanism:** Cluster Q/K via online k-means on unit sphere. Attend only within clusters. Centroids updated via EMA: `μ_new = γ·μ_old + (1-γ)·mean(assigned vectors)`. Commitment loss `L_commit = λ·||z - sg(μ)||²` prevents drift.
- **Complexity:** O(N√N·d) vs O(N²d), with balanced cluster assignment (top-k by distance to centroid).
- **Our adaptation:** Replace fixed Hamming threshold with learned weighted Hamming using soft gate (see §3). The per-bit weight vector `w` plays the role of learned centroid — it defines which dimensions of the binary code are relevant for routing.

### 2.2 Reformer (Kitaev et al., 2020) — LSH as Sparse Attention
- **Mechanism:** Angular LSH via random hyperplane projections. Multi-round (L=8) for recall. Hash: `h(x) = argmax([xR; -xR])` with random R.
- **Key insight for Route C:** For `z ∈ {0,1}^k`, bit-sampling is a standard LSH family for Hamming space. Hash = select random subset of b bits. Collision probability = `(1 - d_H/k)^b`. Simple, zero-overhead on binary codes.
- **Our adaptation:** Bit-sampling LSH as a scalable candidate generation method. On small grids (7×7, N=49), fixed spatial offsets or brute-force top-K may be equally fast. LSH becomes essential when scaling to larger grids (e.g., 32×32 tokens for higher-resolution inputs). We include both fixed-offset and LSH baselines for fair comparison.

### 2.3 Modern Hopfield Networks (Ramsauer et al., 2021) — Energy Perspective
- **Energy:** `E(ξ) = -β⁻¹ log Σ exp(β·ξᵀxμ) + ½ξᵀξ`
- **Update = attention:** `ξ_new = Xᵀ softmax(β·X·ξ)`. One update step = one attention layer.
- **Capacity:** Exponential O(exp(d)) vs classical Hopfield O(d).
- **Our connection:** Our energy-based inference `E(z) = λ_core·E_core + λ_obs·E_obs + λ_rope·E_rope` with Gibbs sampling is analogous to iterative Hopfield retrieval. The amortized inpainting network (Route B) replaces this iteration with a single learned forward pass — equivalent to replacing Hopfield dynamics with a trained attention layer. This is why Route B is the primary path: it amortizes the retrieval.

### 2.4 VQ-VAE / FSQ — Discrete Token Learning
- **VQ-VAE (van den Oord et al., 2017):** Codebook `{e_k}`, commitment loss `β·||z_e - sg(e_k)||²`, EMA updates. Suffers codebook collapse.
- **FSQ (Mentzer et al., 2023, arXiv:2309.15505):** `ẑ_i = round(L/2 · tanh(z_i))`. No codebook, no collapse. Implicit codebook size = ∏L_i.
- **Our position:** We use Gumbel-Sigmoid binary quantization (STE). FSQ is a natural extension for multi-level tokens. Binary STE is simpler and maps to hardware (XNOR gates).

### 2.5 BinaryConnect / XNOR-Net — STE for Binary Parameters
- **BinaryConnect (Courbariaux et al., 2015):** Forward `z = sign(w)`, backward `∂L/∂w ≈ ∂L/∂z · 1[|w|≤1]`. Maintains real-valued shadow weights.
- **XNOR-Net (Rastegari et al., 2016):** `W*X ≈ α·(sign(W) ⊛ sign(X))`, computed via `popcount(XNOR(W,X))`. Scaling factor α = ||W||_1/n.
- **Our use:** The weighted Hamming `w^T(z_i XOR z_j)` is a learned XNOR-popcount with per-bit weights. Gradients flow through `w` directly (continuous); through `z` via STE.

### 2.6 MaskGIT (Chang et al., 2022) — Parallel Masked Prediction
- **Training:** Mask random tokens with ratio γ(r) ~ cosine schedule. Predict masked tokens from context: `L = -Σ_{masked} log p(z_i | z_unmasked)`.
- **Inference:** Start all-masked, iteratively unmask highest-confidence predictions. T=8-12 steps vs N autoregressive.
- **Our use (with Route C modifications):** Our amortized net learns approximate MAP inference, not a pure generative model. Training loss includes:
  - `BCE(predicted, target)` on masked bits (MaskGIT-style)
  - `E_core` consistency penalty (generated z must satisfy local discrete rules)
  - `E_obs` data term (generated z must be consistent with observed pixels)
  This makes it a **constrained amortized inference** network, not a free generative model.

## 3. Two Technical Routes

### Priority: Route B is primary (speed + generalization), Route A is auxiliary (as a learnable module inside B or standalone energy improvement)

---

### Route A: Learned Discrete Routing (升级 D-RoPE)

**Mathematical formulation — SOFT GATE (critical for trainability):**

Current gate (fixed, hard, non-differentiable):
```
gate(i→j) = 1[popcount(z_i XOR z_j) < τ]
```

Proposed gate (learned, **soft**, differentiable):
```
g_{ij} = σ( (τ - w^T(z_i ⊕ z_j)) / T )
```
where:
- `w = softplus(u)`, `u ∈ R^k` is learnable (ensures w > 0)
- `τ` is learnable threshold
- `T` is temperature (annealed during training: T: 1.0 → 0.1)
- `σ` is sigmoid

At test time, optionally harden: `gate = 1[g > 0.5]`.

**Energy with learned routing (differentiable form):**

Instead of `min` (non-differentiable), use **weighted sum**:
```
E_rope(z; w) = Σ_j Σ_{i∈C(j)} g_{ij} · w^T(z_i ⊕ z_j)
```
This is fully differentiable w.r.t. w, τ, and (via STE) z.

For evaluation/hard inference, can revert to:
```
E_rope(z; w) = Σ_j softmin_{i∈C(j)} [w^T(z_i ⊕ z_j)]  with weights g_{ij}
```

**Bit importance grouping (optional, for interpretability):**
```
w = [w_coarse · 1_{bits 0-3}, w_fine · 1_{bits 4-7}]
```
Two scalar parameters instead of k, if interpretability > expressivity.

**Candidate generation — three baselines:**

| Method | Mechanism | Complexity | When to use |
|--------|-----------|------------|-------------|
| Fixed offsets | Spatial neighbors at radii {1,2} | O(C·N) | Small grids (N<100) |
| LSH buckets | Bit-sampling hash, multi-round | O(L·N + N·N/2^b) | Large grids (N>100) |
| Brute-force top-K | Compute all pairwise w^T XOR, take top-K | O(N²·k) | Small grids, exact |

On our 7×7 grid (N=49), fixed offsets and brute-force are equally fast. LSH is included as a **scalability design**, validated in the benchmark suite.

**Training:** Contrastive energy + sparsity regularization:
```
L = E[relu(E(z_clean) - E(z_corrupt) + margin)] + δ·||w||_1
```

**Code:** `learned_routing/__init__.py` — `LearnedHammingGate`, `LSHCandidateGenerator`, `LearnedDRoPEEnergy`, `CombinedEnergyLearned`

---

### Route B: Amortized Inpainting (替代 test-time MCMC) — PRIMARY

**Mathematical formulation:**

Current inference (MCMC):
```
z* = argmin_z E(z)  via 30 sweeps of block Gibbs (~2-5s/sample)
```

Proposed inference (amortized):
```
z* = f_φ(z_observed, mask)  via single forward pass (~5-50ms/sample)
```

**Two latent-level variants (must match downstream classification head):**

**B1: Grid-latent inpainting (7×7×k)** — matches exp09 world-model architecture
- Input: `z_masked (k, H, W)` + `mask (1, H, W)` → concatenated `(k+1, H, W)`
- Output: `logits (k, H, W)` per-bit predictions
- Architecture: 3-layer residual CNN with circular padding (torus topology)
- Classifier receives: predicted `z` grid → `classifier(z_flat)` → class logits

**B2: Histogram-latent inpainting (future, for 95% baseline)**
- Input: partial spatial histogram `(n_bins × vocab)` with mask indicating missing bins
- Output: completed histogram
- This variant directly targets the log-odds classifier path (spatial bins + 512-vocab histogram)
- Not implemented in v1 (requires separate histogram-based pipeline)

**Training loss — constrained amortized MAP inference:**
```
L = L_mask + α·L_core + β·L_obs
```
where:
- `L_mask = BCE(f_φ(z⊙(1-m), m), z⊙m)` — masked bit prediction (MaskGIT-style)
- `L_core = E_core(z_predicted)` — local discrete consistency (predicted z should satisfy neighborhood rules)
- `L_obs = ||decode(z_predicted) - x_clean||² on unmasked pixels` — observation consistency

**Note:** The base v1 implementation uses `L_mask` only for simplicity. Energy-aware training is an immediate extension.

**Iterative refinement (MaskGIT-style):**
1. Predict all masked bits
2. Compute confidence = |σ(logit) - 0.5| × 2
3. Unmask top-confidence fraction (cosine schedule: γ(t) = cos(πt/2T))
4. Re-predict remaining with updated context
5. 2-4 steps typically sufficient (vs 30 MCMC sweeps)

**Code:** `inpainting/__init__.py` — `InpaintNet`, `InpaintTrainer`, `amortized_inpaint`, `iterative_inpaint`

---

### Route A+B integration (future): Learned discrete attention layer inside InpaintNet

The ideal architecture is: InpaintNet with a **learned discrete routing layer** that replaces the CNN's local receptive field with content-addressed long-range connections. This makes Route A a **module inside Route B**, not a standalone MCMC energy term.

```
InpaintNet_v2:
  input: z_masked (k+1, H, W)
  → Conv layer (local features)
  → LearnedDiscreteAttention(z_flat, w, τ)  ← Route A as attention layer
  → Conv layer (output)
  → logits (k, H, W)
```

This is the "Transformer killer feature" path: discrete content-addressed routing inside a compile-friendly network.

---

### Comparison Table

| Aspect | Current (D-RoPE MCMC) | Route A (Learned Gate) | Route B (Amortized) | A+B (Future) |
|--------|----------------------|----------------------|---------------------|---------------|
| Metric | Fixed Hamming | Weighted Hamming (soft) | N/A | Learned routing layer |
| Routing | Fixed spatial offsets | Soft gate + offsets/LSH | CNN receptive field | Discrete attention |
| Inference | 30-sweep MCMC | MCMC (better landscape) | Single forward pass | Forward + routing |
| Speed | ~2000-5000 ms | ~2000-5000 ms | ~5-50 ms | ~10-100 ms |
| OOD | Sensitive to mask type | Adapts via learned w | Generalizes via training | Best of both |

## 4. Evaluation Suite

### Dimensions:
1. **Mask types:** center, random, multi-hole (5×4×4), stripes (2px/6px)
2. **Noise:** clean, noise(σ=0.1), bias(+0.2), dropout(p=0.2)
3. **OOD:** Train mask A → test mask B (all cross-combinations)
4. **Speed:** ms/sample inference latency; parameter count

### Metrics per configuration:
- `acc_before`: classification accuracy on occluded input (before inference)
- `acc_after`: accuracy after inference/inpainting
- `Δacc`: improvement (key metric)
- `mse_before/after`: reconstruction MSE on occluded region
- `runtime_ms`: inference time per sample
- `params`: model parameter count
- **`acceptance_rate`** (MCMC methods only): fraction of proposals accepted — explains mixing quality
- **`energy_curve`** (MCMC methods only): per-sweep energy values — shows convergence trajectory

### Structural diagnostics (future, to explain center vs random divergence):
- Token KL divergence to class prototypes (does inference move z toward correct class?)
- Per-position bit flip rate during inference (where is the solver active?)
- Gate activation map (which positions does the learned gate connect?)

### Output: `outputs/benchmark/results.csv` + pretty-printed console summary

## 5. Narrative Corrections (from reviewer feedback)

1. **Mixing time:** Exp09 data shows D-RoPE **increases** sweeps_95% (22-23 vs 20). We do NOT claim "faster mixing." Instead: "D-RoPE changes the energy landscape and proposal distribution, with benefits that depend on occlusion geometry."

2. **LSH optimality claim:** Removed "provably optimal" phrasing. Instead: "Bit-sampling is a standard LSH family for Hamming space; we use it as a simple, zero-overhead candidate generator for binary codes."

3. **Priority:** Route B (amortized) is the primary path for speed and generalization. Route A (learned routing) is auxiliary — its value is as a learnable module, not as an MCMC energy improvement.
