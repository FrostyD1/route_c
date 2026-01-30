# Route C: Learned Discrete Routing + Amortized Inpainting — Design Document

**Version 2.1 — incorporates reviewer feedback round 2 (2026-01-30)**

## 0. Representation Contract

All modules share these interfaces. This section is the **ground truth** for what `z`, `encode`, `decode`, and energy terms mean.

### Latent representation

```
z ∈ {0,1}^{k × H × W}    where k=8, H=W=7
```
Total: 392 binary bits per image. Each spatial position (i,j) holds a k-bit token.

### Encoder: `encode(x) → z`

```
x ∈ R^{1×28×28}  (normalized MNIST image, [0,1])
    → BinaryEncoder (CNN: Conv(stride=2)→Conv(stride=2)→Conv → logits ∈ R^{k×7×7})
    → GumbelSigmoid (forward: hard {0,1} via STE; backward: soft sigmoid)
    → z ∈ {0,1}^{k×7×7}
```
Ref: Bengio et al. 2013 (STE), Hubara et al. 2016 (saturated STE with `1{|a|≤1}`).

### Decoder: `decode(z) → x̂`

```
z ∈ {0,1}^{k×7×7}
    → BinaryDecoder (Transposed CNN: 7→14→28)
    → x̂ ∈ [0,1]^{1×28×28}   (sigmoid output)
```

### E_obs: Observation energy

```
E_obs(z, x_obs, mask) = (1 / 2σ²) · ||mask ⊙ (decode(z) - x_obs)||²
```
This is pixel-MSE weighted by observation mask. **Known limitation:** implicitly assumes Gaussian observation noise. For binarized MNIST, the correct likelihood is **Bernoulli** (Kingma & Welling, 2014):
```
E_obs_bernoulli(z, x) = -Σ_i [x_i·log(μ_i) + (1-x_i)·log(1-μ_i)]    where μ = decode(z)
```
See §9.2 for planned migration to Bernoulli E_obs. (Ref: Loaiza-Ganem & Cunningham, NeurIPS 2019 — "The Continuous Bernoulli" for [0,1]-valued data.)

### E_core: Local structure energy

```
E_core(z) = -Σ_{i,j} log p_θ(z_{:,i,j} | neigh(z)_{i,j})
```
where `neigh` = 3×3 spatial window (9k-1 = 71 context features), and `p_θ` is a small MLP (`Linear(71,32)→ReLU→Linear(32,8)`). E_core is the negative log-likelihood of each token under its local neighborhood — a discrete analog of an MRF prior.

### Classifier: `classifier(z) → logits`

```
z.reshape(B, k*H*W) → Linear(392, 10) → logits ∈ R^{10}
```
Frozen during InpaintNet training; its gradient provides the L_cls signal.

### Summary table

| Interface | Input | Output | Implementation |
|-----------|-------|--------|----------------|
| `encode(x)` | (1,28,28) float | (8,7,7) binary | `learning/quantizer.py:BinaryEncoder` + `GumbelSigmoid` |
| `decode(z)` | (8,7,7) binary | (1,28,28) float | `learning/quantizer.py:BinaryDecoder` |
| `E_obs(z,x,m)` | z + observed x + mask | scalar | `inference/energy.py:ObservationEnergy` |
| `E_core(z)` | z | scalar | `core/local_energy.py:LocalBitPredictor.energy()` |
| `classifier(z)` | z flat (392,) | (10,) logits | `learning/__init__.py:LogOddsClassifier` |

---

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

**Hardened findings from mid-term benchmark (§8):**

- **噪声下 MCMC：Δmse 往往仍下降，但 Δacc 会变负。** Pixel-MSE type E_obs is NOT a valid proxy for task-consistent inference. When observation is noisy, MCMC pushes z toward a "smooth average" that lowers pixel MSE but destroys discriminative structure. This is a fundamental objective misalignment, not a hyperparameter issue. (Ref: Stuhr et al. 2022, "Don't Miss the Mismatch" — reconstruction vs. classification objectives can diverge by 25–59% on downstream tasks.)

- **D-RoPE 2× 慢且不增益。** Fixed Hamming gate should be treated as a "candidate generation / structural prior" module, NOT an effective energy term. Either it must become learned (Route A: weighted Hamming), or it should be removed from the energy and used only as a candidate set for Route A+B integration.

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

## 5. Detailed Response to Reviewer Feedback

### Point 3 (Reviewer): "Route B 的 latent 层级 — grid vs histogram"

**Our response:** Fully agree. The current v1 InpaintNet operates on grid-latent (7×7×k=8), which is the same representation the classifier consumes (`classifier(z.reshape(B, -1))`). This ensures the inpainted z feeds directly into the classification head without variable mismatch.

The histogram-latent variant (B2) targets the 95.4% log-odds baseline, which uses `spatial_bins(4×4) × vocab(512)` features. This is a different pipeline entirely (VQ tokens → histogram → log-odds), not the learnable encoder pipeline. We defer B2 to future work; v1 focuses on the grid-latent path where MCMC comparison is direct.

**Risk mitigated:** The reviewer correctly identified that pixel-flip MCMC proposals failed because they operated at the wrong abstraction level (pixel vs token). Our InpaintNet operates at the correct level: it predicts z-bits, which is what the classifier consumes.

### Point 5 (Reviewer): "Amortized net 学的是近似 MAP 推断，不是纯粹的 token 语言模型"

**Our response:** Implemented. The InpaintTrainer now includes energy-aware losses:
```
L = L_mask + α_core·L_core + α_obs·L_obs
```
- `L_core`: local predictor consistency — the composed z (observed + predicted) should satisfy neighborhood rules learned by `local_pred`
- `L_obs`: reconstruction consistency — `decode(z_composed)` should match the input image

This makes the network learn **constrained MAP inference**: it doesn't just predict likely bits, it predicts bits that are consistent with the Route C energy landscape. The base version (L_mask only) serves as ablation baseline.

### Point 8 (Reviewer): "Route A 作为 B 的模块进入网络，变成离散注意力层"

**Our response:** This is the most important architectural insight. We have designed (but not yet implemented in v1) `InpaintNet_v2` with a learned discrete attention layer:

```
InpaintNet_v2:
  Conv(k+1, hidden) → ReLU
  → LearnedDiscreteAttention(hidden, w, τ)  ← Route A as attention layer
  → Conv(hidden, k) + skip
```

The attention layer computes:
```
z_out[j] = z_local[j] + Σ_{i∈C(j)} g_{ij} · MLP(z[i])
```
where `g_{ij} = σ((τ - w^T(z_i ⊕ z_j)) / T)` is the learned soft gate.

This is the "Transformer killer feature" path: a feedforward network with **content-addressed long-range routing** that is:
1. Discrete (operates on binary z)
2. Sparse (O(C·N) not O(N²))
3. Compile-friendly (XOR + weighted popcount + threshold)
4. Amortized (single forward pass, no MCMC iteration)

v1 validates that amortized inpainting works at all (Route B alone). v2 adds learned routing inside the network (Route A+B).

## 6. Reviewer v2.1 Feedback — Additional Implementation Points

### §1: L_cls in InpaintNet training — IMPLEMENTED
Added `gamma_cls * CE(classifier(z_pred), y)` to training loss. This ensures the inpainted z is useful for the downstream classification task, not just "looks-like" bit fill. The classifier is frozen (no gradient back to its weights), so L_cls acts as a task-aware auxiliary head guiding the inpainting network.

### §2: MaskGIT decode — train/test consistency
- **Training:** Uses logits → BCE (stable gradients)
- **Inference:** Hard sample `(σ(logit) > 0.5)` for bit predictions
- **Confidence:** `|σ(logit) - 0.5| × 2` — measures distance from decision boundary
- **Gap mitigation:** The InpaintNet uses circular padding + residual skip, which reduces distribution shift between soft-training and hard-inference. Temperature annealing in the gate (Route A) provides an additional smooth → hard transition if needed.

### §3: Discrete attention operator — minimum viable definition
```
LearnedDiscreteAttention(z, w, τ):
  For each position j:
    1. Compute g_{ij} = σ((τ - w^T(z_i ⊕ z_j)) / T) for all i ∈ C(j)
    2. Select top-K candidates by g_{ij} (K~4-8)
    3. Value aggregation: v_j = majority_vote(z_{top-K})  [hardware: popcount + compare]
       OR soft: v_j = Σ g_{ij} · z_i / Σ g_{ij}  [differentiable version for training]
    4. Output: z_out[j] = z_local[j] + project(v_j)  [residual]
```
Hardware path: XOR → weighted popcount → top-K compare → majority vote. No floating-point multiply.

**Training/deployment are the same operator, two implementations:**
- **Training (soft):** `v_j = Σ_{i∈C(j)} g_{ij} · z_i / Σ_{i∈C(j)} g_{ij}` — differentiable weighted average, gradients flow through `g` and (via STE) through `z`.
- **Deployment (hard):** `v_j = majority_vote(z_{top-K})` where top-K selected by hard gate `1[g > 0.5]` — integer-only, no FP multiply.

This is analogous to how attention in Transformers uses softmax during training but can be approximated by top-K sparse attention at inference (Correia et al., 2019). The soft→hard transition is controlled by temperature annealing (T: 1.0 → 0.1).

### §4: Route A training objective — task-supervised, not just contrastive
Current: Contrastive energy ranking (E(z_clean) < E(z_corrupt)).
**Improvement:** Add task-supervision — backpropagate L_cls through the gate weights w:
```
L_A = L_contrastive + λ_task · CE(classifier(z_gated), y)
```
This ensures `w` learns "which bits matter for classification routing," not just "which bits minimize energy."
**Self-supervised augmentation:** Different occlusion views of same image should produce similar gate activations on shared visible regions.

### §5: Calibration metrics for Route B — TODO
- ECE (Expected Calibration Error) on masked region predictions
- Per-step acc_after curve (1, 2, 3, 4 iterative steps)
- Confidence-accuracy correlation plot

### §6: Torus topology ablation — TODO
- `padding = 'zeros'` vs `padding = 'circular'` in InpaintNet
- Rotation augmentation during training (D4 group)
- Both as ablation rows in benchmark suite

## 7. Narrative Corrections (from reviewer feedback)

1. **Mixing time:** Exp09 data shows D-RoPE **increases** sweeps_95% (22-23 vs 20). We do NOT claim "faster mixing." Instead: "D-RoPE changes the energy landscape and proposal distribution, with benefits that depend on occlusion geometry."

2. **LSH optimality claim:** Removed "provably optimal" phrasing. Instead: "Bit-sampling is a standard LSH family for Hamming space; we use it as a simple, zero-overhead candidate generator for binary codes."

3. **Priority:** Route B (amortized) is the primary path for speed and generalization. Route A (learned routing) is auxiliary — its value is as a learnable module, not as an MCMC energy improvement.

4. **B1 vs B2:** B1 (grid-latent 7×7×k) for exp09 world-model comparison. B2 (histogram-latent) targets the 95.4% log-odds baseline and is the faster/more stable path for MNIST. B2 should be implemented soon after v1 validation.

## 8. Mid-Term Benchmark Findings (100 samples/config)

### Raw Results (partial — baseline + drope complete)
```
method    | mask       | noise | acc_before → after | Δacc  | Δmse    | ms/sample
----------|------------|-------|--------------------|-------|---------|----------
baseline  | center     | clean | 34% → 37%          | +3.0% | -0.010  | 370
baseline  | center     | noise | 25% → 24%          | -1.0% | -0.009  | 415
baseline  | random     | clean | 48% → 55%          | +7.0% | -0.012  | 399
baseline  | random     | noise | 42% → 43%          | +1.0% | -0.014  | 411
baseline  | multi_hole | clean | 83% → 84%          | +1.0% | -0.001  | 290
baseline  | multi_hole | noise | 82% → 81%          | -1.0% | +0.004  | 359
baseline  | stripes    | clean | 78% → 78%          | +0.0% | -0.000  | 78
baseline  | stripes    | noise | 80% → 80%          | +0.0% | -0.000  | 82
drope     | center     | clean | 34% → 33%          | -1.0% | -0.007  | 784
drope     | center     | noise | 25% → 24%          | -1.0% | -0.009  | 891
drope     | random     | clean | 48% → 54%          | +6.0% | -0.010  | 743
```

> **⚠️ ERRATA (v2.1):** The stripes rows above (Δacc=0, Δmse=0) were caused by a **confirmed implementation bug** in `pixel_to_bit_mask()`. The function used `patch.mean() < 0.5` (strict less-than) to decide if a latent position should be masked. With `stripe_width=2` and `gap=6`, every 4×4 latent patch has exactly 2/4 rows occluded → `mean = 0.5` → no latent positions were masked → inference was a no-op. **Fixed** in v2.1 by changing threshold to `< 1.0 - ε` (any partial occlusion marks the position). Results pending re-run.

### Key Findings

**Finding 1: Mask geometry determines marginal value of inference.**
- random (+7%) >> center (+3%) >> multi_hole (+1%) >> stripes (~~0%~~ BUG — pending re-run)
- Multi-hole before accuracy is already 83% — small holes don't destroy discriminative structure
- This means evaluation should focus on center/random/stripes (harder) for Route B validation

**Finding 2: Noise causes MSE↓ but acc↓ — objective misalignment.**
- center+noise: MSE improves (-0.009) but acc drops (-1%)
- This is NOT a hyperparameter issue — it's a fundamental mismatch: E_obs (pixel MSE) pushes z toward "smooth average" which lowers MSE but destroys discriminative features
- **Critical implication:** Route B MUST include L_cls or a semantic observation term, not just pixel MSE
- **Quantification (v2.1):** benchmark now reports per-config `corr(Δmse, Δacc)` to make this misalignment visible as a number, not just a narrative claim

**Finding 3: D-RoPE confirms exp09 pattern.**
- center: Δacc = -1% (worse than baseline +3%)
- random: Δacc = +6% (similar to baseline +7%)
- D-RoPE is 2× slower (784ms vs 370ms) with no accuracy benefit
- Fixed Hamming gate is not a useful energy term — it needs learned weights or should be dropped

**Finding 4: Speed baseline established.**
- MCMC: 78-891 ms/sample depending on mask size and method
- Route B target: <50ms (10-100× improvement)
- **v2.1:** latency now reported with hardware info (`outputs/benchmark/hardware_info.txt`), P10/P50/P90 percentiles, batch=1 per-sample

### Actionable Next Steps (from mid-term analysis)
1. ✅ **L_cls in InpaintNet** — implemented; `amortized` = full loss, `amortized_maskonly` = L_mask only ablation
2. **Discrete observation likelihood** — replace pixel MSE with Bernoulli likelihood for E_obs (see §0 and §9.2)
3. ✅ **Diagnostic metrics** — `corr(Δmse, Δacc)`, `mask_ratio`, `mse_before_abs`, percentile runtimes all added to CSV
4. **Center/stripes as primary benchmark** — multi_hole too easy; stripes now valid after bug fix
5. ✅ **Iterative steps curve** — `acc_after` at n_steps=1,2,3,4 saved to `iterative_steps_curve.csv`

## 9. Hard Metrics & Sanity Checks (v2.1)

These metrics were added to prevent false conclusions from implementation artifacts.

### 9.1 Per-config sanity checks (assert on every evaluation run)

| Check | Assert condition | What it catches |
|-------|-----------------|-----------------|
| `pixel_mask_ratio > 0` | Mask must occlude some pixels | Mask generator returning all-ones |
| `bit_mask_ratio > 0` | Latent mask must mark some positions | pixel→bit threshold too strict (stripes bug) |
| `mse_before > 0` | MSE must be positive before inference | decode or MSE computation not running |

### 9.2 Extended CSV columns (outputs/benchmark/results.csv)

| Column | Definition | Why it matters |
|--------|-----------|----------------|
| `pixel_mask_ratio` | Fraction of pixels occluded | Confirms mask is applied; enables cross-mask comparison |
| `bit_mask_ratio` | Fraction of latent positions masked | Direct input to MCMC/InpaintNet scope |
| `corr_dmse_dacc` | Pearson(Δmse, Δacc) per sample | Quantifies objective misalignment (should be positive if MSE is a good proxy) |
| `mse_before_abs` / `mse_after_abs` | Absolute MSE values | Confirms computation is non-trivial |
| `runtime_p10/p50/p90` | Latency percentiles | Reveals tail latency; prevents P90 outlier from hiding behind mean |

### 9.3 Planned: Bernoulli E_obs (replacing pixel MSE)

Current `E_obs = ||decode(z) - x||²` assumes Gaussian observations. For MNIST (binary/near-binary pixels), the correct energy is:
```
E_obs_bernoulli(z, x) = -Σ_i [x_i · log(decode(z)_i) + (1-x_i) · log(1 - decode(z)_i)]
```
This is BCE, not MSE. The decoder already outputs sigmoid values in [0,1], so this is a drop-in replacement. Expected effect: noisy inputs should no longer cause "MSE↓ but acc↓" because BCE penalizes confident-wrong predictions more than MSE does.

**Priority:** This should be implemented **before** training Route A's learned gate, to avoid learning `w` on a misaligned E_obs.

### 9.4 L_mask vs L_mask+L_cls comparison matrix

The benchmark now runs both:
- `amortized` = InpaintNet trained with `L_mask + L_core + L_obs + L_cls` (full loss)
- `amortized_maskonly` = InpaintNet trained with `L_mask` only (ablation)

**Key comparison:** On noisy configs, `amortized` should show Δacc > 0 where `amortized_maskonly` shows Δacc ≤ 0. If this does NOT happen, the "L_cls fixes misalignment" narrative is invalidated and we need to investigate further.

### 9.5 Hardware & environment reporting

All benchmark runs output `outputs/benchmark/hardware_info.txt` with:
- Device (CPU/GPU), GPU model, memory
- PyTorch version, platform
- Batch size (always 1 for per-sample evaluation)

The `<50ms` KPI in §3 is meaningless without specifying hardware. All reported latencies are per-sample (batch=1) on the device recorded in this file.
