# Route C: Discrete-Core Inference Paradigm — Research Report

## 1. Literature Distillation (Paradigm-Relevant Only)

### 1.1 Energy-Based Learning (LeCun 2006)

**Core claim:** A model assigns a scalar energy E(Y,X;W) to every (input, output) pair; inference = argmin_Y E; training = shape the energy surface (dig holes at correct answers, build hills elsewhere).

**Key mechanism:** Loss functions that never require partition function Z:
- Contrastive: E(correct) - E(incorrect) + margin
- Pseudo-likelihood: local normalization (2^k terms, not 2^d)
- Energy difference: ΔE = E(z') - E(z), computed locally

**Route C mapping:** E_core (pseudo-likelihood MRF) + E_obs (observation likelihood) form E(z, o; W). MCMC minimizes E directly. Amortized q_φ compiles the minimization into a single forward pass.

**Verified:** MCMC diagnostic confirms energy landscape structure (accept rates, E_drop). Amortized inference beats MCMC 900× in speed and produces positive Δacc.

**Missing:** Contrastive/margin training for E_core (currently pseudo-likelihood only). Could improve energy landscape quality.

### 1.2 Wake-Sleep / Amortized Inference (Hinton 1995)

**Core claim:** Sleep phase trains recognition network q_φ(z|x) by "dreaming" — sampling from generative model, simulating observations, training q_φ to invert. This is forward-KL (mass-covering, no mode collapse).

**Key mechanism:**
```
Sleep: sample z ~ p(z), simulate x̃ = observe(z), train q_φ to recover z from x̃
Wake:  sample x ~ data, infer z ~ q_φ(z|x), update generative model p(x|z)
```

**Route C mapping:** InpaintNet training IS sleep phase:
1. Sample z = encode(x) from training data
2. Apply random mask m → simulate partial observation
3. Train q_φ(z|z_obs, m) to recover z

**Verified:** Mask mixture training = expanding sleep-phase observation coverage → +5% gain. L_mask_only > L_mask+L_cls (sleep phase purity matters). This is not task-specific — it's a property of forward-KL coverage.

**Missing:** Wake-phase update of E_core/decoder using q_φ's inferred z (currently encoder and inference are trained in separate phases, not jointly iterated).

### 1.3 MaskGIT (Chang et al., CVPR 2022)

**Core claim:** Iterative parallel decoding with confidence-based unmasking. T=8-12 steps vs N=256 autoregressive. Mask distribution during training determines generalization boundary.

**Key mechanism:**
```
for t = 1..T:
    predict all masked positions
    confidence = softmax prob of sampled token
    unmask top-confidence fraction (cosine schedule: γ(t) = cos(πt/2T))
    re-mask lowest confidence
```

**Route C mapping:** `iterative_inpaint` is simplified MaskGIT. Our confidence = |σ(logit)-0.5|×2. Cosine schedule matches.

**Verified:** iterative_4 is best on center+clean (+15%). But n_steps=1 nearly optimal at 7×7 — grid too small for multi-step benefit. Mask distribution is the decisive factor for generalization (confirmed independently via mixture experiment).

**Missing:** Scale to 14×14+ where multi-step should genuinely outperform single-pass (more tokens → more sequential dependencies → more iterative value).

### 1.4 Modern Hopfield Networks (Ramsauer et al., ICLR 2021)

**Core claim:** Continuous Hopfield energy E(ξ) = -lse(β, X^T ξ) + ½ξ^Tξ yields update rule ξ_new = X·softmax(β·X^T·ξ), which IS one transformer attention layer. Exponential storage capacity O(2^{d/2}).

**Key mechanism:**
```
Query = current state ξ
Keys/Values = stored patterns X
Update = softmax attention over stored patterns
```
Three fixed-point types: global average (low β), metastable (medium β), single-pattern retrieval (high β).

**Route C mapping:** GDA = Hamming-space Hopfield retrieval. XOR/popcount distance replaces dot-product similarity. InpaintNet single forward pass ≅ single-step compiled Hopfield update.

**Verified:** GDA +9% on center (retrieval from observed tokens works). Stripes fail = collapsed queries, degenerate Hamming distances (all masked codes → 0 → identical "queries").

**Missing:** Sparse/scalable variant for N>49. At 14×14 (N=196), full N² attention costs 38K entries — still feasible but approaching the boundary. Evidence gating (only use observed tokens as K/V) is the right direction. LSH candidate generation (already implemented in `learned_routing/`) could provide O(N log N) scaling.

### 1.5 Continuous Bernoulli (Loaiza-Ganem & Cunningham, NeurIPS 2019)

**Core claim:** Using Bernoulli likelihood (BCE) on continuous [0,1] data is mathematically incorrect — missing normalization constant C(λ). The Continuous Bernoulli distribution fixes this: p(x|λ) = C(λ)·λ^x·(1-λ)^{1-x}.

**Key mechanism:** C(λ) = 2·atanh(1-2λ)/(1-2λ) ≥ 2. Counterintuitively, C(λ) diverges as λ→0 or λ→1 (more binary data → bigger error from ignoring C).

**Route C mapping:** Sigmoid decoder output implies Bernoulli assumption → E_obs must be BCE, not MSE. For truly continuous [0,1] data, CB correction needed.

**Verified:**
- MSE metrics frozen after repair (geometric mismatch — "blind" to repair)
- BCE decoder reconstruction 4× better than MSE decoder
- BCE/CB metrics move dramatically after repair (geometric alignment)
- This is NOT a task-specific finding — it's a property of observation-space geometry

**Missing:** CB with tuned hyperparameters (current baseline accuracy lower). Extension to RGB: per-channel Bernoulli/Gaussian mixture likelihood.

---

## 2. Abstract Paradigm Interface

The discrete-core inference paradigm has four components. Any system fitting this interface can use the same inference, training, and evaluation machinery.

```
┌─────────────────────────────────────────────────────┐
│                  Paradigm Interface                   │
├──────────────┬──────────────────────────────────────┤
│ ADC/DAC      │ encode(o) → z ∈ {0,1}^{k×H×W}      │
│              │ decode(z) → ô ∈ R^{c×H_o×W_o}       │
│              │ Parameterized by (in_channels,        │
│              │   out_channels, latent_size, n_bits)   │
├──────────────┼──────────────────────────────────────┤
│ E_core       │ energy(z) → R                         │
│              │ Pseudo-likelihood MRF on z             │
│              │ Task-agnostic: local consistency only  │
├──────────────┼──────────────────────────────────────┤
│ E_obs        │ energy(z, o) → R                      │
│              │ = -log p(o | decode(z))                │
│              │ Must match decoder output distribution │
│              │ Bernoulli→BCE, Gaussian→MSE, CB→CB-NLL │
├──────────────┼──────────────────────────────────────┤
│ Inference    │ q_φ(z | z_obs, m) → z_repaired        │
│  (compiled)  │ Amortized: single forward pass         │
│              │ Iterative: T-step MaskGIT decode       │
│              │ MCMC: diagnostic/teacher only           │
├──────────────┼──────────────────────────────────────┤
│ Probe        │ classifier(z) → class logits           │
│ (diagnostic) │ Frozen. Never enters world model loss. │
│              │ Only measures z's information content.  │
└──────────────┴──────────────────────────────────────┘
```

### What changes per dataset

| Component | MNIST/FMNIST/KMNIST | SVHN/CIFAR-10 |
|-----------|-------------------|---------------|
| `in_channels` | 1 | 3 |
| `out_channels` | 1 | 3 |
| E_obs likelihood | Bernoulli (BCE) | Gaussian (MSE) or logistic mixture |
| Image size | 28×28 | 32×32 |
| Latent size options | 7×7, 14×14 | 8×8, 16×16 |
| Decoder activation | Sigmoid | Sigmoid or none (if Gaussian) |

### What does NOT change

- E_core architecture (LocalBitPredictor: 3×3 MRF, pseudo-likelihood)
- InpaintNet architecture (k+1 input, residual CNN, circular padding)
- Inference protocol (amortized + optional iterative)
- Training protocol (sleep-phase mask mixture)
- Evaluation protocol (E_obs metric, E_core violation rate, probe accuracy)
- Mask types and mixture distribution

---

## 3. Unified Benchmark Design

### 3.1 Evaluation Metrics (Three Categories)

All metrics measured before and after inference (Δ is the paradigm signal):

| Category | Metric | Definition | Paradigm role |
|----------|--------|------------|---------------|
| **Observation** | E_obs (BCE/MSE) | -log p(o\|decode(z)) on occluded pixels | Does repair improve observation fit? |
| **Observation** | ΔMSE | MSE change on occluded region | Pixel-level reconstruction quality |
| **Structure** | E_core | -Σ log p(z_i\|neigh) | Does repair improve local consistency? |
| **Structure** | core_viol | fraction of positions where encoder ≠ local_pred | How many E_core violations remain? |
| **Probe** | Δacc | classification accuracy change | Does z carry more task information? |
| **Probe** | ΔBCE_cls | cross-entropy change on probe | Continuous version of Δacc |
| **Cost** | runtime_ms | wall-clock inference time | Practical deployment cost |
| **Cost** | bit_mask_ratio | fraction of z positions repaired | How much work was needed? |

### 3.2 Difficulty Axes

| Axis | Levels | What it tests |
|------|--------|--------------|
| **Dataset** | MNIST → FMNIST → KMNIST | Visual complexity (binary → texture → strokes) |
| **Mask type** | center, stripes, random_block, multi_hole | Occlusion geometry diversity |
| **Mask training** | single-type vs mixture | Sleep-phase coverage (paradigm prediction: mixture always wins) |
| **Corruption** | clean, Gaussian noise σ=0.1, blur | Input quality degradation |
| **Grid scale** | 7×7 (N=49) vs 14×14 (N=196) | Global relation operator value |
| **Inference** | amortized, iterative_1, iterative_4, MCMC(diagnostic) | Operator comparison |

### 3.3 Experiment Matrix (Priority Order)

**Experiment 1: Cross-dataset generalization (amortized + mixture)**

Tests: Does the paradigm work beyond MNIST?

| | MNIST | FMNIST | KMNIST |
|---|---|---|---|
| center+clean | ✓ | ✓ | ✓ |
| center+noise | ✓ | ✓ | ✓ |
| stripes+clean | ✓ | ✓ | ✓ |
| stripes+noise | ✓ | ✓ | ✓ |

Inference: amortized only (fastest, proven best).
Training: mask mixture (proven dominant factor).
Expected if paradigm holds: positive Δacc on center for all datasets, near-zero on stripes.
Expected if paradigm fails: Δacc degrades with visual complexity.

**Experiment 2: Scale-up to 14×14 with GDA**

Tests: Does the global relation operator become necessary at larger scale?

| | 7×7 amortized | 7×7 +GDA | 14×14 amortized | 14×14 +GDA |
|---|---|---|---|---|
| center+clean | baseline | +GDA value | scale effect | GDA×scale interaction |

Expected if Hopfield hypothesis holds: GDA gap grows with grid size (more tokens → more retrieval value).
Expected if wrong: GDA gap stays flat or shrinks.

**Experiment 3: MCMC as diagnostic/teacher**

Tests: Can MCMC-generated pseudo-labels calibrate amortized inference?

Setup: Run 100-step MCMC on small subset (50 samples). Use resulting z as pseudo-labels to fine-tune InpaintNet. Measure calibration improvement.

Expected if useful: Δacc improves 1-3% on hardest configs. Energy-accuracy correlation becomes less negative.
Expected if not: No improvement (amortized already captures what matters).

---

## 4. Dataset-Specific Minimal Changes

### MNIST (current, baseline)
- Encoder: `in_channels=1`, stride-2 CNN → 7×7
- Decoder: `out_channels=1`, Sigmoid, BCE loss
- No changes needed

### FashionMNIST
- Same architecture as MNIST (28×28 grayscale)
- E_obs: BCE (same decoder)
- Expected challenge: more complex textures → E_core needs to capture richer local structure
- Minimal change: **none** (same hyperparameters, different `datasets.FashionMNIST`)

### KMNIST (Kuzushiji)
- Same as MNIST/FMNIST (28×28 grayscale)
- Expected challenge: cursive strokes → long-range dependencies → GDA should help more
- Minimal change: **none**

### SVHN (future, stretch goal)
- `in_channels=3`, `out_channels=3`
- Image size 32×32 → latent 8×8 or 16×16
- E_obs: Gaussian (MSE) or per-channel Bernoulli
- Decoder: remove Sigmoid if using Gaussian
- Encoder: adjust strides for 32×32 → 8×8 (stride 4) or 16×16 (stride 2)
- Minimal change: modify `BinaryEncoder`/`BinaryDecoder` init params

### CIFAR-10 (future, stretch goal)
- Same as SVHN architecture-wise
- Expected challenge: much more visual complexity → may need more bits (k=16) or larger hidden_dim
- This is the hardest test: if paradigm works here, it's genuinely general

---

## 5. Paradigm Predictions ("If Right / If Wrong")

| Prediction | If paradigm is correct | If paradigm is wrong |
|------------|----------------------|---------------------|
| **Cross-dataset** | Positive Δacc on center for FMNIST/KMNIST without architecture changes | Δacc drops to 0 or negative on new datasets |
| **Mask mixture** | Mixture training dominates on all datasets (not MNIST-specific) | Mixture helps on MNIST only |
| **E_obs geometry** | BCE >> MSE on all datasets with sigmoid decoder | MSE works fine on some datasets |
| **Scale-up** | iterative_4 > amortized gap grows at 14×14 | Gap stays same or shrinks |
| **GDA value** | GDA gap grows at 14×14 (more tokens to retrieve from) | GDA gap stays ~0 |
| **MCMC diagnostic** | MCMC Δacc stays negative everywhere (compiled inference always better for deployment) | MCMC becomes positive at some config |

---

## 6. Next Steps: The 2 Critical Experiments

The two experiments that would most strongly validate or falsify the paradigm:

### Critical Experiment A: Cross-Dataset with Zero Architecture Change
Run the exact same model (same hyperparameters, same training protocol, same mask mixture) on MNIST, FashionMNIST, and KMNIST. If positive Δacc on center holds across all three, the paradigm generalizes. If it breaks on FMNIST/KMNIST, we need to understand why — is it E_core capacity? E_obs geometry? Sleep-phase coverage?

### Critical Experiment B: Scale to 14×14
Change only `latent_size` from 7 to 14 (adjust encoder strides). Run amortized and amortized+GDA on center mask. If GDA gap grows from +9% (at 7×7) to +15%+ (at 14×14), the Hopfield retrieval hypothesis is confirmed and the global relation operator is a necessary paradigm component, not a trick.

Everything else (MCMC distillation, CB tuning, threshold sweep) is secondary to these two.
