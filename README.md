# Route C: Boundary Translation + Discrete Core Closure

A unified discrete-world modeling framework that is:
- **(a) Scalable**: Token-based representation with learned codebook
- **(b) Supports inference-time dynamics**: Iterative solving (Gibbs/Metropolis/ICM)
- **(c) Supports interventions and structural priors**: Symmetry/topology enforcement
- **(d) Can degrade to continuous-style computation**: As performance lower bound

## Latest Interpretation (Key Insights)

### 1. Expressivity Lower Bound
Discretization is **NOT** inherently weaker than continuous. If discrete core includes (or can simulate) add/mul, it can emulate ResNet etc. Any observed weakness is due to **effective capacity / learnability** caused by restricted latent family or weak learner, NOT fundamental expressivity limit.

### 2. Training vs Inference
- **Training**: May use differentiable surrogates (STE/Gumbel/soft logic)
- **Inference/Deployment**: Must remain discrete iterative solving
- **Critical**: Proposal scale must match energy scale (block/token moves >> pixel flips)

### 3. DNF and Log-Odds
Same "readout layer" slot R(z)→y, different parameterizations. DNF is valuable for:
- Interpretability (human-readable rules)
- Logic hardening (convert to fast lookup tables)
- Rule extraction from trained models

## Mathematical Model

### Spaces and Symbols
- **Continuous observation**: `o(d) ∈ R^{28×28}` (MNIST image)
- **Discrete latent**: `z ∈ {1..K}^{H'×W'}` (token grid)
- **Codebook**: `C = {c_1, ..., c_K}` (learned patch prototypes)

### Boundary Translation
```
Encoder (ADC): z = Q(E(o))  # patchify + quantize
Decoder (DAC): ô = D(z)     # replace tokens with codebook patches
```

### Energy-Based Inference
```
E_total = λ1·E_conditional + λ2·E_classifier + ...
z* = argmin E(z) via Gibbs/Metropolis in discrete z-space
```

## Project Structure

```
route_c/
├── __init__.py         # Package root
├── core/               # Symmetry, topology, neighborhood
├── boundary/           # ADC/DAC, patchification, VQ codebook
├── learning/           # Token conditional model, log-odds, DNF
│   ├── __init__.py     # TokenConditionalModel, LogOddsClassifier
│   └── dnf.py          # Teacher distillation, differentiable DNF
├── inference/          # Gibbs/Metropolis sampling
│   ├── __init__.py     # Basic inference functions
│   └── energy.py       # EnergyModel interface + implementations
├── logic/              # DNF optimization
│   └── __init__.py     # Subsumption, factoring, CSE
├── mnist/              # Data loading (auto-downloads MNIST)
├── experiments/        # Main experiment scripts
│   ├── exp_mnist_vq_tokens.py   # Original experiments
│   └── exp_unified.py           # Full unified experiment
└── outputs/            # Results and visualizations
```

## Installation

```bash
pip install numpy matplotlib torchvision
```

## Running Experiments

### Full Unified Experiment (Recommended)
```bash
cd route_c
python -m experiments.exp_unified
```

Or from project root:
```bash
python -m route_c.experiments.exp_unified
```

### Original Tokenization Experiment
```bash
cd route_c/experiments
python exp_mnist_vq_tokens.py
```

## Experiments

| Experiment | Description |
|------------|-------------|
| **A: Tokenization** | VQ codebook + reconstruction quality (MSE/PSNR) |
| **B: Masked Completion** | Token conditional model + Gibbs fill accuracy |
| **C: Classification** | Spatial histogram + log-odds (95%+ on MNIST) |
| **D: Energy Inference** | Combined energy for occlusion robustness |
| **E: DNF Distillation** | Teacher→rules with logic optimization |
| **F: Logic Optimization** | Subsumption, factoring, CSE statistics |

## Configuration

Edit `Config` class in the experiment file:

```python
class Config:
    patch_size: int = 4          # Patch size for tokenization
    codebook_size: int = 64      # Number of VQ codes (K)
    kmeans_iters: int = 50       # K-means iterations
    smoothing: float = 1.0       # Laplace smoothing
    boundary: str = "standard"   # "standard" or "torus"
    gibbs_steps: int = 20        # Inference iterations
    spatial_bins: tuple = (2, 2) # Classification feature bins
    mask_ratio: float = 0.25     # Fraction of tokens to mask
```

## Expected Results

### Tokenization (Exp A)
- With K=64, 4×4 patches: MSE ~800-1200, PSNR ~17-20 dB
- Larger K improves reconstruction quality

### Masked Completion (Exp B)
- Random baseline: ~1.5% (1/K for K=64)
- Gibbs fill: ~10-25% (depends on context strength)
- Improvement shows the model captures local structure

### Classification (Exp C)
- Global histogram only: ~60-70%
- With spatial bins (2×2): ~85-92%
- Spatial structure is crucial

### Occlusion Robustness (Exp D)
- Clean accuracy: ~85-92%
- After 14×14 occlusion (no infer): ~70-80%
- After Gibbs inference: +2-5pp improvement

## Key Insights

1. **Tokenization preserves structure**: VQ encoding captures essential digit structure even with modest codebook size.

2. **Local context matters**: The conditional model learns meaningful token co-occurrence patterns.

3. **Inference recovers information**: Gibbs sampling can partially recover masked tokens using learned local statistics.

4. **Proposal scale must match energy scale**: Token-level proposals work better than pixel-level because the energy function is defined over token histograms.

## Extending the Framework

### Custom Codebook
```python
from boundary import kmeans_codebook_fit
codebook, _, _ = kmeans_codebook_fit(patches, K=128, n_iters=100)
```

### Different Neighborhoods
```python
model = TokenConditionalModel(
    vocab_size=K,
    context_size=8,    # 3×3 Moore neighborhood
    boundary="torus"   # Wrap-around edges
)
```

### Template-Based Proposals
```python
from inference import template_proposal_fill
filled, best_idx = template_proposal_fill(
    token_grid, mask, training_tokens, model
)
```

## References

- Route C mathematical model (see header comments in experiment file)
- Previous GoL experiments validated equivariance, intervention, causal cone
- MNIST experiments showed importance of spatial structure and block-level inference
