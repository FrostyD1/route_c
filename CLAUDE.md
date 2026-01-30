# Route C — 项目记忆 (CLAUDE.md)

## 项目定位

Route C 是一个 **task-agnostic 的离散核推断范式**（discrete-core inference paradigm）：
**离散核心 (E_core) + 观测协议 (E_obs) + 推断编译 (amortized q_φ)**

任何数据集、任何下游任务都只是探针（probe），用来验证范式是否 work，不是优化目标。分类准确率仅作 readout/诊断，不参与世界模型定义。

## 范式接口（Dataset-Agnostic）

```
┌──────────────┬─────────────────────────────────────────┐
│ ADC/DAC      │ encode(o) → z ∈ {0,1}^{k×H×W}          │
│              │ decode(z) → ô ∈ R^{c×H_o×W_o}           │
│              │ Params: in_channels, out_channels,        │
│              │         latent_size, n_bits                │
├──────────────┼─────────────────────────────────────────┤
│ E_core       │ energy(z) → R                            │
│              │ Pseudo-likelihood MRF, task-agnostic      │
│              │ 3×3 neighborhood, local consistency only  │
├──────────────┼─────────────────────────────────────────┤
│ E_obs        │ energy(z, o) → R  = -log p(o|decode(z))  │
│              │ Must match decoder output distribution    │
│              │ Sigmoid→BCE, Linear→MSE, CB for [0,1]    │
├──────────────┼─────────────────────────────────────────┤
│ Inference    │ q_φ(z | z_obs, m) → z_repaired           │
│ (compiled)   │ Amortized: single forward pass            │
│              │ Iterative: T-step MaskGIT decode          │
│              │ MCMC: diagnostic/teacher only              │
├──────────────┼─────────────────────────────────────────┤
│ Probe        │ classifier(z) → logits                    │
│ (diagnostic) │ Frozen. Never enters world model loss.    │
└──────────────┴─────────────────────────────────────────┘
```

### 跨数据集变化矩阵

| 组件 | MNIST/FMNIST/KMNIST | SVHN/CIFAR-10 |
|------|-------------------|---------------|
| `in_channels` | 1 | 3 |
| `out_channels` | 1 | 3 |
| E_obs | BCE (Bernoulli) | Gaussian (MSE) or logistic mixture |
| Image size | 28×28 | 32×32 |
| Latent options | 7×7, 14×14 | 8×8, 16×16 |
| Decoder activation | Sigmoid | Sigmoid or none |

### 不变项

E_core 架构、InpaintNet 架构、推断协议、训练协议（sleep-phase mask mixture）、评估协议（E_obs/E_core/probe 三类指标）

## 三层分离架构

| 层 | 对象 | 要求 | 当前实现 |
|---|------|------|---------|
| **Discrete core** | E_core | task-agnostic、局部可组合 | LocalBitPredictor MRF, -Σ log p_θ(z_i \| neigh) |
| **Observation model** | E_obs | 必须匹配解码器分布假设 | BCE (已验证) |
| **Inference operator** | q_φ(z\|o,m) | amortized 为主 | InpaintNet (3层残差CNN) |

**红线规则：** 任何 task-specific loss（如 L_cls）都必须降级为 probe/diagnostic。

## 核心组件

- **ADC/DAC**: BinaryEncoder/BinaryDecoder, Gumbel-Sigmoid STE, `in_channels` parameterized
- **z**: z ∈ {0,1}^{k×H×W} (当前 k=8, H=W=7, 392 bits)
- **E_core**: LocalBitPredictor MRF, 3×3 邻域, pseudo-likelihood
- **E_obs**: BCE (sigmoid decoder → Bernoulli assumption → geometrically matched)
- **InpaintNet**: 摊销推断, k+1 input channels, residual CNN, circular padding
- **GDA**: GlobalDiscreteAttention, XOR/popcount Hamming, evidence gating
- **LearnedHammingGate**: g_ij = σ((τ - w^T(z_i⊕z_j))/T), LSH candidate generation
- **分类器**: Linear probe only, frozen, never enters world model

## 已验证的范式结论

| # | 结论 | 证据 | 范式意义 |
|---|------|------|---------|
| 1 | **推断 = 编译推断 (amortized)** | +7~+15% Δacc, 2.4ms, MCMC 快 900× | 范式三：sleep-phase 编译 |
| 2 | **MCMC = 诊断/teacher only** | 全负 Δacc (即使 informed proposal) | 范式二：尺度不匹配 |
| 3 | **E_obs = BCE (几何匹配)** | MSE 度量"失明"，BCE 4× 更好重建 | 范式四：观测几何 |
| 4 | **L_cls 从训练移除** | forward-KL 覆盖式已隐含语义 | 范式三：sleep phase 纯粹性 |
| 5 | **Mask mixture 是主导泛化因子** | +5% 独立增益，超过 GDA 架构贡献 | 范式三：编译覆盖 |
| 6 | **GDA 在 center mask 无增益** | 7×7 gap=0%, 14×14 gap=0%, Hopfield 假说未确认 | 范式五需重新评估 |
| 7 | **Stripes 根因 = bit_mask 过度激进** | confidence policy Δacc 0% (vs -17%) | 操作层面，非范式层面 |
| 8 | **跨数据集泛化成立** | MNIST/FMNIST/KMNIST 全部 center Δacc > 0，零架构改动 | 范式通用性 |
| 9 | **Scale-up 有效，GDA 无增益** | 14×14 Δacc=+39% (vs 7×7 +25%)，GDA gap=0% | 范式五需重新定位 |
| 10 | **E_obs 驱动修复 >> E_core 驱动** | evidence_fixed total=+13.0% vs adaptive_pp +0.0% | 范式四：观测几何决定修复 |

## 五大计算范式

### 范式一：配分函数壁垒
z ∈ {0,1}^d, 2^d 构型不可穷举。必须用能量差分/局部归一化/对比学习。
**来源：** LeCun (2006)

### 范式二：尺度匹配
推断算子感受野 ≥ 最大能量尺度。MCMC bit-flip 对 E_obs "失明"。
**来源：** Ramsauer et al. (2021)

### 范式三：睡眠期编译
forward-KL sleep phase 将离散优化编译为前馈计算。mask 分布 = 编译覆盖域。
**来源：** Hinton et al. (1995); MaskGIT (Chang 2022)

### 范式四：观测几何
E_obs 是观测空间的度量张量，不是 loss。Sigmoid decoder → BCE, not MSE.
**来源：** Loaiza-Ganem & Cunningham (2019)

### 范式五：Hopfield 检索
argmin E(z) subject to z_obs fixed ≅ Hopfield 模式完成 ≅ attention layer。
**来源：** Ramsauer et al. (2021); Hopfield (1982)

```
范式五(检索) → 范式四(几何) → 范式三(编译) → 范式二(尺度) → 范式一(配分函数)
每层约束其上层。
```

## 实验结果汇总

### E_obs 几何消融 (exp_eobs.py)

| E_obs | Δacc | MSE重建 | 度量运动性 |
|-------|------|---------|----------|
| MSE | +8.5% | 0.114 | **冻结**（修复前后不变） |
| BCE | +9.0% | 0.028 | 大幅改善 (bce 3.94→1.58) |
| CB | +9.0% | 0.054 | 大幅改善 (cb 2.04→0.33) |

### Phase 1: MCMC block + 契约 (exp_phase1.py)

| 方法 | center Δacc | stripes Δacc | 速度 |
|------|------------|-------------|------|
| mcmc_bit | -17% | -45% | 894ms |
| mcmc_block | -3% | -31% | 2226ms |
| amortized | +7% | -16% | 2.4ms |
| iterative_4 | **+15%** | -20% | 87ms |

### GDA v1 (exp_gda.py)

| config | v1 (local) | v2 (+GDA) | GDA 增益 |
|--------|-----------|-----------|---------|
| center+clean | +4% | **+13%** | +9% |
| stripes+clean | -11% | -16% | -5% |

### GDA v2: evidence gating + mixture (exp_gda_v2.py)

| config | v1_rand | v1_mix | v3_mix |
|--------|---------|--------|--------|
| center+clean | +9% | **+14%** | +14% |
| stripes+noise | -16% | -14% | **-9%** |

核心发现：mask mixture (+5%) >> GDA architecture (~0%) at 7×7

### 14×14 Scale-Up (exp_scale14.py)

| Grid | Method | center+clean Δacc | center+noise Δacc | Clean probe |
|------|--------|-------------------|-------------------|-------------|
| 7×7 | local | +25.0% | +35.0% | 67.2% |
| 7×7 | +GDA | +25.0% | +34.0% | 67.2% |
| **14×14** | **local** | **+39.0%** | **+39.0%** | **73.6%** |
| 14×14 | +GDA | +39.0% | +37.0% | 73.6% |

**Scale-up confirmed:** 14×14 → larger Δacc (+39% vs +25%), higher probe accuracy (73.6% vs 67.2%)
**GDA gap = 0% at both scales:** Hopfield 检索假说未确认。Local 3×3 MRF 对 contiguous occlusion 已足够。GDA 可能仅在 scattered/sparse mask 下有价值。

### Cross-Dataset Generalization (exp_generalization.py)

| Dataset | center+clean Δacc | center+noise Δacc | stripes+clean Δacc |
|---------|-------------------|-------------------|--------------------|
| MNIST | +13.5% | +16.5% | -10.0% |
| **FashionMNIST** | **+30.0%** | **+28.0%** | -15.5% |
| KMNIST | +4.5% | +4.0% | -3.5% |

全部数据集 center Δacc > 0，零架构改动。范式跨数据集泛化已验证。

### Smart Mask Policy (exp_smart_mask.py)

| policy | center Δacc | stripes Δacc | bit_ratio |
|--------|-----------|-------------|-----------|
| any | **+12%** | -17% | 51%/71% |
| majority | -6% | **0%** | 18%/0% |
| confidence | 0% | **0%** | 1%/1% |

### Adaptive Policy (exp_adaptive_policy.py)

| policy | center avg | stripes avg | **total** |
|--------|-----------|-------------|-----------|
| any | +11.0% | -14.5% | -3.5% |
| confidence | +0.0% | -1.5% | -1.5% |
| **adaptive_pp** | **+1.0%** | **-1.0%** | **+0.0%** |

根因：E_core 一致性信号不能检测 correlated corruption（center 整个区域同时被遮挡 → 邻居都错 → 互相"同意"错误编码）。需要 E_obs 级别的信号来检测。

### Evidence-Strength Repair (exp_evidence_strength.py)

| policy | center_clean | center_noise | stripes_clean | stripes_noise | **total** |
|--------|-------------|-------------|--------------|--------------|-----------|
| any (全修) | +13.0% | +11.0% | -16.0% | -13.0% | **-5.0%** |
| evidence_adaptive | +8.0% | +10.0% | -16.0% | -6.0% | -4.0% |
| **evidence_fixed(th=1.0)** | **+6.0%** | **+7.0%** | **+0.0%** | **+0.0%** | **+13.0%** |

**突破：** E_obs 残差信号成功分离 center 和 stripes。Total=+13.0%，是所有 policy 中最好的。
核心机制：全遮挡 patch 的 E_obs 残差 = ∞（无观测信号 → 必须修复），部分观测 patch 残差低（观测匹配 → 不修复）。
范式结论：**修复决策应由观测空间信号 (E_obs) 驱动，不是编码空间信号 (E_core)**。

## 范式契约（已固化）

```json
{
  "E_obs": "BCE (geometrically matched to sigmoid decoder)",
  "L_cls": "REMOVED — classifier is probe only",
  "inference_main": "amortized / iterative (compiled)",
  "MCMC": "diagnostic/teacher only, never deployment",
  "mask_training": "mixture (random 40%: center 20%: stripes 20%: multi_hole 20%)"
}
```

## 当前优先级

### 已完成
1. ~~跨数据集泛化~~：✅ MNIST/FMNIST/KMNIST 全部 center Δacc > 0
2. ~~Scale to 14×14~~：✅ +39% Δacc，GDA gap=0%（Hopfield 假说未确认）
3. ~~Evidence-strength repair~~：✅ E_obs 残差 total=+13.0%，远超 E_core 一致性 (+0.0%)

### 当前优先级
4. **KMNIST 表征容量**：增加 k bits / grid size / encoder depth
5. **CIFAR-10 小规模验证**：in_channels=3，验证范式在 RGB 上的可行性
6. **MCMC 作为诊断工具**：确认 MCMC 永远不该用于部署
7. **Paradigm narrative**：为论文准备范式叙事框架

## 构建与运行

```bash
# 单实验
python benchmarks/exp_eobs.py --device cuda --eval_samples 200
python benchmarks/exp_phase1.py --device cuda --eval_samples 100
python benchmarks/exp_gda.py --device cuda --eval_samples 100
python benchmarks/exp_gda_v2.py --device cuda --eval_samples 100
python benchmarks/exp_smart_mask.py --device cuda --eval_samples 100
python benchmarks/exp_adaptive_policy.py --device cuda --eval_samples 100

# 跨数据集泛化
python benchmarks/exp_generalization.py --device cuda --eval_samples 100

# 14×14 scale-up
python benchmarks/exp_scale14.py --device cuda
```

## 文件结构

```
route_c/
├── core/local_energy.py       # E_core: LocalBitPredictor, RouteCModel
├── inference/energy.py         # 能量模型接口 + Gibbs/MH 推断
├── inpainting/__init__.py      # InpaintNet + 训练 + 摊销/迭代推断
├── learned_routing/__init__.py # LearnedHammingGate + LSH + LearnedDRoPEEnergy
├── learning/quantizer.py       # BinaryEncoder, BinaryDecoder, LearnableADC_DAC
├── benchmarks/
│   ├── run_suite.py            # 原始基准测试
│   ├── exp_eobs.py             # E_obs 几何消融
│   ├── exp_phase1.py           # MCMC block + 范式契约
│   ├── exp_gda.py              # GDA v1
│   ├── exp_gda_v2.py           # GDA v2 (evidence gating + mixture)
│   ├── exp_smart_mask.py       # Smart bit_mask policy
│   ├── exp_adaptive_policy.py  # Adaptive policy + threshold sweep
│   ├── exp_generalization.py   # 跨数据集泛化（已完成）
│   ├── exp_scale14.py          # 14×14 scale-up + GDA 验证（已完成）
│   └── exp_evidence_strength.py # E_obs 残差驱动修复（已完成，total=+13%）
├── PARADIGM_REPORT.md          # 范式研究报告（文献+benchmark+实验矩阵）
├── DESIGN_DOC.md               # 设计文档 v2.1
└── CLAUDE.md                   # 本文件
```
