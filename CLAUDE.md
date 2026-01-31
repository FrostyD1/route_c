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
| 11 | **Evidence repair 跨数据集 Pareto** | MNIST/FMNIST/KMNIST 9/9 configs Pareto PASS | 范式四：通用修复策略 |
| 12 | **像素 E_obs 优于 token/freq（7×7）** | pixel total=+7%, token=-2%, freq=-2% | 范式四：像素空间信息最丰富（低分辨率） |
| 13 | **INT8 量化可行** | probe drop 0.4%, Δacc 反升 +7% | 范式五：离散化路径存在 |
| 14 | **D4 约束接口有效但有限** | acc_var 降低 3%, viol_var 最低 | 范式六：约束 API 可行 |
| 15 | **协议化观测在正确分辨率下成立** | 14×14: token +43%, freq +43%, pixel +42% | 范式四：z-space 观测可脱离像素 |
| 16 | **INT4 量化可行，QAT 最优** | INT4 PTQ probe +1.4%, QAT Δacc +31% | 范式五：硬件化现实路径 |
| 17 | **Gabor/DCT bank > Sobel/LoG** | probe 55.6% vs 39.8%，但仍不足 | 范式五：固定滤波器需更多基 |
| 18 | **OperatorSelector 正确保守** | 7×7 全选 local，与 FGO 触发条件一致 | 范式架构：条件调度 |
| 19 | **Feature Bridge 需同域+多级量化** | binary 10%→INT8 97.6%, VQ 94-96% | 逆映射：域匹配+幅值保留 |
| 20 | **z 是稳定协议（cycle contract）** | Hamming(z,ẑ)=1.4%, 5-cycle drift<2.8% | 范式核心：协议可逆 |
| 21 | **Repair 提升 cycle 稳定性** | 遮挡后 Hamming 1.87%→1.57%, gain=+0.003 | 范式四：repair 增强协议 |
| 22 | **Feature Bridge 全配置可行** | 8/8 configs ≥94% agreement, repair +16-22% | 逆映射：INT/VQ 双路均可 |
| 23 | **INT4 协议可闭环迭代** | z_cycle=1.85%, drift5=3.07%, agree=95.8% | 范式核心：可读写计算总线 |
| 24 | **时序动力学成立** | 1-step/5-step MSE 赢 baseline, E_dyn gap=0.31 | 范式扩展：离散核心可做时间演化 |
| 25 | **去噪编译生成最优** | violation 0.178（最低），cycle 0.012（最低） | 范式生成：sleep-phase 编译可做无条件采样 |

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

### FGO v1 (exp_fgo_v1.py)

| Grid | Variant | center gap | multi_hole gap | random_sparse gap |
|------|---------|-----------|---------------|------------------|
| 7×7 | fgo_multiband | **+7%** | +1% | +1% |
| 7×7 | fgo_adaptive | **+7%** | -1% | 0% |
| 14×14 | fgo_multiband | +1% | 0% | -1% |
| 14×14 | fgo_adaptive | 0% | 0% | 0% |

7×7 有增益但 14×14 消失。原因：local conv 多层堆叠在小 grid 上已近似全局混合。
random_sparse@7×7 = -59%（灾难性）：bit_mask 标记所有位置为 masked → 零证据。

### FGO-Trigger (exp_fgo_trigger.py) — 28×28 grid, 784 tokens

| Mask | local Δacc | fgo_adaptive Δacc | **gap** | evidence% |
|------|-----------|-------------------|---------|-----------|
| two_block | +6.0% | +6.0% | 0% | 88% |
| **checkerboard** | +23.0% | **+26.0%** | **+3%** | 49% |
| missing_quadrant | +8.0% | +7.0% | -1% | 75% |
| center | +28.0% | **+33.0%** | **+5%** | 75% |

**FGO 触发条件确认：** 28×28 grid + checkerboard/center = fgo_adaptive 有 +3~5% gap。
触发三要素：大 grid (≥28×28) + 分布式遮挡 + 足够证据密度。
contiguous 远程（two_block, missing_quadrant）不触发 — local propagation 足够。

### Phase 3: Evidence Repair 模块化 (exp_phase3_evidence_module.py)

跨数据集 × 3 mask × 2 policy（any vs evidence_fixed th=1.0）：

| Dataset | Policy | center Δacc | stripes Δacc | multi_hole Δacc | **total** | Pareto |
|---------|--------|------------|-------------|----------------|-----------|--------|
| MNIST | any | +9% | -14% | -5% | -10% | — |
| MNIST | evidence | +8% | +0% | +0% | **+8%** | ✅ PASS |
| FMNIST | any | +15% | -14% | -3% | -2% | — |
| FMNIST | evidence | +13% | +5% | +8% | **+26%** | ✅ PASS |
| KMNIST | any | +0% | -9% | -1% | -10% | — |
| KMNIST | evidence | +0% | +2% | +2% | **+4%** | ✅ PASS |

**全部 9 个 config Pareto PASS。** Evidence repair 是范式级突破。

### Phase 4: 观测协议迁移 (exp_phase4_observation_protocol.py)

| Protocol | center Δacc | stripes Δacc | **total** |
|----------|------------|-------------|-----------|
| **pixel_bce** | **+20.0%** | **-13.0%** | **+7.0%** |
| token_bce | +19.0% | -21.0% | -2.0% |
| freq_dct | +19.0% | -21.0% | -2.0% |

**像素 E_obs 仍然最强。** Token/Freq 协议 center 接近（-1%），但 stripes 大幅退化。
原因：token/freq E_obs 在 z-space 或 DCT-space 定义，对 stripes 像素级部分遮挡信息丢失更严重。

### Phase 5: 边界翻译离散化 (exp_phase5_discretize.py)

| Config | Probe Acc | center Δacc | Enc Params | Dec Params |
|--------|----------|------------|-----------|-----------|
| **learned_fp32** | **69.0%** | +20.0% | 42,184 | 74,433 |
| **learned_int8** | 68.6% | **+27.0%** | 42,184 | 74,433 |
| fixed_filter_fp32 | 39.8% | -6.0% | 216 | 20,833 |

**INT8 量化可行：** probe 仅降 0.4%，Δacc 反而提升（+27% vs +20%，量化噪声可能起正则化作用）。
**固定滤波器不够：** probe 39.8%（瓶颈是编码器容量），params 仅 0.5%，但信息损失过大。

### Phase 6: 约束写入接口 (exp_phase6_constraints.py)

| Config | rot0 acc | rot90 acc | rot180 acc | rot270 acc | acc_var |
|--------|---------|----------|-----------|-----------|---------|
| no_constraint | 71.0% | 1.0% | 36.0% | 5.0% | 0.0793 |
| **d4_constraint** | 69.0% | 2.0% | 40.0% | 4.0% | **0.0769** |
| aug_rotate | 70.0% | 1.0% | 38.0% | 4.0% | 0.0792 |
| d4_plus_aug | 71.0% | 1.0% | 37.0% | 5.0% | 0.0797 |

| Config | rot0 viol | rot90 viol | rot180 viol | rot270 viol | viol_var |
|--------|----------|-----------|-----------|-----------|---------|
| no_constraint | 0.073 | 0.076 | 0.072 | 0.079 | 8e-6 |
| **d4_constraint** | 0.276 | 0.276 | 0.278 | 0.275 | **1e-6** |
| aug_rotate | 0.077 | 0.075 | 0.077 | 0.079 | 2e-6 |

**D4 约束降低旋转灵敏度**（acc_var 0.0793→0.0769），violation variance 最低（1e-6）。
但效果有限 — 旋转准确率整体很差（rot90/rot270 ~1-5%），根因是编码器不具备旋转等变性。
**约束接口 API 已实现：** `ConstraintInterface.add_symmetry('D4')` + `compile()` 可声明约束。

### Phase 7: 高分辨率协议观测 (exp_phase7_hires_protocol.py)

| 网格 | 协议 | center Δacc | stripes Δacc | **total** |
|------|------|------------|-------------|-----------|
| 7×7 | pixel_bce | +23% | -13% | +10% |
| 7×7 | token_bce | +20% | -22% | -2% |
| 7×7 | freq_dct | +20% | -22% | -2% |
| **14×14** | **pixel_bce** | **+43%** | **-1%** | **+42%** |
| **14×14** | **token_bce** | **+44%** | **-1%** | **+43%** |
| **14×14** | **freq_dct** | **+44%** | **-1%** | **+43%** |

**GPT 诊断完全正确：** 分辨率是 token/freq 失败的根因。14×14 下三种协议完全收敛。
Stripes 从 -22% → -1%。z-space 观测协议在 2×2 patch 分辨率下可行。

### Phase 8: INT4 量化 + Gabor/DCT (exp_phase8_int4_quant.py)

| 配置 | Probe | center Δacc | Enc Params |
|------|-------|------------|-----------|
| learned_fp32 | 68.2% | +22% | 42,184 |
| learned_int8 | 68.6% | +18% | 42,184 |
| **learned_int4_ptq** | **69.6%** | +21% | 42,184 |
| **qat_int4** | 68.8% | **+31%** | 42,184 |
| gabor_dct_bank | 55.6% | +2% | 808 |

**INT4 全 VIABLE。** QAT Δacc +31% 是所有配置最高。Gabor/DCT bank probe 55.6%（比 Phase 5 Sobel/LoG 39.8% 提升 40%）。

### Phase 9: 条件算子调度 (exp_phase9_operator_selector.py)

| Mask | Selected | local Δacc | fgo Δacc | gap | evidence% |
|------|----------|-----------|---------|-----|-----------|
| center | local | +14% | +22% | +8% | 0.49 |
| stripes | local | -22% | -23% | -1% | 0.29 |
| checkerboard | local | -2% | -1% | +1% | 0.49 |
| multi_hole | local | -10% | -6% | +4% | 0.76 |

| 策略 | total |
|------|-------|
| Always local | -20% |
| Always FGO | -8% |
| Auto-dispatch | -20% |
| Oracle (best) | -7% |

7×7 下 selector 正确全选 local（grid<14）。FGO 在 center/multi_hole 有 gap，但小 grid 不可靠。
**OperatorSelector 的价值在大 grid：** 在 28×28 + distributed mask 时才触发 FGO（见 Phase 2）。

### Phase 10: Feature Translator (exp_phase10_feature_translator.py)

| bits | cosine(h,ĥ) | agreement | KL_div | Q params | R params |
|------|-------------|-----------|--------|----------|----------|
| 8 | 0.497 | 10.3% | 2.32 | 33,928 | 34,176 |
| 4 | 0.510 | 12.7% | 1.65 | 33,412 | 33,664 |
| 2 | 0.501 | 9.0% | 2.12 | 33,154 | 33,408 |

**全部 DEGRADED。** 原因：ResNet18 layer3 = 256 通道 → 8/4/2 bits 压缩比太极端（256:8=32×）。

### Phase 10A: Bridge v2 — KL-dominant + 3×3 conv (exp_phase10a_bridge_v2.py)

| k | ratio | agree% | cosine | KL | Δlogit |
|---|-------|--------|--------|-----|--------|
| 64 | 4× | 27.0% | 0.726 | 0.74 | 1.89 |
| 32 | 8× | 30.7% | 0.682 | 0.86 | 2.05 |
| 16 | 16× | 34.0% | 0.609 | 1.05 | 2.28 |
| 8 | 32× | 30.3% | 0.588 | 0.98 | 2.19 |

**仍然 DEGRADED。** 3×3 conv + KL-dominant 改善到 27-34%（从 10%），但 binary {0,1} 根本无法表示连续特征幅值/相位。

### Phase 10B: Bridge v3 — Same-domain Teacher + INT/VQ (exp_phase10b_bridge_v3.py) ✅

**突破！** 同域教师（小 CNN 直接训练在 FMNIST，特征 7×7×64）+ 多级量化。

**Route A: INT Token Bridge**

| config | bits/pos | agree% | cosine | KL | drift1 | drift5 | repair |
|--------|---------|--------|--------|-----|--------|--------|--------|
| INT8_k32 | 256 | **97.6%** | **0.9634** | 0.004 | 0.010 | 0.092 | +19.5% |
| INT8_k16 | 128 | 97.4% | 0.9534 | 0.006 | 0.011 | 0.120 | +22.0% |
| INT4_k32 | 128 | 97.2% | 0.9558 | 0.006 | 0.023 | 0.137 | +16.5% |
| INT4_k16 | 64 | 96.6% | 0.9455 | 0.011 | 0.024 | 0.184 | +20.5% |

**Route B: VQ Codebook Bridge**

| config | bits/pos | agree% | cosine | KL | drift1 | drift5 | repair |
|--------|---------|--------|--------|-----|--------|--------|--------|
| VQ_K256_d16 | 8 | **96.0%** | 0.608 | 0.025 | 0.024 | 0.124 | +22.5% |
| VQ_K512_d32 | 9 | 95.0% | 0.582 | 0.028 | 0.017 | 0.103 | +17.5% |
| VQ_K512_d16 | 9 | 94.8% | 0.593 | 0.026 | 0.026 | 0.141 | +18.5% |
| VQ_K256_d32 | 8 | 94.0% | 0.586 | 0.031 | 0.018 | 0.058 | +21.0% |

**全部 8 配置 agreement ≥ 90%！** Best: INT8_k32 = 97.6%（vs Phase 10 binary 10.3%，提升 +87.3%）。
**三个关键修复：**
1. **同域教师**：FMNIST 小 CNN（81.8% acc）vs ImageNet ResNet18（域不匹配）
2. **多级量化**：INT4/INT8 保留幅值信息 vs binary {0,1} 丢失一切
3. **VQ codebook**：离散聚类而非二值化 — 8 bits/position 仍达 94-96%

**Repair 全部有效：** +16-22% 修复增益，证明 bridge 后的离散表示可修复。

### Phase 11: Feature Repairability (exp_phase11_repairability.py)

| corruption | agree_no_repair | agree_repaired | cosine_no_rep | cosine_repair | gain |
|-----------|----------------|---------------|---------------|---------------|------|
| center | 14.0% | 16.3% | 0.459 | 0.465 | +2.3% |
| block | 13.7% | 15.3% | 0.460 | 0.464 | +1.7% |
| channels | 15.3% | 10.3% | 0.463 | 0.467 | -5.0% |

受 Phase 10 translator 瓶颈影响，基线 agreement 仅 ~14%。Repair 在 spatial corruption 上有微弱增益。
**需在 Phase 10 translator 改进后重新测试。**

### Phase 12: Cycle Contract (exp_phase12_cycle_contract.py)

| Test | Result |
|------|--------|
| **Clean cycle Hamming(z,ẑ)** | **1.38%** (5.4/392 bits flip) → **STABLE** |
| Occluded no repair | 1.87% |
| Occluded repaired | 1.57% → **Repair IMPROVES stability** |
| 1-cycle drift | 1.48% |
| 2-cycle drift | 2.16% (+0.67%) |
| 3-cycle drift | 2.51% (+1.03%) |
| 5-cycle drift | 2.81% (+1.32%) → **STABLE (drift < 2× single)** |

**z 是稳定协议。** 单次 cycle 仅 1.38% bit flip，5 次 cycle 后仍 < 3%。
Repair 让 cycle 更稳定（1.87% → 1.57%）。多轮 cycle 漂移线性增长而非指数，协议不发散。

### Phase 10C-2: Deterministic Protocol Bridge (exp_phase10c2_deterministic_bridge.py) ✅

| config | agree% | z_cycle | drift1 | drift5 | bounded |
|--------|--------|---------|--------|--------|---------|
| **INT4_k16_cyc10** | **95.8%** | **1.85%** | 1.85% | 3.07% | **Y** |
| INT4_k32_cyc20 | 97.4% | 3.02% | 3.02% | 5.35% | Y |
| INT4_k32_cyc10 | 96.8% | 4.17% | 4.17% | 7.54% | Y |
| INT4_k32_cyc05 | 97.2% | 4.86% | 4.86% | 9.26% | Y |
| INT8_k32_cyc10 | 96.8% | 66.5% | 66.5% | 84.5% | Y |
| INT8_k16_cyc10 | 97.0% | 82.8% | 82.8% | 93.0% | Y |

**INT4_k16 PROTOCOL-STABLE!** z_cycle=1.85%（≤2%），drift5=3.07%，agreement=95.8%。
三个关键修复：(1) 确定性量化（EMA 校准固定 scale），(2) 多轮 cycle loss（训练展开 3 cycles），(3) 边界余量损失（推离 rounding 边界）。
INT4 > INT8：16 级量化比 256 级天然更确定（边界更少）。VQ 路线暂停（index 跳变不可控）。

### Phase 13B: Temporal Dynamics v3 (exp_phase13b_temporal_v3.py) ✅

| 指标 | Baseline | Phase 13B | +Projection |
|------|----------|-----------|-------------|
| 1-step Hamming | 0.168 | 0.106 | **0.105** |
| 1-step MSE | 0.047 | 0.024 | 0.024 |
| 5-step MSE | 0.047 | 0.040 | **0.039** |
| E_dyn gap | — | **0.31** | — |
| Var(z) | — | 0.25 | — |

**5/6 checks PASS, TEMPORAL DYNAMICS CONFIRMED.** 1-step 和 5-step MSE 都赢 baseline。
E_dyn 区分真实 vs 打乱序列（gap=0.31）。Energy projection 有正向增益。漂移亚线性。

### Generative: Unconditional Image Generation (exp_gen_unconditional.py)

**FMNIST 首轮结果：**

| method | violation | token_KL | diversity | cycle | 1NN_dist |
|--------|-----------|----------|-----------|-------|----------|
| bernoulli | 0.360 | 0.753 | 0.339 | 0.217 | 32.2 |
| token_cat | 0.261 | **0.035** | 0.340 | 0.139 | 49.2 |
| ar | 0.229 | 0.306 | 0.305 | 0.055 | 52.3 |
| **denoise** | **0.178** | 0.286 | 0.269 | **0.012** | 33.3 |

**Denoising compilation（Route C 范式方法）赢在协议指标：** 最低 violation（0.178），最低 cycle error（0.012）。
AR 在 violation 上第二。Token categorical 有最佳 token KL（但 violation 高）。

### Generative: Freq-as-Evidence A1→A2 路线 (exp_gen_freq_*.py)

**A1v2（freq-aware denoiser training）FMNIST 结果：**
- freq_train_03: violation -14%, connectedness=0.862 ≈ 真实(0.863), BEP_d 改善 20%
- freq_train_03_ms: diversity 0.256（≈baseline 0.258），Gate 全 PASS

**A2（structured HF generation）FMNIST 8-config 结果：**

| config | viol | div | conn | HF_coh | HF_noise |
|--------|------|-----|------|--------|----------|
| baseline | 0.146 | 0.260 | 0.966 | -0.324 | 284 |
| **freq_full** | 0.136 | 0.246 | **0.997** | **-0.319** | 342 |
| freq_sched_coh_ms | 0.152 | 0.248 | 0.982 | -0.320 | 359 |
| **freq_full_ms** | 0.142 | **0.273** | 0.835 | -0.336 | 345 |

freq_full: HF coherence 最接近真实，connectedness 0.997，Gate PASS。
freq_full_ms: diversity 超 baseline(0.273>0.260)，低频 energy gap 最小(0.034)。
所有配置 HF_noise_index 远高于真实(284-484 vs 113)——离散 z→decoder 管线固有的块状梯度问题。

**CIFAR-10 首轮结果：**

| method | violation | diversity | cycle | conn |
|--------|-----------|-----------|-------|------|
| denoise_base | 0.212 | 0.189 | 0.009 | 1.000 |
| denoise_freq | **0.149** | 0.071 | **0.005** | 1.000 |
| denoise_freq_ms | 0.229 | **0.184** | 0.016 | 1.000 |

Denoise compilation 在 CIFAR-10 上也赢（violation 最低），freq 训练进一步改善，但 diversity 仍有坍塌风险。

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

## Session 连续性指令

**这个项目任务量很大，必须连续推进。** 如果 context compact 或换窗口：
1. 读 CLAUDE.md 恢复进度（本文件）
2. 检查 `当前执行阶段` 标记，从断点继续
3. 每完成一个 Phase 立刻更新本文件并 git commit
4. 每个 Phase 写成独立脚本 `benchmarks/exp_xxx.py`，可独立运行
5. 用后台任务并行跑实验，不要等

## 频域路线图 (Phase 0-6)

### 已完成的前置实验
- ~~跨数据集泛化~~：✅ MNIST/FMNIST/KMNIST 全部 center Δacc > 0
- ~~Scale to 14×14~~：✅ +39% Δacc，GDA gap=0%（Hopfield 假说未确认）
- ~~Evidence-strength repair~~：✅ E_obs 残差 total=+13~22%，远超 E_core 一致性 (+0.0%)

### 当前执行阶段：生成实验 A2(structured HF) FMNIST 完成 ✅ → CIFAR-10 freq_full 配置验证中

---

### Phase 0：范式合同冻结（把接口变成不可破坏的 API）
**目标：** Route C 变成可复用计算栈，不是实验脚本堆。

冻结三份合同：
- **表示合同**：z 的形状(k×H×W)、位宽(k=8)、网格(7/14)、mask 语义(evidence policy)
- **能量合同**：E_core(局部规则) + E_obs(观测似然) + 可选 E_relation(全局关系项)，不允许 task head 混入
- **推断合同**：推断器只学 q_φ(z|o,m)，训练目标只来自一致性(sleep-phase)

过关标准：
- 任意换数据集不用改接口
- 任意新算子(FGO/GDA)必须插在同一位置，不破坏合同
- 已完成 70%（Representation Contract + evidence policy），需要固化为 API

---

### Phase 1：FGO v1（频域全局算子）
**目标：** 建立与 Transformer 类似的"全局混合原语"，用 FFT/DCT 走硬件最强路径。

在 InpaintNet 内：local conv → **FGO** → local conv

FGO 三个版本（先做这三种，足够论文）：
1. **Low-pass only**：只开低频（全局形状/结构）
2. **Multi-band gate**：低/中/高频 3 段门控（结构/边缘/纹理）
3. **Data-adaptive gate**：门控由输入产生（task-agnostic）

**关键：** GDA 在 contiguous mask 上无增益 → FGO 必须靠 **multi-hole / random-sparse** 证明价值。
Center 只做 sanity check，不是主判据。

过关标准：
- random-sparse / multi-hole 上 FGO 明显优于 local-only（Δacc + BCE_after + 结构一致性）
- 复杂度 O(N log N)，比 O(N²) attention 更能 scale
- 14×14 上跑通

**实验脚本：** `benchmarks/exp_fgo_v1.py`
**测试 mask：** multi_hole, random_sparse, center(sanity check)
**测试网格：** 7×7, 14×14

---

### Phase 2：FGO v2（内容相关的频域算子）
**目标：** 在不回到 O(N²) 的前提下引入 content-based 能力。

两条路（选一条）：
- **2A) Content-conditioned spectral gating**：门控 G(f) 由 z 的全局摘要生成（按内容选频带）
- **2B) 频域互相关**：FFT(h)·conj(FFT(h_ref))，用可见区域做 reference，全局匹配响应图

过关标准：
- OOD mask（条纹、稀疏结构化遮挡）下，content-conditioned 比固定频带更稳
- 保持 FFT 吞吐优势

**实验脚本：** `benchmarks/exp_fgo_v2.py`

---

### Phase 3：Evidence repair 固化为范式模块
**目标：** 把 evidence_fixed/evidence_adaptive 写成标准模块，消灭 stripes 策略依赖。

实现：
- `RepairMask = f(residual_map, mask_geometry, uncertainty)` 标准接口
- residual_map 来源：decoder 在可见像素的 BCE residual / token likelihood residual

过关标准：
- 不同 mask 类型无需单独设策略
- total Δacc 不被某种 mask 拉垮（Pareto 改善）

**注意：** Phase 6 的 constraint interface 也在这里体现 — 不是所有东西都要不变性，而是提供"可声明约束"的 API。

---

### Phase 4：从"像素观测"迁移到"协议化观测"
**目标：** E_obs 从像素空间搬到离散协议空间，脱 MNIST 偶然性。

两条路（选一条）：
- **4A) Token likelihood**：观测转 tokens，E_obs 在 token 空间定义
- **4B) Frequency-domain observation**：观测 = 低频系数 + 稀疏高频系数

过关标准：
- CIFAR-10/SVHN 子集上保持"修复=观测一致性+结构一致性"闭环
- 不需要 L_cls 对齐语义

---

### Phase 5：边界翻译层可离散化迁移
**目标：** CNN → 可网表化算子库（逐步替换，不一刀切）。

步骤：
1. 局部卷积 → 固定滤波(Sobel/LoG/Gabor/DCT) + 可学习门控
2. 逐层量化约束（STE/LSQ/PTQ）：FP32→INT8→更低
3. 输出协议固定化：z 的 bit-budget、频带结构

过关标准：
- FLOPs 下降、数据类型降低，范式核心仍工作
- 部分算子可替换为 LUT/bitwise 近似

---

### Phase 6：对称性约束接口（不是全面改造）
**目标：** 提供 **constraint writing interface**，让用户声明约束，系统自动编入能量函数。

**这不是"所有东西都要不变性"，而是提供写入 constraint 的 API。**

三种实现（按范式纯度排序）：
1. **E_core 加群等变约束(D4)**：E_core(z) + E_core(R·z) + …
2. **频域协议天然等变**：DCT/FFT 低频能量对旋转/缩放有规律（径向能量谱）
3. **连续侧 augmentation**：最不范式，但最容易，仅作 baseline

FMNIST 验证（形状主体稳定）：
- rotation consistency (0/90/180/270)
- scale jitter (0.8-1.2)
- 定义为 latent consistency 而非 label invariance

过关标准：
- 旋转/缩放扰动下 E_core violation 更低
- 修复性能对变换更鲁棒（Δacc 波动变小）

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
│   ├── exp_evidence_strength.py # E_obs 残差驱动修复（已完成，total=+13%）
│   ├── exp_fgo_trigger.py       # FGO 触发条件验证（28×28, 已完成）
│   ├── exp_phase3_evidence_module.py # Phase 3: Evidence repair 跨数据集（已完成）
│   ├── exp_phase4_observation_protocol.py # Phase 4: Token/Freq/Pixel E_obs（已完成）
│   ├── exp_phase5_discretize.py  # Phase 5: INT8 + 固定滤波（已完成）
│   ├── exp_phase6_constraints.py # Phase 6: D4 约束接口（已完成）
│   ├── exp_phase7_hires_protocol.py # Phase 7: 高分辨率协议观测（已完成）
│   ├── exp_phase8_int4_quant.py  # Phase 8: INT4 量化 + Gabor/DCT（已完成）
│   ├── exp_phase9_operator_selector.py # Phase 9: 条件算子调度（已完成）
│   ├── exp_phase10_feature_translator.py # Phase 10: Feature Translator（binary baseline）
│   ├── exp_phase10a_bridge_v2.py      # Phase 10A: KL-dominant + 3×3 conv（仍 DEGRADED）
│   ├── exp_phase10b_bridge_v3.py      # Phase 10B: Same-domain + INT/VQ（✅ 突破）
│   ├── exp_phase11_repairability.py   # Phase 11: 特征修复（已完成）
│   └── exp_phase12_cycle_contract.py  # Phase 12: Cycle Contract（已完成 ✅）
├── PARADIGM_REPORT.md          # 范式研究报告（文献+benchmark+实验矩阵）
├── DESIGN_DOC.md               # 设计文档 v2.1
└── CLAUDE.md                   # 本文件
```
