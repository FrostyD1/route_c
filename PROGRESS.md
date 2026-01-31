# Route C — 实验进度记录

> 从此文件开始记录新实验结果，CLAUDE.md 仅保留范式定义和接口规范。

---

## 2026-01-31

### X-CORE: 统一 E_prior 跨 Repair + Generation (exp_xcore_global_prior.py) ✅

**核心实验：** 2×3 矩阵 (16/24 bit × none/pos_marg/spatial_cov)，同一 pipeline 同时测 repair + generation。

#### Repair 模式

| Config | probe_clean | probe_repair | **gap** | KL_marg_center | Δcov_center | ham_unmasked |
|--------|------------|-------------|---------|----------------|-------------|--------------|
| 16bit_none | 0.420 | 0.386 | 0.034 | 0.009 | 0.117 | **0.000** |
| 16bit_pos_marg | 0.422 | 0.382 | 0.040 | 0.020 | 0.158 | **0.000** |
| 16bit_spatial_cov | 0.416 | 0.370 | 0.046 | 0.017 | 0.202 | **0.000** |
| **24bit_none** | 0.420 | **0.400** | **0.020** | 0.036 | 0.332 | **0.000** |
| 24bit_pos_marg | **0.458** | 0.420 | 0.038 | 0.082 | 0.591 | **0.000** |
| 24bit_spatial_cov | 0.440 | 0.392 | 0.048 | 0.125 | 0.787 | **0.000** |

#### Generation 模式 (real: HF_noise=264, HueVar=0.019)

| Config | viol | div | HF_noise | HueVar | ColorKL | conn |
|--------|------|-----|----------|--------|---------|------|
| 16bit_none | 0.005 | 0.455 | 735 | 0.001 | 1.16 | 0.871 |
| 16bit_pos_marg | 0.005 | 0.464 | 576 | 0.002 | 1.01 | **0.988** |
| 16bit_spatial_cov | 0.006 | **0.477** | 457 | 0.002 | **0.99** | 0.967 |
| **24bit_none** | 0.023 | 0.429 | **259** | 0.002 | 1.06 | 0.610 |
| 24bit_pos_marg | 0.021 | 0.438 | **260** | 0.002 | **0.69** | 0.884 |
| 24bit_spatial_cov | 0.023 | 0.380 | **197** | 0.001 | 11.94 | **1.000** |

#### 因果链分析: prior → KL_marg ↓ → gap ↓ ?

| Config | KL_marg 变化 | gap 变化 | 因果 |
|--------|-------------|---------|------|
| 16bit pos_marg | 0.009→0.020 ↑ | 0.034→0.040 ↑ | **NO** |
| 16bit spatial_cov | 0.009→0.017 ↑ | 0.034→0.046 ↑ | **NO** |
| 24bit pos_marg | 0.036→0.082 ↑ | 0.020→0.038 ↑ | **NO** |
| 24bit spatial_cov | 0.036→0.125 ↑ | 0.020→0.048 ↑ | **NO** |

#### 结论

**结论 #63: 修复合同零泄漏（架构性保证）** — 全 6 config ham_unmasked=0.000
**结论 #64: 24bit HF_noise 精确匹配真实 (259 vs 264)** — 带宽是 HF_noise 的决定因素
**结论 #65: Prior 在 repair 中有害** — 所有 prior 都增加 KL_marg 和 gap，因果链不成立
**结论 #66: Prior 在 generation 中有效** — spatial_cov 16bit: HF_noise 735→457, div +5%
**结论 #67: Repair 和 Generation 对 prior 的需求相反** — repair 需要忠实还原，prior 推偏分布；generation 需要全局结构，prior 提供约束
**结论 #68: 24bit_spatial_cov ColorKL 崩溃 (11.94)** — λ=0.3 对 24bit 仍太强，spatial_cov 在高带宽下需要更低 λ

**范式意义：** E_prior 不能统一 repair 和 generation。两个模式需要不同的先验策略：
- Repair: 零先验（λ=0），纯 E_obs + E_core 驱动
- Generation: 弱先验（λ=0.1-0.3），提供全局结构约束

---

### v3 Classify: 层级协议 + 对比学习 (exp_cifar10_classify_v3.py) ✅

| Config | Clean | Repair | Gap | Notes |
|--------|-------|--------|-----|-------|
| flat_baseline | 0.476 | 0.417 | 0.059 | 32×32×8, mixed probe |
| hier_sem_only | 0.346 | 0.346 | **0.000** | 8×8×16, perfectly stable |
| hier_tex_only | **0.479** | **0.430** | 0.049 | 32×32×8 tex channel |
| hier_dual | 0.388 | 0.401 | 0.013 | tex+sem combined |
| contra_sem_only | 0.373 | 0.373 | **0.000** | +SimCLR on z_sem |
| contra_dual | 0.426 | 0.418 | 0.008 | +SimCLR on dual |

**结论 #59:** z_sem 完全 repair-stable (gap=0.000)——语义层天然不受 repair 干扰
**结论 #60:** Contrastive 对 z_sem 有效 (+2.7%: 34.6→37.3%)，但低于 z_tex 天花板 (47.9%)
**结论 #61:** 层级结构不提升分类精度 (hier_tex_only 47.9% ≈ flat 47.6%)，ADC 瓶颈

### E2b-combined: 24-bit + Spatial Cov (exp_e2b_combined.py) ✅

| Config | bits | λ | HF_noise(real=264) | HueVar(real=0.019) | ColorKL | div | conn |
|--------|------|---|-----------|---------|---------|-----|------|
| 16bit_baseline | 16 | 0.0 | 420 | 0.0033 | 0.61 | 0.448 | 0.861 |
| 16bit_spatial_cov_0.3 | 16 | 0.3 | 643 | 0.0017 | 0.80 | 0.470 | 0.865 |
| 24bit_baseline | 24 | 0.0 | 381 | 0.0015 | 1.45 | 0.440 | 0.437 |
| **24bit_spatial_cov_0.3** | **24** | **0.3** | **208** | 0.0030 | **0.64** | 0.405 | 0.768 |
| 24bit_spatial_cov_1.0 | 24 | 1.0 | 514 | 0.0007 | 20.4 | 0.243 | 1.000 |
| 24bit_spatial_cov_3.0 | 24 | 3.0 | 705 | 0.0001 | 21.3 | 0.071 | 1.000 |

**结论 #56:** 24bit+spatial_cov_0.3 是 Pareto 最优 (HF_noise=208, ColorKL=0.64)
**结论 #57:** 高 λ 先验摧毁生成质量 (λ=3.0: div=0.07, ColorKL=21)
**结论 #58:** HueVar 仍未解决 (所有配置 <0.003 vs real 0.019)

### E2b-light: Mixture Start (exp_e2b_factorized_prior.py) — 跳过

device bug 崩溃，用户决定跳过直接跑 X-CORE。

### G1-lite: dt/T/Schedule Sweep (exp_g1_dt_schedule.py) ✅

| dt | violation | diversity | HF_noise | delta_u |
|----|-----------|-----------|----------|---------|
| 0.10 | 0.0092 | 0.469 | 510 | 2.2 |
| 0.25 | 0.0059 | 0.471 | 604 | 2.6 |
| 0.50 | 0.0049 | 0.457 | 749 | 3.6 |
| 1.00 | 0.0038 | 0.442 | 911 | 7.3 |

**结论 #55:** dt/T/schedule 调优改善 violation 但恶化 HF_noise。根本权衡，不是操作层面能解决的。

---

## 实验状态

- [x] E2a: 全局先验 — spatial_cov 打破齐次相 (#52-54)
- [x] G1-lite: dt/T/schedule sweep (#55)
- [x] E2b-combined: 24bit+spatial_cov (#56-58)
- [x] v3 Classify: 层级+对比 (#59-61)
- [x] E2b-light: mixture start (#62) — 之前已跑出部分结果
- [x] **X-CORE: 统一 E_prior** (#63-68) ✅
- [ ] Git push 所有结果 + 图片
