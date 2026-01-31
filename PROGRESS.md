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

### R0: Routing vs Encoder Capacity — Causal Separation (exp_r0_routing_encoder.py) ✅

**核心实验：** 5-config 因果分离矩阵，隔离 content routing 效果 vs encoder 容量效果。

| Config | Encoder | Routing | acc_clean | acc_repair | gap | div | HF_noise | ColorKL | conn |
|--------|---------|---------|-----------|-----------|-----|-----|----------|---------|------|
| A baseline | Encoder16 (159K) | none | 0.414 | 0.374 | +0.040 | 0.457 | 989 | 1.00 | 0.953 |
| B content | Encoder16 | content K=4 | 0.416 | 0.386 | +0.030 | 0.468 | **509** | **0.82** | 0.937 |
| C random | Encoder16 | random K=4 | **0.448** | **0.410** | +0.038 | 0.464 | 802 | 1.28 | 0.647 |
| D deep | EncoderDeep16 (834K) | none | 0.416 | 0.372 | +0.044 | 0.372 | 772 | 1.20 | 1.000 |
| E deep+route | EncoderDeep16 | content K=4 | 0.412 | 0.360 | +0.052 | 0.430 | 709 | 0.90 | 0.999 |

#### 因果分析

| 效应 | diversity Δ | classify Δ | 判定 |
|------|-----------|-----------|------|
| Routing (B vs A) | +0.010 | +0.002 | 微弱 |
| Random control (C vs A) | +0.007 | **+0.034** | Random > Content! |
| Encoder (D vs A) | **-0.086** | +0.002 | 有害 |
| Interaction (E-D vs B-A) | +0.049 | -0.006 | 无协同 |

#### 结论

**结论 #69: Content routing 不改善分类（+0.2%），random routing 反而更好（+3.4%）** — 分类增益来自正则化/噪声注入，不是学到的内容相似性。GPT 的 "content-dependent routing 是缺件" 假说被否定。
**结论 #70: Content routing 大幅改善 HF_noise（989→509，-49%）** — 全局信息混合确实帮助生成纹理质量，但机制是频率分布对齐而非语义路由。
**结论 #71: 深度 encoder（5.2× 参数）不改善分类（0.416 ≈ 0.414）** — 反驳 "encoder 容量是瓶颈" 假说。更多参数不等于更好的 z 语义。
**结论 #72: 深度 encoder 损害生成多样性（0.457→0.372，-18.6%）** — 过参数化 encoder 在小数据上过拟合，导致 z 分布坍塌。
**结论 #73: 分类瓶颈在训练协议而非架构** — 所有 5 个架构变体 acc 都在 41-45% 范围内（spread 仅 3.6%），远低于 TinyCNN 的 61%。瓶颈是 reconstruction-only 的 sleep-phase 训练不产生判别性 z，而非 encoder/flow 容量不足。

**范式意义：** GPT 提出的 "Transformer-like content routing" 路线被实验否定。分类精度的根本限制不在架构（encoder 深度、routing 机制），而在训练目标（纯重建 vs 判别性）。这是范式契约的固有代价：z 是为重建/修复优化的协议，不是为分类优化的表示。

---

## 实验状态

- [x] E2a: 全局先验 — spatial_cov 打破齐次相 (#52-54)
- [x] G1-lite: dt/T/schedule sweep (#55)
- [x] E2b-combined: 24bit+spatial_cov (#56-58)
- [x] v3 Classify: 层级+对比 (#59-61)
- [x] E2b-light: mixture start (#62) — 之前已跑出部分结果
- [x] **X-CORE: 统一 E_prior** (#63-68) ✅
- [x] **R0: Routing vs Encoder Capacity** (#69-73) ✅
- [ ] Git push 所有结果 + 图片

---

## 历史实验记录（从 CLAUDE.md 迁移）

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

核心发现：mask mixture (+5%) >> GDA architecture (~0%) at 7×7

### 14×14 Scale-Up (exp_scale14.py)

| Grid | Method | center+clean Δacc | Clean probe |
|------|--------|-------------------|-------------|
| 7×7 | local | +25.0% | 67.2% |
| **14×14** | **local** | **+39.0%** | **73.6%** |
| 14×14 | +GDA | +39.0% | 73.6% |

GDA gap = 0% at both scales.

### Cross-Dataset Generalization (exp_generalization.py)

| Dataset | center+clean Δacc | center+noise Δacc |
|---------|-------------------|-------------------|
| MNIST | +13.5% | +16.5% |
| FashionMNIST | +30.0% | +28.0% |
| KMNIST | +4.5% | +4.0% |

### Evidence-Strength Repair (exp_evidence_strength.py)

| policy | **total** |
|--------|-----------|
| any (全修) | **-5.0%** |
| **evidence_fixed(th=1.0)** | **+13.0%** |

E_obs 残差信号成功分离 center 和 stripes。

### Phase 3: Evidence Repair 模块化 — 全 9 config Pareto PASS

### Phase 4-9: 观测协议/量化/约束/高分辨率/算子调度

- Phase 4: 像素 E_obs 最强(7×7)，14×14 下三协议收敛
- Phase 5: INT8 可行(probe -0.4%)
- Phase 6: D4 约束接口有效但有限
- Phase 7: 14×14 下 token/freq/pixel 全收敛
- Phase 8: INT4 全 viable, QAT Δacc +31%
- Phase 9: OperatorSelector 7×7 正确全选 local

### Phase 10-10C2: Feature Bridge

- Phase 10: Binary bridge DEGRADED (10% agree)
- Phase 10A: KL-dominant 仍 DEGRADED (27-34%)
- **Phase 10B: Same-domain + INT/VQ 突破** — INT8_k32=97.6% agree
- Phase 10C-2: INT4_k16 PROTOCOL-STABLE (z_cycle=1.85%)

### Phase 11-12: Repairability + Cycle Contract

- Phase 11: 受 translator 瓶颈限制
- **Phase 12: z 是稳定协议** — clean cycle 1.38%, 5-cycle drift <3%

### Phase 13B: Temporal Dynamics — CONFIRMED (5/6 checks PASS)

### Generation: Unconditional + Freq-as-Evidence

- Denoising compilation 赢协议指标 (viol 最低, cycle 最低)
- freq_full_ms HF_noise=295 最接近真实(264)

### CIFAR-10 A2: freq_full_ms 唯一 "BETTER" 频率评价

### CIFAR-10 Classification v1-v2

- Conv probe >> Linear (+13%)
- **Repair 是分布迁移非语义破坏** — mixed training 最优 (clean 44.9% / repair 40.7%)

### CIFAR-10 C1 Bandwidth + C2 Staged Encoder

- 带宽有帮助但非主瓶颈 (spread=2.8%)
- flat_resblock=51.5% 但 gap=7.5%; staged gap≈0% 但 clean 降至 40-45%

### G1+G2: 带宽升级 + 频率带采样

- 32×32×16: div +70%, viol -24%, 但 HF_noise 爆炸(1838)
- 频率带调度: E_gap_low 近半

### G4 + Flow F0/F0b/F0c

- **Flow > one-shot**: diversity 3.8×
- **Energy hinge (D1) 最优**: E_gap_L=0.087
- **flat_norm (GroupNorm) 是正解**: delta_u=6.0 自然收敛
- σ schedule 完全无效
- U-Net delta_u 恒定(稳定), flat T>10 发散

### C1: Unified Operator — Op-D (energy hinge) Pareto 最优

- Evidence clamping = 架构性零泄漏 (全4算子 ham_unmasked=0.000)
- Gen-first 算子通过全部部署门

### G2-lite: Protocol Density

- **24-bit HF_noise Pareto 最优** (231 vs real 264)
- INT4 token HF 最低(115) 但 cycle 崩溃

### E2a: Global Statistics Prior

- **Spatial covariance 打破齐次相** (HueVar 0.044→2.785, real=2.44)
- Marginal prior 有害, channel-only 无效

### G1-lite: dt/T/Schedule

- Violation ↔ HF_noise 根本权衡, 不可调优解决

### E2b: 24bit + Spatial Cov Combined

- **24bit+spatial_cov_0.3 Pareto 最优** (HF=204, ColorKL=0.98)
- HueVar 仍未解决

### E2b-light: Factorized Prior Start

- **Mixture K8 HueVar=0.026** (超过 real 0.020), 首次突破

### X-CORE: 统一先验跨 repair+gen (#63-68)

- E_prior 对 repair 有害(gap↑), 对 gen 有益(HF=259≈real)
- 两模式需求相反，E_prior 不能统一

### R0: Routing vs Encoder 因果分离 (#69-73)

- Content routing 不改善分类(+0.2%), random 更好(+3.4%)
- 深度 encoder(5.2×) 不改善分类，反损害 diversity(-18.6%)
- 分类瓶颈在训练协议(reconstruction-only)，非架构

### O0: Observability Floor (#74-75)

- obs_floor 杀死 dead bits (1.0→0.025) 但 diversity 坍塌(0.47→0.002)
- 24bit+obs_floor 不如 16bit+obs_floor

### S0: System Identification Diagnostic (#76-77)

- 离散 Gramian 秩 ~94%, 可控秩 99.6% — 系统非真"死"
- Spectral gap 56万：rank 高但 per-bit influence 极低(~0.02)

### ICC-1: Information Circulation Constraint (#78-81) ✅

| Config | dead% | entropy | Fisher | div | acc | gap | HueVar | cycle |
|--------|-------|---------|--------|-----|-----|-----|--------|-------|
| ICC1A base | 1.000 | 0.216 | 0.041 | 0.480 | 0.420 | 0.040 | 0.0009 | 0.035 |
| ICC1B +entropy | 0.880 | 0.667 | 0.032 | 0.466 | 0.392 | 0.014 | 0.0040 | 0.180 |
| ICC1C +ctrl_port | 1.000 | 0.209 | 0.021 | 0.464 | 0.432 | 0.014 | 0.0013 | 0.035 |
| ICC1D +ctrl_full | 0.865 | 0.655 | **2.433** | 0.436 | 0.392 | 0.050 | 0.0024 | 0.150 |

- **条件熵(B)**: HueVar 4.4×, gap -65%, div 仅降3% — 最温和的正则化
- **ControlPort 无损失(C)**: 无效果，Fisher 反降 — 架构通路不够需损失驱动
- **ANOVA ctrl_full(D)**: Fisher 59×突破! 但 acc 降, gap 升, cycle 退化
- **全部 HARD_FAIL**: cycle 退化(0.035→0.15-0.18)破坏协议稳定性
- **结论**: O+C 约束方向正确但 λ 过强，需保守调优(λ_ent≤0.1, λ_ctrl≤0.3)
