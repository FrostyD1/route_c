# Route C — 实验进度记录

> 从此文件开始记录新实验结果，CLAUDE.md 仅保留范式定义和接口规范。

---

## 2026-01-31

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

### G1-lite: dt/T/Schedule Sweep (exp_g1_dt_schedule.py) ✅

| dt | violation | diversity | HF_noise | delta_u |
|----|-----------|-----------|----------|---------|
| 0.10 | 0.0092 | 0.469 | 510 | 2.2 |
| 0.25 | 0.0059 | 0.471 | 604 | 2.6 |
| 0.50 | 0.0049 | 0.457 | 749 | 3.6 |
| 1.00 | 0.0038 | 0.442 | 911 | 7.3 |

**结论 #55:** dt/T/schedule 调优改善 violation 但恶化 HF_noise。根本权衡，不是操作层面能解决的。

### E2b-light: Factorized Prior Start — 待修 bug 重跑

`compute_z_mmd` device mismatch 崩溃。已定位修复方案。

---

## 待跑实验

- [ ] E2b-light (exp_e2b_factorized_prior.py) — device bug 修复后重跑
- [ ] X-CORE (exp_xcore_global_prior.py) — 统一 E_prior 跨 repair + generation
- [ ] Git push 所有结果 + 图片
