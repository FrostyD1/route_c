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
| 26 | **频率证据改善生成连贯性** | conn 0.335→0.997, HF_coh 接近真实, Gate PASS | 观测几何升级：频率是证据通道不是 task loss |
| 27 | **CIFAR-10 彩色生成可行** | freq_full_ms HF_noise 最接近真实(295 vs 264) | 范式可扩展到 RGB，离散核通用 |
| 28 | **Repair 是分布迁移，非语义破坏** | train_repair→test_repair 40.6% vs train_clean→test_repair 16.2% | 范式四：可写协议保持语义 |
| 29 | **混合训练恢复 repair 读出** | mixed probe: clean 44.9% / repair 40.7% (Δ=-4.2%) | 范式三：编译覆盖域需包含 repair 分布 |
| 30 | **带宽有帮助但非主瓶颈** | 32×32×16=51.0% vs 32×32×8=49.5%（spread 仅2.8%） | 协议容量：更多 bits 有增益但需配合更强 encoder |
| 31 | **ResBlock encoder 提升分类至 51.5%** | flat_resblock=51.5% vs plain_flat=45.1%（+6.4%） | 感受野/深度对协议翻译层很重要 |
| 32 | **Staging 实现完美 repair 稳定但牺牲精度** | sem gap=0.000, dual gap=0.2% vs flat gap=7.5%；但 clean -6.2% | 准确度-稳定性权衡：结构隔离有效但有代价 |
| 33 | **G1: 带宽大幅改善生成多样性和一致性** | div 0.201→0.342(+70%), viol 0.381→0.290(-24%), HF_coh接近真实；但HF_noise爆炸(136→1838) | 生成容量：16384 bits 解锁多样性，但32×32 grid引入噪声 |
| 34 | **G2: 频率带调度采样改善低频结构** | E_gap_low 0.335→0.181(近半), div+12%, HF_noise 136→187(更接近264) | 生成采样：粗到细频率调度有效，DCT空间位置代理足够 |
| 35 | **G3: 16×16×16 stride-2 最优生成z规格** | E_gap_low=0.109(最低), HF_noise=187(可控), viol=0.322 | stride-2空间抽象+高channel = 最佳平衡，1:1 mapping (32×32) 不可取 |
| 36 | **Regression denoiser 赢 HF_noise** | HF_noise=297(最接近真实264), div=0.304(最高), viol=0.301(最低) | 连续残差预测比二值分类更适合生成,但仍无细腻纹理 |
| 37 | **G4: U-Net one-shot 改善结构但非纹理** | div +63%(0.144→0.236), E_gap_L 近半(0.250→0.136), MaskGIT div 3×(0.470) HF_coh≈真实(-0.305) | 架构深度改善结构,MaskGIT改善多样性,但HF_noise仍高 |
| 38 | **F0: Flow operator 是范式级突破** | flat flow div 3.8×(0.118→0.448), HF_coh≈真实(-0.311); 能量100%单调下降; 首次出现类漫画细节 | 范式核心：迭代下降算子 > one-shot 预测器 |
| 39 | **能量 hinge 训练改善频率分布** | D1: E_gap_L=0.087(最低), E_gap_H=0.236(C的一半), viol/cycle最低 | E_core参与训练 > 推理时projection(D2≈C) |
| 40 | **T=10 对 U-Net 不够** | U-Net converge_step=10/10(刚好用完), flat=6/10(有余量); U-Net 视觉反而比 flat 杂乱 | 需要 T=20-50 让 U-Net 充分收敛 |
| 41 | **Flat flow T>10 发散（delta_u 指数爆炸）** | delta_u: T10=14.7, T30=5.5M, T50=4.5e13; div: 0.466→0.262(T50坍塌44%) | Flat 无收敛机制, T=10 视觉好纯属甜蜜点 |
| 42 | **U-Net 完全稳定，delta_u 恒定** | delta_u: T5=2.6→T50=3.4; div: 0.471→0.451(仅-4%); E_gap_H@T20=0.017(最佳) | Energy hinge 赋予 U-Net 真正的动力学稳定性 |
| 43 | **HF_noise ∝ T（Langevin 噪声累积）** | flat: 424→1320(T5→T30); unet: 494→953(T5→T50); 真实=264 | σ schedule 后期应更快衰减 |
| 44 | **GroupNorm 是收敛机制正解** | flat_norm: delta_u=6.0(稳定), div=0.483(最高), E_gap_H=0.007(最低); tanh: 61(不够), tanhskip: 0.99(过约束div↓) | 稳定中间层特征 > 限制输出范围 |
| 45 | **σ schedule 对所有模型无效** | 同一模型4种schedule差距<3%; F0c 16配置验证 | HF_noise根因是ADC/DAC管线，非Langevin |
| 46 | **Evidence clamping 保证零证据泄漏** | 全4种算子 ham_unmasked=0.000, cycle<0.04 | 范式核心：修复合同是架构性保证，非训练依赖 |
| 47 | **Gen-first 算子通过全部部署门** | Op-B/D: G1 Cost PASS, G2 Contract PASS, G3 ModeSwitch PASS | 范式统一算子：生成训练的算子可直接做修复 |
| 48 | **Energy-aware 是 Pareto 最优算子** | Op-D: div=0.427(最高), E_gap_H=0.44(最低), 全门PASS; Op-A div=0.03(坍塌) | 能量 hinge + flow 训练 = 统一修复/生成/分类 |
| 49 | **24-bit 是 HF_noise Pareto 最优** | L1_bits24: HF_noise=231(real=264), div=0.435; L1_bits16=921, L1_bits32=659 | 协议密度：24bit/pos 是信息瓶颈的甜蜜点 |
| 50 | **INT4 token HF_noise 最低但 cycle 崩溃** | L3_int4_ch4: HF_noise=115, cycle=0.483; binary→INT4 round-trip 不可控 | INT4 出路需确定性量化(参见 Phase 10C-2) |
| 51 | **残差解码器(L2)多样性坍塌** | L2_main16_res8: div=0.112(坍塌), HF_noise=549; 冻结主路+可训残差不够 | 主+残差分离架构不如平铺 24bit |
| 52 | **Spatial covariance 是正确的全局先验** | HueVar 0.044→2.785(real=2.44), ColorKL -73%, act_KL -80% | 范式：缺件是"方差匹配"不是"均值匹配" |
| 53 | **Marginal prior 有害（推向均值=更齐次）** | div 0.47→0.21, HueVar 0.044→0.010; 匹配均值 ≠ 匹配分布 | 全局先验必须保留方差结构 |
| 54 | **Channel-only prior 完全无效** | 所有指标 Δ<0.001, 与 baseline 无差异 | 缺的是空间非齐次性，不是通道统计 |
| 55 | **dt/T/schedule 调优改善 violation 但恶化 HF_noise** | warmup best viol=0.0034, 但 HF_noise=1165(real=234); 更多步/更大步长→更齐次 | 采样超参不解决结构问题 |
| 56 | **24-bit+spatial_cov_0.3 是组合最优** | HF_noise=204(real=264), ColorKL=0.98(最佳); 两修复互补 | 带宽+先验组合有效 |
| 57 | **高 λ 先验摧毁生成质量** | λ=1.0: HF_noise=514, ColorKL=10; λ=3.0: div=0.15(坍塌), ColorKL=21 | 先验强度必须极保守（λ≤0.3） |
| 58 | **HueVar 仍未解决（需更强的非齐次机制）** | 所有E2b配置 HueVar<0.003(real=0.019); spatial_cov 改善 ColorKL 但不改善像素级色调方差 | 当前先验在协方差层面工作，不在像素纹理层面 |
| 59 | **层级 z 分离：z_sem 完全 repair-stable** | z_sem gap=0.000, z_tex gap=0.049; dual gap=0.013 | 语义层天然稳定，纹理层承载 repair 风险 |
| 60 | **Contrastive 对 z_sem 有效（+2.7%）但低于 z_tex 天花板** | sem_only 37.3%, tex_only 47.9%, dual_contra 42.6% | z_tex 空间场仍是主要信息载体 |
| 61 | **层级结构不提升分类精度（+0.3%）** | hier_tex_only 47.9% vs flat 47.6%, 几乎无差 | 当前 ADC 瓶颈下层级不增加信息量 |
| 62 | **Mixture start 突破 HueVar 屏障** | K8 HueVar=0.026(real=0.020), 首次接近真实; marginal_start 有害(div 坍塌) | 多模态起始点打破色调均匀性 |
| 63-68 | **X-CORE: 统一 E_prior 跨 repair+gen** | repair: prior 全有害(gap↑); gen: 24bit HF=259≈real; prior 需求相反 | E_prior 不能统一两模式 |
| 69 | **Content routing 不改善分类，random 更好** | content +0.2%, random +3.4%; 增益来自正则化非语义路由 | GPT routing 假说被否定 |
| 70 | **Content routing 改善 HF_noise（-49%）** | 989→509, 全局混合帮助纹理质量 | 频率对齐而非语义路由 |
| 71 | **深度 encoder（5.2×参数）不改善分类** | 0.416≈0.414, 834K vs 159K 无差异 | encoder 容量非瓶颈 |
| 72 | **深度 encoder 损害生成多样性（-18.6%）** | div 0.457→0.372, 过拟合导致 z 坍塌 | 小数据过参数化有害 |
| 73 | **分类瓶颈在训练协议而非架构** | 5 架构变体 acc 41-45%(spread 3.6%), 远低于 TinyCNN 61% | reconstruction-only z 的固有限制 |

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

> **进度更新统一在 PROGRESS.md**，CLAUDE.md 仅保留范式定义、结论表和接口规范。
> 详细实验数据表已迁移至 PROGRESS.md。

结论汇总见上方 #1-#73。

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

## 已完成阶段（全部 ✅）

Phase 0-6（频域路线图）、G1-G4（生成）、F0（Flow）、C1-C2（分类）、E2a-E2b（先验）、X-CORE、R0 均已完成。
详细路线图和结果见 PROGRESS.md。

### 当前状态
- 73 条范式结论已固化
- R0 实验否定了 "content routing" 和 "encoder capacity" 假说
- 分类瓶颈确认在训练协议（reconstruction-only），非架构

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
│   ├── exp_phase12_cycle_contract.py  # Phase 12: Cycle Contract（已完成 ✅）
│   ├── exp_gen_unconditional.py       # 无条件生成（已完成）
│   ├── exp_gen_freq_*.py              # 频率证据生成 A1/A2（已完成）
│   ├── exp_gen_cifar10_a2.py          # CIFAR-10 A2 生成（已完成）
│   ├── exp_gen_cifar10_bw_32x32x16.py # 32×32×16 带宽测试
│   ├── exp_gen_cifar10_int4_v2.py     # INT4 token 4-denoiser 对比
│   ├── exp_cifar10_classify.py        # CIFAR-10 分类 Probe v1（已完成）
│   ├── exp_cifar10_classify_v2.py     # CIFAR-10 分类 Probe v2 语义稳定性（已完成 ✅）
│   ├── exp_cifar10_classify_v3.py     # CIFAR-10 分类 v3 双总线+对比（部分完成）
│   ├── exp_cifar10_bandwidth.py       # C1: 带宽扫描 4 configs（已完成 ✅）
│   ├── exp_cifar10_staged_encoder.py  # C2: Staged ResBlock + VICReg（已完成 ✅）
│   ├── exp_gen_cifar10_g1g2.py       # G1+G2: 带宽升级 + 频率带调度采样（已完成 ✅）
│   ├── exp_gen_cifar10_g3.py        # G3: 16×16×16 stride-2 + regression denoiser（已完成 ✅）
│   ├── exp_gen_cifar10_g4_energy_unet.py # G4: U-Net + E_core + MaskGIT（部分完成）
│   ├── exp_flow_f0.py               # F0: 统一下降算子 flow（已完成 ✅）
│   ├── exp_flow_f0b_tsweep.py       # F0b: T-sweep 收敛步数（已完成 ✅）
│   ├── exp_flow_f0c_fixes.py       # F0c: 发散修复+σ schedule（已完成 ✅）
│   ├── exp_c1_operator_modes.py    # C1: 统一算子三模式兼容性（已完成 ✅）
│   ├── exp_g2_protocol_density.py  # G2-lite: 协议密度/分层（已完成 ✅）
│   ├── exp_g1_dt_schedule.py      # G1-lite: dt/T/schedule sweep（已完成 ✅）
│   ├── exp_e2b_combined.py        # E2b: 24bit + spatial_cov combined（运行中）
│   ├── exp_e2b_factorized_prior.py # E2b-light: factorized prior start（待跑）
│   ├── exp_e2a_global_prior.py    # E2a: 全局先验能量（已完成 ✅）
│   └── exp_r0_routing_encoder.py  # R0: Routing vs Encoder 因果分离（已完成 ✅）
├── PARADIGM_REPORT.md          # 范式研究报告（文献+benchmark+实验矩阵）
├── DESIGN_DOC.md               # 设计文档 v2.1
└── CLAUDE.md                   # 本文件
```
