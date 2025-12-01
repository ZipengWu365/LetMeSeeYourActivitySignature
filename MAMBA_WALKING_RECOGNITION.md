# Mamba-Based Walking State Recognition
## 从HMM到Mamba: 现代状态空间模型的时序平滑方案

---

## 1. 背景与动机

### 1.1 当前方案性能分析 (Benchmark Results)

根据`Benchmark.ipynb`的实验结果，HMM对walking状态识别有**显著提升**:

| 方法 | Macro F1 | 提升幅度 | 说明 |
|------|---------|---------|------|
| **Random Forest** | 0.706 | baseline | 手工特征 + RF |
| **XGBoost** | 0.694 | -0.012 | 手工特征 + XGB |
| **RF + HMM** | **0.812** | **+0.106** | ⭐ **15% 相对提升** |
| **XGB + HMM** | **0.805** | **+0.111** | ⭐ **16% 相对提升** |

**关键发现**:
- HMM 平滑使 F1 从 ~0.70 提升到 ~0.81
- 这是一个**巨大的提升** (绝对提升 +10.6%)
- HMM的作用在于**时序平滑**和**状态连续性约束**

### 1.2 HMM 的工作原理 (来自`hmm.py`分析)

```python
class HMM:
    # 三个核心概率矩阵:
    # 1. startprob: π - 初始状态分布 (4维向量, 对应: sleep, sedentary, light, MVPA)
    # 2. transmat: A - 状态转移矩阵 (4×4)
    # 3. emissionprob: B - 发射概率矩阵 (4×4) - 从预测到真实标签的条件概率
    
    def fit(self, Y_pred, Y_true, groups):
        # 从数据中估计:
        # - transmat: 统计相邻时间窗的状态转移频率
        # - emissionprob: 统计 "预测标签→真实标签" 的混淆矩阵
        # - startprob: 均匀分布 (默认)
    
    def predict(self, Y, groups):
        # Viterbi 解码: 找到最优状态序列
        # argmax P(states | observations) 
        # = argmax P(obs | states) × P(states)
```

**HMM 优势**:
1. ✅ **物理可解释性**: 转移矩阵反映活动切换规律 (如 sleep → sedentary 概率高)
2. ✅ **全局优化**: Viterbi算法考虑整个序列,而非逐点决策
3. ✅ **概率建模**: 输出后验概率,可用于不确定性估计
4. ✅ **轻量高效**: 参数量少 (4×4转移矩阵),推理极快

**HMM 局限性**:
1. ❌ **马尔科夫假设**: 当前状态只依赖前一状态 (一阶马尔科夫)
2. ❌ **离散状态**: 无法捕捉状态内的连续变化 (如步态加速过程)
3. ❌ **固定转移矩阵**: 不随输入变化 (如不同个体的活动模式)
4. ❌ **无法学习复杂模式**: 手工设定状态数,无法自适应

---

## 2. Mamba: 现代状态空间模型

### 2.1 什么是 Mamba?

**Mamba** (Gu & Dao, 2023) 是一种**选择性状态空间模型 (Selective SSM)**，核心特点:

```
传统 SSM (Linear State-Space Model):
    x(t) = A·x(t-1) + B·u(t)    # 状态更新
    y(t) = C·x(t)               # 观测输出
    
Mamba (Selective SSM):
    Δ, B, C = MLP(input)        # 🔥 参数依赖输入 (selectivity)
    x(t) = 离散化(A, B, Δ)·x(t-1) + …
    y(t) = C·x(t)
```

**关键创新**:
1. **Selectivity (选择性)**: 模型参数Δ、B、C根据输入动态调整
   - Δ控制"记忆衰减速度"(类似forget gate)
   - B、C控制"输入/输出门控"
   
2. **Hardware-aware算法**: 通过并行扫描算法,在GPU上高效实现
   - 训练速度接近Transformer
   - 推理速度远超Transformer (线性复杂度 vs 二次复杂度)

3. **长程依赖**: 可捕捉远距离时序关系 (不受马尔科夫假设限制)

### 2.2 Mamba vs HMM 核心对比

| 维度 | HMM | Mamba | 对walking识别的影响 |
|------|-----|-------|-------------------|
| **状态表示** | 离散(4个状态) | 连续(隐藏维度d) | Mamba可捕捉状态内部变化 |
| **记忆长度** | 1步(马尔科夫) | 长程(可达数百步) | Mamba可利用更长历史 |
| **参数化** | 固定转移矩阵 | 输入自适应 | Mamba可对不同个体调整 |
| **学习方式** | 统计估计 | 端到端训练 | Mamba可从数据学习复杂模式 |
| **可解释性** | ⭐⭐⭐⭐⭐ | ⭐⭐ | HMM更直观 |
| **计算复杂度** | O(K²T) | O(dT) | K=状态数,d=隐藏维度,T=序列长 |

---

## 3. 三种架构方案对比

### 方案 A: 端到端 Mamba (替换 HMM)

```
┌──────────────────────────────────────────────────────────────────┐
│                  End-to-End Mamba Pipeline                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Raw Time Series X:(N, 1000, 3)                                  │
│           │                                                       │
│           ▼                                                       │
│  ┌──────────────────┐                                            │
│  │  Preprocessing   │  (可选: ENMO / Multi-Scale / Raw)          │
│  │  X → X':(N,C,T)  │                                            │
│  └────────┬─────────┘                                            │
│           │                                                       │
│           ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Mamba Encoder (Stack of Mamba Blocks)            │   │
│  │  ┌──────────────────────────────────────────────────┐    │   │
│  │  │ MambaBlock 1: (C,T) → (d,T)                      │    │   │
│  │  │  - Selective SSM with Δ, B, C from input         │    │   │
│  │  │  - LayerNorm + Residual                          │    │   │
│  │  └──────────────────────────────────────────────────┘    │   │
│  │  ┌──────────────────────────────────────────────────┐    │   │
│  │  │ MambaBlock 2-6: (d,T) → (d,T)                    │    │   │
│  │  │  - Deep temporal feature extraction              │    │   │
│  │  └──────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────┬───────────────────────────┘   │
│                                  │                                │
│                                  ▼                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Classification Head                              │   │
│  │  Linear(d → num_classes) + Softmax                       │   │
│  └──────────────────────────────┬───────────────────────────┘   │
│                                  │                                │
│                                  ▼                                │
│  y_pred: (N,) - 每个10s窗口的预测标签                            │
└──────────────────────────────────────────────────────────────────┘
```

**优点**:
- ✅ 端到端可训练,特征+分类器联合优化
- ✅ 自动学习时序依赖,无需手工设计HMM参数
- ✅ 可处理长程依赖 (如"睡眠后更可能sedentary")

**缺点**:
- ❌ 完全替换现有方案,风险大
- ❌ 需要大量标注数据 (深度模型)
- ❌ 可解释性差
- ❌ 计算成本高

---

### 方案 B: Mamba 作为 HMM 的后处理 (混合方案) ⭐ **推荐**

```
┌──────────────────────────────────────────────────────────────────┐
│              Hybrid: HandCrafted Features + Mamba Smoother        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Step 1: Window-level Classification (保持现有方案)               │
│  ┌──────────────────┐      ┌───────────────┐                    │
│  │ X_feats:(N, 136) │  ──► │  RF / XGBoost │ ──► Y_pred:(N, 4)  │
│  │  (手工特征)      │      │  (现有模型)   │      (概率分布)    │
│  └──────────────────┘      └───────────────┘                    │
│                                     │                             │
│                                     ▼                             │
│  Step 2: Sequence-level Smoothing (Mamba替换HMM)                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Mamba Smoother (轻量版)                      │   │
│  │  Input: Y_pred:(N,4) + Optional Raw:(N,C,T)              │   │
│  │                                                           │   │
│  │  ┌───────────────────────────────────────────────────┐   │   │
│  │  │  1. Feature Fusion Layer                          │   │   │
│  │  │     - Concat [Y_pred, Auxiliary_features]         │   │   │
│  │  │     - Auxiliary可选: ENMO均值/步频/姿态角度       │   │   │
│  │  │     → (N, 4+k)                                    │   │   │
│  │  └────────────────────┬──────────────────────────────┘   │   │
│  │                       ▼                                  │   │
│  │  ┌───────────────────────────────────────────────────┐   │   │
│  │  │  2. Mamba Temporal Encoder (2-3 layers)          │   │   │
│  │  │     - MambaBlock: (4+k, N) → (d, N)              │   │   │
│  │  │     - d=64~128 (较小隐藏维度,降低计算)            │   │   │
│  │  │     - 学习:                                       │   │   │
│  │  │       • 状态切换的平滑性 (如walking持续时间)      │   │   │
│  │  │       • 个体特异性模式 (不同人的活动节律)         │   │   │
│  │  │       • 长程依赖 (如早晨更可能light活动)          │   │   │
│  │  └────────────────────┬──────────────────────────────┘   │   │
│  │                       ▼                                  │   │
│  │  ┌───────────────────────────────────────────────────┐   │   │
│  │  │  3. Classification Head                           │   │   │
│  │  │     Linear(d → 4) + Softmax                       │   │   │
│  │  │     → Y_smoothed:(N, 4)                           │   │   │
│  │  └───────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                     │                             │
│                                     ▼                             │
│  Final: Y_final = argmax(Y_smoothed, axis=1)                     │
└──────────────────────────────────────────────────────────────────┘
```

**训练策略**:
```python
# 两阶段训练:
# Stage 1: 训练 RF/XGB (已有)
rf_model.fit(X_feats_train, y_train)
y_pred_train_proba = rf_model.predict_proba(X_feats_train)

# Stage 2: 训练 Mamba Smoother
mamba_smoother = MambaSmoother(d_model=64, n_layers=2)
# 损失函数: CrossEntropy + Smoothness Regularization
loss = CE_loss(y_pred_smooth, y_true) + λ * temporal_variation_penalty(y_pred_smooth)

mamba_smoother.fit(
    y_pred_train_proba,  # (N, 4) 概率输入
    y_train,             # (N,) 真实标签
    groups_train         # 按participant分组
)
```

**优点**:
- ✅ **渐进式迁移**: 可复用现有RF/XGB模型
- ✅ **数据高效**: 只需训练轻量Mamba (参数少)
- ✅ **可对比验证**: 直接与HMM对比 (公平竞争)
- ✅ **灵活度高**: 可加入辅助特征 (如原始信号统计)

**缺点**:
- ⚠️ 两阶段训练略复杂
- ⚠️ 需要调参 (Mamba层数、隐藏维度、正则化系数)

---

### 方案 C: Mamba + Attention Hybrid (召回导向)

```
┌──────────────────────────────────────────────────────────────────┐
│          Mamba-Attention Hybrid (for High-Recall Walking)        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  目标: 极致提升 Walking 召回率 (避免漏检)                         │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Multi-Head Architecture                                 │   │
│  │                                                           │   │
│  │  Raw X:(N,C,T)                                           │   │
│  │       │                                                   │   │
│  │       ├────────────┬──────────────┐                      │   │
│  │       │            │               │                      │   │
│  │       ▼            ▼               ▼                      │   │
│  │  ┌────────┐  ┌────────┐      ┌────────┐                 │   │
│  │  │ Mamba  │  │ Attn   │      │  RF    │                 │   │
│  │  │ Stream │  │ Stream │      │ Stream │                 │   │
│  │  │  (局部) │  │ (全局) │      │ (统计) │                 │   │
│  │  └───┬────┘  └───┬────┘      └───┬────┘                 │   │
│  │      │           │                │                      │   │
│  │      └───────────┴────────────────┘                      │   │
│  │                  │                                        │   │
│  │                  ▼                                        │   │
│  │         Fusion Layer (Gating/Concat)                     │   │
│  │                  │                                        │   │
│  │                  ▼                                        │   │
│  │         Final Classifier (4-class)                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

**适用场景**: 如果 Walking 是关键类别,需要高召回 (如步态生物标志物提取)

**优点**:
- ✅ Mamba捕捉局部节律 (步态周期~2Hz)
- ✅ Attention捕捉全局上下文 (活动转换)
- ✅ RF提供可解释基线

**缺点**:
- ❌ 模型复杂度最高
- ❌ 训练数据需求大

---

## 4. 实验设计: Mamba vs HMM

### 4.1 研究问题

**RQ1**: Mamba能否超越HMM的时序平滑效果?  
**RQ2**: Mamba的性能提升是否值得额外的计算成本?  
**RQ3**: Mamba在不同预处理方法(ENMO vs Multi-Scale)下的表现差异?

### 4.2 对比实验设置

| 实验组 | Window Classifier | Smoother | 预处理方法 | 备注 |
|--------|------------------|----------|-----------|------|
| **E1** | RF (手工特征) | None | ENMO | Baseline 1 |
| **E2** | RF (手工特征) | HMM | ENMO | Baseline 2 (最强HMM) |
| **E3** | RF (手工特征) | **Mamba-Light** | ENMO | 方案B变体1 |
| **E4** | RF (手工特征) | **Mamba-Medium** | ENMO | 方案B变体2 |
| **E5** | Hydra (aeon) | None | Multi-Scale | 新预处理baseline |
| **E6** | Hydra (aeon) | **Mamba-Light** | Multi-Scale | 混合方案 |
| **E7** | **End2End Mamba** | - | Multi-Scale | 方案A (纯Mamba) |

**Mamba配置**:
- **Mamba-Light**: d_model=64, n_layers=2, params~50K
- **Mamba-Medium**: d_model=128, n_layers=3, params~200K

### 4.3 评估指标

```python
metrics = {
    # 核心指标 (与HMM对比)
    'macro_f1': ...,
    'walking_f1': ...,  # 单独报告walking类的F1
    'walking_recall': ...,  # 步态生物标志物关键
    
    # 时序平滑效果
    'avg_segment_length': ...,  # 平均连续段长度 (越长越平滑)
    'num_transitions': ...,  # 状态切换次数 (越少越平滑)
    'temporal_consistency': ...,  # 自定义: P(y_t == y_{t-1})
    
    # 效率指标
    'train_time': ...,
    'inference_time_per_sample': ...,
    'memory_footprint': ...,
}
```

### 4.4 消融实验

**Ablation 1: Mamba的选择性机制是否必要?**
- 对比: Mamba (Selective SSM) vs Linear SSM (不依赖输入的固定参数)
- 假设: 选择性机制可自适应调整记忆,优于固定SSM

**Ablation 2: 辅助特征的贡献**
```python
# 输入变体:
# Variant A: 仅 Y_pred (概率分布)
# Variant B: Y_pred + ENMO统计量 (mean, std)
# Variant C: Y_pred + 原始信号embedding
```

**Ablation 3: 正则化的影响**
```python
loss = CE_loss + λ1 * L_smooth + λ2 * L_consistent
# L_smooth: sum((y_t - y_{t-1})^2)  # 惩罚突变
# L_consistent: KL(y_pred_smooth || y_pred_raw)  # 保持与RF预测接近
```

---

## 5. Mamba 版本选择与性能对比 🚀

### 5.1 可用Mamba版本对比 (2024最新)

| 版本 | 训练速度 | 参数规模 | GPU需求 | 推荐场景 |
|------|---------|---------|---------|----------|
| **Mamba-2** (SSD) | ⭐⭐⭐⭐⭐ +50% | 可变 | 中-高 | **推荐** - SOTA性能 |
| **pytorch-mamba** | ⭐⭐⭐⭐ | 可变 | 中 | **推荐** - 平衡性能与易用 |
| mamba-minimal | ⭐⭐ | 小 | 低 | 教育/理解用途 |
| MiniMamba | ⭐⭐⭐⭐ +3x | 小-中 | 低-中 | 轻量快速原型 |
| Official Mamba | ⭐⭐⭐⭐ | 可变 | 中-高 | 完整功能 |

**⭐ 最终选择: `pytorch-mamba` (带Mamba-2优化)**

**理由**:
1. **训练速度**: 集成了Mamba-2的SSD (State Space Dual) 框架,比原版快50%
2. **易用性**: 纯PyTorch实现,无需自定义CUDA算子
3. **轻量化**: 支持小模型 (d_model=64-128),适合我们的任务
4. **GPU友好**: 优化的并行扫描算法,在单GPU上即可高效训练

### 5.2 Mamba-2 关键优化

```python
# Mamba-2 的核心改进:
# 1. State Space Dual (SSD) - 连接SSM和Attention
# 2. Structured Masked Attention (SMA) - 更大状态空间
# 3. Tensor Core优化 - 硬件加速

# 状态空间大小对比:
Mamba-1: N = 16  (受限)
Mamba-2: N = 64-256  (大幅提升,无额外成本)

# 训练速度提升:
# - 50% faster than Mamba-1
# - 2-5x faster than Transformer (长序列)
```

### 5.3 针对Walking识别的优化配置

```python
# 我们的Light配置 (优化速度)
Mamba_Light_Config = {
    'd_model': 64,        # 隐藏维度 (vs Hydra的9k特征)
    'd_state': 16,        # SSM状态维度 (Mamba-2可用64)
    'd_conv': 4,          # 卷积核大小
    'n_layers': 2,        # 层数 (足够捕捉10s窗口)
    'expand': 2,          # FFN扩展因子
    'use_mamba2': True,   # ⚠️ 启用Mamba-2优化
}

# 预期性能:
# - 训练时间: ~10-20 min (vs HMM的5 min)
# - 推理速度: ~1-2ms/sample (vs HMM的0.5ms)
# - 内存占用: ~500MB (单GPU)
```

---

## 6. RF+HMM 深度剖析: 为什么它如此有效? 🔬

### 6.1 HMM的三个成功要素

#### 要素1: 转移矩阵捕捉活动规律

```python
# 从数据中学到的转移矩阵 (简化示例):
transmat = [
#          sleep  sedentary  light  MVPA
  [sleep] [ 0.95,    0.04,    0.01,  0.00 ],  # 睡眠极稳定
  [sed. ] [ 0.01,    0.85,    0.12,  0.02 ],  # 久坐易→light
  [light] [ 0.00,    0.20,    0.70,  0.10 ],  # light较动态
  [MVPA ] [ 0.00,    0.05,    0.25,  0.70 ],  # MVPA短暂但稳定
]

# 关键洞察:
# 1. 主对角线值高 → 状态持久性 (减少抖动)
# 2. sleep→MVPA ≈ 0 → 物理约束 (不可能直接转换)
# 3. MVPA→sedentary低 → 运动后更可能light活动
```

**量化分析**:从`hmm.py`的`compute_transition`函数:
```python
# 统计相邻窗口的转移频率
transition[i,j] = count(state_t=i → state_{t+1}=j) / count(state_t=i)

# 这捕捉了:
# - 时序依赖: P(y_t | y_{t-1})
# - 物理约束: 某些转移概率为0
# - 个体差异: 通过群组分别统计
```

#### 要素2: 发射概率校准RF的误差

```python
# 发射矩阵 B: P(RF预测=j | 真实状态=i)
emissionprob = [
#           RF_pred: sleep  sed.  light  MVPA
  [True:sleep]     [ 0.92,  0.06,  0.02,  0.00 ],  # RF对sleep很准
  [True:sed. ]     [ 0.05,  0.75,  0.18,  0.02 ],  # sed.易混淆light
  [True:light]     [ 0.01,  0.25,  0.65,  0.09 ],  # light最难
  [True:MVPA ]     [ 0.00,  0.03,  0.27,  0.70 ],  # MVPA中等准
]

# 关键作用:
# 1. 校准RF的系统性偏差 (如过度预测sedentary)
# 2. 提供"不确定性估计" (混淆矩阵的概率版)
```

**核心优势**: 当RF犯错时,HMM通过历史状态进行"事后修正"。

例子:
```
时刻:      t=0    t=1    t=2    t=3    t=4
RF预测:   light  sed.   light  light  light
HMM平滑:  light  light  light  light  light
         ↑
      修正了t=1的噪声预测 (因为light→sed.→light不太可能)
```

#### 要素3: Viterbi全局优化

```python
# Viterbi vs 逐点决策:
# 逐点: argmax P(y_t | RF_t) for each t  ← RF原始做法
# Viterbi: argmax P(y_1...y_T | RF_1...RF_T)  ← HMM做法
#         = argmax ∏_{t=1}^T P(RF_t|y_t) × P(y_t|y_{t-1})

# 效果:
# - 全局最优路径 (动态规划)
# - 考虑整个序列,而非局部贪心
```

### 6.2 HMM的局限性: Mamba的突破口

| HMM局限 | 具体问题 | Mamba优势 |
|---------|---------|----------|
| **1. 马尔科夫假设** | 只看t-1时刻,忽略更长历史 | 可回溯数百步 (线性复杂度) |
| **2. 离散状态** | 无法表达"加速中""减速中"等过渡状态 | 连续隐藏状态 |
| **3. 固定转移矩阵** | 所有人共享同一转移矩阵 | **选择性SSM**: 参数依赖输入 |
| **4. 无法利用原始信号** | 只看RF的概率输出,丢失细节 | 可融合辅助特征 (如ENMO统计) |
| **5. 发射概率的粗糙建模** | 仅用训练集平均混淆矩阵 | 学习复杂的发射分布 |

**最关键突破点: 个性化转移概率**
```python
# HMM: 所有participant共享转移矩阵A
# → 忽略个体差异 (如老年人 sleep→sedentary 概率更高)

# Mamba: 转移"矩阵"依赖输入
Δ, B, C = MLP(RF_proba, participant_embedding)
# → 自适应调整记忆衰减速度Δ (类似动态转移矩阵)
```

### 6.3 针对性改进策略

**策略1: 模拟HMM的转移约束**
```python
# 在Mamba loss中添加转移平滑正则:
loss_transition = λ_trans * sum(
    CE(transition_probs[t], transition_probs[t-1])  
    for t in 1..T
)
# 目标: 学习到类似HMM转移矩阵的平滑性
```

**策略2: 利用Mamba的长程依赖**
```python
# HMM只看1步,Mamba可回溯整个序列
# 场景: 检测"虚假MVPA"
# - 如果前30秒都是sedentary,突然1个MVPA窗口 → 可能是噪声
# - HMM只看前1个sedentary,Mamba看前30个 → 更robust
```

**策略3: 融合原始信号特征**
```python
# Auxiliary features增强:
aux_features = [
    ENMO_mean,      # 运动强度
    ENMO_std,       # 运动变异性  
    dominant_freq,  # 主频 (步态~2Hz)
    postural_angle, # 姿态角度 (区分站立/坐)
]
# HMM无法用到这些,Mamba可以!
```

---

## 7. 详细实验设计 📊

### 7.1 实验目标与假设

**主目标**: Mamba Smoother 的 Macro F1 **≥ 0.820** (vs HMM 0.812)

**假设验证**:
- **H1**: Mamba的长程依赖可减少"孤立噪声窗口" → Recall提升
- **H2**: Mamba的选择性机制可适应个体差异 → 泛化性提升  
- **H3**: 融合辅助特征可弥补RF特征的不足 → Precision提升

### 7.2 数据划分策略

```python
# 遵循Benchmark的划分 (保持公平对比)
train_participants = 101  # 前101人
test_participants = 50    # 后50人

# 关键: 按participant分组,避免数据泄露
for train_idx, test_idx in GroupShuffleSplit(n_splits=1, test_size=0.2):
    # 训练集再划分:
    X_train_rf, X_val = X[train_idx[:80%]], X[train_idx[80%:]]
    
    # X_train_rf: 训练RF/XGB (已完成)
    # X_val: 训练Mamba smoother (从RF概率输出)
    # X_test: 最终评估 (公平对比)
```

### 7.3 基线实验组

| ID | Window Clf | Smoother | 辅助特征 | 预期F1 | 训练时间 | 用途 |
|----|-----------|----------|---------|-------|---------|------|
| **E0** | RF | None | - | 0.706 | - | Baseline (已有) |
| **E1** | RF | HMM | - | **0.812** | ~5 min | ⭐ **竞争目标** |
| **E2** | XGB | HMM | - | 0.805 | ~8 min | 次要对比 |

### 7.4 Mamba实验组 (核心)

| ID | Window Clf | Smoother | d_model | n_layers | 辅助特征 | 正则化 | 预期F1 | 训练时间 |
|----|-----------|----------|---------|----------|---------|--------|-------|----------|
| **M1** | RF | Mamba-Light | 64 | 2 | ❌ | 标准 | 0.815 | 12 min |
| **M2** | RF | Mamba-Light | 64 | 2 | ✅ ENMO | 标准 | **0.822** | 15 min |
| **M3** | RF | Mamba-Medium | 128 | 3 | ✅ ENMO | 标准 | **0.828** | 25 min |
| **M4** | RF | Mamba-Medium | 128 | 3 | ✅ Full | 强平滑 | **0.825** | 30 min |
| **M5** | XGB | Mamba-Light | 64 | 2 | ✅ ENMO | 标准 | 0.818 | 15 min |

**辅助特征详情**:
- **ENMO**: `[mean, std, max]` 3维
- **Full**: `[ENMO, dominant_freq, postural_angle, jerk]` 6维

**正则化配置**:
```python
# 标准:
lambda_smooth = 0.01  # 时序平滑
lambda_consistent = 0.1  # 与RF一致性

# 强平滑:
lambda_smooth = 0.05  # 增强平滑 (可能降低对快速变化的响应)
lambda_consistent = 0.05
```

### 7.5 消融实验

**Ablation 1: 选择性机制的贡献**
| ID | SSM类型 | Selectivity | 预期F1 | 说明 |
|----|---------|-------------|--------|------|
| A1 | Mamba (Selective) | ✅ | 0.822 | 完整Mamba |
| A2 | S4 (Non-selective) | ❌ | 0.810 | 固定参数SSM |

**Ablation 2: 辅助特征的贡献**
| ID | 辅助特征 | Δ F1 vs M1 | 说明 |
|----|---------|-----------|------|
| M1 | None | baseline | 仅RF概率 |
| M2 | +ENMO | +0.007 | 运动强度信息 |
| M4 | +Full | +0.010 | 完整物理特征 |

**Ablation 3: 模型深度的影响**
| n_layers | d_model | 参数量 | 预期F1 | 训练时间 |
|----------|---------|--------|-------|----------|
| 1 | 64 | 25K | 0.810 | 8 min |
| 2 | 64 | 50K | 0.815 | 12 min |
| 3 | 64 | 75K | 0.817 | 18 min |
| 2 | 128 | 200K | 0.822 | 15 min |
| 3 | 128 | 300K | **0.828** | 25 min |

### 7.6 评估指标 (完整)

```python
metrics = {
    # === 核心指标 (主要优化目标) ===
    'macro_f1': ...,           # ⭐ 主要指标 (必须 ≥ 0.820)
    'macro_f1_ci': ...,        # 95% 置信区间 (bootstrap)
    
    # === 各类别F1 (诊断用) ===
    'f1_sleep': ...,
    'f1_sedentary': ...,
    'f1_light': ...,
    'f1_mvpa': ...,
    
    # === Walking相关 (如果做二分类) ===
    'walking_f1': ...,         # light+MVPA vs others
    'walking_recall': ...,     # 步态生物标志物关键
    'walking_precision': ...,
    
    # === 时序平滑质量 ===
    'avg_segment_length': ..., # 平均连续段长度 (秒)
    'transition_rate': ...,    # 每分钟状态切换次数
    'smoothness_score': 1 - transition_rate / theoretical_max,
    
    # === 计算效率 ===
    'train_time_sec': ...,
    'inference_time_per_sample_ms': ...,
    'memory_peak_mb': ...,
    
    # === 鲁棒性分析 ===
    'per_participant_f1_std': ...,  # F1的个体间标准差
    'worst_case_f1': ...,           # 最差participant的F1
}
```

### 7.7 统计显著性检验

```python
# McNemar's Test: 成对比较HMM vs Mamba
from statsmodels.stats.contingency_tables import mcnemar

# 构建混淆矩阵:
# |         | Mamba正确 | Mamba错误 |
# | HMM正确 |     a     |     b     |
# | HMM错误 |     c     |     d     |

table = [[a, b], [c, d]]
result = mcnemar(table)

if result.pvalue < 0.05:
    print("Mamba显著优于HMM (p < 0.05)")
```

### 7.8 失败案例分析

```python
# 收集Mamba预测错误但HMM正确的样本:
error_samples = [
    (idx, y_true[idx], y_pred_hmm[idx], y_pred_mamba[idx])
    for idx in range(len(y_true))
    if y_pred_hmm[idx] == y_true[idx] and y_pred_mamba[idx] != y_true[idx]
]

# 分析错误模式:
# 1. 是否集中在某些participant? → 个体差异问题
# 2. 是否集中在某些活动类别? → 特定类别建模不足
# 3. 是否集中在边界窗口? → 过度平滑问题
```

---

## 8. 任务列表 (可执行) ✅

### Phase 0: 环境准备 (1小时)

- [ ] **Task 0.1**: 安装依赖
  ```bash
  pip install torch>=2.0 numpy pandas scikit-learn
  pip install mamba-ssm  # 官方Mamba (需CUDA 11.8+)
  # 或使用纯PyTorch版本:
  pip install causal-conv1d>=1.1.0  # Mamba依赖
  ```

- [ ] **Task 0.2**: 验证Mamba安装
  ```python
  # test_mamba_install.py
  from mamba_ssm import Mamba
  import torch
  
  model = Mamba(d_model=64, d_state=16)
  x = torch.randn(1, 100, 64)  # (B, L, D)
  y = model(x)
  print(f"Mamba output shape: {y.shape}")  # 应该是 (1, 100, 64)
  ```

- [ ] **Task 0.3**: 加载现有数据
  ```python
  # 确保以下文件存在:
  # - prepared_data/X_feats.pkl  (手工特征)
  # - prepared_data/Y_Walmsley2020.npy  (标签)
  # - prepared_data/P.npy  (participant IDs)
  ```

### Phase 1: 复现RF+HMM基线 (2小时)

- [ ] **Task 1.1**: 训练RF模型
  ```bash
  cd experiments/gait_filter
  python -c "
  from classifier import Classifier
  import numpy as np
  import pandas as pd
  
  X_feats = pd.read_pickle('../../prepared_data/X_feats.pkl').values
  Y = np.load('../../prepared_data/Y_Walmsley2020.npy')
  P = np.load('../../prepared_data/P.npy')
  
  # 前101人训练
  train_mask = P < 'P102'  # 字符串比较
  X_train, y_train, P_train = X_feats[train_mask], Y[train_mask], P[train_mask]
  
  rf_model = Classifier('rf', verbose=1)
  rf_model.fit(X_train, y_train, P_train)
  
  # 保存模型
  import joblib
  joblib.dump(rf_model, 'models/rf_baseline.pkl')
  "
  ```

- [ ] **Task 1.2**: 训练RF+HMM (目标基线)
  ```bash
  python -c "
  from classifier import Classifier
  # ... (同上加载数据)
  
  rf_hmm_model = Classifier('rf_hmm', verbose=1)
  rf_hmm_model.fit(X_train, y_train, P_train)
  
  # 在测试集评估
  test_mask = P >= 'P102'
  X_test, y_test, P_test = X_feats[test_mask], Y[test_mask], P[test_mask]
  
  y_pred_rf_hmm = rf_hmm_model.predict(X_test, P_test)
  
  from sklearn.metrics import f1_score
  f1_macro = f1_score(y_test, y_pred_rf_hmm, average='macro')
  print(f'RF+HMM F1 Macro: {f1_macro:.4f}')  # 期望: ~0.812
  
  joblib.dump(rf_hmm_model, 'models/rf_hmm_baseline.pkl')
  "
  ```

- [ ] **Task 1.3**: 生成RF概率输出 (Mamba训练用)
  ```bash
  python -c "
  import joblib
  import numpy as np
  
  rf_model = joblib.load('models/rf_baseline.pkl')
  
  # 生成训练集概率 (用于训练Mamba)
  y_train_proba = rf_model.window_classifier.predict_proba(X_train)
  np.save('prepared_data/y_train_proba_rf.npy', y_train_proba)
  
  # 生成测试集概率
  y_test_proba = rf_model.window_classifier.predict_proba(X_test)
  np.save('prepared_data/y_test_proba_rf.npy', y_test_proba)
  "
  ```

### Phase 2: 实现Mamba Smoother (1天)

- [ ] **Task 2.1**: 创建`mamba_smoother.py` (核心模块)
  - 复制第5节的`MambaSmoother`类代码
  - 复制`MambaSmootherTrainer`类代码
  - 添加辅助特征计算函数

- [ ] **Task 2.2**: 实现辅助特征提取
  ```python
  # auxiliary_features.py
  def compute_auxiliary_features(X_raw, feature_type='enmo'):
      """
      从原始信号提取辅助特征
      Args:
          X_raw: (N, 1000, 3) 原始加速度
          feature_type: 'none', 'enmo', 'full'
      Returns:
          aux_feats: (N, k) 辅助特征
      """
      if feature_type == 'none':
          return None
      
      import numpy as np
      
      # ENMO统计
      enmo = np.linalg.norm(X_raw, axis=2) - 1.0  # (N, 1000)
      enmo_mean = enmo.mean(axis=1)  # (N,)
      enmo_std = enmo.std(axis=1)
      enmo_max = enmo.max(axis=1)
      
      if feature_type == 'enmo':
          return np.column_stack([enmo_mean, enmo_std, enmo_max])  # (N, 3)
      
      elif feature_type == 'full':
          # 主频
          from scipy.fft import rfft, rfftfreq
          fft_vals = np.abs(rfft(enmo, axis=1))
          freqs = rfftfreq(1000, 1/100)  # 100Hz采样率
          dominant_freq = freqs[fft_vals.argmax(axis=1)]  # (N,)
          
          # 姿态角度 (粗略估计)
          gravity_vec = X_raw.mean(axis=1)  # (N, 3)
          postural_angle = np.arctan2(
              np.linalg.norm(gravity_vec[:, :2], axis=1),
              gravity_vec[:, 2]
          )  # (N,) 弧度
          
          # Jerk
          jerk = np.linalg.norm(np.diff(X_raw, axis=1), axis=2).mean(axis=1)  # (N,)
          
          return np.column_stack([
              enmo_mean, enmo_std, enmo_max,
              dominant_freq, postural_angle, jerk
          ])  # (N, 6)
  ```

- [ ] **Task 2.3**: 创建训练脚本`train_mamba_smoother.py`
  ```python
  # train_mamba_smoother.py
  import argparse
  import numpy as np
  import torch
  from mamba_smoother import MambaSmoother, MambaSmootherTrainer
  from auxiliary_features import compute_auxiliary_features
  
  def main(args):
      # 加载数据
      y_train_proba = np.load('prepared_data/y_train_proba_rf.npy')
      y_train = np.load('prepared_data/Y_Walmsley2020.npy')[train_mask]
      P_train = np.load('prepared_data/P.npy')[train_mask]
      
      # 辅助特征
      if args.aux_features != 'none':
          X_raw_train = np.load('prepared_data/X.npy')[train_mask]
          aux_feats_train = compute_auxiliary_features(X_raw_train, args.aux_features)
          aux_dim = aux_feats_train.shape[1]
      else:
          aux_feats_train = None
          aux_dim = 0
      
      # 初始化模型
      model = MambaSmoother(
          n_classes=4,
          d_model=args.d_model,
          n_layers=args.n_layers,
          use_aux_features=(aux_dim > 0),
          aux_dim=aux_dim,
      )
      
      # 训练
      trainer = MambaSmootherTrainer(
          model,
          lr=args.lr,
          lambda_smooth=args.lambda_smooth,
          lambda_consistent=args.lambda_consistent,
      )
      
      trainer.fit(
          y_train_proba,
          y_train,
          P_train,
          aux_features=aux_feats_train,
          epochs=args.epochs,
          batch_size=args.batch_size,
      )
      
      # 保存模型
      torch.save(model.state_dict(), f'models/mamba_smoother_{args.exp_id}.pt')
  
  if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--exp_id', type=str, required=True)
      parser.add_argument('--d_model', type=int, default=64)
      parser.add_argument('--n_layers', type=int, default=2)
      parser.add_argument('--aux_features', choices=['none', 'enmo', 'full'], default='enmo')
      parser.add_argument('--lambda_smooth', type=float, default=0.01)
      parser.add_argument('--lambda_consistent', type=float, default=0.1)
      parser.add_argument('--epochs', type=int, default=50)
      parser.add_argument('--batch_size', type=int, default=32)
      parser.add_argument('--lr', type=float, default=1e-3)
      args = parser.parse_args()
      main(args)
  ```

### Phase 3: 运行实验 (1天)

- [ ] **Task 3.1**: 训练M1 (Mamba-Light, 无辅助特征)
  ```bash
  python train_mamba_smoother.py \
    --exp_id M1 \
    --d_model 64 \
    --n_layers 2 \
    --aux_features none \
    --epochs 50
  ```

- [ ] **Task 3.2**: 训练M2 (Mamba-Light + ENMO) ⭐ **关键实验**
  ```bash
  python train_mamba_smoother.py \
    --exp_id M2 \
    --d_model 64 \
    --n_layers 2 \
    --aux_features enmo \
    --epochs 50
  ```

- [ ] **Task 3.3**: 训练M3 (Mamba-Medium + ENMO) ⭐ **最优配置**
  ```bash
  python train_mamba_smoother.py \
    --exp_id M3 \
    --d_model 128 \
    --n_layers 3 \
    --aux_features enmo \
    --epochs 50
  ```

- [ ] **Task 3.4**: 训练M4 (Mamba-Medium + Full特征)
  ```bash
  python train_mamba_smoother.py \
    --exp_id M4 \
    --d_model 128 \
    --n_layers 3 \
    --aux_features full \
    --epochs 50
  ```

### Phase 4: 评估与对比 (半天)

- [ ] **Task 4.1**: 创建评估脚本`evaluate_smoothers.py`
  ```python
  # evaluate_smoothers.py
  import argparse
  import numpy as np
  import pandas as pd
  from sklearn.metrics import f1_score, classification_report
  import joblib
  import torch
  from mamba_smoother import MambaSmoother
  
  def evaluate_model(y_true, y_pred, model_name):
      f1_macro = f1_score(y_true, y_pred, average='macro')
      f1_per_class = f1_score(y_true, y_pred, average=None)
      
      print(f"\n{'='*50}")
      print(f"{model_name}")
      print(f"{'='*50}")
      print(f"Macro F1: {f1_macro:.4f}")
      print(f"Per-class F1: {f1_per_class}")
      print(classification_report(y_true, y_pred))
      
      return {
          'model': model_name,
          'macro_f1': f1_macro,
          'f1_sleep': f1_per_class[0],
          'f1_sedentary': f1_per_class[1],
          'f1_light': f1_per_class[2],
          'f1_mvpa': f1_per_class[3],
      }
  
  def main():
      # 加载测试数据
      y_test = np.load('prepared_data/Y_Walmsley2020.npy')[test_mask]
      P_test = np.load('prepared_data/P.npy')[test_mask]
      
      results = []
      
      # 1. RF baseline
      rf_model = joblib.load('models/rf_baseline.pkl')
      y_pred_rf = rf_model.predict(X_test, P_test)
      results.append(evaluate_model(y_test, y_pred_rf, 'RF (baseline)'))
      
      # 2. RF+HMM (目标)
      rf_hmm_model = joblib.load('models/rf_hmm_baseline.pkl')
      y_pred_rf_hmm = rf_hmm_model.predict(X_test, P_test)
      results.append(evaluate_model(y_test, y_pred_rf_hmm, 'RF+HMM ⭐'))
      
      # 3-6. Mamba实验组
      for exp_id in ['M1', 'M2', 'M3', 'M4']:
          # 加载配置和模型
          # ... (根据exp_id加载相应模型)
          mamba_model = load_mamba_model(exp_id)
          y_pred_mamba = mamba_model.predict(y_test_proba, P_test)
          results.append(evaluate_model(y_test, y_pred_mamba, f'RF+Mamba-{exp_id}'))
      
      # 保存结果
      df_results = pd.DataFrame(results)
      df_results.to_csv('results/smoother_comparison.csv', index=False)
      print("\n" + "="*60)
      print("FINAL RESULTS (sorted by Macro F1)")
      print("="*60)
      print(df_results.sort_values('macro_f1', ascending=False))
  ```

- [ ] **Task 4.2**: 运行完整评估
  ```bash
  python evaluate_smoothers.py > results/evaluation_log.txt
  ```

- [ ] **Task 4.3**: 统计显著性检验
  ```python
  # statistical_test.py
  from statsmodels.stats.contingency_tables import mcnemar
  
  # HMM vs Mamba-M3 (最优配置)
  y_pred_hmm = ...
  y_pred_mamba = ...
  y_true = ...
  
  # McNemar表
  both_correct = ((y_pred_hmm == y_true) & (y_pred_mamba == y_true)).sum()
  hmm_only = ((y_pred_hmm == y_true) & (y_pred_mamba != y_true)).sum()
  mamba_only = ((y_pred_hmm != y_true) & (y_pred_mamba == y_true)).sum()
  both_wrong = ((y_pred_hmm != y_true) & (y_pred_mamba != y_true)).sum()
  
  table = [[both_correct, hmm_only], [mamba_only, both_wrong]]
  result = mcnemar(table)
  
  print(f"McNemar p-value: {result.pvalue:.4f}")
  if result.pvalue < 0.05:
      print("✅ Mamba显著优于HMM (p < 0.05)")
  ```

### Phase 5: 分析与优化 (1天)

- [ ] **Task 5.1**: 失败案例分析
  - 找出Mamba预测错误但HMM正确的样本
  - 分析是否有系统性模式 (特定participant/活动/时间段)

- [ ] **Task 5.2**: 超参数调优 (如果M3未达到0.820)
  ```python
  # 使用Optuna进行贝叶斯优化
  import optuna
  
  def objective(trial):
      d_model = trial.suggest_categorical('d_model', [64, 96, 128, 160])
      n_layers = trial.suggest_int('n_layers', 2, 4)
      lambda_smooth = trial.suggest_float('lambda_smooth', 0.001, 0.1, log=True)
      
      # 训练模型并返回验证F1
      ...
      return val_f1
  
  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=30)
  ```

- [ ] **Task 5.3**: 可视化对比
  ```python
  # 绘制时序预测对比图
  import matplotlib.pyplot as plt
  
  # 选择一个participant的序列
  participant_id = 'P120'
  mask = P_test == participant_id
  
  fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
  
  axes[0].plot(y_test[mask], label='Ground Truth', marker='o')
  axes[0].set_title('Ground Truth')
  
  axes[1].plot(y_pred_hmm[mask], label='RF+HMM', marker='s', alpha=0.7)
  axes[1].set_title('RF+HMM Prediction')
  
  axes[2].plot(y_pred_mamba[mask], label='RF+Mamba', marker='^', alpha=0.7)
  axes[2].set_title('RF+Mamba Prediction')
  
  plt.xlabel('Time (10s windows)')
  plt.savefig('results/prediction_comparison.png')
  ```

---

## 8.5. ESN (Echo State Network): 最佳Sweet Spot? 🎯

### 8.5.1 ESN vs HMM vs Mamba 对比

基于最新研究,**ESN可能是硬件受限情况下的最优方案**！

```
性能-效率权衡图:

精度
  ^
  │                              ⭐ Mamba (高精度,高成本)
  │                             ╱
  │                            ╱
0.83│              ◉ ESN (sweet spot!)  
  │             ╱   \
  │            ╱      \
0.81│      ⬤ HMM       \
  │                    \
0.71│  ● RF             \
  │
  └──────────────────────────────────────> 训练时间
     5min   10min        25min
     CPU    CPU/GPU      GPU only
```

| 维度 | HMM | **ESN** ⭐ | Mamba |
|------|-----|---------|-------|
| **训练速度** | 5 min (CPU) | **8-12 min (CPU)** | 15-25 min (GPU) |
| **推理速度** | 0.5 ms/sample | **1 ms/sample** | 2 ms/sample |
| **内存占用** | <100 MB | **~500 MB** | ~1-2 GB |
| **GPU需求** | ❌ | ❌ (可选加速) | ✅ 必需 |
| **预期F1** | 0.812 | **0.818-0.825** | 0.822-0.828 |
| **实现复杂度** | 简单 | **极简** | 中等 |
| **可解释性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

### 8.5.2 ESN的核心优势

**1. Reservoir Computing原理**
```python
# ESN架构:
# 1. Input Layer: (4,) - RF概率输出
# 2. Reservoir (固定随机连接): (N,) - N=500-1000神经元
# 3. Output Layer (唯一训练): (4,) - 线性回归

class ESN:
    def __init__(self, n_reservoir=800, spectral_radius=0.9):
        # 随机初始化reservoir (固定!)
        self.W_in = random_matrix(n_reservoir, 4)   # 输入权重
        self.W_res = random_matrix(n_reservoir, n_reservoir)  # reservoir权重
        
        # 调整spectral radius (控制记忆长度)
        self.W_res *= spectral_radius / max_eigenvalue(W_res)
        
        # 输出权重 (唯一需要训练的!)
        self.W_out = None  # 通过Ridge回归学习
    
    def fit(self, y_pred_proba_seq, y_true_seq):
        # 1. 收集reservoir状态 (无需梯度!)
        states = self._collect_states(y_pred_proba_seq)
        
        # 2. Ridge回归 (秒级!)
        from sklearn.linear_model import Ridge
        self.W_out = Ridge(alpha=1e-6).fit(states, y_true_seq)
    
    def _collect_states(self, inputs):
        # 动态reservoir激活 (recurrent)
        h = np.zeros(n_reservoir)
        states = []
        for t in range(len(inputs)):
            h = np.tanh(W_in @ inputs[t] + W_res @ h)
            states.append(h)
        return np.array(states)
```

**为什么ESN这么快?**
- ✅ **固定Reservoir**: 不需要训练10000+个权重
- ✅ **仅线性输出层**: Ridge回归极快 (闭式解)
- ✅ **无反向传播**: 不需要GPU
- ✅ **Recurrent记忆**: 依然能捕捉时序依赖

**2. ESN vs HMM: 为什么ESN更强?**

| HMM局限 | ESN解决方案 |
|---------|------------|
| 离散状态 (4个) | 连续状态空间 (800维reservoir) |
| 一阶马尔科夫 | **动态reservoir记忆** (可回溯数十步) |
| 固定转移矩阵 | **输入驱动的状态演化** (W_res·h + W_in·x) |
| 线性发射概率 | **非线性tanh激活** (更复杂的发射建模) |

**3. ESN vs Mamba: 为什么ESN更快?**

```
Mamba (Selective SSM):
  - 参数依赖输入 (MLP计算Δ, B, C)  ← GPU密集
  - 深度网络 (2-3层) × 端到端训练    ← 需要梯度
  - 优化算法: Adam + 50 epochs      ← 时间长
  
ESN (Fixed Reservoir):
  - 参数固定随机 (无需计算)         ← CPU友好
  - 单层输出 + 闭式解              ← 无梯度
  - Ridge回归: 1次矩阵求逆         ← 秒级
```

### 8.5.3 ESN实验组设计

| ID | Reservoir Size | Spectral Radius | 辅助特征 | 正则化 | 预期F1 | 训练时间 |
|----|----------------|-----------------|---------|--------|-------|----------|
| **E1** | 500 | 0.9 | ❌ | α=1e-6 | 0.815 | 8 min |
| **E2** | 800 | 0.9 | ✅ ENMO | α=1e-6 | **0.820** | 10 min |
| **E3** | 1000 | 0.95 | ✅ ENMO | α=1e-7 | **0.823** | 12 min |
| **E4** | 800 | 0.9 | ✅ Full | α=1e-6 | **0.822** | 12 min |

**超参数说明**:
- **Reservoir Size**: 神经元数量 (越大越强但越慢)
- **Spectral Radius**: 控制记忆长度 (0.9=短期, 0.99=长期)
- **正则化α**: Ridge回归的L2惩罚 (防止过拟合)

### 8.5.4 ESN实现 (极简!)

```python
# esn_smoother.py
import numpy as np
from sklearn.linear_model import Ridge
from scipy import sparse

class ESNSmoother:
    def __init__(
        self,
        n_classes=4,
        n_reservoir=800,
        spectral_radius=0.9,
        sparsity=0.9,  # reservoir稀疏性
        input_scaling=1.0,
        ridge_alpha=1e-6,
        random_state=42,
    ):
        self.n_classes = n_classes
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.ridge_alpha = ridge_alpha
        
        np.random.seed(random_state)
        
        # 初始化固定权重
        # 输入权重: (n_reservoir, n_classes + aux_dim)
        self.W_in = np.random.randn(n_reservoir, n_classes) * input_scaling
        
        # Reservoir权重: 稀疏随机矩阵
        self.W_res = sparse.random(
            n_reservoir, n_reservoir, 
            density=1-sparsity,
            random_state=random_state
        ).toarray()
        
        # 调整spectral radius
        eigenvalues = np.linalg.eigvals(self.W_res)
        self.W_res *= spectral_radius / np.max(np.abs(eigenvalues))
        
        # 输出权重 (训练后填充)
        self.W_out = None
    
    def fit(self, y_pred_proba, y_true, groups, aux_features=None):
        """
        训练ESN smoother
        
        Args:
            y_pred_proba: (N, 4) - RF概率输出
            y_true: (N,) - 真实标签 (0-3)
            groups: (N,) - participant IDs
            aux_features: (N, k) - 辅助特征 (可选)
        """
        # 拼接辅助特征
        if aux_features is not None:
            # 扩展W_in
            aux_dim = aux_features.shape[1]
            W_in_aux = np.random.randn(self.n_reservoir, aux_dim) * 0.5
            self.W_in = np.hstack([self.W_in, W_in_aux])
            
            inputs = np.hstack([y_pred_proba, aux_features])
        else:
            inputs = y_pred_proba
        
        # 收集reservoir状态 (按participant分组)
        all_states = []
        all_targets = []
        
        unique_groups = np.unique(groups)
        for g in unique_groups:
            mask = groups == g
            seq_inputs = inputs[mask]
            seq_targets = y_true[mask]
            
            # 运行reservoir
            states = self._run_reservoir(seq_inputs)
            all_states.append(states)
            all_targets.append(seq_targets)
        
        # 合并所有状态
        X_train = np.vstack(all_states)  # (total_timesteps, n_reservoir)
        y_train = np.concatenate(all_targets)  # (total_timesteps,)
        
        # One-hot编码
        y_train_onehot = np.eye(self.n_classes)[y_train]  # (N, 4)
        
        # Ridge回归训练输出层
        print(f"Training Ridge regression on {X_train.shape[0]} samples...")
        self.ridge = Ridge(alpha=self.ridge_alpha)
        self.ridge.fit(X_train, y_train_onehot)
        
        print(f"ESN training complete. Reservoir size: {self.n_reservoir}")
    
    def predict(self, y_pred_proba, groups, aux_features=None):
        """
        预测
        """
        if aux_features is not None:
            inputs = np.hstack([y_pred_proba, aux_features])
        else:
            inputs = y_pred_proba
        
        # 按participant预测
        all_predictions = []
        unique_groups = np.unique(groups)
        
        for g in unique_groups:
            mask = groups == g
            seq_inputs = inputs[mask]
            
            # 运行reservoir
            states = self._run_reservoir(seq_inputs)
            
            # 预测
            proba = self.ridge.predict(states)  # (T, 4)
            preds = np.argmax(proba, axis=1)  # (T,)
            
            all_predictions.append(preds)
        
        return np.concatenate(all_predictions)
    
    def _run_reservoir(self, inputs):
        """
        运行reservoir动态
        
        Args:
            inputs: (T, input_dim)
        Returns:
            states: (T, n_reservoir)
        """
        T = len(inputs)
        states = np.zeros((T, self.n_reservoir))
        h = np.zeros(self.n_reservoir)  # 初始状态
        
        for t in range(T):
            # Reservoir更新: h_t = tanh(W_in·x_t + W_res·h_{t-1})
            h = np.tanh(self.W_in @ inputs[t] + self.W_res @ h)
            states[t] = h
        
        return states
```

### 8.5.5 ESN vs Mamba: 最终推荐

```
决策树:

Q1: GPU是否可用且稳定?
    ├── NO  → ✅ ESN (CPU训练,性能接近Mamba)
    └── YES → Q2

Q2: 训练时间是否敏感? (需要快速迭代)
    ├── YES → ✅ ESN (10 min vs Mamba 25 min)
    └── NO  → Q3

Q3: 是否追求绝对最高性能? (牺牲速度)
    ├── YES → Mamba-Medium (F1 ~ 0.828)
    └── NO  → ✅ ESN (F1 ~ 0.823, 速度2.5x)

推荐优先级 (针对你的3080Ti + 32GB):
1. ESN-E3 (n_reservoir=1000)      ⭐⭐⭐⭐⭐ 最平衡
2. Mamba-M3 (d_model=128, n=3)    ⭐⭐⭐⭐   最高精度
3. ESN-E2 (n_reservoir=800)       ⭐⭐⭐⭐   最快
```

**最终建议**: 
- **先跑ESN-E2和E3** (20分钟内完成,确保超过HMM)
- **如果E3达到0.820+**: 可能不需要Mamba了!
- **如果E3接近但未达标**: 再跑Mamba-M2作为backup

---

## 9. 成功标准与应急方案

### 9.1 成功标准 (优先级)

| 优先级 | 标准 | 达成条件 |
|--------|------|----------|
| **P0** | 超越HMM | Mamba Macro F1 **≥ 0.820** (vs HMM 0.812) |
| **P1** | 统计显著性 | McNemar p-value < 0.05 |
| **P2** | 计算效率可接受 | 训练时间 < 30 min, 推理 < 5ms/sample |
| **P3** | 鲁棒性 | per-participant F1 std < HMM |

### 9.2 应急方案

**场景1: M3配置未达到0.820**
```
原因诊断:
├── 1. 训练不充分?
│   → 增加epochs到100, 降低学习率到5e-4
├── 2. 正则化过强?
│   → 减小lambda_smooth到0.005
├── 3. 模型容量不足?
│   → 尝试d_model=192, n_layers=4
└── 4. 辅助特征无效?
    → 移除辅助特征,简化为M1配置
```

**场景2: Mamba比HMM慢太多 (>1小时训练)**
```
优化策略:
├── 使用Mamba-2优化版本 (50% speedup)
├── 减少batch_size (降低内存占用)
├── 使用混合精度训练 (torch.cuda.amp)
└── 考虑只在验证集的20%上训练Mamba
```

**场景3: Mamba过拟合 (train F1 >> test F1)**
```
正则化增强:
├── 增大lambda_smooth到0.05
├── 添加Dropout (0.2-0.3)
├── 数据增强: 时间抖动, participant sampling
└── Early stopping (耐心值=10 epochs)
```

---

## 10. 实现计划

### 5.1 技术栈

```python
# 核心依赖
import torch
import torch.nn as nn
from mamba_ssm import Mamba  # https://github.com/state-spaces/mamba

# 或使用 transformers 集成
from transformers import MambaModel, MambaConfig
```

### 5.2 代码结构

```
experiments/gait_filter/
├── mamba_smoother.py          # Mamba平滑器实现
│   ├── class MambaSmoother(nn.Module)
│   │   ├── __init__(d_model, n_layers, n_classes)
│   │   ├── forward(y_pred_proba, aux_feats=None)
│   │   └── viterbi_decode()  # 可选: 结合Viterbi
│   └── class MambaSmootherTrainer
│       ├── fit(y_pred_train, y_train, groups)
│       ├── predict(y_pred_test, groups)
│       └── evaluate(y_true, y_pred)
│
├── train_mamba_smoother.py    # 训练脚本
│   └── 两阶段训练:
│       1. 加载已训练RF模型
│       2. 训练Mamba smoother
│
├── evaluate_smoothers.py       # 对比HMM vs Mamba
│   └── 并行对比E1-E7所有实验组
│
└── MAMBA_WALKING_RECON.md     # 本文档
```

### 5.3 MambaSmoother 实现草图

```python
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaSmoother(nn.Module):
    def __init__(
        self,
        n_classes=4,          # sleep, sedentary, light, MVPA
        d_model=64,           # 隐藏维度
        n_layers=2,           # Mamba层数
        d_state=16,           # SSM状态维度
        d_conv=4,             # 卷积核大小
        expand=2,             # FFN expansion
        dropout=0.1,
        use_aux_features=False,  # 是否使用辅助特征
        aux_dim=0,            # 辅助特征维度
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model
        self.use_aux_features = use_aux_features
        
        # 输入投影: (n_classes + aux_dim) → d_model
        input_dim = n_classes + aux_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Mamba层堆叠
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ) for _ in range(n_layers)
        ])
        
        # LayerNorm
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # 输出投影: d_model → n_classes
        self.output_proj = nn.Linear(d_model, n_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, y_pred_proba, aux_features=None, return_logits=False):
        """
        Args:
            y_pred_proba: (batch_size, seq_len, n_classes) - RF概率输出
            aux_features: (batch_size, seq_len, aux_dim) - 可选辅助特征
        
        Returns:
            y_smoothed: (batch_size, seq_len, n_classes) - 平滑后的概率
        """
        batch_size, seq_len, _ = y_pred_proba.shape
        
        # 拼接辅助特征
        if self.use_aux_features and aux_features is not None:
            x = torch.cat([y_pred_proba, aux_features], dim=-1)
        else:
            x = y_pred_proba
        
        # 输入投影
        x = self.input_proj(x)  # (B, T, d_model)
        
        # Mamba层
        for mamba, norm in zip(self.mamba_layers, self.norms):
            # Mamba期望输入: (B, d_model, T)
            x_transposed = x.transpose(1, 2)  # (B, T, d) → (B, d, T)
            x_out = mamba(x_transposed)        # (B, d, T)
            x_out = x_out.transpose(1, 2)     # (B, d, T) → (B, T, d)
            
            # Residual + Norm
            x = norm(x + self.dropout(x_out))
        
        # 输出投影
        logits = self.output_proj(x)  # (B, T, n_classes)
        
        if return_logits:
            return logits
        
        # Softmax归一化
        y_smoothed = torch.softmax(logits, dim=-1)
        return y_smoothed

class MambaSmootherTrainer:
    def __init__(
        self,
        model,
        lr=1e-3,
        weight_decay=1e-5,
        lambda_smooth=0.01,     # 平滑正则化系数
        lambda_consistent=0.1,  # 与原始预测一致性系数
        device='cuda',
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.lambda_smooth = lambda_smooth
        self.lambda_consistent = lambda_consistent
        
    def compute_loss(self, y_pred_smooth, y_pred_raw, y_true):
        """
        三项损失:
        1. CrossEntropy: 与真实标签匹配
        2. Smoothness: 惩罚时序突变
        3. Consistency: 与RF原始预测保持接近
        """
        # 1. CE Loss
        ce_loss = nn.CrossEntropyLoss()(
            y_pred_smooth.reshape(-1, self.model.n_classes),
            y_true.reshape(-1)
        )
        
        # 2. Smoothness Loss: L2 norm of temporal difference
        diff = y_pred_smooth[:, 1:, :] - y_pred_smooth[:, :-1, :]
        smooth_loss = torch.mean(diff ** 2)
        
        # 3. Consistency Loss: KL(smooth || raw)
        kl_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log_softmax(y_pred_smooth, dim=-1),
            y_pred_raw  # 原始RF概率 (已softmax)
        )
        
        total_loss = (
            ce_loss + 
            self.lambda_smooth * smooth_loss + 
            self.lambda_consistent * kl_loss
        )
        
        return total_loss, {
            'ce': ce_loss.item(),
            'smooth': smooth_loss.item(),
            'kl': kl_loss.item(),
        }
    
    def fit(
        self,
        y_pred_train,  # (N, 4) numpy array - RF概率输出
        y_train,       # (N,) numpy array - 真实标签
        groups_train,  # (N,) numpy array - participant IDs
        epochs=50,
        batch_size=32,
    ):
        """
        训练Mamba smoother
        """
        from torch.utils.data import TensorDataset, DataLoader
        
        # 转为tensor
        y_pred_train = torch.FloatTensor(y_pred_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        
        # 按participant分组构建序列
        # 每个participant的所有时间窗组成一个序列
        unique_groups = np.unique(groups_train)
        
        dataset = []
        for g in unique_groups:
            mask = groups_train == g
            y_pred_seq = y_pred_train[mask]  # (T_g, 4)
            y_seq = y_train[mask]            # (T_g,)
            dataset.append((y_pred_seq, y_seq))
        
        # DataLoader (使用collate_fn处理变长序列)
        def collate_fn(batch):
            # batch: list of (y_pred_seq, y_seq)
            y_pred_batch = [item[0] for item in batch]
            y_batch = [item[1] for item in batch]
            
            # Padding到最长序列
            max_len = max(len(seq) for seq in y_pred_batch)
            
            y_pred_padded = torch.zeros(len(batch), max_len, 4)
            y_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
            
            for i, (y_pred, y) in enumerate(zip(y_pred_batch, y_batch)):
                length = len(y_pred)
                y_pred_padded[i, :length] = y_pred
                y_padded[i, :length] = y
            
            return y_pred_padded, y_padded
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for y_pred_batch, y_batch in dataloader:
                y_pred_batch = y_pred_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward
                y_pred_smooth = self.model(y_pred_batch, return_logits=True)
                
                # Loss
                loss, loss_dict = self.compute_loss(
                    y_pred_smooth,
                    y_pred_batch,  # 原始RF概率
                    y_batch
                )
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, y_pred_test, groups_test):
        """
        推理: 对测试集进行平滑预测
        """
        self.model.eval()
        
        y_pred_smoothed = []
        
        with torch.no_grad():
            unique_groups = np.unique(groups_test)
            for g in unique_groups:
                mask = groups_test == g
                y_pred_seq = torch.FloatTensor(y_pred_test[mask]).unsqueeze(0).to(self.device)
                
                # Mamba平滑
                y_smooth_proba = self.model(y_pred_seq)  # (1, T, 4)
                
                # Argmax
                y_smooth_labels = torch.argmax(y_smooth_proba, dim=-1).squeeze(0)
                
                y_pred_smoothed.append(y_smooth_labels.cpu().numpy())
        
        return np.concatenate(y_pred_smoothed)
```

---

## 6. 预期结果与假设

### 6.1 性能假设

| 方法 | 预期 Macro F1 | 相对HMM提升 | 推理速度 |
|------|--------------|------------|---------|
| **RF** | 0.706 | baseline | Fast |
| **RF + HMM** | 0.812 | baseline (HMM) | Very Fast |
| **RF + Mamba-Light** | **0.820 - 0.830** | +1~2% | Medium |
| **RF + Mamba-Medium** | **0.825 - 0.835** | +1.5~3% | Slower |
| **End2End Mamba** | 0.800 - 0.850 | 不定 (高方差) | Medium |

**理由**:
1. Mamba可捕捉**长程依赖** → 更准确的状态转移
2. Mamba的**选择性**可适应不同个体 → 个性化平滑
3. 但Mamba需要更多数据和调参 → 可能不稳定

### 6.2 失败模式分析

**如果Mamba未超越HMM,可能原因**:
1. **数据不足**: Mamba是深度模型,需要更多训练数据
2. **过平滑**: Mamba可能过度平滑,错过短暂活动 (如brief standing)
3. **超参数敏感**: d_model、正则化系数需要精细调整

**缓解策略**:
- 数据增强: 滑动窗口采样,增加训练样本
- 早停 + 验证集监控
- 贝叶斯超参数优化 (Optuna)

---

## 7. 讨论: Mamba的适用场景

### 7.1 何时用Mamba取代HMM?

```
决策树:

Q1: 数据集规模是否足够? (参与者数 > 100, 总样本 > 100k)
    ├── NO  → ❌ 继续用HMM (Mamba易过拟合)
    └── YES → Q2

Q2: 是否需要个性化建模? (不同个体活动模式差异大)
    ├── NO  → ⚠️ HMM可能足够
    └── YES → Q3

Q3: 计算资源是否充足? (GPU训练 + 可接受推理延迟)
    ├── NO  → ❌ HMM更轻量
    └── YES → ✅ 尝试Mamba (方案B)

Q4: 是否需要端到端优化? (特征提取+分类联合训练)
    ├── NO  → 方案B (Mamba Smoother)
    └── YES → 方案A (End2End Mamba)
```

### 7.2 HMM的不可替代优势

**可解释性案例**:
```python
# HMM转移矩阵示例 (可直观理解):
#              sleep  sedentary  light  MVPA
# sleep      [  0.95    0.04     0.01   0.00 ]  ← 睡眠很稳定
# sedentary  [  0.01    0.85     0.12   0.02 ]  ← 久坐易转light
# light      [  0.00    0.20     0.70   0.10 ]  ← light较动态
# MVPA       [  0.00    0.05     0.25   0.70 ]  ← MVPA较短暂

# Mamba的"转移矩阵"是隐式的,难以可视化和解释
```

**实时监控**: HMM可在低功耗设备 (MCU) 上实时运行,Mamba需GPU

---

## 8. 结论与建议

### 8.1 总结

| 方案 | 推荐度 | 适用场景 | 风险 |
|------|-------|---------|------|
| **方案B: Mamba Smoother** | ⭐⭐⭐⭐⭐ | 研究导向、有GPU、追求SOTA | 调参成本 |
| **保持HMM** | ⭐⭐⭐⭐ | 生产部署、可解释性优先、资源受限 | 无新意 |
| **方案A: End2End Mamba** | ⭐⭐⭐ | 充足数据、愿意重构pipeline | 高风险 |

### 8.2 阶段性路线图

**Phase 0 (当前)**: 复现HMM基线
- 运行`Benchmark.ipynb`,确认 F1~0.81

**Phase 1 (1-2周)**: 实现Mamba Smoother (方案B)
- 代码实现: `mamba_smoother.py`
- 对比实验: E1-E4 (见4.2节)
- **里程碑**: Mamba F1 ≥ 0.815 (超越HMM)

**Phase 2 (可选, 1周)**: 消融实验
- 测试辅助特征、正则化、Selective SSM的贡献
- 分析失败案例 (哪些样本Mamba预测错误?)

**Phase 3 (可选, 2周)**: End2End Mamba (方案A)
- 仅在Phase 1成功后考虑
- 探索联合优化的潜力

---

## 9. 参考文献

### 核心论文

1. **Mamba原论文**:  
   Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.  
   arXiv:2312.00752. [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)

2. **State-Space Models综述**:  
   Gu, A., Goel, K., & Ré, C. (2021). *Efficiently Modeling Long Sequences with Structured State Spaces*.  
   ICLR 2022. [https://arxiv.org/abs/2111.00396](https://arxiv.org/abs/2111.00396)

3. **HMM在HAR中的应用**:  
   Willetts, M., et al. (2018). *Statistical machine learning of sleep and physical activity phenotypes from sensor data in 96,220 UK Biobank participants*.  
   Scientific Reports.

### 代码仓库

- **Mamba官方实现**: [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
- **Hugging Face Transformers集成**: `transformers>=4.38.0` (支持MambaModel)
- **当前项目**: `capture24-master/`

---

## 附录: 快速开始代码

```bash
# 安装Mamba
pip install mamba-ssm  # 需要 CUDA 11.8+

# 或使用Hugging Face
pip install transformers>=4.38.0

# 训练Mamba Smoother
cd experiments/gait_filter
python train_mamba_smoother.py \
    --datadir prepared_data \
    --annot Walmsley2020 \
    --d_model 64 \
    --n_layers 2 \
    --epochs 50 \
    --output models/mamba_smoother.pt

# 评估对比
python evaluate_smoothers.py \
    --methods hmm mamba-light mamba-medium \
    --output results/smoother_comparison.csv
```
