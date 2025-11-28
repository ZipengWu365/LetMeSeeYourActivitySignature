# CAPTURE-24 Gait Filter: 特征提取方法对比与实验计划

## 📊 数据流水线概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         原始数据 X.npy (934,762 × 1000 × 3)                  │
│                              10秒窗口 @ 100Hz × 3轴                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              共享预处理步骤                                   │
│  v = np.linalg.norm(xyz, axis=1)    # 三轴合成向量模 → (1000,)              │
│  v = median_filter(v, size=5)       # 中值滤波去噪                           │
│  v = v - 1                          # 减去重力 (~1g)                         │
│  v = np.clip(v, -2, 2)              # 截断异常值                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────┴──────────────┐
                        │     向量模序列 V.npy        │
                        │     (934,762 × 1000)        │
                        └──────────────┬──────────────┘
                                       │
           ┌───────────────┬───────────┼───────────┬───────────────┐
           ▼               ▼           ▼           ▼               ▼
    ┌────────────┐  ┌────────────┐ ┌────────┐ ┌────────────┐ ┌────────────┐
    │ Hand-craft │  │ MiniRocket │ │  SAX   │ │    SFA     │ │   ABBA     │
    │  Features  │  │   (PPV)    │ │Symbols │ │  Words     │ │  Symbols   │
    │   31-dim   │  │ 9996-dim   │ │ string │ │  string    │ │  string    │
    └─────┬──────┘  └─────┬──────┘ └───┬────┘ └─────┬──────┘ └─────┬──────┘
          │               │            │            │              │
          │               │            └─────┬──────┴──────────────┘
          │               │                  │
          ▼               ▼                  ▼
    ┌────────────┐  ┌────────────┐    ┌────────────┐
    │ RF/XGBoost │  │ Ridge/LR   │    │   MrSQM    │
    │  (树模型)   │  │ (线性模型)  │    │ (符号方法)  │
    └────────────┘  └────────────┘    └────────────┘
```

---

## 🔬 特征提取方法详解

### 共同起点：向量模预处理

所有特征提取方法**共享同一预处理**，输入为 1D 时间序列 `v` (长度 1000):

```python
# 输入: xyz shape (1000, 3)
v = np.linalg.norm(xyz, axis=1)  # → (1000,)
v = median_filter(v, size=5)
v = v - 1  # 去重力
v = np.clip(v, -2, 2)
```

---

### 方法 1: Hand-crafted Features (Benchmark 基线)

**输出维度**: 31

| 类别 | 特征数 | 计算方法 |
|------|--------|---------|
| 时域统计 | 4 | mean, std, skew, kurtosis |
| 分位数 | 5 | min, q25, median, q75, max |
| 自相关 | 5 | ACF峰/谷位置、零交叉 |
| 频谱 | 8 | spectral entropy, power, top-3 frequencies |
| FFT | 6 | Welch PSD @ 0-5Hz |
| 峰值 | 4 | peak count, prominence stats |

**优点**: 领域知识编码，低维高效，可解释性强
**缺点**: 需要人工设计，可能遗漏未知模式

---

### 方法 2: MiniRocket

**输出维度**: ~9,996

**原理**: 对向量模序列 `v` 应用 84 个随机卷积核，每个核使用多种 dilation：

$$\text{PPV}_k = \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}[(v * w_k)_t > b_k]$$

| 参数 | 值 |
|------|-----|
| 卷积核数量 | 84 |
| 核长度 | 9 (固定) |
| 权重取值 | {-1, 0, 1, 2} |
| Dilation | 指数增长序列 |
| 聚合方式 | PPV (正值比例) |

**优点**: 自动特征学习，无需领域知识
**缺点**: 高维，需要更多训练数据

---

### 方法 3: SAX (Symbolic Aggregate approXimation)

**输出**: 符号字符串

**原理**: 将时间序列离散化为符号序列

```python
# 1. PAA降维: 1000点 → w个段 (如 w=20)
paa = [mean(v[i*50:(i+1)*50]) for i in range(20)]

# 2. 正态分位数切分为 alphabet_size 个符号 (如 a=4 → 'a','b','c','d')
breakpoints = norm.ppf([0.25, 0.5, 0.75])

# 3. 映射为符号
sax_word = ''.join([chr(ord('a') + bin_idx) for bin_idx in digitize(paa, breakpoints)])
# 结果如: "aabbccddaabbccdd..."
```

**参数**:
- `word_length (w)`: PAA 段数 (常用 8-32)
- `alphabet_size (a)`: 符号数 (常用 4-8)

---

### 方法 4: SFA (Symbolic Fourier Approximation)

**输出**: 频域符号字符串

**原理**: 在频域而非时域进行符号化

```python
# 1. DFT 取前 l 个系数
dft_coeffs = fft(v)[:word_length]

# 2. 对每个系数的实部/虚部分别量化
# 使用数据驱动的 breakpoints (MCB - Multiple Coefficient Binning)
sfa_word = quantize_fourier_coeffs(dft_coeffs, breakpoints)
```

**与 SAX 区别**:
- SAX 基于时域 PAA → 保留形状
- SFA 基于频域 DFT → 保留频率特性

---

### 方法 5: ABBA (Adaptive Brownian Bridge-based Aggregation)

**输出**: 自适应符号字符串

**原理**: 将时间序列压缩为 (长度, 增量) 对，然后聚类

```python
# 1. 线性分段近似 → 提取 (length, increment) 对
pieces = compress_to_pieces(v, tolerance=0.1)
# 如: [(50, 0.3), (30, -0.1), (40, 0.5), ...]

# 2. 对 pieces 进行聚类 → 符号
symbols = cluster_pieces(pieces, k=10)
# 结果如: "AABCDEABC..."
```

**优点**: 自适应分辨率，短序列表示长模式
**缺点**: 计算复杂度较高

---

## 🧪 实验计划 Task List

### Phase 0: 数据预处理与中间特征缓存

| Task | 描述 | 输出 |
|------|------|------|
| 0.1 | 加载 X.npy，计算向量模，保存 V.npy | `prepared_data/V.npy` (934762, 1000) |
| 0.2 | 构建二分类标签 (walking vs rest) | `prepared_data/Y_binary.npy` |
| 0.3 | Group-based train/test split | `prepared_data/split_indices.pkl` |

### Phase 1: 特征提取 (可并行)

| Task | 方法 | 输出文件 | 预估维度 |
|------|------|---------|---------|
| 1.1 | Hand-crafted | `features/handcraft.npy` | (N, 31) |
| 1.2 | MiniRocket | `features/minirocket.npy` | (N, 9996) |
| 1.3 | SAX (w=16, a=4) | `features/sax.pkl` | 字符串列表 |
| 1.4 | SFA (w=16, a=4) | `features/sfa.pkl` | 字符串列表 |
| 1.5 | ABBA (tol=0.1) | `features/abba.pkl` | 字符串列表 |

### Phase 2: 分类模型训练

| Task | 特征 | 分类器 | 评估指标 |
|------|------|--------|---------|
| 2.1 | Hand-crafted | Random Forest | PR-AUC, F1, Confusion Matrix |
| 2.2 | Hand-crafted | XGBoost | PR-AUC, F1, Confusion Matrix |
| 2.3 | Hand-crafted | Logistic Regression | PR-AUC, F1, Confusion Matrix |
| 2.4 | MiniRocket | Ridge Classifier | PR-AUC, F1, Confusion Matrix |
| 2.5 | MiniRocket | Logistic Regression | PR-AUC, F1, Confusion Matrix |
| 2.6 | SAX | MrSQM | PR-AUC, F1, Confusion Matrix |
| 2.7 | SFA | MrSQM | PR-AUC, F1, Confusion Matrix |
| 2.8 | Combined | Stacking Ensemble | PR-AUC, F1, Confusion Matrix |

### Phase 3: 评估与导出

| Task | 描述 | 输出 |
|------|------|------|
| 3.1 | Hard-negative 测试 (Walking vs Chores) | 专项 PR 曲线 |
| 3.2 | Calibration 曲线 | Reliability Diagram |
| 3.3 | 导出最佳模型 | `models/gait_filter_best.pkl` |
| 3.4 | 推理速度测试 | seconds per 1000 windows |

---

## 📁 目录结构设计

```
experiments/gait_filter/
├── config.yaml                 # 实验配置
├── run_pipeline.py             # 一键运行入口
├── preprocess.py               # Phase 0: 预处理
├── extract_features.py         # Phase 1: 特征提取
├── train_classifiers.py        # Phase 2: 训练
├── evaluate.py                 # Phase 3: 评估
└── logs/
    └── {project_id}_execution.log

artifacts/gait_filter/
├── features/                   # 中间特征缓存
│   ├── V.npy
│   ├── handcraft.npy
│   ├── minirocket.npy
│   ├── sax.pkl
│   ├── sfa.pkl
│   └── abba.pkl
├── models/                     # 训练好的模型
│   └── gait_filter_best.pkl
├── plots/                      # 可视化
│   ├── pr_curve_comparison.png
│   ├── calibration_curve.png
│   └── confusion_matrix.png
└── results/
    └── metrics_summary.csv
```

---

## 🚀 一键运行命令

```bash
# 完整流水线
python experiments/gait_filter/run_pipeline.py --project-id GF002 --phases all

# 仅特征提取
python experiments/gait_filter/run_pipeline.py --project-id GF002 --phases extract

# 仅训练
python experiments/gait_filter/run_pipeline.py --project-id GF002 --phases train

# 快速测试模式 (小数据集)
python experiments/gait_filter/run_pipeline.py --project-id GF002_test --quick-test
```

---

## ✅ 当前完成状态

| 组件 | 状态 | 备注 |
|------|------|------|
| `train_gait_filter.py` (旧版) | ⚠️ 需重构 | 仅 MiniRocket，未使用向量模预处理 |
| `GF001_capture24_log.md` | ✅ 已生成 | PR-AUC=0.52 (需改进) |
| 向量模预处理 | ❌ 未实现 | 需要新增 |
| SAX/SFA/ABBA | ❌ 未实现 | 需要新增 |
| MrSQM 集成 | ❌ 未实现 | 需要安装依赖 |
| 统一 Pipeline | ❌ 未实现 | 本次任务重点 |

---

## 📦 依赖项

```yaml
# 核心依赖
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
joblib

# 时间序列特征
sktime>=0.24          # MiniRocket
pyts>=0.12            # SAX, SFA
fABBA>=0.1            # ABBA (可选)
mrsqm>=0.0.3          # MrSQM 分类器

# 分类器
xgboost>=2.0
lightgbm>=4.0         # 可选

# 可视化
matplotlib>=3.7
seaborn>=0.12
```
