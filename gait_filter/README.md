# CAPTURE-24 Gait Filter Pipeline
## 一键运行指南

---

## 🐧 Linux 服务器一键运行

```bash
# 1. 进入项目目录
cd capture24-master

# 2. 赋予执行权限
chmod +x experiments/gait_filter/run.sh

# 3. 运行 (自动安装依赖)
./experiments/gait_filter/run.sh

# 快速测试模式 (10k样本)
./experiments/gait_filter/run.sh --quick-test

# 跳过MiniRocket (节省40GB内存)
./experiments/gait_filter/run.sh --skip-minirocket

# 查看帮助
./experiments/gait_filter/run.sh --help
```

脚本会自动:
- ✅ 检测并安装 Python 依赖
- ✅ 安装 FFTW3 库 (MrSQM依赖)
- ✅ 安装 mrsqm, pyts, sktime 等
- ✅ 检查数据文件
- ✅ 运行完整 pipeline

---

## 🚀 快速开始 (手动方式)

### 1. 安装依赖
```bash
# 方式1: pip安装 (推荐)
pip install numpy scipy pandas scikit-learn joblib matplotlib statsmodels sktime xgboost pyts

# 方式2: 从requirements.txt安装
pip install -r experiments/gait_filter/requirements.txt

# 方式3: conda环境 (如果有)
conda activate your_env
pip install sktime pyts  # conda环境可能缺少这两个
```

### 2. 运行Pipeline
```bash
# 完整运行 (全部93万样本)
python experiments/gait_filter/run_pipeline.py --project-id GF002 --phases all

# 快速测试 (1万样本，验证环境)
python experiments/gait_filter/run_pipeline.py --project-id GF_TEST --quick-test
```

---

## ⏱️ 预估运行时间

### 硬件配置建议

| 配置级别 | CPU | RAM | 预估时间 (全量93万) |
|---------|-----|-----|-------------------|
| **最低配置** | 4核 | 16GB | ~4-6 小时 |
| **推荐配置** | 8核 | 32GB | ~2-3 小时 |
| **高性能** | 16核+ | 64GB+ | ~1 小时 |

### 各阶段耗时分解

| 阶段 | 操作 | 1万样本 | 93万样本(预估) |
|------|------|--------|---------------|
| **Phase 0: Preprocess** | 计算ENMO | 2s | ~3 min |
| **Phase 1: Extract** | 手工特征 | 5s | ~8 min |
| | MiniRocket | 30s | ~45 min |
| | SAX/SFA | 2s | ~3 min |
| **Phase 2: Train** | RF (500树) | 1s | ~5 min |
| | XGBoost | 0.5s | ~3 min |
| | LR | 0.5s | ~2 min |
| | MiniRocket+Ridge | 0.5s | ~2 min |
| | MiniRocket+LR | 5min | ~30 min |
| **Phase 3: Evaluate** | 绘图+导出 | 3s | ~5 min |
| **总计** | | ~6 min | **~1.5-2 小时** |

### 内存使用估算

| 数据 | 形状 | 内存占用 |
|------|------|---------|
| X.npy (原始) | (934762, 1000, 3) | ~11 GB (mmap) |
| V.npy (ENMO) | (934762, 1000) | ~3.7 GB |
| handcraft.npy | (934762, 32) | ~120 MB |
| minirocket.npy | (934762, 9996) | ~37 GB ⚠️ |

**注意**: MiniRocket 特征很大！建议：
- 服务器至少 64GB RAM，或
- 使用 `--max-samples 100000` 限制样本数

---

## 📦 依赖包说明

| 包 | 版本 | 用途 | 必需? |
|----|------|------|------|
| `numpy` | >=1.24,<2.0 | 数值计算 | ✅ |
| `scipy` | >=1.10 | 信号处理、滤波 | ✅ |
| `pandas` | >=2.0 | 数据处理 | ✅ |
| `scikit-learn` | >=1.3 | 分类器、评估 | ✅ |
| `joblib` | >=1.3 | 模型保存、并行 | ✅ |
| `statsmodels` | >=0.14 | ACF计算 | ✅ |
| `matplotlib` | >=3.7 | 可视化 | ✅ |
| `sktime` | >=0.24 | MiniRocket | ⚠️ 可选 |
| `pyts` | >=0.12 | SAX/SFA | ⚠️ 可选 |
| `xgboost` | >=2.0 | XGBoost分类器 | ⚠️ 可选 |

---

## 🔧 命令行参数

```bash
python experiments/gait_filter/run_pipeline.py [OPTIONS]

选项:
  --project-id TEXT      项目ID，用于日志和输出命名 [default: GF002]
  --phases TEXT          运行阶段: all, preprocess, extract, train, evaluate
                         可组合: --phases preprocess,extract
  --prepared-dir PATH    prepared_data目录路径 [default: prepared_data]
  --artifacts-dir PATH   输出目录 [default: artifacts/gait_filter]
  --quick-test          快速测试模式 (限制1万样本)
  --max-samples INT     最大样本数 (用于调试)
  --seed INT            随机种子 [default: 42]
  --n-jobs INT          并行进程数 [default: -1 (全部CPU)]
```

### 使用示例

```bash
# 1. 完整运行
python experiments/gait_filter/run_pipeline.py --project-id GF002

# 2. 仅预处理和特征提取 (可在低内存机器上先做)
python experiments/gait_filter/run_pipeline.py --project-id GF002 --phases preprocess,extract

# 3. 限制样本数 (用于开发测试)
python experiments/gait_filter/run_pipeline.py --project-id GF_DEV --max-samples 50000

# 4. 在已有特征上重新训练
python experiments/gait_filter/run_pipeline.py --project-id GF002 --phases train,evaluate

# 5. 使用4个CPU核心
python experiments/gait_filter/run_pipeline.py --project-id GF002 --n-jobs 4
```

---

## 📁 输出文件

运行完成后，生成以下文件：

```
artifacts/gait_filter/
├── features/
│   ├── V.npy                    # ENMO序列 (N, 1000)
│   ├── Y_binary.npy             # 二分类标签
│   ├── split_indices.npz        # 训练/测试索引
│   ├── handcraft.npy            # 手工特征 (N, 32)
│   ├── minirocket.npy           # MiniRocket特征 (N, ~10000)
│   ├── sax.pkl                  # SAX符号序列
│   └── sfa.pkl                  # SFA符号序列
├── models/
│   ├── handcraft_rf.pkl         # Random Forest
│   ├── handcraft_xgb.pkl        # XGBoost
│   ├── handcraft_lr.pkl         # Logistic Regression
│   ├── minirocket_ridge.pkl     # Ridge Classifier
│   ├── minirocket_lr.pkl        # Logistic Regression
│   └── gait_filter_best.pkl     # 最佳模型 (自动选择)
├── plots/
│   ├── pr_curves_comparison.png # PR曲线对比
│   ├── calibration_curves.png   # 校准曲线
│   └── confusion_matrix_*.png   # 混淆矩阵
└── training_summary.pkl         # 训练结果汇总

experiments/gait_filter/
├── logs/
│   └── {project_id}_{timestamp}_execution.log  # 执行日志
└── {project_id}_final_report.md                # 最终报告
```

---

## ⚠️ 常见问题

### Q1: sktime 安装失败
```bash
# 尝试指定版本
pip install sktime==0.24.0

# 或者跳过 MiniRocket (仍可运行手工特征)
# Pipeline 会自动跳过不可用的特征
```

### Q2: 内存不足 (MemoryError)
```bash
# 方案1: 限制样本数
python run_pipeline.py --max-samples 100000

# 方案2: 分阶段运行
python run_pipeline.py --phases preprocess  # 先预处理
python run_pipeline.py --phases extract     # 再提取
python run_pipeline.py --phases train       # 最后训练
```

### Q3: MiniRocket 训练太慢
```bash
# 跳过 MiniRocket，只用手工特征
# 编辑 train_classifiers.py，注释掉 MiniRocket 相关代码
```

### Q4: NumPy 版本冲突
```bash
# 强制使用 NumPy 1.x
pip install "numpy>=1.24,<2.0"
```

---

## 📊 ENMO 计算说明

**ENMO (Euclidean Norm Minus One)** 是可穿戴加速度计标准指标：

$$\text{ENMO} = \max\left(\sqrt{x^2 + y^2 + z^2} - 1g, 0\right)$$

| 活动 | ENMO范围 (mg) |
|------|--------------|
| 静止/睡眠 | 0-20 |
| 久坐 | 20-50 |
| 轻度活动 | 50-100 |
| 步行 | 100-300 |
| 跑步 | 300-1000+ |

本Pipeline中：
- 输入 X.npy 已经是 g 单位 (1g ≈ 9.8 m/s²)
- 输出 ENMO 也是 g 单位
- 步行典型值: 0.1-0.3g
