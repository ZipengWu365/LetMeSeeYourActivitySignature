# LetMeSeeYourActivitySignature

Human Activity Recognition and Gait Detection Pipeline using accelerometer data.

## 📁 Project Structure

```
├── gait_filter/          # Walking (Gait) Detection Pipeline
│   ├── run.sh            # 🐧 Linux one-click runner
│   ├── run_pipeline.py   # Main entry point
│   ├── preprocess.py     # ENMO computation, binary labels, train/test split
│   ├── extract_features.py  # Hand-crafted, MiniRocket, SAX, SFA features
│   ├── train_classifiers.py # RF, XGBoost, LR, Ridge, MrSQM classifiers
│   ├── evaluate.py       # PR curves, confusion matrices, model export
│   ├── requirements.txt  # Python dependencies
│   └── README.md         # Detailed documentation
```

## 🚀 Quick Start (Linux)

```bash
# Clone this repo
git clone https://github.com/ZipengWu365/LetMeSeeYourActivitySignature.git
cd LetMeSeeYourActivitySignature

# Run gait filter pipeline (auto-installs dependencies)
chmod +x gait_filter/run.sh
./gait_filter/run.sh --quick-test
```

## 🔬 Gait Filter Pipeline

A binary classifier to detect **walking/running activities** from accelerometer data.

### Features
- **Hand-crafted features** (31-dim): moments, quantiles, ACF, spectral features
- **MiniRocket** (~10k-dim): random convolutional kernels
- **SAX/SFA**: Symbolic representations for time-series

### Classifiers
- Random Forest, XGBoost, Logistic Regression, Ridge
- **MrSQM**: Multiple Representations Sequence Miner (symbolic classifier)

### Data
Designed for [CAPTURE-24](https://github.com/OxWearables/capture24) dataset:
- 151 participants, ~2000 hours of accelerometer data
- 100Hz sampling rate, 10-second windows

## 📊 Expected Results

| Method | PR-AUC | F1-Score |
|--------|--------|----------|
| Handcraft + RF | ~0.85 | ~0.81 |
| Handcraft + XGBoost | ~0.84 | ~0.80 |
| MiniRocket + Ridge | ~0.82 | ~0.78 |
| SAX + MrSQM | TBD | TBD |

## 📝 License

MIT License
