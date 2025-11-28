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

---

## 📦 Step 0: Get CAPTURE-24 Data

Before running the pipeline, you need to download and preprocess the CAPTURE-24 dataset.

### Option A: Download Pre-processed Data (Recommended)

The CAPTURE-24 benchmark provides pre-processed numpy arrays:

```bash
# Clone the CAPTURE-24 benchmark repo
git clone https://github.com/OxWearables/capture24.git
cd capture24

# Download the pre-processed data (~4GB compressed)
# The data is hosted on Oxford's servers
wget https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16f0f17064/files/dcj82k7829 -O capture24.zip
unzip capture24.zip

# This creates:
# data/
#   ├── P001.csv.gz ... P151.csv.gz  (raw accelerometer + annotations)
```

### Option B: Generate from Raw Data

If you have the raw `.csv.gz` files, run the preprocessing script:

```bash
cd capture24

# Install dependencies
pip install numpy scipy pandas tqdm

# Run preprocessing (generates X.npy, Y.npy, etc.)
python prepare_data.py

# This creates:
# prepared_data/
#   ├── X.npy          # (934762, 1000, 3) - 10s windows at 100Hz, 3-axis
#   ├── Y.npy          # (934762,) - Activity labels (string)
#   ├── T.npy          # (934762,) - Timestamps
#   ├── P.npy          # (934762,) - Participant IDs
#   └── pid_lookup.pkl # Participant ID mapping
```

### Data Format

| File | Shape | Description |
|------|-------|-------------|
| `X.npy` | (934762, 1000, 3) | Accelerometer windows: 10s × 100Hz × 3-axis (x,y,z in g units) |
| `Y.npy` | (934762,) | Activity labels: "sleep", "sit-stand", "walking", etc. |
| `P.npy` | (934762,) | Participant IDs for group-based train/test split |
| `T.npy` | (934762,) | Timestamps |

### Directory Structure After Setup

```
your_project/
├── capture24/                    # CAPTURE-24 benchmark repo
│   ├── data/                     # Raw CSV files (optional)
│   ├── prepared_data/            # ⬅️ Required: X.npy, Y.npy, P.npy
│   │   ├── X.npy
│   │   ├── Y.npy
│   │   └── P.npy
│   └── prepare_data.py
│
└── LetMeSeeYourActivitySignature/  # This repo
    └── gait_filter/
        └── run.sh                # Point to ../capture24/prepared_data
```

---

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
