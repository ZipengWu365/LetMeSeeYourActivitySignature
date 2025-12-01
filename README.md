# LetMeSeeYourGaitSignature

Walking State Recognition from Wearable Accelerometer Data  
**HMM vs ESN vs Mamba** - Comparing temporal smoothing approaches

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place the following files in `prepared_data/` directory:
- `X_feats.pkl` - Extracted features (N, 32)
- `Y_Walmsley2020.npy` - Activity labels
- `P.npy` - Participant IDs

### 3. Run Experiments

```bash
# Quick mode (~20 min): HMM + 2 ESN configs
python run_hmm_esn_mamba_walking_recognition_experiments.py --mode quick

# Standard mode (~40 min): HMM + ESN + Mamba-Light
python run_hmm_esn_mamba_walking_recognition_experiments.py --mode standard

# Full mode (~2 hours): All experiments
python run_hmm_esn_mamba_walking_recognition_experiments.py --mode full
```

---

## File Structure

```
 run_hmm_esn_mamba_walking_recognition_experiments.py  # Main experiment script
 train_baseline.py            # RF + HMM baseline
 train_esn_smoother.py        # ESN (Echo State Network) smoother
 train_mamba_smoother.py      # Mamba SSM smoother
 evaluate_smoothers.py        # Model evaluation & comparison
 classifier.py                # Classifier utilities
 hmm.py                       # HMM implementation
 fix_pickle_compat.py         # NumPy pickle compatibility fix
 requirements.txt             # Python dependencies
 MAMBA_WALKING_RECOGNITION.md # Detailed documentation
 readmeforgaitfilter.md       # Gait filter pipeline docs
```

---

## Experiment Modes

| Mode | Duration | Experiments |
|------|----------|-------------|
| quick | ~20 min | HMM + 2 ESN configs |
| standard | ~40 min | HMM + ESN + Mamba-Light |
| full | ~2 hours | All 7 experiment combinations |

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NVIDIA GPU recommended (for Mamba)
- ~16GB RAM minimum

---

## Documentation

See `MAMBA_WALKING_RECOGNITION.md` for:
- Theoretical background (HMM vs Mamba)
- Architecture comparisons
- Implementation details
