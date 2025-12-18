# Next-Step Experiment Design: HMM-Based Mamba (P1 → P1-R → P3)
For Code Agent Execution — Strict, Engineering-Grade Plan

## 0. Scope and Non-Negotiable Rules
1. The agent must **not change**: dataset semantics, label set, windowing, grouping logic, split protocol, or metric definitions.
2. The agent must **not silently improve** the design. Implement exactly what is specified here.
3. The agent must **not refactor** unrelated code. Only implement the new method(s) specified here and the minimal orchestration needed.
4. The agent must **not introduce** new baselines or modify baseline hyperparameters. Baselines are rerun only under the unified evaluator.
5. All runs must be reproducible: fixed seeds, saved configs, deterministic participant splits, saved artifacts.
6. Any ambiguity must be logged as an explicit **ASSUMPTION** in `REPORT_NEXT.md` and kept conservative.

This document defines the **next stage** after your current progress (P1 implemented plus smoke test). It expands the roadmap to include:
- **P1 full runs and baseline reruns (mandatory)**
- **P1-R: HMM-regularized emission training (new method; lightweight)**
- **P3: Markov-gated Mamba (new method; lightweight MoE)**
with strict “go/no-go” gates.

---

## 1. Current Status (Inputs to This Stage)
You already have:
- `models/tiny_mamba_emission.py` (P1 emission network)
- `models/hmm_decode.py` (HMM transition fit plus Viterbi decode)
- `evaluation/evaluate_sequence_labeling.py` (unified evaluator)
- `train/train_p1_mamba_hmm.py` (P1 trainer)

Smoke test indicates:
- Decoding provides a positive gain over raw predictions
- Class 1 (moderate-vigorous) has near-zero F1, requiring immediate diagnostic checks

This stage assumes those modules run end-to-end.

---

## 2. Mandatory Diagnostics (Must Happen Before Any New Model)
Before running any sweeps or implementing P1-R/P3, the agent must produce a short diagnostic report and artifacts.

### 2.1 Label Mapping Consistency Check
Hard requirement: verify that label mapping is consistent across:
- training labels encoding
- evaluation label encoding
- HMM transition fitting (from labels)
- any saved `predictions.*`

Deliverable:
- Save `artifacts/<run>/label_mapping.json` containing:
  - `name_to_index` and `index_to_name`
- Print it verbatim in `REPORT_NEXT.md`

### 2.2 Prediction Frequency Check (Class 1 Collapse Detection)
Compute and report, on **test split**:
- histogram of `y_true`
- histogram of `y_pred_raw`
- histogram of `y_pred_decoded`

Deliverable:
- Save `artifacts/<run>/class_histograms.json`
- Include the table in `REPORT_NEXT.md`

### 2.3 Emission Confidence Check (Class 1 Signal)
For all test timesteps where `y_true == 1`:
- compute mean and quantiles of `proba_raw[:,1]`
- compare to mean of max probability over other classes

Deliverable:
- Save `artifacts/<run>/class1_proba_stats.json`
- Include summary in `REPORT_NEXT.md`

**Only after these three diagnostics are complete** may the agent proceed.

---

## 3. Stage A: Reproduce Full P1 and Baselines (Mandatory)
### 3.1 Runs to Execute
1. **P1 Full Training (20 epochs max, early stopping)**
   - inference reports both Raw and Decoded
2. **Baselines under unified evaluator** (no changes)
   - RF (if already available)
   - RF+HMM
   - ESN smoother (optional if already stable)
   - Mamba smoother on RF probabilities (existing)

### 3.2 Output Contract (All Methods)
Every method must produce, for test split:
- `metrics.json`
- `predictions.csv` or `predictions.parquet` with:
  - `group_id, t, y_true, y_pred`
  - optional: `y_pred_raw`, `y_pred_decoded`, `proba_raw[4]`
- `transition_matrix.npy` (if decoding is used)

All metrics must be computed by `evaluation/evaluate_sequence_labeling.py`.

### 3.3 Acceptance Gate for Proceeding
After Stage A, compute:
- `Δ = MacroF1_decoded(P1) - MacroF1(RF+HMM)`

Proceed rules:
- If `Δ >= -0.03` (within 3% absolute), do **not** implement P3 yet.
  - First run **P1-R** (Section 4) for a controlled gain attempt.
- If `Δ < -0.03`, implement **P1-R** first; only proceed to P3 if P1-R fails.

---

## 4. Stage B: New Method P1-R (HMM-Regularized Emission Training)
P1-R is the preferred “second method” because it keeps the architecture fixed and only changes the training objective.

### 4.1 Concept
We keep the same emission model as P1:
- Mamba emits per-step probabilities `p_t = softmax(e_t/τ)`

We add a transition-consistency regularizer that encourages:
- `p_t` to align with the Markov prediction from the previous step: `p_{t-1} A`

### 4.2 Loss Function
Define:
- `CE = CrossEntropy(y_t, e_t)` (standard per-step)
- `A` = transition matrix computed from **training labels only** (same as P1)

Regularizer:
- `R = mean_{t=2..T} KL(p_t || p_{t-1} A)`

Total loss:
- `L = CE + λ * R`

Where:
- `λ` is tuned on validation only.

### 4.3 Implementation Requirements
Add a new trainer script:
- `train/train_p1r_mamba_hmm.py`

Constraints:
- Do not change dataset, splits, evaluator, decode path.
- Reuse `models/tiny_mamba_emission.py` and `models/hmm_decode.py`.

Artifacts:
- Save `lambda`, `tau`, and `epsilon` in config.
- Save both raw and decoded metrics.

### 4.4 Hyperparameter Plan (Minimal, Pre-Registered)
- Fix P1 best config from Stage A (d_model, layers, dropout, lr, wd).
- Sweep only:
  - `λ ∈ {0.0, 0.1, 0.3, 1.0}`
- If Stage A diagnostics show calibration issues, optionally tune:
  - `τ ∈ {0.7, 1.0, 1.3, 1.6}` (validation only)
  - `ε ∈ {0.0, 0.01, 0.02}` (validation only)

Selection rule:
- Select best config by **validation decoded Macro-F1**.
- Evaluate only the selected config on test.

### 4.5 Acceptance Gate to Proceed to P3
Compute:
- `ΔR = MacroF1_decoded(P1-R) - MacroF1_decoded(P1)`

Proceed rules:
- If `ΔR >= +0.01` or `MacroF1_decoded(P1-R) >= MacroF1(RF+HMM) - 0.02`, stop and finalize report.
- If `ΔR < +0.01` AND still behind RF+HMM by > 0.03, proceed to P3.

---

## 5. Stage C: New Method P3 (Markov-Gated Mamba, Lightweight)
P3 introduces a second structural method while controlling parameter growth.

### 5.1 Design Principle
Do **not** implement K separate full Mamba experts.

Instead implement:
- one shared Tiny-Mamba trunk producing `h_t`
- state-conditioned modulation producing state-specific logits contributions

### 5.2 Architecture
Given trunk output `h_t ∈ R^{d_model}`.

Create K=4 state embeddings `s_k ∈ R^{d_gate}`.

Compute a gating modulation for each state:
- `m_k = sigmoid(W_g s_k + b_g) ∈ R^{d_model}`
- state-conditioned representation: `h_t^{(k)} = h_t ⊙ m_k`

State-specific logits:
- `e_t^{(k)} = W_o h_t^{(k)} + b_o`  (output dimension 4)

Combine using Markov posterior weights `γ_t(k)` from HMM forward-backward:
- `e_t = Σ_k γ_t(k) * e_t^{(k)}`
- `p_t = softmax(e_t/τ)`

Decoding:
- Option A (preferred): decode using the same HMM Viterbi on `p_t`
- Option B (optional): argmax without decode as raw metric

### 5.3 Training and Inference Loop
Key constraint: `γ_t(k)` depends on emissions; to keep implementation stable:
- Use an alternating scheme per epoch:
  1. Compute emissions `p_t` from current model
  2. Run forward-backward with fixed `A` to obtain `γ_t`
  3. Compute combined logits and CE loss using `γ_t` as weights
- Do not differentiate through forward-backward in the first implementation.

### 5.4 Implementation Requirements
New model file:
- `models/markov_gated_mamba.py`

New trainer:
- `train/train_p3_markov_gated_mamba.py`

Re-use:
- transition fit from training labels only
- unified evaluator
- same split protocol

### 5.5 Hyperparameter Plan (Strict and Small)
Start from best P1 config:
- `d_model` unchanged
- `n_layers` unchanged

New P3-specific params:
- `d_gate ∈ {8, 16}`
- `gate_init ∈ {neutral, conservative}` (document exact initialization)
- Do not combine P1-R and P3 in the first pass (no extra regularizers)

Budget:
- at most 6 runs for P3.

Proceed to final report after best P3 run.

---

## 6. Final Reporting Requirements (REPORT_NEXT.md)
The agent must produce `REPORT_NEXT.md` containing:
1. Exact split (participant counts; seed)
2. Label mapping JSON and verification notes
3. Diagnostics outputs (histograms, class-1 proba stats)
4. Baseline results table under unified evaluator
5. P1 results (raw vs decoded; per-class; confusion matrix)
6. P1-R results with λ sweep summary and final selected config
7. P3 results if executed (run budget and configs)
8. Decision log: which gates triggered Stage B/C and why
9. Artifacts index: directories and files saved for reproduction

No narrative fluff; strictly technical and reproducible.

---

## 7. Required CLI and Config Conventions
All new scripts must support:
- `--config <path>` or equivalent
- `--seed`
- `--output_dir`

Each run must write:
- `config_resolved.json`
- `metrics.json`
- `predictions.*`
- `label_mapping.json`
- `class_histograms.json`
- `class1_proba_stats.json`

---

## 8. Default Immediate Execution Plan (Next Actions)
1. Run Stage A:
   - P1 full (20 epochs cap)
   - RF+HMM baseline under unified evaluator
2. Produce diagnostics outputs for class 1 collapse
3. If P1 is behind RF+HMM:
   - Run P1-R sweep on λ
4. Only if still behind by >0.03 Macro-F1 after P1-R:
   - Implement and run P3

This is the strict order. Do not skip steps.
