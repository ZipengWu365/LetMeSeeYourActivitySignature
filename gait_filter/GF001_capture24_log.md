# GF001 MiniRocket Walking Filter

## Dataset
- Total filtered windows: 895299
- Train participants (approx): 120
- Train windows used: 16000 (pos 4000, neg 12000)
- Test windows: 184036 (pos 11923, neg 172113)

## Metrics
- PR AUC: 0.506
- Precision at recall ¡Ý 0.70: 0.358
- Hard-negative PR AUC: 0.708
- Hard-negative precision at recall ¡Ý 0.70: 0.584

## Artifacts
- Confusion matrix: artifacts\gait_filter\plots\confusion_matrix.png
- PR curve: artifacts\gait_filter\plots\precision_recall_curve.png
- Calibration: artifacts\gait_filter\plots\calibration_curve.png
- Hard-negative PR: artifacts\gait_filter\plots\precision_recall_hard.png
- Model path: artifacts\gait_filter\models\gait_filter.pkl (0.10 MB)
- Inference time: 7.0635 s per 1000 windows

## JSON summary
```json
{
  "project_id": "GF001",
  "train_windows": 16000,
  "test_windows": 184036,
  "train_positive": 4000,
  "train_negative": 12000,
  "test_positive": 11923,
  "test_negative": 172113,
  "precision_recall": {
    "pr_auc": 0.5056974326053674,
    "precision_at_recall_ge_0.70": 0.3578839063705736
  },
  "hard_negative_precision_recall": {
    "pr_auc": 0.7080552421097579,
    "precision_at_recall_ge_0.70": 0.5839094431360632
  },
  "confusion_matrix_row_normalized": [
    [
      0.9287619180422164,
      0.07123808195778356
    ],
    [
      0.34932483435377004,
      0.65067516564623
    ]
  ],
  "model_size_mb": 0.0998697280883789,
  "seconds_per_1000_windows": 7.0634606999810785
}
```