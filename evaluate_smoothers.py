#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate and compare all trained smoothers
Generate final comparison report
"""

import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, classification_report, balanced_accuracy_score

def load_data():
    """Load test data"""
    data_dir = Path('./prepared_data')

    # Load split info
    split_info = joblib.load(data_dir / 'train_test_split.pkl')
    test_mask = split_info['test_mask']

    # Load data
    Y = np.load(data_dir / 'Y_Walmsley2020.npy')
    P = np.load(data_dir / 'P.npy')

    y_test = Y[test_mask]
    P_test = P[test_mask]

    return y_test, P_test

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate a single model"""
    # Get unique labels
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)

    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"{'='*70}")
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")

    # Build result dict with actual label names
    result = {
        'model': model_name,
        'macro_f1': f1_macro,
        'balanced_accuracy': bacc,
    }
    
    # Add per-class F1 scores with correct label names
    for i, label in enumerate(labels):
        if i < len(f1_per_class):
            result[f'f1_{label}'] = f1_per_class[i]

    return result

def main():
    print(f"\n{'='*70}")
    print(f"  Evaluate All Smoother Models")
    print(f"{'='*70}\n")

    try:
        # Load test data
        print("[1/3] Loading test data...")
        y_test, P_test = load_data()
        print(f"  Test samples: {len(y_test)}")
        print(f"  Labels: {np.unique(y_test)}")

        # Load and evaluate all models
        print("\n[2/3] Evaluating models...")
        data_dir = Path('./prepared_data')
        model_dir = Path('./models')
        results = []

        # 1. RF baseline (if exists)
        rf_model_path = model_dir / 'rf_baseline.pkl'
        if rf_model_path.exists():
            print(f"\n[*] Evaluating: RF baseline")
            rf_model = joblib.load(rf_model_path)

            # Load features
            X_feats = pd.read_pickle(data_dir / 'X_feats.pkl').values
            split_info = joblib.load(data_dir / 'train_test_split.pkl')
            X_test = X_feats[split_info['test_mask']]

            y_pred_rf = rf_model.predict(X_test, P_test)
            results.append(evaluate_model(y_test, y_pred_rf, 'RF'))

        # 2. RF+HMM (if exists)
        rf_hmm_path = model_dir / 'rf_hmm_baseline.pkl'
        if rf_hmm_path.exists():
            print(f"\n[*] Evaluating: RF+HMM")
            rf_hmm_model = joblib.load(rf_hmm_path)

            X_feats = pd.read_pickle(data_dir / 'X_feats.pkl').values
            split_info = joblib.load(data_dir / 'train_test_split.pkl')
            X_test = X_feats[split_info['test_mask']]

            y_pred_rf_hmm = rf_hmm_model.predict(X_test, P_test)
            results.append(evaluate_model(y_test, y_pred_rf_hmm, 'RF+HMM [BASELINE]'))

        # 3. ESN models
        for esn_file in sorted(model_dir.glob('esn_*.pkl')):
            exp_id = esn_file.stem.replace('esn_', '')
            print(f"\n[*] Evaluating: ESN-{exp_id}")

            try:
                esn_model = joblib.load(esn_file)

                # Load RF probabilities
                y_test_proba = np.load(data_dir / 'y_test_proba_rf.npy')

                # No auxiliary features for simplicity
                aux_test = None

                y_pred_esn = esn_model.predict(y_test_proba, P_test, aux_test)
                results.append(evaluate_model(y_test, y_pred_esn, f'ESN-{exp_id}'))
            except Exception as e:
                print(f"  [ERROR] Failed to evaluate ESN-{exp_id}: {e}")
                traceback.print_exc()

        # 4. Mamba models
        for mamba_file in sorted(model_dir.glob('mamba_*.pkl')):
            exp_id = mamba_file.stem.replace('mamba_', '')
            print(f"\n[*] Evaluating: Mamba-{exp_id}")

            try:
                mamba_model = joblib.load(mamba_file)

                # Load RF probabilities
                y_test_proba = np.load(data_dir / 'y_test_proba_rf.npy')

                # No auxiliary features for simplicity
                aux_test = None

                y_pred_mamba = mamba_model.predict(y_test_proba, P_test, aux_test)
                results.append(evaluate_model(y_test, y_pred_mamba, f'Mamba-{exp_id}'))
            except Exception as e:
                print(f"  [ERROR] Failed to evaluate Mamba-{exp_id}: {e}")
                traceback.print_exc()

        # Generate comparison report
        print("\n[3/3] Generating comparison report...")
        if len(results) == 0:
            print("  [WARN] No trained models found")
            return

        df = pd.DataFrame(results)
        df = df.sort_values('macro_f1', ascending=False)

        # Save results
        results_dir = Path('./results')
        results_dir.mkdir(exist_ok=True, parents=True)
        results_path = results_dir / 'smoother_comparison.csv'
        df.to_csv(results_path, index=False)

        print(f"\n{'='*70}")
        print(f"  Final Results (sorted by Macro F1)")
        print(f"{'='*70}\n")
        
        # Print only key columns to avoid width issues
        display_cols = ['model', 'macro_f1', 'balanced_accuracy']
        print(df[display_cols].to_string(index=False))

        # Highlight best result
        best_model = df.iloc[0]
        print(f"\n{'='*70}")
        print(f"[BEST] Best Model: {best_model['model']}")
        print(f"       Macro F1: {best_model['macro_f1']:.4f}")

        # Compare with HMM
        hmm_rows = df[df['model'].str.contains('HMM')]
        if len(hmm_rows) > 0:
            hmm_f1 = hmm_rows.iloc[0]['macro_f1']
            if best_model['macro_f1'] > hmm_f1:
                improvement = (best_model['macro_f1'] - hmm_f1) * 100
                print(f"       [OK] Beats HMM: +{improvement:.2f}% (absolute: +{best_model['macro_f1'] - hmm_f1:.4f})")
            else:
                gap = hmm_f1 - best_model['macro_f1']
                print(f"       [WARN] Did not beat HMM (gap: {gap:.4f})")

        print(f"{'='*70}\n")
        print(f"[OK] Results saved: {results_path}")

    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
