#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估和对比所有训练好的平滑器
生成最终对比报告
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def load_data():
    """加载测试数据"""
    data_dir = Path('./prepared_data')
    
    # 加载划分信息
    split_info = joblib.load(data_dir / 'train_test_split.pkl')
    test_mask = split_info['test_mask']
    
    # 加载数据
    Y = np.load(data_dir / 'Y_Walmsley2020.npy')
    P = np.load(data_dir / 'P.npy')
    
    y_test = Y[test_mask]
    P_test = P[test_mask]
    
    return y_test, P_test

def evaluate_model(y_true, y_pred, model_name):
    """评估单个模型"""
    from sklearn.metrics import balanced_accuracy_score
    
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None)
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"{'='*70}")
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    
    return {
        'model': model_name,
        'macro_f1': f1_macro,
        'balanced_accuracy': bacc,
        'f1_sleep': f1_per_class[0] if len(f1_per_class) > 0 else 0,
        'f1_sedentary': f1_per_class[1] if len(f1_per_class) > 1 else 0,
        'f1_light': f1_per_class[2] if len(f1_per_class) > 2 else 0,
        'f1_mvpa': f1_per_class[3] if len(f1_per_class) > 3 else 0,
    }

def main():
    print(f"\n{'='*70}")
    print(f"  评估所有平滑器模型")
    print(f"{'='*70}\n")
    
    # 加载测试数据
    print("[1/3] 加载测试数据...")
    y_test, P_test = load_data()
    print(f"  测试集样本数: {len(y_test)}")
    
    # 加载所有模型并评估
    print("\n[2/3] 评估模型...")
    data_dir = Path('./prepared_data')
    model_dir = Path('./models')
    results = []
    
    # 1. RF baseline (if exists)
    rf_model_path = model_dir / 'rf_baseline.pkl'
    if rf_model_path.exists():
        print(f"\n[*] 评估: RF baseline")
        rf_model = joblib.load(rf_model_path)
        
        # 加载特征
        X_feats = pd.read_pickle(data_dir / 'X_feats.pkl').values
        split_info = joblib.load(data_dir / 'train_test_split.pkl')
        X_test = X_feats[split_info['test_mask']]
        
        y_pred_rf = rf_model.predict(X_test, P_test)
        results.append(evaluate_model(y_test, y_pred_rf, 'RF'))
    
    # 2. RF+HMM (if exists)
    rf_hmm_path = model_dir / 'rf_hmm_baseline.pkl'
    if rf_hmm_path.exists():
        print(f"\n[*] 评估: RF+HMM")
        rf_hmm_model = joblib.load(rf_hmm_path)
        
        X_feats = pd.read_pickle(data_dir / 'X_feats.pkl').values
        split_info = joblib.load(data_dir / 'train_test_split.pkl')
        X_test = X_feats[split_info['test_mask']]
        
        y_pred_rf_hmm = rf_hmm_model.predict(X_test, P_test)
        results.append(evaluate_model(y_test, y_pred_rf_hmm, 'RF+HMM [BASELINE]'))
    
    # 3. ESN models
    for esn_file in model_dir.glob('esn_*.pkl'):
        exp_id = esn_file.stem.replace('esn_', '')
        print(f"\n[*] 评估: ESN-{exp_id}")
        
        esn_model = joblib.load(esn_file)
        
        # 加载RF概率
        y_test_proba = np.load(data_dir / 'y_test_proba_rf.npy')
        
        # 加载辅助特征 (如果需要)
        # 简化版本: 暂时不使用辅助特征
        aux_test = None
        
        y_pred_esn = esn_model.predict(y_test_proba, P_test, aux_test)
        results.append(evaluate_model(y_test, y_pred_esn, f'ESN-{exp_id}'))
    
    # 生成对比报告
    print("\n[3/3] 生成对比报告...")
    if len(results) == 0:
        print("  [WARN] 没有找到任何训练好的模型")
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values('macro_f1', ascending=False)
    
    # 保存结果
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True, parents=True)
    results_path = results_dir / 'smoother_comparison.csv'
    df.to_csv(results_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"  [*] 最终结果对比 (按Macro F1排序)")
    print(f"{'='*70}\n")
    print(df.to_string(index=False))
    
    # 高亮最优结果
    best_model = df.iloc[0]
    print(f"\n{'='*70}")
    print(f"[BEST] 最优模型: {best_model['model']}")
    print(f"       Macro F1: {best_model['macro_f1']:.4f}")
    
    # 与HMM对比
    hmm_rows = df[df['model'].str.contains('HMM')]
    if len(hmm_rows) > 0:
        hmm_f1 = hmm_rows.iloc[0]['macro_f1']
        if best_model['macro_f1'] > hmm_f1:
            improvement = (best_model['macro_f1'] - hmm_f1) * 100
            print(f"       [OK] 超越HMM: +{improvement:.2f}% (绝对提升: +{best_model['macro_f1'] - hmm_f1:.4f})")
        else:
            gap = hmm_f1 - best_model['macro_f1']
            print(f"       [WARN] 未超越HMM (差距: {gap:.4f})")
    
    print(f"{'='*70}\n")
    print(f"[OK] 结果已保存: {results_path}")

if __name__ == '__main__':
    main()
