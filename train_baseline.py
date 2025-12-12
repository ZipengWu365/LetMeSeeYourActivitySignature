#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练基线模型 (RF / RF+HMM)
用于建立walking状态识别的baseline性能
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from classifier import Classifier
from sklearn.metrics import f1_score, classification_report

def main():
    parser = argparse.ArgumentParser(description='训练基线模型')
    parser.add_argument('--model', type=str, default='rf_hmm', 
                       choices=['rf', 'rf_hmm', 'xgb', 'xgb_hmm'],
                       help='模型类型')
    parser.add_argument('--data_dir', type=str, 
                       default=str(project_root / 'prepared_data'),
                       help='数据目录')
    parser.add_argument('--output_dir', type=str,
                       default=str(Path(__file__).parent / 'models'),
                       help='模型输出目录')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  训练基线模型: {args.model}")
    print(f"{'='*70}\n")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载数据
    print("[1/4] 加载数据...")
    data_dir = Path(args.data_dir)
    
    X_feats = pd.read_pickle(data_dir / 'X_feats.pkl').values
    Y = np.load(data_dir / 'Y_Walmsley2020.npy')
    P = np.load(data_dir / 'P.npy')
    
    print(f"  数据形状: X={X_feats.shape}, Y={Y.shape}, P={P.shape}")
    print(f"  类别分布: {dict(zip(*np.unique(Y, return_counts=True)))}")
    
    # 划分训练/测试集 (按participant)
    print("\n[2/4] 划分数据集...")
    unique_participants = np.unique(P)
    n_train = 101  # 前101人训练
    
    train_participants = unique_participants[:n_train]
    test_participants = unique_participants[n_train:]
    
    train_mask = np.isin(P, train_participants)
    test_mask = np.isin(P, test_participants)
    
    X_train = X_feats[train_mask]
    y_train = Y[train_mask]
    P_train = P[train_mask]
    
    X_test = X_feats[test_mask]
    y_test = Y[test_mask]
    P_test = P[test_mask]
    
    print(f"  训练集: {X_train.shape[0]} samples, {len(train_participants)} participants")
    print(f"  测试集: {X_test.shape[0]} samples, {len(test_participants)} participants")
    
    # 训练模型
    print(f"\n[3/4] 训练 {args.model} 模型...")
    model = Classifier(args.model, verbose=1)
    model.fit(X_train, y_train, P_train)
    
    # 评估
    print("\n[4/4] 评估模型...")
    y_pred = model.predict(X_test, P_test)
    
    # 计算指标
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    print("\n" + "="*70)
    print(f"  {args.model.upper()} 测试集结果")
    print("="*70)
    print(classification_report(y_test, y_pred))
    print(f"\nMacro F1: {f1_macro:.4f}")
    print(f"Per-class F1: {f1_per_class}")
    
    # 保存模型
    model_path = output_dir / f'{args.model}_baseline.pkl'
    joblib.dump(model, model_path)
    print(f"\n[OK] 模型已保存: {model_path}")
    
    # 生成RF概率输出 (用于训练ESN/Mamba)
    if 'rf' in args.model.lower():
        print("\n[*] 生成RF概率输出...")
        
        # 训练集概率
        y_train_proba = model.window_classifier.predict_proba(X_train)
        proba_train_path = output_dir.parent / 'prepared_data' / 'y_train_proba_rf.npy'
        proba_train_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(proba_train_path, y_train_proba)
        print(f"  训练集概率: {proba_train_path}")
        
        # 测试集概率
        y_test_proba = model.window_classifier.predict_proba(X_test)
        proba_test_path = output_dir.parent / 'prepared_data' / 'y_test_proba_rf.npy'
        np.save(proba_test_path, y_test_proba)
        print(f"  测试集概率: {proba_test_path}")
        
        # 保存划分信息
        split_info = {
            'train_mask': train_mask,
            'test_mask': test_mask,
            'train_participants': train_participants,
            'test_participants': test_participants,
        }
        split_path = output_dir.parent / 'prepared_data' / 'train_test_split.pkl'
        joblib.dump(split_info, split_path)
        print(f"  划分信息: {split_path}")
    
    print(f"\n{'='*70}")
    print(f"  [OK] 训练完成! Macro F1 = {f1_macro:.4f}")
    print(f"{'='*70}\n")
    
    return f1_macro

if __name__ == '__main__':
    main()

