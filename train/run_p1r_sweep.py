#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1-R Hyperparameter Sweep

Executes the λ sweep for HMM-regularized training.

Design: Section 4.4 of next_step_experiment_design_hmm_based_mamba.md
- λ ∈ {0.0, 0.1, 0.3, 1.0}
- All other hyperparams fixed at P1 best config (d_model=16, n_layers=1, etc.)
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent

def run_p1r_experiment(lambda_val: float, output_base: Path):
    """Run P1-R experiment with given λ"""
    
    config_name = f"p1r_lambda_{lambda_val:.1f}".replace('.', '_')
    output_dir = output_base / config_name
    output_dir.mkdir(exist_ok=True, parents=True)
    
    cmd = [
        str(Path(__file__).parent.parent / '.conda' / 'bin' / 'python3') 
            if not sys.executable.endswith('python3')
            else sys.executable,
        str(project_root / 'train' / 'train_p1r_mamba_hmm.py'),
        '--output_dir', str(output_dir),
        '--lambda', str(lambda_val),
        '--tau', '1.0',
        '--epsilon', '0.0',
        '--seed', '42',
    ]
    
    print(f"\n[*] Running P1-R with λ={lambda_val}")
    print(f"  Output dir: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"  ✓ Completed successfully")
            return True, output_dir
        else:
            print(f"  ✗ Failed with return code {result.returncode}")
            print("  STDOUT:", result.stdout[-500:] if result.stdout else "")
            print("  STDERR:", result.stderr[-500:] if result.stderr else "")
            return False, output_dir
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timed out after 3600 seconds")
        return False, output_dir
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False, output_dir


def extract_metrics(output_dir: Path) -> dict:
    """Extract best metrics from output"""
    metrics_file = output_dir / 'test_results' / 'metrics.json'
    
    if not metrics_file.exists():
        return None
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    return {
        'raw_macro_f1': metrics['raw']['macro_f1'],
        'decoded_macro_f1': metrics['decoded']['macro_f1'],
        'per_class_f1_raw': metrics['raw'].get('f1_class_1', None),  # Class 1
        'per_class_f1_decoded': metrics['decoded'].get('f1_class_1', None),
    }


def main():
    print("=" * 70)
    print("P1-R HYPERPARAMETER SWEEP: λ sweep")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Setup
    output_base = project_root / 'artifacts' / 'p1r_experiments'
    output_base.mkdir(exist_ok=True, parents=True)
    
    # Lambda values to sweep (from Section 4.4)
    lambda_values = [0.0, 0.1, 0.3, 1.0]
    
    results = {}
    
    # Run experiments
    for lambda_val in lambda_values:
        success, output_dir = run_p1r_experiment(lambda_val, output_base)
        
        if success:
            metrics = extract_metrics(output_dir)
            if metrics:
                results[lambda_val] = {
                    'output_dir': str(output_dir),
                    'metrics': metrics,
                }
                print(f"    Raw F1: {metrics['raw_macro_f1']:.4f}, Decoded F1: {metrics['decoded_macro_f1']:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    
    if not results:
        print("✗ No successful runs")
        return
    
    # Find best config
    best_lambda = max(results.keys(), 
                     key=lambda l: results[l]['metrics']['decoded_macro_f1'])
    best_metrics = results[best_lambda]['metrics']
    
    print(f"\nBest λ: {best_lambda}")
    print(f"  Raw Macro F1:     {best_metrics['raw_macro_f1']:.4f}")
    print(f"  Decoded Macro F1: {best_metrics['decoded_macro_f1']:.4f}")
    print(f"  Class 1 F1:       {best_metrics['per_class_f1_decoded']:.4f}")
    
    print(f"\nAll results:")
    for lambda_val in sorted(results.keys()):
        m = results[lambda_val]['metrics']
        print(f"  λ={lambda_val:.1f}: Raw F1={m['raw_macro_f1']:.4f}, Decoded F1={m['decoded_macro_f1']:.4f}, Class1={m['per_class_f1_decoded']:.4f}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'sweep_params': {'lambda_values': lambda_values},
        'results': results,
        'best_lambda': best_lambda,
        'best_metrics': best_metrics,
    }
    
    with open(output_base / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Sweep summary saved to {output_base / 'sweep_summary.json'}")


if __name__ == '__main__':
    main()
