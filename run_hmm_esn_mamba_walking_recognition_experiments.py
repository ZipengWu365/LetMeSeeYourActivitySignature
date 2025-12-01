#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行实验Pipeline - Walking State Recognition
全自动对比 HMM vs ESN vs Mamba

硬件要求:
- GPU: NVIDIA 3080Ti (12GB) 或更好 [OK] 你的配置满足
- RAM: 32GB [OK] 你的配置满足
- OS: Windows/Linux with CUDA 11.8+

运行方式:
    python run_hmm_esn_mamba_walking_recognition_experiments.py --mode quick        # 快速模式 (仅ESN)
    python run_hmm_esn_mamba_walking_recognition_experiments.py --mode standard     # 标准模式 (ESN + Mamba-Light)
    python run_hmm_esn_mamba_walking_recognition_experiments.py --mode full         # 完整模式 (所有实验)

作者: AI Assistant
日期: 2024-12
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import json

# === 配置 ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'prepared_data'
MODEL_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

# 确保目录存在
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 实验配置
EXPERIMENTS = {
    'quick': {
        'name': '快速模式 (推荐首次运行)',
        'duration': '~20分钟',
        'experiments': ['E1_HMM', 'E2_ESN_medium', 'E3_ESN_large'],
        'description': '复现HMM基线 + 运行2个ESN配置,确保快速超越HMM'
    },
    'standard': {
        'name': '标准模式',
        'duration': '~40分钟',
        'experiments': ['E1_HMM', 'E2_ESN_medium', 'E3_ESN_large', 'M2_Mamba_light'],
        'description': '包含ESN和轻量Mamba,平衡速度与性能'
    },
    'full': {
        'name': '完整模式',
        'duration': '~2小时',
        'experiments': [
            'E1_HMM', 
            'E2_ESN_medium', 'E3_ESN_large', 'E4_ESN_full',
            'M1_Mamba_light_noaux', 'M2_Mamba_light', 'M3_Mamba_medium'
        ],
        'description': '运行所有实验组,获得完整对比'
    }
}

# === 工具函数 ===
def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def run_command(cmd, description, critical=True):
    """
    运行命令并处理错误
    
    Args:
        cmd: 命令列表
        description: 描述
        critical: 是否关键步骤 (失败则退出)
    """
    print(f"[*] {description}...")
    print(f"    Command: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"    [OK] 成功 (耗时: {elapsed:.1f}秒)")
        if result.stdout:
            print(f"    输出: {result.stdout[:200]}")  # 前200字符
        return True
    else:
        print(f"    [FAIL] 失败 (returncode: {result.returncode})")
        print(f"    错误: {result.stderr}")
        if critical:
            print("\n[!] 关键步骤失败,终止实验")
            sys.exit(1)
        return False

def check_environment():
    """检查环境依赖"""
    print_section("Phase 0: 环境检查")
    
    # 检查Python版本
    py_version = sys.version_info
    print(f"[*] Python版本: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print("    [WARN] 建议Python 3.8+")
    
    # 检查GPU
    try:
        import torch
        print(f"[*] PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"    [OK] GPU可用: {torch.cuda.get_device_name(0)}")
            print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("    [WARN] GPU不可用,将使用CPU (ESN仍可运行,Mamba会较慢)")
    except ImportError:
        print("    [FAIL] PyTorch未安装")
        return False
    
    # 检查关键包
    required_packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'scipy': 'SciPy',
    }
    
    missing = []
    for pkg, name in required_packages.items():
        try:
            __import__(pkg)
            print(f"    [OK] {name}")
        except ImportError:
            print(f"    [FAIL] {name} 未安装")
            missing.append(pkg)
    
    # 检查Mamba (可选)
    try:
        import mamba_ssm
        print(f"    [OK] Mamba-SSM (版本: {mamba_ssm.__version__ if hasattr(mamba_ssm, '__version__') else '未知'})")
    except ImportError:
        print("    [WARN] Mamba-SSM未安装 (仅ESN + HMM可运行)")
    
    if missing:
        print(f"\n[!] 缺少依赖: {', '.join(missing)}")
        print(f"安装命令: pip install {' '.join(missing)}")
        return False
    
    # 检查数据
    data_files = ['X_feats.pkl', 'Y_Walmsley2020.npy', 'P.npy']
    for f in data_files:
        path = DATA_DIR / f
        if path.exists():
            print(f"    [OK] 数据文件: {f}")
        else:
            print(f"    [FAIL] 数据文件缺失: {f}")
            return False
    
    return True

def install_dependencies():
    """安装缺失依赖"""
    print_section("安装依赖")
    
    packages = [
        'torch>=2.0',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'joblib',
        'matplotlib',
        'statsmodels',
    ]
    
    print("[*] 安装基础依赖...")
    run_command(
        [sys.executable, '-m', 'pip', 'install'] + packages,
        "安装基础包",
        critical=False
    )
    
    # 尝试安装Mamba (可选)
    print("\n[*] 尝试安装Mamba-SSM (需要CUDA 11.8+)...")
    mamba_success = run_command(
        [sys.executable, '-m', 'pip', 'install', 'mamba-ssm', 'causal-conv1d>=1.1.0'],
        "安装Mamba-SSM",
        critical=False
    )
    
    if not mamba_success:
        print("    [INFO] Mamba安装失败,将仅运行ESN实验")
        print("    如需运行Mamba,请手动安装: pip install mamba-ssm")

def run_experiment_group(exp_id):
    """
    运行单个实验组
    
    Args:
        exp_id: 实验ID (如 'E2_ESN_medium')
    """
    print(f"\n{'─'*70}")
    print(f"运行实验: {exp_id}")
    print(f"{'─'*70}")
    
    if exp_id == 'E1_HMM':
        # Phase 1: 训练RF+HMM基线
        script_path = Path(__file__).parent / 'train_baseline.py'
        run_command(
            [sys.executable, str(script_path), '--model', 'rf_hmm'],
            f"训练 {exp_id}",
        )
    
    elif exp_id.startswith('E') and 'ESN' in exp_id:
        # ESN实验
        config_map = {
            'E2_ESN_medium': ('800', '0.9'),
            'E3_ESN_large': ('1000', '0.95'),
            'E4_ESN_full': ('800', '0.9'),
        }
        
        n_res, spec_rad = config_map.get(exp_id, ('800', '0.9'))
        aux_features = 'full' if 'full' in exp_id else 'enmo'
        
        script_path = Path(__file__).parent / 'train_esn_smoother.py'
        run_command(
            [
                sys.executable, str(script_path),
                '--exp_id', exp_id,
                '--n_reservoir', n_res,
                '--spectral_radius', spec_rad,
                '--aux_features', aux_features,
            ],
            f"训练 {exp_id}",
        )
    
    elif exp_id.startswith('M'):
        # Mamba实验
        config_map = {
            'M1_Mamba_light_noaux': ('64', '2', 'none'),
            'M2_Mamba_light': ('64', '2', 'enmo'),
            'M3_Mamba_medium': ('128', '3', 'enmo'),
        }
        
        d_model, n_layers, aux_features = config_map.get(exp_id, ('64', '2', 'enmo'))
        
        script_path = Path(__file__).parent / 'train_mamba_smoother.py'
        run_command(
            [
                sys.executable, str(script_path),
                '--exp_id', exp_id,
                '--d_model', d_model,
                '--n_layers', n_layers,
                '--aux_features', aux_features,
                '--epochs', '50',
            ],
            f"训练 {exp_id}",
        )

def evaluate_all():
    """评估所有训练好的模型"""
    print_section("Phase 4: 模型评估")
    
    script_path = Path(__file__).parent / 'evaluate_smoothers.py'
    run_command(
        [sys.executable, str(script_path)],
        "评估所有模型并生成对比报告",
    )

def generate_report():
    """生成最终报告"""
    print_section("生成最终报告")
    
    results_file = RESULTS_DIR / 'smoother_comparison.csv'
    if not results_file.exists():
        print("    [WARN] 结果文件不存在,跳过报告生成")
        return
    
    import pandas as pd
    df = pd.read_csv(results_file)
    
    print("\n" + "="*70)
    print("  [*] 最终结果对比 (按Macro F1排序)")
    print("="*70)
    print(df.sort_values('macro_f1', ascending=False).to_string(index=False))
    
    # 高亮最优结果
    best_model = df.loc[df['macro_f1'].idxmax()]
    print(f"\n[BEST] 最优模型: {best_model['model']}")
    print(f"   Macro F1: {best_model['macro_f1']:.4f}")
    
    # 检查是否超越HMM
    hmm_f1 = df[df['model'].str.contains('HMM')]['macro_f1'].values
    if len(hmm_f1) > 0:
        hmm_baseline = hmm_f1[0]
        if best_model['macro_f1'] > hmm_baseline:
            improvement = (best_model['macro_f1'] - hmm_baseline) * 100
            print(f"   [OK] 超越HMM: +{improvement:.2f}% (绝对提升: +{best_model['macro_f1'] - hmm_baseline:.4f})")
        else:
            print(f"   [FAIL] 未超越HMM (差距: {hmm_baseline - best_model['macro_f1']:.4f})")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Walking State Recognition - 一键运行实验Pipeline'
    )
    parser.add_argument(
        '--mode',
        choices=['quick', 'standard', 'full'],
        default='quick',
        help='实验模式 (默认: quick)'
    )
    parser.add_argument(
        '--skip-env-check',
        action='store_true',
        help='跳过环境检查 (不推荐)'
    )
    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='自动安装缺失依赖'
    )
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("\n" + "="*70)
    print("  [*] Walking State Recognition - 自动化实验Pipeline")
    print("="*70)
    print(f"模式: {EXPERIMENTS[args.mode]['name']}")
    print(f"预计时间: {EXPERIMENTS[args.mode]['duration']}")
    print(f"说明: {EXPERIMENTS[args.mode]['description']}")
    print("="*70)
    
    # 环境检查
    if not args.skip_env_check:
        if not check_environment():
            if args.install_deps:
                install_dependencies()
                if not check_environment():
                    print("\n[!] 环境检查仍未通过,请手动解决依赖问题")
                    sys.exit(1)
            else:
                print("\n[!] 环境检查未通过")
                print("建议运行: python run_all_experiments.py --install-deps")
                sys.exit(1)
    
    # 运行实验
    experiments = EXPERIMENTS[args.mode]['experiments']
    total_start = time.time()
    
    for i, exp_id in enumerate(experiments, 1):
        print(f"\n{'█'*70}")
        print(f"  进度: {i}/{len(experiments)} - {exp_id}")
        print(f"{'█'*70}")
        
        try:
            run_experiment_group(exp_id)
        except Exception as e:
            print(f"\n[!] 实验 {exp_id} 失败: {e}")
            print("继续下一个实验...")
    
    # 评估
    evaluate_all()
    
    # 生成报告
    generate_report()
    
    # 总结
    total_time = time.time() - total_start
    print(f"\n\n{'='*70}")
    print(f"  [OK] 所有实验完成!")
    print(f"  总耗时: {total_time/60:.1f} 分钟")
    print(f"  结果保存在: {RESULTS_DIR}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
