#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P3: Markov-Gated Mamba + HMM Decode

Lightweight structural extension with state-conditioned modulation.

Design: Section 5 of next_step_experiment_design_hmm_based_mamba.md
"""

import argparse
import sys
import json
import yaml
import importlib.util
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load markov_gated_mamba
spec = importlib.util.spec_from_file_location("markov_gated_mamba", 
    project_root / "models" / "markov_gated_mamba.py")
mgm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mgm_module)
MarkovGatedMamba = mgm_module.MarkovGatedMamba
create_markov_gated_mamba = mgm_module.create_markov_gated_mamba

# Load hmm_decode
spec = importlib.util.spec_from_file_location("hmm_decode", project_root / "models" / "hmm_decode.py")
hmm_decode_module = importlib.util.module_from_spec(spec)
sys.modules['hmm'] = __import__('hmm', fromlist=['HMM'])
sys.modules['utils'] = __import__('utils', fromlist=['ordered_unique'])
spec.loader.exec_module(hmm_decode_module)
HMMDecoder = hmm_decode_module.HMMDecoder
create_hmm_decoder = hmm_decode_module.create_hmm_decoder

# Load evaluator
spec = importlib.util.spec_from_file_location("evaluate_sequence_labeling",
    project_root / "evaluation" / "evaluate_sequence_labeling.py")
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)
create_evaluator = eval_module.create_evaluator


class SequenceDataset(Dataset):
    """Dataset for grouped sequences"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            groups: Group identifiers (n_samples,)
        """
        self.X = X
        self.y = y
        self.groups = groups
        
        # Get unique groups and build sequences
        self.unique_groups = np.unique(groups)
        self.sequences = []
        self.targets = []
        
        for group in self.unique_groups:
            mask = groups == group
            X_g = X[mask]
            y_g = y[mask]
            
            if y_g.dtype.kind not in ['i', 'u']:
                raise ValueError(f"Expected integer labels, got {y_g.dtype}")
            
            self.sequences.append(torch.from_numpy(X_g).float())
            self.targets.append(torch.from_numpy(y_g).long())
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> tuple:
        return self.sequences[idx], self.targets[idx]


class P3MarkovGatedMambaTrainer:
    """
    Trainer for P3: Markov-Gated Mamba + HMM Decode
    
    Key features:
    - State-conditioned modulation of shared Mamba trunk
    - Alternating training: compute γ_t, then update with γ_t as fixed weights
    - Standard CE loss (no additional regularizers)
    - HMM decoding in inference
    
    Args:
        config: Configuration dictionary
        device: Torch device
        output_dir: Output directory
    """
    
    def __init__(
        self,
        config: dict,
        device: torch.device = None,
        output_dir: Path = None,
    ):
        self.config = config
        self.device = device or torch.device('cpu')
        self.output_dir = Path(output_dir) if output_dir else Path('./artifacts/p3_markov')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get P3-specific params
        self.d_gate = config.get('d_gate', 8)
        self.gate_init = config.get('gate_init', 'neutral')
        
        # Initialize model
        self.model = create_markov_gated_mamba(
            input_dim=config.get('input_dim', 32),
            n_classes=config.get('n_classes', 4),
            d_model=config.get('d_model', 16),
            n_layers=config.get('n_layers', 1),
            d_gate=self.d_gate,
            dropout=config.get('dropout', 0.2),
            device=self.device,
        )
        
        self.decoder = create_hmm_decoder(n_classes=config.get('n_classes', 4))
        self.evaluator = create_evaluator(n_classes=config.get('n_classes', 4))
        
        # Transition matrix (will be set during training)
        self.transition_matrix = None
        self.forward_backward_alpha = None
        self.forward_backward_beta = None
        self.gamma = None
        self.n_classes = config.get('n_classes', 4)
        
        # Training state
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_raw_f1': [],
            'val_decoded_f1': [],
        }
        self.best_val_decoded_f1 = -np.inf
        self.best_epoch = 0
        self.patience_counter = 0
        
        self.log_file = self.output_dir / 'training.log'
    
    def _log(self, msg: str):
        """Log message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def _compute_forward_backward(
        self,
        proba: np.ndarray,
        groups: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Markov posterior γ_t(k) using forward-backward
        
        Args:
            proba: Emission probabilities (n_samples, n_classes)
            groups: Group identifiers (n_samples,)
        
        Returns:
            gamma: Posterior weights (n_samples, n_classes)
        """
        unique_groups = np.unique(groups)
        gamma_all = np.zeros_like(proba)
        
        for group in unique_groups:
            mask = groups == group
            proba_g = proba[mask]
            
            # Run forward-backward with fixed A
            # Using sklearn HMM forward_pass and backward_pass
            A = self.transition_matrix  # (n_classes, n_classes)
            start_prob = np.ones(self.n_classes) / self.n_classes
            
            # Forward pass
            alpha = np.zeros_like(proba_g)
            alpha[0] = start_prob * proba_g[0]
            
            for t in range(1, len(proba_g)):
                alpha[t] = proba_g[t] * (alpha[t-1] @ A)
            
            # Backward pass
            beta = np.zeros_like(proba_g)
            beta[-1] = 1.0
            
            for t in range(len(proba_g) - 2, -1, -1):
                beta[t] = (beta[t+1] * proba_g[t+1]) @ A.T
            
            # Posterior: γ_t(k) = α_t(k) β_t(k) / Z_t
            gamma_g = (alpha * beta)
            gamma_g /= (gamma_g.sum(axis=1, keepdims=True) + 1e-8)
            
            gamma_all[mask] = gamma_g
        
        return gamma_all
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        groups_val: np.ndarray,
    ) -> dict:
        """Train P3 model"""
        self._log("\n" + "="*70)
        self._log("P3 TRAINING: Markov-Gated Mamba + HMM Decode")
        self._log("="*70)
        
        # Label mapping
        label_to_idx = None
        if y_train.dtype.kind in ['U', 'S', 'O']:
            unique_labels = np.unique(y_train)
            label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            self._log(f"\nLabel mapping: {label_to_idx}")
            y_train_idx = np.array([label_to_idx[label] for label in y_train])
            y_val_idx = np.array([label_to_idx[label] for label in y_val])
        else:
            y_train_idx = y_train
            y_val_idx = y_val
        
        # Fit HMM transition matrix
        self._log("\n[1/3] Fitting HMM transition matrix on training labels...")
        self.decoder.fit_transition(y_train_idx, groups_train)
        self.transition_matrix = self.decoder.transition_matrix
        self._log(f"  Transition matrix shape: {self.transition_matrix.shape}")
        
        # Datasets
        train_dataset = SequenceDataset(X_train, y_train_idx, groups_train)
        val_dataset = SequenceDataset(X_val, y_val_idx, groups_val)
        
        self._log(f"  Training sequences: {len(train_dataset)}")
        self._log(f"  Validation sequences: {len(val_dataset)}")
        
        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('lr', 3e-4),
            weight_decay=self.config.get('weight_decay', 1e-4),
        )
        criterion = nn.CrossEntropyLoss()
        
        epochs = self.config.get('epochs', 20)
        early_stopping_patience = self.config.get('early_stopping_patience', 5)
        gradient_clip = self.config.get('gradient_clip', 1.0)
        
        self._log(f"\n[2/3] Training Markov-Gated Mamba...")
        self._log(f"  Epochs: {epochs}")
        self._log(f"  LR: {self.config.get('lr', 3e-4)}")
        self._log(f"  d_gate: {self.d_gate}")
        self._log(f"  gate_init: {self.gate_init}")
        
        # Training loop with alternating scheme
        self.model.train()
        for epoch in range(epochs):
            # 1. Compute gamma from current model
            train_logits = []
            with torch.no_grad():
                for idx in range(len(train_dataset)):
                    X_seq, _ = train_dataset[idx]
                    X_seq = X_seq.to(self.device)
                    logits = self.model(X_seq)
                    train_logits.append(torch.softmax(logits, dim=-1).cpu().numpy())
            
            train_proba = np.concatenate(train_logits)
            gamma_t_train = self._compute_forward_backward(train_proba, groups_train)
            
            # 2. Training step with fixed gamma
            train_loss = self._train_epoch_with_gamma(
                train_dataset, groups_train, gamma_t_train,
                optimizer, criterion, gradient_clip
            )
            
            # 3. Validation step
            val_raw_f1, val_decoded_f1 = self._validate_epoch(val_dataset, y_val_idx, groups_val)
            
            # Log
            self.train_history['epoch'].append(epoch)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_raw_f1'].append(val_raw_f1)
            self.train_history['val_decoded_f1'].append(val_decoded_f1)
            
            self._log(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.4f} | Val Raw F1: {val_raw_f1:.4f} | Val Decoded F1: {val_decoded_f1:.4f}")
            
            # Early stopping
            if val_decoded_f1 > self.best_val_decoded_f1:
                self.best_val_decoded_f1 = val_decoded_f1
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint('best_model')
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    self._log(f"\n  [EARLY STOPPING] No improvement for {early_stopping_patience} epochs")
                    break
        
        # Load best model
        self._load_checkpoint('best_model')
        self._log(f"\n  [OK] Best model from epoch {self.best_epoch+1}, Val Decoded F1: {self.best_val_decoded_f1:.4f}")
        
        self._save_training_history()
        
        return {
            'best_epoch': self.best_epoch,
            'best_val_decoded_f1': self.best_val_decoded_f1,
            'total_epochs': epoch + 1,
        }
    
    def _train_epoch_with_gamma(
        self,
        dataset: SequenceDataset,
        groups_all: np.ndarray,
        gamma_t: np.ndarray,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        gradient_clip: float,
    ) -> float:
        """Train one epoch using fixed γ_t"""
        self.model.train()
        total_loss = 0
        n_samples = 0
        
        # Random shuffle
        indices = np.random.permutation(len(dataset))
        
        start_idx = 0
        for idx in indices:
            X_seq, y_seq = dataset[idx]
            X_seq = X_seq.to(self.device)
            y_seq = y_seq.to(self.device)
            
            # Get gamma for this sequence
            seq_len = len(y_seq)
            gamma_seq = torch.from_numpy(gamma_t[start_idx:start_idx+seq_len]).float().to(self.device)
            start_idx += seq_len
            
            # Forward pass
            optimizer.zero_grad()
            logits = self.model(X_seq, gamma_t=gamma_seq)
            loss = criterion(logits, y_seq)
            
            # Backward
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            
            total_loss += loss.item() * len(y_seq)
            n_samples += len(y_seq)
        
        return total_loss / n_samples if n_samples > 0 else 0
    
    def _validate_epoch(
        self,
        dataset: SequenceDataset,
        y_true_all: np.ndarray,
        groups_all: np.ndarray,
    ) -> tuple:
        """Validation step"""
        self.model.eval()
        
        y_pred_raw_list = []
        y_proba_list = []
        
        with torch.no_grad():
            for idx in range(len(dataset)):
                X_seq, _ = dataset[idx]
                X_seq = X_seq.to(self.device)
                
                logits = self.model(X_seq)
                proba = torch.softmax(logits, dim=-1).cpu().numpy()
                y_pred_raw = np.argmax(proba, axis=-1)
                
                y_pred_raw_list.append(y_pred_raw)
                y_proba_list.append(proba)
        
        y_pred_raw = np.concatenate(y_pred_raw_list)
        y_proba_all = np.concatenate(y_proba_list)
        y_pred_decoded = self.decoder.predict(y_proba_all, groups=groups_all)
        
        from sklearn.metrics import f1_score
        raw_f1 = f1_score(y_true_all, y_pred_raw, average='macro', zero_division=0)
        decoded_f1 = f1_score(y_true_all, y_pred_decoded, average='macro', zero_division=0)
        
        return raw_f1, decoded_f1
    
    def _save_checkpoint(self, name: str = 'best_model'):
        """Save checkpoint"""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, checkpoint_dir / f'{name}.pt')
    
    def _load_checkpoint(self, name: str = 'best_model'):
        """Load checkpoint"""
        checkpoint_path = self.output_dir / 'checkpoints' / f'{name}.pt'
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def _save_training_history(self):
        """Save history"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
    
    def predict(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        groups_test: np.ndarray,
    ) -> dict:
        """Inference on test set"""
        self._log("\n[3/3] Testing on test set...")
        
        # Label mapping
        label_to_idx = None
        if y_test.dtype.kind in ['U', 'S', 'O']:
            unique_labels = np.unique(y_test)
            label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            y_test_idx = np.array([label_to_idx[label] for label in y_test])
        else:
            y_test_idx = y_test
        
        self.model.eval()
        test_dataset = SequenceDataset(X_test, y_test_idx, groups_test)
        
        y_pred_raw_list = []
        y_proba_list = []
        
        with torch.no_grad():
            for idx in range(len(test_dataset)):
                X_seq, _ = test_dataset[idx]
                X_seq = X_seq.to(self.device)
                
                logits = self.model(X_seq)
                proba = torch.softmax(logits, dim=-1).cpu().numpy()
                y_pred_raw = np.argmax(proba, axis=-1)
                
                y_pred_raw_list.append(y_pred_raw)
                y_proba_list.append(proba)
        
        y_pred_raw = np.concatenate(y_pred_raw_list)
        y_proba_all = np.concatenate(y_proba_list)
        y_pred_decoded = self.decoder.predict(y_proba_all, groups=groups_test)
        
        # Evaluate
        results = self.evaluator.evaluate(
            y_test_idx,
            y_pred_raw,
            y_pred_decoded,
            groups=groups_test,
            return_predictions=True,
        )
        
        # Save results
        test_results_dir = self.output_dir / 'test_results'
        test_results_dir.mkdir(exist_ok=True, parents=True)
        
        with open(test_results_dir / 'metrics.json', 'w') as f:
            metrics_serializable = {
                'raw': results['raw'],
                'decoded': results['decoded'],
            }
            json.dump(metrics_serializable, f, indent=2)
        
        predictions_df = pd.DataFrame({
            'y_true': results['predictions']['y_true'],
            'y_pred_raw': results['predictions']['y_pred_raw'],
            'y_pred_decoded': results['predictions']['y_pred_decoded'],
            'groups': results['predictions']['groups'],
        })
        predictions_df.to_csv(test_results_dir / 'predictions.csv', index=False)
        
        self._log(f"\n  Raw Macro F1:     {results['raw']['macro_f1']:.4f}")
        self._log(f"  Decoded Macro F1: {results['decoded']['macro_f1']:.4f}")
        self._log(f"  HMM Gain:         {results['decoded']['macro_f1'] - results['raw']['macro_f1']:+.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='P3: Markov-Gated Mamba')
    parser.add_argument('--data_dir', type=str, default='prepared_data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='artifacts/p3_markov', help='Output directory')
    parser.add_argument('--config', type=str, default=None, help='Config YAML')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d_gate', type=int, default=8, help='Gate dimension')
    parser.add_argument('--gate_init', type=str, default='neutral', help='Gate initialization')
    
    args = parser.parse_args()
    
    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load data
    print("\n[*] Loading data...")
    data_dir = Path(args.data_dir)
    X_feats = pd.read_pickle(data_dir / 'X_feats.pkl').values
    Y = np.load(data_dir / 'Y_Walmsley2020.npy')
    P = np.load(data_dir / 'P.npy')
    
    print(f"  X shape: {X_feats.shape}, Y shape: {Y.shape}")
    
    # Split
    print("\n[*] Splitting data...")
    unique_participants = sorted(np.unique(P))
    n_total = len(unique_participants)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    train_participants = unique_participants[:n_train]
    val_participants = unique_participants[n_train:n_train + n_val]
    test_participants = unique_participants[n_train + n_val:]
    
    train_mask = np.isin(P, train_participants)
    val_mask = np.isin(P, val_participants)
    test_mask = np.isin(P, test_participants)
    
    X_train, y_train, P_train = X_feats[train_mask], Y[train_mask], P[train_mask]
    X_val, y_val, P_val = X_feats[val_mask], Y[val_mask], P[val_mask]
    X_test, y_test, P_test = X_feats[test_mask], Y[test_mask], P[test_mask]
    
    print(f"  Train: {X_train.shape[0]} samples, {len(train_participants)} participants")
    print(f"  Val:   {X_val.shape[0]} samples, {len(val_participants)} participants")
    print(f"  Test:  {X_test.shape[0]} samples, {len(test_participants)} participants")
    
    # Config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'input_dim': 32,
            'n_classes': 4,
            'd_model': 16,
            'n_layers': 1,
            'dropout': 0.2,
            'lr': 3e-4,
            'weight_decay': 1e-4,
            'epochs': 20,
            'early_stopping_patience': 5,
            'gradient_clip': 1.0,
            'd_gate': args.d_gate,
            'gate_init': args.gate_init,
        }
    
    # Device
    device = torch.device(args.device)
    
    # Train
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    trainer = P3MarkovGatedMambaTrainer(config, device=device, output_dir=output_dir)
    train_result = trainer.train(X_train, y_train, P_train, X_val, y_val, P_val)
    
    # Test
    test_result = trainer.predict(X_test, y_test, P_test)
    
    print("\n[OK] P3 Training and Testing Complete!")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
