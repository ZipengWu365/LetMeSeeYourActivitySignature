#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markov-Gated Mamba Architecture

State-conditioned modulation of Mamba trunk to produce state-specific logits.

Design: Section 5 of next_step_experiment_design_hmm_based_mamba.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class MarkovGatedMamba(nn.Module):
    """
    Markov-Gated Mamba for Activity Sequence Classification
    
    Architecture:
    - Shared Tiny-Mamba trunk producing h_t ∈ R^{d_model}
    - State embeddings s_k ∈ R^{d_gate} for each state k
    - Gating modulation: m_k = sigmoid(W_g s_k + b_g)
    - State-conditioned: h_t^{(k)} = h_t ⊙ m_k
    - Per-state logits: e_t^{(k)} = W_o h_t^{(k)} + b_o
    - Combined: e_t = Σ_k γ_t(k) * e_t^{(k)}
    
    Args:
        input_dim: Input feature dimension (default: 32)
        n_classes: Number of output classes (default: 4)
        d_model: Mamba hidden dimension (default: 16)
        n_layers: Number of Mamba layers (default: 1)
        d_gate: Gate dimension (default: 8)
        d_state: SSM state dimension (default: 16)
        d_conv: Convolution dimension (default: 4)
        expand: Expansion factor (default: 2)
        dropout: Dropout rate (default: 0.1)
        gate_init: Initialization strategy for gates (default: 'neutral')
    """
    
    def __init__(
        self,
        input_dim: int = 32,
        n_classes: int = 4,
        d_model: int = 16,
        n_layers: int = 1,
        d_gate: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        gate_init: str = 'neutral',
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.d_model = d_model
        self.d_gate = d_gate
        self.gate_init = gate_init
        self.device = device or torch.device('cpu')
        
        # 1. Feature projection
        self.feat_proj = nn.Linear(input_dim, d_model)
        
        # 2. Shared Mamba trunk
        if not MAMBA_AVAILABLE:
            raise ImportError("Mamba not available. Install: pip install mamba-ssm")
        
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])
        
        # 3. State embeddings (one per class)
        self.state_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(d_gate))
            for _ in range(n_classes)
        ])
        
        # 4. Gating network: s_k -> m_k
        self.gate_network = nn.Sequential(
            nn.Linear(d_gate, d_gate),
            nn.ReLU(),
            nn.Linear(d_gate, d_model),
            nn.Sigmoid(),
        )
        
        # 5. Per-state output heads
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, n_classes)
            for _ in range(n_classes)
        ])
        
        # Initialize
        self._init_gates(gate_init)
        self.to(self.device)
    
    def _init_gates(self, strategy: str):
        """Initialize gate parameters"""
        if strategy == 'neutral':
            # Start with neutral gates (all ones)
            for se in self.state_embeddings:
                nn.init.normal_(se, mean=0.0, std=0.1)
        elif strategy == 'conservative':
            # Conservative: gates closer to zero (smaller modulation)
            for se in self.state_embeddings:
                nn.init.normal_(se, mean=-1.0, std=0.1)
    
    def forward(self, x: torch.Tensor, gamma_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (seq_len, input_dim)
            gamma_t: Markov posterior weights (seq_len, n_classes)
                    If None, uses uniform weights
        
        Returns:
            logits: Combined logits (seq_len, n_classes)
        """
        seq_len = x.shape[0]
        
        # 1. Feature projection
        h = self.feat_proj(x)  # (seq_len, d_model)
        
        # 2. Mamba trunk (needs 3D input: batch=1, seqlen, dim)
        h = h.unsqueeze(0)  # (1, seq_len, d_model)
        for mamba_layer in self.mamba_layers:
            h = mamba_layer(h)  # (1, seq_len, d_model)
        h = h.squeeze(0)  # (seq_len, d_model)
        
        # 3. State-conditioned gating and per-state logits
        logits_per_state = []
        
        for k in range(self.n_classes):
            # Get gate modulation for state k
            s_k = self.state_embeddings[k]  # (d_gate,)
            m_k = self.gate_network(s_k)  # (d_model,)
            
            # Apply modulation
            h_k = h * m_k.unsqueeze(0)  # (seq_len, d_model)
            
            # Generate state-specific logits
            e_k = self.output_heads[k](h_k)  # (seq_len, n_classes)
            logits_per_state.append(e_k)
        
        # 4. Combine using Markov posterior weights
        if gamma_t is None:
            # Uniform weights
            gamma_t = torch.ones(seq_len, self.n_classes, device=self.device) / self.n_classes
        else:
            gamma_t = gamma_t.to(self.device)  # (seq_len, n_classes)
        
        # Weighted combination: e_t = Σ_k γ_t(k) * e_t^{(k)}
        logits = torch.zeros(seq_len, self.n_classes, device=self.device)
        for k in range(self.n_classes):
            logits += gamma_t[:, k:k+1] * logits_per_state[k]  # (seq_len, n_classes)
        
        return logits


def create_markov_gated_mamba(
    input_dim: int = 32,
    n_classes: int = 4,
    d_model: int = 16,
    n_layers: int = 1,
    d_gate: int = 8,
    dropout: float = 0.1,
    device: Optional[torch.device] = None,
) -> MarkovGatedMamba:
    """Factory function for creating Markov-Gated Mamba"""
    return MarkovGatedMamba(
        input_dim=input_dim,
        n_classes=n_classes,
        d_model=d_model,
        n_layers=n_layers,
        d_gate=d_gate,
        dropout=dropout,
        device=device,
    )
