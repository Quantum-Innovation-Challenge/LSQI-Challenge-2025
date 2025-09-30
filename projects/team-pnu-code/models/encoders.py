import math
import numpy as np
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

from .qml import qnn_basic


# =========================
# Pooling Layer
# =========================
class Pooling(nn.Module):
    def __init__(self, in_dim: int, hidden: int, mode: str = "attn"):
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in ["mean", "max", "min", "attn"]:
            raise ValueError(f"Unknown pooling mode: {self.mode}. Choose from 'mean', 'max', 'min', 'attn'.")

        if self.mode == "attn":
            self.proj = nn.Linear(in_dim, hidden)
            self.score = nn.Linear(hidden, 1)
            self.norm = nn.LayerNorm(hidden)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            return x

        if self.mode == "mean":
            return x.mean(dim=1)
        
        elif self.mode == "max":
            return x.max(dim=1).values
        
        elif self.mode == "min":
            return x.min(dim=1).values

        elif self.mode == "attn":
            # [B, T, F] -> [B, T, H]
            h = self.proj(x)
            
            # [B, T, H] -> [B, T, 1] -> [B, T]
            a = self.score(torch.tanh(h)).squeeze(-1)
            
            # [B, T] -> [B, T, 1]
            w = torch.softmax(a, dim=1).unsqueeze(-1)
            
            # [B, T, H] * [B, T, 1] -> [B, T, H] -> [B, H]
            z = (h * w).sum(dim=1)
            
            # [B, H]
            return self.norm(z)

# =========================
# Base Encoder
# =========================
class BaseEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# =========================
# MLP Encoder
# =========================
class MLPEncoder(BaseEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        time_pool: str = None,  # "mean"/"max"/"attn"/None
        use_input_ln: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.time_pool = time_pool

        if time_pool in ["mean", "max", "min", "attn"]:
            self.pooling = Pooling(in_dim, hidden, mode=time_pool)
            d = hidden
        else:
            self.pooling = None
            d = in_dim

        layers = []
        if use_input_ln:
            layers.append(nn.LayerNorm(d))
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if x.dim() == 3:
            x = self.pooling(x) # [B, N, F] -> [B, F]
        return self.net(x) # [B, F] -> [B, H]

# =========================
# ResNet Block
# =========================
class ResBlock(nn.Module):
    def __init__(
            self, d: int, 
            dropout: float = 0.0
        ):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.ln(x)
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        return x + self.drop(h)

# =========================
# ResMLP Encoder (Basic)
# =========================
class ResMLPEncoder(BaseEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.1,
        time_pool: str = None,  # Changed default to None
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden
        self.pooling = Pooling(in_dim, hidden, mode=time_pool) if time_pool in ["mean", "max", "min", "attn"] else None

        stem_in = hidden if self.pooling is not None else in_dim
        
        self.stem = nn.Sequential(nn.Linear(stem_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(
            *[ResBlock(hidden, dropout=dropout) for _ in range(n_blocks)]
        )
        self.final_ln = nn.LayerNorm(hidden)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and self.pooling is not None:
            x = self.pooling(x)
        elif x.dim() == 2 and self.pooling is not None:

            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Linear(x.size(-1), self.hidden_dim).to(x.device)
            x = self.input_proj(x)

            x = self.blocks(x)
            return self.final_ln(x)

        x = self.stem(x)
        x = self.blocks(x)
        return self.final_ln(x)

# =========================
# MoE (Mixture of Experts) - Basic
# =========================
class MoEBlock(nn.Module):
    """Basic Mixture of Experts block with residual connection."""
    
    def __init__(self, in_dim: int, hidden_dim: int, num_experts: int = 4, 
                 top_k: int = 2, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU() if activation == "relu" else nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, in_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(in_dim, num_experts)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(in_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get gating scores
        gate_scores = F.softmax(self.gate(x), dim=-1)  # [B, num_experts]
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = F.softmax(top_k_scores, dim=-1)  # Renormalize top-k scores
        
        # Apply experts
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(self.experts[i](x))
        
        # Combine expert outputs
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            # Handle both 1D and 2D cases
            if top_k_indices.dim() == 1:
                expert_idx = top_k_indices[i:i+1]  # Keep as 1D tensor
                expert_score = top_k_scores[i:i+1]  # Keep as 1D tensor
            else:
                expert_idx = top_k_indices[:, i]  # [B]
                expert_score = top_k_scores[:, i:i+1]  # [B, 1]
            
            # Gather expert outputs
            if expert_idx.dim() == 1 and expert_idx.size(0) == 1:  # single sample case
                expert_output = expert_outputs[expert_idx[0].item()]
            else:
                expert_output = torch.stack([expert_outputs[j][b] for b, j in enumerate(expert_idx)])
            output += expert_score * expert_output
        
        # Residual connection
        return self.layer_norm(x + output)


class MoEEncoder(BaseEncoder):
    """Basic Mixture of Experts encoder with residual connections."""
    def __init__(self, in_dim: int, hidden_dims: List[int], num_experts: int = 4, 
                 top_k: int = 2, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = hidden_dims[-1]
        
        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(MoEBlock(prev_dim, hidden_dim, num_experts, top_k, dropout, activation))
            # MoEBlock maintains the same dimension, so prev_dim stays the same
            # Only change dimension if we want to project
            if hidden_dim != prev_dim:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =========================
# Advanced ResMLP + MoE Components
# =========================
class ResidualMLPBlock(nn.Module):
    """Advanced Residual MLP Block with LayerNorm and Dropout"""
    def __init__(
        self,
        hidden_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # MLP layers
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout for residual connection
        self.dropout_layer = nn.Dropout(dropout)
    
    def _get_activation(self, activation: str):
        if activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "swish":
            return nn.SiLU()
        else:
            return nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connection
        residual = x
        x = self.norm1(x)
        x = self.mlp(x)
        x = self.dropout_layer(x)
        x = x + residual
        
        # Second residual connection
        residual = x
        x = self.norm2(x)
        return x + residual


class AdvancedMoEBlock(nn.Module):
    """Advanced MoE Block with Top-K routing and load balancing"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_capacity_factor: float = 1.25,
        dropout: float = 0.1,
        jitter_noise: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise

        # Router
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])

        # Aux loss weight
        self.aux_loss_weight = 0.01

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, H]

        B, L, H = x.shape
        x_flat = x.view(-1, H)  # [N, H]
        N = x_flat.size(0)

        # Router logits
        logits = self.router(x_flat)  # [N, E]
        if self.training and self.jitter_noise > 0:
            logits = logits + torch.randn_like(logits) * self.jitter_noise

        probs = F.softmax(logits, dim=-1)  # [N, E]
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)  # [N, K]

        # Normalize
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Dispatch
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_ids = topk_idx[:, k]   # [N]
            expert_wts = topk_probs[:, k] # [N]

            for e, expert in enumerate(self.experts):
                mask = (expert_ids == e)
                if mask.any():
                    y = expert(x_flat[mask]) * expert_wts[mask].unsqueeze(-1)
                    output[mask] += y

        # Reshape
        output = output.view(B, L, H)
        if L == 1:
            output = output.squeeze(1)

        # Aux loss: encourage balanced expert usage
        counts = torch.bincount(topk_idx.flatten(), minlength=self.num_experts).float()
        load = counts / counts.sum()
        aux_loss = self.aux_loss_weight * (load * torch.log(load + 1e-9)).sum()

        return output, aux_loss


class ResMLPMoEBlock(nn.Module):
    """Combined ResMLP + Advanced MoE Block"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.resmlp = ResidualMLPBlock(
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            activation=activation
        )
        
        self.moe = AdvancedMoEBlock(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout
        )
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ResMLP processing
        x = self.resmlp(x)
        
        # MoE processing
        x, aux_loss = self.moe(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x, aux_loss


class ResMLPMoEEncoder(BaseEncoder):
    """
    Advanced ResMLP + MoE Hybrid Encoder
    Transformer-like stacking: [ResMLP->MoE]->[ResMLP->MoE]->[ResMLP->MoE]
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_experts: int = 8,
        top_k: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        time_pool: Optional[str] = None,  # "mean"/"max"/"attn"/None
        use_input_projection: bool = True,
        use_output_projection: bool = True
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.num_layers = num_layers
        self.time_pool = time_pool
        
        # Input projection
        if use_input_projection:
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            )
        else:
            self.input_proj = nn.Identity()
        
        # ResMLP + MoE blocks
        self.blocks = nn.ModuleList([
            ResMLPMoEBlock(
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                activation=activation
            ) for _ in range(num_layers)
        ])
        
        # Time pooling (if needed)
        if time_pool in ["mean", "max", "min", "attn"]:
            if time_pool == "attn":
                self.pooling = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                    nn.Softmax(dim=1)
                )
            else:
                self.pooling = time_pool
        else:
            self.pooling = None
        
        # Output projection
        if use_output_projection:
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout)
            )
        else:
            self.output_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, in_dim] or [batch_size, in_dim]
        Returns:
            output: [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
            total_aux_loss: total auxiliary loss from all MoE blocks
        """
        # Input projection
        x = self.input_proj(x)
        
        # Process through ResMLP + MoE blocks
        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss += aux_loss
        
        # Time pooling (if needed)
        if self.pooling is not None and x.dim() == 3:
            if self.pooling == "mean":
                x = x.mean(dim=1)
            elif self.pooling == "max":
                x = x.max(dim=1).values
            elif self.pooling == "min":
                x = x.min(dim=1).values
            elif isinstance(self.pooling, nn.Module):
                # Attention pooling
                weights = self.pooling(x)  # [batch_size, seq_len, 1]
                x = (x * weights).sum(dim=1)  # [batch_size, hidden_dim]
        
        # Output projection
        x = self.output_proj(x)
        
        return x

class CNNEncoder(BaseEncoder):
    """CNN-based encoder for sequence data"""
    
    def __init__(
        self, 
        in_dim: int, 
        hidden: int = 64, 
        depth: int = 3, 
        dropout: float = 0.1,
        kernel_size: int = 3,
        num_filters: int = 64
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        
        # Input projection to create channels
        self.input_proj = nn.Linear(in_dim, num_filters)
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            in_channels = num_filters if i == 0 else num_filters
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection
        self.final_proj = nn.Linear(num_filters, hidden)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Handle 2D input (add sequence dimension)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, in_dim]
        
        # Input projection
        x = self.input_proj(x)  # [batch_size, seq_len, num_filters]
        
        # Transpose for Conv1d: [batch_size, num_filters, seq_len]
        x = x.transpose(1, 2)
        
        # CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global pooling: [batch_size, num_filters, 1]
        x = self.global_pool(x)
        
        # Flatten: [batch_size, num_filters]
        x = x.squeeze(-1)
        
        # Final projection: [batch_size, hidden]
        x = self.final_proj(x)
        x = self.dropout(x)
        
        return x


def _quantum_circuit(inputs, weights):
    """Quantum circuit function that can be pickled"""
    # Get dimensions from weights shape
    n_layers, n_qubits = weights.shape[:2]
    
    # Encode classical data
    for i in range(n_qubits):
        if i < len(inputs):
            qml.RY(inputs[i], wires=i)
    # Variational layers
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.Rot(*weights[l, q], wires=q)
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_qubits: int = 4, n_layers: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        
        # Create circuit as a separate method to avoid pickle issues
        self.circuit = qml.QNode(
            _quantum_circuit,
            self.dev, 
            interface="torch"
        )
        self.readout = nn.Linear(n_qubits, out_features)

    def forward(self, x: torch.Tensor):
        outputs = []
        for sample in x:
            q_out = self.circuit(sample, self.params)
            # Preserve device information
            if isinstance(q_out, list):
                q_out = torch.stack([torch.tensor(val, dtype=torch.float32, device=x.device) for val in q_out])
            else:
                q_out = torch.tensor(q_out, dtype=torch.float32, device=x.device)
            outputs.append(q_out)
        q_out = torch.stack(outputs)
        return self.readout(q_out)



class QMLPEncoder(BaseEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        time_pool: str = None,
        use_input_ln: bool = False,
        n_qubits: int = 4,
        n_layers: int = 1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.time_pool = time_pool

        if time_pool in ["mean", "max", "min", "attn"]:
            self.pooling = Pooling(in_dim, hidden, mode=time_pool)
            d = hidden
        else:
            self.pooling = None
            d = in_dim

        layers = []
        if use_input_ln:
            layers.append(nn.LayerNorm(d))

        layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]

        layers += [QuantumLinear(hidden, hidden, n_qubits=n_qubits, n_layers=n_layers), nn.ReLU()]

        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]

        layers += [nn.Linear(hidden, hidden)]
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and self.pooling is not None:
            x = self.pooling(x)
        return self.net(x)


class QResMLPEncoder(BaseEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        time_pool: str = None,
        n_qubits: int = 4,
        n_layers: int = 1
    ):
        super().__init__()
        self.hidden = hidden
        self.pooling = Pooling(in_dim, hidden, mode=time_pool) if time_pool in ["mean", "max", "min", "attn"] else None
        stem_in = hidden if self.pooling else in_dim

        # Stem: classical
        self.stem = nn.Sequential(nn.Linear(stem_in, hidden), nn.ReLU(), nn.Dropout(dropout))

        self.qblock = QuantumLinear(hidden, hidden, n_qubits=n_qubits, n_layers=n_layers)

        # Residual MLP blocks
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(num_layers)])
        self.final_ln = nn.LayerNorm(hidden)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and self.pooling is not None:
            x = self.pooling(x)
        x = self.stem(x)
        x = self.qblock(x)
        x = self.blocks(x)
        return self.final_ln(x)

class QMoEEncoder(BaseEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden: List[int],
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        n_qubits: int = 4,
        n_layers: int = 1
    ):
        super().__init__()
        self.out_dim = hidden[-1]

        class QuantumMoEBlock(MoEBlock):
            def __init__(self, in_dim, hidden, num_experts, top_k, dropout, activation):
                super().__init__(in_dim, hidden, num_experts, top_k, dropout, activation)
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        QuantumLinear(in_dim, hidden, n_qubits=n_qubits, n_layers=n_layers),
                        nn.ReLU() if activation == "relu" else nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden, in_dim)  # readout은 classical
                    ) for _ in range(num_experts)
                ])

        layers = []
        prev_dim = in_dim
        for hidden in hidden:
            layers.append(QuantumMoEBlock(prev_dim, hidden, num_experts, top_k, dropout, activation))
            if hidden != prev_dim:
                layers.append(nn.Linear(prev_dim, hidden))
                prev_dim = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QResMLPMoEEncoder(BaseEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        num_layers: int = 6,
        num_experts: int = 8,
        top_k: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        time_pool: Optional[str] = None,
        use_input_projection: bool = True,
        use_output_projection: bool = True,
        n_qubits: int = 4,
        n_layers: int = 1
    ):
        super().__init__()
        self.out_dim = hidden

        # 입력 projection (classical)
        if use_input_projection:
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout)
            )
        else:
            self.input_proj = nn.Identity()

        # 중간 quantum block
        self.qblock = QuantumLinear(hidden, hidden, n_qubits=n_qubits, n_layers=n_layers)

        # ResMLP + MoE blocks
        self.blocks = nn.ModuleList([
            ResMLPMoEBlock(hidden, num_experts, top_k, mlp_ratio, dropout, activation)
            for _ in range(num_layers)
        ])

        # Time pooling
        if time_pool in ["mean", "max", "min", "attn"]:
            if time_pool == "attn":
                self.pooling = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, 1),
                    nn.Softmax(dim=1)
                )
            else:
                self.pooling = time_pool
        else:
            self.pooling = None

        # Output projection (classical)
        if use_output_projection:
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.Dropout(dropout)
            )
        else:
            self.output_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x = self.qblock(x)

        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss += aux_loss

        if self.pooling is not None and x.dim() == 3:
            if self.pooling == "mean":
                x = x.mean(dim=1)
            elif self.pooling == "max":
                x = x.max(dim=1).values
            elif self.pooling == "min":
                x = x.min(dim=1).values
            elif isinstance(self.pooling, nn.Module):
                weights = self.pooling(x)
                x = (x * weights).sum(dim=1)

        x = self.output_proj(x)
        return x, total_aux_loss


## Quantum Models
class QNNEncoder(BaseEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        time_pool: str = None,  # "mean"/"max"/"attn"/None
        use_input_ln: bool = False,
        n_qubits=8, n_layers=2
    ):
        super().__init__()
        self.in_dim = in_dim
        self.time_pool = time_pool

        if time_pool in ["mean", "max", "min", "attn"]:
            self.pooling = Pooling(in_dim, hidden, mode=time_pool)
            d = hidden
        else:
            self.pooling = None
            d = in_dim

        layers = []
        if use_input_ln:
            layers.append(nn.LayerNorm(d))
        for _ in range(depth):
            layers += [qnn_basic(d, hidden, n_qubits, n_layers), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = self.pooling(x) # [B, N, F] -> [B, F]
        return self.net(x) # [B, F] -> [B, H]
    
class ResidualQNNBlock(nn.Module):
    """Advanced Residual MLP Block with LayerNorm and Dropout"""
    
    def __init__(
        self,
        hidden_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        n_qubits=8, n_layers=2
    ):
        super().__init__()   
        
        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # MLP layers
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            qnn_basic(hidden_dim, mlp_hidden, n_qubits, n_layers),
            self._get_activation(activation),
            nn.Dropout(dropout),
            qnn_basic(mlp_hidden, hidden_dim, n_qubits, n_layers),
            nn.Dropout(dropout)
        )
        
        # Dropout for residual connection
        self.dropout_layer = nn.Dropout(dropout)
    
    def _get_activation(self, activation: str):
        if activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "swish":
            return nn.SiLU()
        else:
            return nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connection
        residual = x
        x = self.norm1(x)
        x = self.mlp(x)
        x = self.dropout_layer(x)
        x = x + residual
        
        # Second residual connection
        residual = x
        x = self.norm2(x)
        return x + residual

class ResQNNMoEBlock(nn.Module):
    """Combined ResQNN + Advanced MoE Block"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        n_qubits: int = 8, 
        n_layers: int = 2
    ):
        super().__init__()
        
        self.resmlp = ResidualQNNBlock(
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            activation=activation,
            n_qubits=n_qubits, 
            n_layers=n_layers
        )
        
        self.moe = AdvancedMoEBlock(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout
        )
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ResMLP processing
        x = self.resmlp(x)
        
        # MoE processing
        x, aux_loss = self.moe(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x, aux_loss
    

    
class ResQNNMoEEncoder(BaseEncoder):
    """
    Advanced ResQNN + MoE Hybrid Encoder
    Transformer-like stacking: [ResMLP->MoE]->[ResMLP->MoE]->[ResMLP->MoE]
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_experts: int = 8,
        top_k: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        time_pool: Optional[str] = None,  # "mean"/"max"/"attn"/None
        use_input_projection: bool = True,
        use_output_projection: bool = True,
        n_qubits: int = 8, 
        n_layers: int = 2
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.num_layers = num_layers
        self.time_pool = time_pool
        
        # Input projection
        if use_input_projection:
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            )
        else:
            self.input_proj = nn.Identity()
        
        # ResMLP + MoE blocks
        self.blocks = nn.ModuleList([
            ResQNNMoEBlock(
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                activation=activation,
                n_qubits=n_qubits,
                n_layers=n_layers
            ) for _ in range(num_layers)
        ])
        
        # Time pooling (if needed)
        if time_pool in ["mean", "max", "min", "attn"]:
            if time_pool == "attn":
                self.pooling = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                    nn.Softmax(dim=1)
                )
            else:
                self.pooling = time_pool
        else:
            self.pooling = None
        
        # Output projection
        if use_output_projection:
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout)
            )
        else:
            self.output_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, in_dim] or [batch_size, in_dim]
        Returns:
            output: [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
            total_aux_loss: total auxiliary loss from all MoE blocks
        """
        # Input projection
        x = self.input_proj(x)
        
        # Process through ResMLP + MoE blocks
        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss += aux_loss
        
        # Time pooling (if needed)
        if self.pooling is not None and x.dim() == 3:
            if self.pooling == "mean":
                x = x.mean(dim=1)
            elif self.pooling == "max":
                x = x.max(dim=1).values
            elif self.pooling == "min":
                x = x.min(dim=1).values
            elif isinstance(self.pooling, nn.Module):
                # Attention pooling
                weights = self.pooling(x)  # [batch_size, seq_len, 1]
                x = (x * weights).sum(dim=1)  # [batch_size, hidden_dim]
        
        # Output projection
        x = self.output_proj(x)
        
        return x
    