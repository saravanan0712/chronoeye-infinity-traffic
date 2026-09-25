"""
ChronoEye Infinity - ST-GNN Neural Network Layers.
Implements Spatial Graph Convolution, Temporal GRU Encoder, and integrated Spatio-Temporal blocks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphSpatialConv(nn.Module):
    """
    Graph Spatial Convolution layer with degree-normalized symmetric message passing:
    H' = activation( D_hat^(-1/2) * A_hat * D_hat^(-1/2) * H * W + bias )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def compute_dynamic_adjacency(
        adj_static: torch.Tensor,
        transition_count: Optional[torch.Tensor] = None,
        speed: Optional[torch.Tensor] = None,
        travel_time: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        has_unobserved_gap: Optional[torch.Tensor] = None,
        ref_speed: float = 60.0,
        ref_travel_time: float = 30.0,
        alpha_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Computes dynamically weighted adjacency matrix based on real traffic observation evidence.
        
        Formula:
        For edges with adj_static[i, j] > 0:
            flow_factor = log(1.0 + max(0, transition_count))
            speed_factor = clamp(speed / ref_speed, 0.1, 2.0) if speed > 0
                           else clamp(ref_travel_time / travel_time, 0.1, 2.0) if travel_time > 0
                           else 1.0
            conf_factor = clamp(confidence, 0.0, 1.0) if confidence is not None else 1.0
            gap_discount = 0.8 if has_unobserved_gap else 1.0
            
            dynamic_mult = 1.0 + alpha_scale * tanh(flow_factor * speed_factor * conf_factor * gap_discount)
            A_dynamic[i, j] = adj_static[i, j] * dynamic_mult

        Guarantees:
        - Strict non-negativity
        - Bounded multiplier in [1.0, 1.0 + alpha_scale] (with alpha_scale=1.0, multiplier in [1.0, 2.0])
        - No phantom edges (where adj_static == 0, A_dynamic == 0)
        - Robust against NaNs, Infs, and negative inputs
        - If no dynamic evidence is supplied, returns exact adj_static
        """
        # Ensure adj_static is non-negative and clean
        adj_clean = torch.nan_to_num(adj_static, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        
        # If no dynamic evidence is passed, return adj_clean directly (static identity)
        if (
            transition_count is None
            and speed is None
            and travel_time is None
            and confidence is None
            and has_unobserved_gap is None
        ):
            return adj_clean

        device = adj_clean.device
        dtype = adj_clean.dtype
        N = adj_clean.size(0)

        # 1. Transition Flow Intensity Factor
        if transition_count is not None:
            tc = transition_count if isinstance(transition_count, torch.Tensor) else torch.tensor(transition_count, device=device, dtype=dtype)
            tc = torch.nan_to_num(tc, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
            flow_factor = torch.log1p(tc)
        else:
            flow_factor = torch.zeros((N, N), device=device, dtype=dtype)

        # 2. Speed / Travel Time Factor
        if speed is not None:
            sp = speed if isinstance(speed, torch.Tensor) else torch.tensor(speed, device=device, dtype=dtype)
            sp = torch.nan_to_num(sp, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
            # When speed > 0, scale relative to ref_speed, clamped to [0.1, 2.0]; when 0, default to 1.0
            sp_safe = torch.where(sp > 1e-4, (sp / max(1.0, ref_speed)).clamp(min=0.1, max=2.0), torch.ones_like(sp))
        elif travel_time is not None:
            tt = travel_time if isinstance(travel_time, torch.Tensor) else torch.tensor(travel_time, device=device, dtype=dtype)
            tt = torch.nan_to_num(tt, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
            # When travel_time > 0, scale relative to ref_travel_time, clamped to [0.1, 2.0]; when 0, default to 1.0
            sp_safe = torch.where(tt > 1e-4, (ref_travel_time / tt.clamp(min=1e-3)).clamp(min=0.1, max=2.0), torch.ones_like(tt))
        else:
            sp_safe = torch.ones((N, N), device=device, dtype=dtype)

        # 3. Transition Confidence Modulation
        if confidence is not None:
            cf = confidence if isinstance(confidence, torch.Tensor) else torch.tensor(confidence, device=device, dtype=dtype)
            cf = torch.nan_to_num(cf, nan=1.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)
        else:
            cf = torch.ones((N, N), device=device, dtype=dtype)

        # 4. Unobserved Gap Discount
        if has_unobserved_gap is not None:
            gap = has_unobserved_gap if isinstance(has_unobserved_gap, torch.Tensor) else torch.tensor(has_unobserved_gap, device=device, dtype=torch.bool)
            gap_discount = torch.where(gap, torch.full((N, N), 0.8, device=device, dtype=dtype), torch.ones((N, N), device=device, dtype=dtype))
        else:
            gap_discount = torch.ones((N, N), device=device, dtype=dtype)

        # 5. Combined Dynamic Edge Multiplier
        evidence_intensity = flow_factor * sp_safe * cf * gap_discount
        dynamic_multiplier = 1.0 + float(alpha_scale) * torch.tanh(evidence_intensity)

        # 6. Apply strictly to topologically valid edges (zero phantom edges)
        adj_dynamic = adj_clean * dynamic_multiplier
        adj_dynamic = torch.nan_to_num(adj_dynamic, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)

        return adj_dynamic

    @staticmethod
    def normalize_adjacency(
        adj: torch.Tensor,
        dynamic_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes symmetric normalized adjacency: A_norm = D^(-1/2) * (A_hat) * D^(-1/2)
        where A_hat = A_dynamic + I if dynamic_weights supplied, else A + I.
        """
        if dynamic_weights is not None:
            adj_effective = dynamic_weights
        else:
            adj_effective = adj

        adj_clean = torch.nan_to_num(adj_effective, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        N = adj_clean.size(0)
        adj_with_self = adj_clean + torch.eye(N, device=adj_clean.device, dtype=adj_clean.dtype)
        deg = torch.sum(adj_with_self, dim=-1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
        deg_mat = torch.diag(deg_inv_sqrt)
        return torch.mm(torch.mm(deg_mat, adj_with_self), deg_mat)

    def forward(
        self, x: torch.Tensor, adj_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [..., N, in_features]
        adj_norm: [N, N] normalized adjacency matrix
        returns: [..., N, out_features]
        """
        x = self.dropout(x)
        # Linear projection
        support = torch.matmul(x, self.weight)  # [..., N, out_features]
        # Graph message passing
        out = torch.matmul(adj_norm, support)   # [..., N, out_features]
        if self.bias is not None:
            out = out + self.bias
        return out


class TemporalGRUBlock(nn.Module):
    """
    Temporal Recurrent Block modeling sequential time dependencies across T_in.
    Processes [B, T_in, N, hidden_dim] -> [B, N, hidden_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T_in, N, input_dim]
        returns: (all_states [B, T_in, N, hidden_dim], last_state [B, N, hidden_dim])
        """
        B, T, N, F = x.shape
        # Reshape to [B * N, T, F] for batch GRU processing
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, F)
        out_gru, h_n = self.gru(x_reshaped)  # [B * N, T, hidden_dim], [num_layers, B * N, hidden_dim]
        
        out = out_gru.view(B, N, T, self.hidden_dim).permute(0, 2, 1, 3).contiguous()  # [B, T, N, hidden_dim]
        last = out[:, -1, :, :]  # [B, N, hidden_dim]
        return out, last


class STGNNBlock(nn.Module):
    """
    Integrated Spatio-Temporal Block combining:
    1. Spatial Graph Convolution at each time step
    2. Temporal GRU across the time horizon
    3. Layer Normalization & Residual connections
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        self.spatial_conv = GraphSpatialConv(hidden_dim, hidden_dim, dropout=dropout)
        self.temporal_gru = TemporalGRUBlock(hidden_dim, hidden_dim, dropout=dropout)
        
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.activation = nn.ReLU()

    def forward(
        self, x: torch.Tensor, adj_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [B, T, N, hidden_dim]
        adj_norm: [N, N]
        returns: [B, T, N, hidden_dim]
        """
        # 1. Spatial Graph Convolution
        residual = x
        B, T, N, H = x.shape
        x_spatial = self.spatial_conv(x, adj_norm)
        x_spatial = self.activation(x_spatial)
        if self.use_residual:
            x_spatial = x_spatial + residual
        x_spatial = self.norm1(x_spatial)

        # 2. Temporal GRU
        residual_temporal = x_spatial
        x_temporal, _ = self.temporal_gru(x_spatial)
        if self.use_residual:
            x_temporal = x_temporal + residual_temporal
        x_out = self.norm2(x_temporal)

        return x_out
