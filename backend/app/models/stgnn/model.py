"""
ChronoEye Infinity - Spatio-Temporal Graph Neural Network (ST-GNN) Model.
Processes spatial graph topologies and temporal sequences to predict multi-horizon traffic states.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List

from app.models.stgnn.config import STGNNConfig
from app.models.stgnn.layers import GraphSpatialConv, STGNNBlock


class SpatioTemporalGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network (ST-GNN).
    Architecture:
    Input [B, T_in, N, F]
        ↓
    Feature Embedding Linear Projection -> [B, T_in, N, hidden_dim]
        ↓
    Sequential ST-GNN Blocks (GraphSpatialConv + Temporal GRU + LayerNorm + Residual)
        ↓
    Multi-Horizon Forecast Head -> [B, H, N, output_dim] (+5, +10, +15 min)
    """

    def __init__(self, config: Optional[STGNNConfig] = None):
        super().__init__()
        self.config = config or STGNNConfig()

        # 1. Feature Embedding Projection
        self.input_projection = nn.Linear(self.config.input_dim, self.config.hidden_dim)

        # 2. Sequential ST-GNN Blocks
        self.st_blocks = nn.ModuleList([
            STGNNBlock(
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
                use_residual=self.config.use_residual,
                use_layer_norm=self.config.use_layer_norm,
            )
            for _ in range(self.config.num_spatial_layers)
        ])

        # 3. Multi-Horizon Forecast Heads (one projection head per horizon, e.g. +5, +10, +15 min)
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim // 2, self.config.output_dim),
            )
            for _ in range(self.config.num_horizons)
        ])

    def get_parameter_count(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
        dynamic_edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, T_in, N, F] input feature tensor
        adj: [N, N] adjacency matrix
        feature_mask: [B, T_in, N, F] optional boolean mask
        dynamic_edge_weights: [N, N] optional dynamically-weighted adjacency matrix
        returns: [B, H, N, output_dim] multi-horizon predictions
        """
        B, T, N, F = x.shape
        
        # Normalize adjacency matrix symmetrically if not already normalized
        if adj.dim() == 2:
            adj_norm = GraphSpatialConv.normalize_adjacency(adj, dynamic_weights=dynamic_edge_weights)
        else:
            adj_norm = adj

        # Mask missing inputs with 0 before linear projection
        if feature_mask is not None:
            x = x * feature_mask.float()
        else:
            x = torch.nan_to_num(x, nan=0.0)

        # 1. Input Linear Projection
        h = self.input_projection(x)  # [B, T, N, hidden_dim]

        # 2. Spatio-Temporal Graph & Temporal Message Passing
        for block in self.st_blocks:
            h = block(h, adj_norm)  # [B, T, N, hidden_dim]

        # Use final temporal state for multi-horizon decoding
        h_last = h[:, -1, :, :]  # [B, N, hidden_dim]

        # 3. Multi-Horizon Prediction Decoding
        horizon_preds: List[torch.Tensor] = []
        for head in self.horizon_heads:
            pred_h = head(h_last)  # [B, N, output_dim]
            horizon_preds.append(pred_h.unsqueeze(1))  # [B, 1, N, output_dim]

        out = torch.cat(horizon_preds, dim=1)  # [B, H, N, output_dim]
        return out
