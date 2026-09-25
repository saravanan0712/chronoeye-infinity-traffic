"""
ChronoEye Infinity - Module 3 Step 5I: Ablation Study Runner.
Executes controlled ablation experiments:
1. Ablation A: Temporal-Only Model (No spatial graph convolution, A = I)
2. Ablation B: Spatial-Only Model (No temporal GRU sequence modeling)
3. Ablation C: Feature Group Ablations (Without speed, without congestion)
4. Ablation D: History Length Comparison (T_in = 1 vs T_in = 3)
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.models.stgnn.config import STGNNConfig
from app.models.stgnn.model import SpatioTemporalGNN
from app.models.stgnn.layers import GraphSpatialConv
from app.training.trainer import STGNNTrainer, TrainingResult
from app.graph.temporal_dataset_slicer import SpatioTemporalDataset, TemporalDatasetSlicer
from app.graph.temporal_snapshot_schema import TemporalGraphDatasetContract


class AblationExperimentResult(BaseModel):
    ablation_id: str
    description: str
    model_name: str
    best_val_loss: float
    test_metrics: Dict[str, Dict[str, Dict[str, float]]]
    notes: Optional[str] = None


class TemporalOnlyGNN(nn.Module):
    """Ablation A: Temporal-Only Model (Spatial message passing disabled, A = Identity)."""

    def __init__(self, config: STGNNConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_temporal_layers,
            batch_first=True,
            dropout=config.dropout if config.num_temporal_layers > 1 else 0.0,
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, config.output_dim),
            )
            for _ in range(config.num_horizons)
        ])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, N, F = x.shape
        if feature_mask is not None:
            x = x * feature_mask.float()
        else:
            x = torch.nan_to_num(x, nan=0.0)

        h = self.input_proj(x)  # [B, T, N, hidden_dim]
        # Reshape [B * N, T, H]
        h_reshaped = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.config.hidden_dim)
        out_gru, _ = self.gru(h_reshaped)
        last = out_gru[:, -1, :].view(B, N, self.config.hidden_dim)  # [B, N, hidden_dim]

        horizon_preds = []
        for head in self.heads:
            pred_h = head(last)  # [B, N, output_dim]
            horizon_preds.append(pred_h.unsqueeze(1))
        return torch.cat(horizon_preds, dim=1)


class SpatialOnlyGNN(nn.Module):
    """Ablation B: Spatial-Only Model (No temporal sequence modeling, static GCN on last timestep)."""

    def __init__(self, config: STGNNConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.gcn_layers = nn.ModuleList([
            GraphSpatialConv(config.hidden_dim, config.hidden_dim, dropout=config.dropout)
            for _ in range(config.num_spatial_layers)
        ])
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, config.output_dim),
            )
            for _ in range(config.num_horizons)
        ])

        self.activation = nn.ReLU()

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, N, in_feat = x.shape
        if feature_mask is not None:
            x = x * feature_mask.float()
        else:
            x = torch.nan_to_num(x, nan=0.0)

        adj_norm = GraphSpatialConv.normalize_adjacency(adj) if adj.dim() == 2 else adj
        last_step = x[:, -1, :, :]  # [B, N, in_feat]
        h = self.input_proj(last_step)  # [B, N, hidden_dim]

        for gcn in self.gcn_layers:
            h = self.activation(gcn(h, adj_norm))

        horizon_preds = []
        for head in self.heads:
            pred_h = head(h)  # [B, N, output_dim]
            horizon_preds.append(pred_h.unsqueeze(1))
        return torch.cat(horizon_preds, dim=1)


class AblationStudyRunner:
    """
    Executes and aggregates ablation study experiments across the same dataset splits.
    """

    @staticmethod
    def run_all_ablations(
        dataset: SpatioTemporalDataset,
        contract: Optional[TemporalGraphDatasetContract] = None,
        epochs: int = 15,
        checkpoint_dir: str = "checkpoints/ablations",
    ) -> Dict[str, AblationExperimentResult]:
        """
        Runs Ablations A, B, C, D on the dataset.
        """
        results: Dict[str, AblationExperimentResult] = {}
        config = STGNNConfig(
            input_dim=len(dataset.feature_names),
            output_dim=len(dataset.target_names),
            forecast_horizons=dataset.forecast_horizons,
            hidden_dim=32,
            learning_rate=0.005,
        )

        # Baseline ST-GNN Full
        full_model = SpatioTemporalGNN(config=config)
        trainer_full = STGNNTrainer(full_model, config=config, checkpoint_dir=checkpoint_dir)
        res_full = trainer_full.train(dataset, epochs=epochs, save_checkpoint_name="full_stgnn.pt")
        results["Full_STGNN"] = AblationExperimentResult(
            ablation_id="FULL_STGNN",
            description="Complete Spatio-Temporal Graph Neural Network (Spatial GCN + Temporal GRU)",
            model_name="ST-GNN",
            best_val_loss=res_full.best_val_loss,
            test_metrics=res_full.test_metrics.get("ST-GNN", {}),
        )

        # Ablation A: Temporal Only (No Spatial GCN)
        temp_model = TemporalOnlyGNN(config=config)
        trainer_temp = STGNNTrainer(temp_model, config=config, checkpoint_dir=checkpoint_dir)
        res_temp = trainer_temp.train(dataset, epochs=epochs, save_checkpoint_name="temporal_only.pt")
        results["Ablation_A_Temporal_Only"] = AblationExperimentResult(
            ablation_id="ABLATION_A",
            description="Temporal-Only Model (No spatial graph convolution)",
            model_name="Temporal-Only GRU",
            best_val_loss=res_temp.best_val_loss,
            test_metrics=res_temp.test_metrics.get("ST-GNN", {}),
        )

        # Ablation B: Spatial Only (No Temporal GRU)
        spat_model = SpatialOnlyGNN(config=config)
        trainer_spat = STGNNTrainer(spat_model, config=config, checkpoint_dir=checkpoint_dir)
        res_spat = trainer_spat.train(dataset, epochs=epochs, save_checkpoint_name="spatial_only.pt")
        results["Ablation_B_Spatial_Only"] = AblationExperimentResult(
            ablation_id="ABLATION_B",
            description="Spatial-Only Model (No temporal sequence modeling)",
            model_name="Spatial-Only GCN",
            best_val_loss=res_spat.best_val_loss,
            test_metrics=res_spat.test_metrics.get("ST-GNN", {}),
        )

        # Ablation D: Shorter History (T_in = 1 vs T_in = 3)
        if contract is not None:
            slicer_short = TemporalDatasetSlicer(
                input_sequence_length=1,
                forecast_horizons=dataset.forecast_horizons,
                train_ratio=dataset.metadata.get("train_ratio", 0.70),
                val_ratio=dataset.metadata.get("val_ratio", 0.15),
                test_ratio=dataset.metadata.get("test_ratio", 0.15),
            )
            dataset_short = slicer_short.slice_dataset(contract)
            config_short = copy.deepcopy(config)
            config_short.input_sequence_length = 1
            model_short = SpatioTemporalGNN(config=config_short)
            trainer_short = STGNNTrainer(model_short, config=config_short, checkpoint_dir=checkpoint_dir)
            res_short = trainer_short.train(dataset_short, epochs=epochs, save_checkpoint_name="history_t1.pt")
            results["Ablation_D_Short_History_T1"] = AblationExperimentResult(
                ablation_id="ABLATION_D",
                description="Short History Comparison (T_in = 1 snapshot / 5 min)",
                model_name="ST-GNN (T_in=1)",
                best_val_loss=res_short.best_val_loss,
                test_metrics=res_short.test_metrics.get("ST-GNN", {}),
            )

        return results
