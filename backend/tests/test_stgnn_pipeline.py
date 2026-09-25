"""
Tests for Module 3: Spatio-Temporal Graph Neural Network (ST-GNN) Pipeline.
Verifies dataset adapter, ST-GNN forward pass, masked loss, trainer execution,
baseline models, evaluation metrics, checkpoint persistence, and ablation configurations.
"""

import pytest
import math
import os
import torch
import numpy as np

from app.adapters.benchmark_adapter import TrafficDatasetAdapter
from app.graph.temporal_dataset_slicer import TemporalDatasetSlicer
from app.models.stgnn.config import STGNNConfig
from app.models.stgnn.layers import GraphSpatialConv, TemporalGRUBlock, STGNNBlock
from app.models.stgnn.loss import MaskedLoss, MaskedMAE, MaskedRMSE, MaskedMAPE, compute_all_metrics
from app.models.stgnn.model import SpatioTemporalGNN
from app.training.baselines import PersistenceBaseline, MovingAverageBaseline, LinearRegressionBaseline
from app.training.trainer import STGNNTrainer
from app.training.ablation import AblationStudyRunner, TemporalOnlyGNN, SpatialOnlyGNN


# =====================================================================
# TEST 01: Benchmark Dataset Adapter generates continuous snapshots
# =====================================================================
def test_01_benchmark_dataset_adapter():
    contract = TrafficDatasetAdapter.generate_research_benchmark_dataset(
        num_nodes=5, num_timesteps=40, seed=42, window_seconds=300.0
    )
    assert len(contract.snapshots) == 40
    assert contract.window_seconds == 300.0
    # Verify node count per snapshot
    snap0 = contract.snapshots[0]
    assert len(snap0.nodes) == 5
    assert snap0.nodes[0].flow_rate is not None
    assert snap0.nodes[0].average_speed is not None


# =====================================================================
# TEST 02: Adapter preserves distance-based graph edges
# =====================================================================
def test_02_adapter_preserves_graph_edges():
    contract = TrafficDatasetAdapter.generate_research_benchmark_dataset(
        num_nodes=4, num_timesteps=20, seed=42
    )
    snap0 = contract.snapshots[0]
    assert len(snap0.edges) > 0
    e0 = snap0.edges[0]
    assert e0.source != e0.target
    assert e0.travel_time is not None


# =====================================================================
# TEST 03: ST-GNN Layer GraphSpatialConv forward pass
# =====================================================================
def test_03_graph_spatial_conv():
    conv = GraphSpatialConv(in_features=8, out_features=16)
    x = torch.randn(2, 3, 5, 8)  # [B, T, N, F]
    adj = torch.eye(5) + torch.ones(5, 5) * 0.2
    adj_norm = GraphSpatialConv.normalize_adjacency(adj)
    out = conv(x, adj_norm)
    assert out.shape == (2, 3, 5, 16)


# =====================================================================
# TEST 04: ST-GNN Layer TemporalGRUBlock forward pass
# =====================================================================
def test_04_temporal_gru_block():
    gru_block = TemporalGRUBlock(input_dim=16, hidden_dim=32, num_layers=1)
    x = torch.randn(4, 3, 5, 16)  # [B, T_in, N, F]
    all_states, last_state = gru_block(x)
    assert all_states.shape == (4, 3, 5, 32)
    assert last_state.shape == (4, 5, 32)


# =====================================================================
# TEST 05: STGNNBlock integrated layer
# =====================================================================
def test_05_stgnn_block():
    block = STGNNBlock(hidden_dim=32, use_residual=True, use_layer_norm=True)
    x = torch.randn(2, 3, 4, 32)
    adj = torch.eye(4)
    out = block(x, adj)
    assert out.shape == (2, 3, 4, 32)


# =====================================================================
# TEST 06: Full SpatioTemporalGNN forward pass & output shape
# =====================================================================
def test_06_spatio_temporal_gnn_forward():
    config = STGNNConfig(
        input_dim=8,
        output_dim=4,
        hidden_dim=32,
        num_spatial_layers=2,
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],
    )
    model = SpatioTemporalGNN(config=config)
    assert model.get_parameter_count() > 0

    x = torch.randn(2, 3, 6, 8)  # B=2, T=3, N=6, F=8
    adj = torch.eye(6)
    out = model(x, adj)
    # Output shape: [B, H, N, output_dim] -> [2, 3, 6, 4]
    assert out.shape == (2, 3, 6, 4)


# =====================================================================
# TEST 07: Masked Loss correctly ignores missing (masked) entries
# =====================================================================
def test_07_masked_loss_ignores_masked_values():
    loss_fn = MaskedLoss(loss_type="mae")
    pred = torch.tensor([[10.0, 5000.0], [20.0, 9999.0]])
    target = torch.tensor([[12.0, float("nan")], [22.0, 0.0]])
    mask = torch.tensor([[True, False], [True, False]])  # ignore the 2nd column

    loss = loss_fn(pred, target, mask=mask)
    # Only evaluates col 0: |10-12| = 2, |20-22| = 2 -> mean = 2.0
    assert abs(loss.item() - 2.0) < 1e-4


# =====================================================================
# TEST 08: Safe MAPE calculation with near-zero denominator
# =====================================================================
def test_08_safe_mape_calculation():
    pred = torch.tensor([0.0, 10.0])
    target = torch.tensor([0.0, 10.0])
    mape = MaskedMAPE(pred, target, epsilon=1.0)
    assert not math.isnan(mape)
    assert mape == 0.0


# =====================================================================
# TEST 09: Persistence Baseline prediction shape and values
# =====================================================================
def test_09_persistence_baseline():
    baseline = PersistenceBaseline()
    x = torch.zeros(2, 3, 4, 8)
    x[:, -1, :, 1] = 150.0  # last step flow = 150.0
    preds = baseline.predict(x, num_horizons=3, num_targets=4, feature_to_target_map=[1, 2, 3, 4])
    assert preds.shape == (2, 3, 4, 4)
    # Target 0 (flow) at all horizons should equal 150.0
    assert torch.all(preds[:, :, :, 0] == 150.0)


# =====================================================================
# TEST 10: Moving Average Baseline prediction
# =====================================================================
def test_10_moving_average_baseline():
    baseline = MovingAverageBaseline()
    x = torch.zeros(2, 3, 4, 8)
    # Step 0: 100, Step 1: 200, Step 2: 300 -> Mean = 200.0
    x[:, 0, :, 1] = 100.0
    x[:, 1, :, 1] = 200.0
    x[:, 2, :, 1] = 300.0
    preds = baseline.predict(x, num_horizons=3, num_targets=4, feature_to_target_map=[1, 2, 3, 4])
    assert torch.all(preds[:, :, :, 0] == 200.0)


# =====================================================================
# TEST 11: Linear Regression Baseline fit and predict
# =====================================================================
def test_11_linear_regression_baseline():
    baseline = LinearRegressionBaseline(alpha=1.0)
    x_train = torch.randn(10, 3, 4, 8)
    y_train = torch.randn(10, 3, 4, 4)
    baseline.fit(x_train, y_train)
    
    preds = baseline.predict(x_train, num_horizons=3, num_targets=4)
    assert preds.shape == (10, 3, 4, 4)


# =====================================================================
# TEST 12: End-to-End STGNNTrainer training loop & checkpoint saving
# =====================================================================
def test_12_trainer_end_to_end_training(tmp_path):
    contract = TrafficDatasetAdapter.generate_research_benchmark_dataset(
        num_nodes=4, num_timesteps=30, seed=42
    )
    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN"],
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
    )
    dataset = slicer.slice_dataset(contract)
    
    config = STGNNConfig(
        input_dim=len(dataset.feature_names),
        output_dim=len(dataset.target_names),
        forecast_horizons=dataset.forecast_horizons,
        hidden_dim=16,
    )
    model = SpatioTemporalGNN(config=config)
    ckpt_dir = str(tmp_path / "checkpoints")
    trainer = STGNNTrainer(model, config=config, checkpoint_dir=ckpt_dir)

    result = trainer.train(dataset, epochs=3, batch_size=8, save_checkpoint_name="test_model.pt")
    assert result.epochs_trained == 3
    assert result.best_val_loss >= 0.0
    assert "ST-GNN" in result.test_metrics
    assert "Persistence" in result.test_metrics
    assert os.path.exists(os.path.join(ckpt_dir, "test_model.pt"))


# =====================================================================
# TEST 13: Ablation Study Runner builds Temporal-Only and Spatial-Only models
# =====================================================================
def test_13_ablation_models():
    config = STGNNConfig(input_dim=8, output_dim=4, hidden_dim=16)
    temp_model = TemporalOnlyGNN(config)
    spat_model = SpatialOnlyGNN(config)
    
    x = torch.randn(2, 3, 4, 8)
    adj = torch.eye(4)
    
    out_temp = temp_model(x, adj)
    out_spat = spat_model(x, adj)
    
    assert out_temp.shape == (2, 3, 4, 4)
    assert out_spat.shape == (2, 3, 4, 4)
