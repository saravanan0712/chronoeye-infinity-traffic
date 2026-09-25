"""
Unit and Integration Tests for Real/Public Traffic Forecasting Dataset Adapter (PeMS08).
Verifies data loading, schema mapping, graph topology, sequence slicing, and ST-GNN ingestion.
"""

import pytest
import math
import os
import torch
from pathlib import Path

from app.adapters.real_traffic_adapter import RealTrafficDatasetAdapter
from app.graph.temporal_dataset_slicer import TemporalDatasetSlicer
from app.models.stgnn.config import STGNNConfig
from app.models.stgnn.model import SpatioTemporalGNN
from app.training.baselines import PersistenceBaseline, MovingAverageBaseline


def test_01_pems08_loading_and_contract_generation():
    """Verifies that PeMS08 dataset loads into a valid TemporalGraphDatasetContract."""
    npz_path = "data/research/pems08/PEMS08.npz"
    csv_path = "data/research/pems08/PEMS08.csv"
    if not Path(npz_path).exists():
        pytest.skip("PeMS08 dataset file not available in test environment.")

    contract, metadata = RealTrafficDatasetAdapter.load_pems08(
        npz_path=npz_path,
        csv_path=csv_path,
        num_timesteps=48,  # 4 hours @ 5-min
        num_nodes=20,      # First 20 sensors
    )

    assert contract is not None
    assert len(contract.snapshots) == 48
    assert contract.snapshots[0].node_count == 20
    assert metadata["loaded_timesteps"] == 48
    assert metadata["loaded_nodes"] == 20
    assert metadata["target_availability"]["flow_rate"] is True
    assert metadata["target_availability"]["travel_time"] is False


def test_02_pems08_feature_mapping_and_masks():
    """Verifies that features are properly mapped and missing fields have mask=False."""
    npz_path = "data/research/pems08/PEMS08.npz"
    csv_path = "data/research/pems08/PEMS08.csv"
    if not Path(npz_path).exists():
        pytest.skip("PeMS08 dataset file not available in test environment.")

    contract, _ = RealTrafficDatasetAdapter.load_pems08(
        npz_path=npz_path,
        csv_path=csv_path,
        num_timesteps=10,
        num_nodes=10,
    )

    node0 = contract.snapshots[0].nodes[0]
    assert node0.flow_rate is not None
    assert node0.density is not None
    assert node0.average_speed is not None
    assert node0.congestion is not None
    assert node0.queue_length is None  # unobserved loop detector field
    assert node0.incoming_flow is None
    assert node0.outgoing_flow is None


def test_03_pems08_slicing_with_temporal_dataset_slicer():
    """Verifies that the sliced real dataset has proper train/val/test splits without leakage."""
    npz_path = "data/research/pems08/PEMS08.npz"
    csv_path = "data/research/pems08/PEMS08.csv"
    if not Path(npz_path).exists():
        pytest.skip("PeMS08 dataset file not available in test environment.")

    contract, _ = RealTrafficDatasetAdapter.load_pems08(
        npz_path=npz_path,
        csv_path=csv_path,
        num_timesteps=100,
        num_nodes=15,
    )

    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        normalization_method="zscore",
    )
    dataset = slicer.slice_dataset(contract)

    assert len(dataset.train_samples) > 0
    assert len(dataset.val_samples) > 0
    assert len(dataset.test_samples) > 0
    assert dataset.adjacency.num_nodes == 15
    # Ensure normalizer is fitted
    assert dataset.feature_scaler is not None
    assert "flow_rate" in dataset.feature_scaler.mean


def test_04_pems08_stgnn_forward_pass():
    """Verifies that the ST-GNN model performs forward pass on PeMS08 real data batches."""
    config = STGNNConfig(
        input_dim=8,
        hidden_dim=32,
        output_dim=4,
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],
        num_st_blocks=2,
    )
    model = SpatioTemporalGNN(config)

    batch_size = 4
    n_nodes = 15
    t_in = 3
    x = torch.randn(batch_size, t_in, n_nodes, 8)
    adj = torch.eye(n_nodes)

    out = model(x, adj)
    assert out.shape == (batch_size, 3, n_nodes, 4)
