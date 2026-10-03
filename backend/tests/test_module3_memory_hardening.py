"""
ChronoEye Infinity - Module 3 Memory Hardening & Scalability Test Suite.
Verifies compact PeMS08 representation, lazy temporal sliding windows,
train-only normalization, missing data masks, PyTorch DataLoader mini-batching,
GPU memory safety, 576-step regression, and frozen Stage 1-4 hash integrity.
"""

import os
import math
import hashlib
import pytest
import numpy as np
import torch
from pathlib import Path

from app.adapters.real_traffic_adapter import RealTrafficDatasetAdapter, CompactTrafficDataset
from app.graph.temporal_dataset_slicer import (
    TemporalDatasetSlicer,
    CompactSpatioTemporalDataset,
    LazySpatioTemporalSplit,
    DatasetSplit,
)
from app.models.stgnn.config import STGNNConfig
from app.models.stgnn.model import SpatioTemporalGNN
from app.models.stgnn.loss import MaskedLoss
from app.training.trainer import STGNNTrainer
from app.training.baselines import PersistenceBaseline, MovingAverageBaseline, LinearRegressionBaseline


NPZ_PATH = "data/research/pems08/PEMS08.npz"
CSV_PATH = "data/research/pems08/PEMS08.csv"


# =====================================================================
# TEST 1: Full PEMS08 compact representation loads (17856 x 170 x 8)
# =====================================================================
def test_01_full_pems08_compact_loading():
    if not Path(NPZ_PATH).exists():
        pytest.skip("PEMS08 dataset not available in test environment.")

    compact_data, meta = RealTrafficDatasetAdapter.load_pems08_compact(
        npz_path=NPZ_PATH,
        csv_path=CSV_PATH,
        num_timesteps=None,  # All 17,856
        num_nodes=None,      # All 170
    )

    assert isinstance(compact_data, CompactTrafficDataset)
    assert compact_data.features.shape == (17856, 170, 8)
    assert compact_data.targets.shape == (17856, 170, 4)
    assert compact_data.features.dtype == np.float32
    assert compact_data.targets.dtype == np.float32
    assert compact_data.num_timesteps == 17856
    assert compact_data.num_nodes == 170
    assert len(compact_data.feature_names) == 8
    assert len(compact_data.target_names) == 4
    assert meta["compact_mode"] is True


# =====================================================================
# TEST 2: Static topology has 170 nodes and 295 edges
# =====================================================================
def test_02_static_topology_preserved():
    if not Path(NPZ_PATH).exists():
        pytest.skip("PEMS08 dataset not available.")

    compact_data, _ = RealTrafficDatasetAdapter.load_pems08_compact(
        npz_path=NPZ_PATH,
        csv_path=CSV_PATH,
        num_timesteps=10,
        num_nodes=170,
    )

    assert compact_data.adjacency_matrix.shape == (170, 170)
    assert len(compact_data.edge_index[0]) == 295
    assert len(compact_data.edge_index[1]) == 295
    assert len(compact_data.edge_weights) == 295
    assert compact_data.node_ids[0] == "PEMS08_SENSOR_000"
    assert compact_data.node_ids[-1] == "PEMS08_SENSOR_169"


# =====================================================================
# TEST 3: Lazy dataset length is correct (17,842 samples for T_in=12, H_max=3)
# =====================================================================
def test_03_lazy_dataset_length_mathematically_exact():
    if not Path(NPZ_PATH).exists():
        pytest.skip("PEMS08 dataset not available.")

    compact_data, _ = RealTrafficDatasetAdapter.load_pems08_compact(
        npz_path=NPZ_PATH,
        csv_path=CSV_PATH,
        num_timesteps=17856,
        num_nodes=170,
    )

    slicer = TemporalDatasetSlicer(
        input_sequence_length=12,
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],  # offsets 1, 2, 3
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    dataset = slicer.slice_dataset(compact_data)

    # Required window length = 12 + 3 = 15
    # Total sliding windows = 17856 - 15 + 1 = 17,842
    expected_total = 17856 - (12 + 3) + 1
    assert expected_total == 17842
    assert dataset.total_samples == 17842


# =====================================================================
# TEST 4: __getitem__ returns correct tensor shapes
# =====================================================================
def test_04_getitem_returns_correct_tensor_shapes():
    if not Path(NPZ_PATH).exists():
        pytest.skip("PEMS08 dataset not available.")

    compact_data, _ = RealTrafficDatasetAdapter.load_pems08_compact(
        npz_path=NPZ_PATH,
        csv_path=CSV_PATH,
        num_timesteps=50,
        num_nodes=170,
    )

    slicer = TemporalDatasetSlicer(
        input_sequence_length=12,
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    dataset = slicer.slice_dataset(compact_data)

    x, x_mask, y, y_mask = dataset.train_dataset[0]

    assert isinstance(x, torch.Tensor)
    assert x.shape == (12, 170, 8)
    assert x_mask.shape == (12, 170, 8)
    assert x_mask.dtype == torch.bool
    assert y.shape == (3, 170, 4)
    assert y_mask.shape == (3, 170, 4)
    assert y_mask.dtype == torch.bool


# =====================================================================
# TEST 5: Chronological split 70 / 15 / 15
# =====================================================================
def test_05_chronological_split_full_dataset():
    if not Path(NPZ_PATH).exists():
        pytest.skip("PEMS08 dataset not available.")

    compact_data, _ = RealTrafficDatasetAdapter.load_pems08_compact(
        npz_path=NPZ_PATH,
        csv_path=CSV_PATH,
        num_timesteps=17856,
        num_nodes=170,
    )

    slicer = TemporalDatasetSlicer(
        input_sequence_length=12,
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    dataset = slicer.slice_dataset(compact_data)

    train_n = len(dataset.train_samples)
    val_n = len(dataset.val_samples)
    test_n = len(dataset.test_samples)

    assert train_n == int(17842 * 0.70)  # 12489
    assert val_n == int(17842 * 0.15)    # 2676
    assert test_n == 17842 - 12489 - 2676 # 2677
    assert train_n + val_n + test_n == 17842


# =====================================================================
# TEST 6: 576-step regression (393 / 84 / 85)
# =====================================================================
def test_06_regression_576_steps_counts():
    if not Path(NPZ_PATH).exists():
        pytest.skip("PEMS08 dataset not available.")

    compact_data, _ = RealTrafficDatasetAdapter.load_pems08_compact(
        npz_path=NPZ_PATH,
        csv_path=CSV_PATH,
        num_timesteps=576,
        num_nodes=30,
    )

    slicer = TemporalDatasetSlicer(
        input_sequence_length=12,
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    dataset = slicer.slice_dataset(compact_data)

    # 576 - 15 + 1 = 562 total samples
    # 562 * 0.70 = 393.4 -> 393
    # 562 * 0.15 = 84.3 -> 84
    # Test = 562 - 393 - 84 = 85
    assert len(dataset.train_samples) == 393
    assert len(dataset.val_samples) == 84
    assert len(dataset.test_samples) == 85
    assert dataset.total_samples == 562


# =====================================================================
# TEST 7: Train-only normalization (Zero Leakage)
# =====================================================================
def test_07_train_only_normalization_strict_isolation():
    # Build synthetic compact dataset with extreme values in val/test
    T, N, F = 100, 5, 8
    features = np.zeros((T, N, F), dtype=np.float32)
    # Train timesteps (0..70): flow = 10.0
    features[:70, :, 1] = 10.0
    # Val/Test timesteps (70..100): extreme anomaly = 9999.0
    features[70:, :, 1] = 9999.0

    targets = np.zeros((T, N, 4), dtype=np.float32)
    targets[:70, :, 0] = 10.0
    targets[70:, :, 0] = 9999.0

    compact_data = CompactTrafficDataset(
        dataset_id="TEST_LEAKAGE",
        features=features,
        feature_mask=np.array([True] * 8, dtype=bool),
        targets=targets,
        target_mask=np.array([True] * 4, dtype=bool),
        adjacency_matrix=np.eye(N, dtype=np.float32),
        edge_index=[[0], [1]],
        edge_weights=[1.0],
        node_ids=[f"NODE_{i}" for i in range(N)],
        node_to_idx={f"NODE_{i}": i for i in range(N)},
        num_nodes=N,
        num_timesteps=T,
        feature_names=["vehicle_count", "flow_rate", "density", "average_speed", "queue_length", "congestion", "incoming_flow", "outgoing_flow"],
        target_names=["flow_rate", "density", "congestion", "travel_time"],
    )

    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,
        forecast_horizons=["PLUS_5MIN"],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    dataset = slicer.slice_dataset(compact_data)

    mean_flow = dataset.feature_scaler.mean["flow_rate"]
    target_mean = dataset.target_scaler.mean["flow_rate"]

    # Must reflect training data (10.0), NOT contaminated by 9999.0
    assert mean_flow == 10.0, f"Leakage violation: mean_flow is {mean_flow}"
    assert target_mean == 10.0, f"Leakage violation: target_mean is {target_mean}"


# =====================================================================
# TEST 8: Masks are preserved (unobserved loop detector fields masked)
# =====================================================================
def test_08_missing_masks_preservation():
    if not Path(NPZ_PATH).exists():
        pytest.skip("PEMS08 dataset not available.")

    compact_data, _ = RealTrafficDatasetAdapter.load_pems08_compact(
        npz_path=NPZ_PATH,
        csv_path=CSV_PATH,
        num_timesteps=30,
        num_nodes=10,
    )

    slicer = TemporalDatasetSlicer(input_sequence_length=12, forecast_horizons=["PLUS_5MIN"])
    dataset = slicer.slice_dataset(compact_data)

    x, x_mask, y, y_mask = dataset.train_dataset[0]

    # Features: queue_length (4), incoming_flow (6), outgoing_flow (7) must be False
    assert x_mask[0, 0, 0].item() is True   # vehicle_count
    assert x_mask[0, 0, 1].item() is True   # flow_rate
    assert x_mask[0, 0, 2].item() is True   # density
    assert x_mask[0, 0, 3].item() is True   # average_speed
    assert x_mask[0, 0, 4].item() is False  # queue_length (unobserved)
    assert x_mask[0, 0, 5].item() is True   # congestion
    assert x_mask[0, 0, 6].item() is False  # incoming_flow (unobserved)
    assert x_mask[0, 0, 7].item() is False  # outgoing_flow (unobserved)

    # Targets: travel_time (3) must be False
    assert y_mask[0, 0, 0].item() is True   # flow_rate
    assert y_mask[0, 0, 1].item() is True   # density
    assert y_mask[0, 0, 2].item() is True   # congestion
    assert y_mask[0, 0, 3].item() is False  # travel_time (unobserved)


# =====================================================================
# TEST 9: DataLoader produces configurable batches
# =====================================================================
def test_09_dataloader_batching():
    if not Path(NPZ_PATH).exists():
        pytest.skip("PEMS08 dataset not available.")

    compact_data, _ = RealTrafficDatasetAdapter.load_pems08_compact(
        npz_path=NPZ_PATH,
        csv_path=CSV_PATH,
        num_timesteps=100,
        num_nodes=20,
    )

    slicer = TemporalDatasetSlicer(input_sequence_length=12, forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"])
    dataset = slicer.slice_dataset(compact_data)

    loader = dataset.get_dataloader("train", batch_size=8, shuffle=True)
    batch = next(iter(loader))
    bx, bx_mask, by, by_mask = batch

    assert bx.shape == (8, 12, 20, 8)
    assert bx_mask.shape == (8, 12, 20, 8)
    assert by.shape == (8, 3, 20, 4)
    assert by_mask.shape == (8, 3, 20, 4)


# =====================================================================
# TEST 10: Mini-batch memory isolation (Only batch is moved to device)
# =====================================================================
def test_10_device_transfer_isolation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = STGNNConfig(input_dim=8, output_dim=4, hidden_dim=16, forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"])
    model = SpatioTemporalGNN(config=config)
    trainer = STGNNTrainer(model, config=config)

    bx = torch.randn(4, 12, 10, 8)
    adj = torch.eye(10)

    # Forward pass with only batch transferred
    bx_dev = bx.to(trainer.device)
    adj_dev = adj.to(trainer.device)
    out = trainer.model(bx_dev, adj_dev)

    assert out.shape == (4, 3, 10, 4)
    assert out.device == trainer.device


# =====================================================================
# TEST 11: End-to-end CPU smoke training test (1-2 epochs)
# =====================================================================
def test_11_trainer_smoke_training(tmp_path):
    if not Path(NPZ_PATH).exists():
        pytest.skip("PEMS08 dataset not available.")

    compact_data, _ = RealTrafficDatasetAdapter.load_pems08_compact(
        npz_path=NPZ_PATH,
        csv_path=CSV_PATH,
        num_timesteps=60,
        num_nodes=10,
    )

    slicer = TemporalDatasetSlicer(
        input_sequence_length=12,
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
    )
    dataset = slicer.slice_dataset(compact_data)

    config = STGNNConfig(
        input_dim=8,
        output_dim=4,
        hidden_dim=16,
        forecast_horizons=dataset.forecast_horizons,
        num_st_blocks=1,
    )
    model = SpatioTemporalGNN(config=config)
    trainer = STGNNTrainer(model, config=config, checkpoint_dir=str(tmp_path / "ckpts"))

    result = trainer.train(dataset, epochs=2, batch_size=8, save_checkpoint_name="smoke_model.pt")

    assert result.epochs_trained == 2
    assert result.best_val_loss >= 0.0
    assert "ST-GNN" in result.test_metrics
    assert "Persistence" in result.test_metrics
    assert "Moving_Average" in result.test_metrics
    assert "Linear_Regression" in result.test_metrics
    assert os.path.exists(tmp_path / "ckpts" / "smoke_model.pt")


# =====================================================================
# TEST 12: CUDA device handling
# =====================================================================
def test_12_cuda_cpu_device_handling():
    config = STGNNConfig(input_dim=8, output_dim=4, hidden_dim=16)
    model = SpatioTemporalGNN(config=config)
    trainer = STGNNTrainer(model, config=config)

    expected_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    assert trainer.device.type == expected_device_type


# =====================================================================
# TEST 13: Numerical equivalence of compact vs eager representation
# =====================================================================
def test_13_numerical_equivalence_compact_vs_eager():
    if not Path(NPZ_PATH).exists():
        pytest.skip("PEMS08 dataset not available.")

    # Load 30 steps with both methods
    contract_eager, _ = RealTrafficDatasetAdapter.load_pems08(
        npz_path=NPZ_PATH,
        csv_path=CSV_PATH,
        num_timesteps=30,
        num_nodes=5,
        compact=False,
    )
    compact_data, _ = RealTrafficDatasetAdapter.load_pems08_compact(
        npz_path=NPZ_PATH,
        csv_path=CSV_PATH,
        num_timesteps=30,
        num_nodes=5,
    )

    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=["PLUS_5MIN"], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    ds_eager = slicer.slice_dataset(contract_eager)
    ds_compact = slicer.slice_dataset(compact_data)

    assert ds_eager.total_samples == ds_compact.total_samples

    # Check feature scaler means match
    for fname in ["flow_rate", "density", "average_speed", "congestion"]:
        e_mean = ds_eager.feature_scaler.mean[fname]
        c_mean = ds_compact.feature_scaler.mean[fname]
        assert abs(e_mean - c_mean) < 1e-3, f"Mismatch on {fname} mean: eager={e_mean}, compact={c_mean}"


# =====================================================================
# TEST 14: Frozen Stage 1–4 hashes remain unchanged
# =====================================================================
def test_14_frozen_stage1_to_4_hashes():
    expected_hashes = {
        "frame_source.py": "A5163E66F80AE669544233F467CC617608A7C2454EE2D1A07A03F3075103FAA3",
        "detector.py": "C5E1051B7AFF67D6939A7D2BE89B83096C6CBCF07A3F9572B4CC34A547291EC2",
        "bytetrack.py": "B6A551BF820610D19396D8482DC6826B4C66A6F8187D18CC3091C8CF6A24771E",
        "plate_association.py": "D04553F319574126735921C36C51E23D92B321A1511CD5173AD7965690E3E588",
    }
    paths = {
        "frame_source.py": "backend/app/perception/frame_source.py",
        "detector.py": "backend/app/perception/detector.py",
        "bytetrack.py": "backend/app/perception/bytetrack.py",
        "plate_association.py": "backend/app/perception/plate_association.py",
    }

    for name, exp_hash in expected_hashes.items():
        filepath = paths[name]
        assert os.path.exists(filepath), f"Frozen file {filepath} not found"
        with open(filepath, "rb") as f:
            content = f.read()
        calc_hash = hashlib.sha256(content).hexdigest().upper()
        assert calc_hash == exp_hash, f"CRITICAL SECURITY VIOLATION: Frozen file {name} hash changed from {exp_hash} to {calc_hash}"
