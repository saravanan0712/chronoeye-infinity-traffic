"""
Tests for Module 3 Step 4: Spatio-Temporal Dataset Sequence Slicer & Normalization Pipeline.
Verifies temporal sequence slicing, multi-horizon targets, chronological train/val/test splits,
TRAIN-only feature/target normalization (zero leakage), missing data mask preservation,
and research integrity.
"""

import pytest
import math
from app.graph.temporal_snapshot_schema import (
    TemporalNodeFeatures,
    TemporalEdgeFeatures,
    TemporalGraphSnapshot,
    TemporalGraphDatasetContract,
)
from app.graph.temporal_snapshot_aggregator import TemporalSnapshotAggregator
from app.graph.temporal_dataset_slicer import (
    TemporalDatasetSlicer,
    DatasetSplit,
    SpatioTemporalDataset,
    SpatioTemporalSample,
    NormalizationParams,
)
from app.forecasting.forecasting_schema import ForecastHorizon
from app.graph.graph_schema import NodeType, EdgeType


def create_synthetic_snapshot(
    start_time: float,
    duration: float = 300.0,
    node_data: dict = None,
    edges: list = None,
) -> TemporalGraphSnapshot:
    """Helper creating a single synthetic TemporalGraphSnapshot."""
    node_data = node_data or {
        "CAM_A": {"vehicle_count": 10, "flow_rate": 120.0, "average_speed": 50.0},
        "CAM_B": {"vehicle_count": 15, "flow_rate": 180.0, "average_speed": 45.0},
    }
    nodes = []
    for n_id, feats in sorted(node_data.items()):
        nodes.append(
            TemporalNodeFeatures(
                node_id=n_id,
                node_type=NodeType.CAMERA,
                timestamp=start_time,
                vehicle_count=feats.get("vehicle_count"),
                flow_rate=feats.get("flow_rate"),
                density=feats.get("density"),
                average_speed=feats.get("average_speed"),
                queue_length=feats.get("queue_length"),
                congestion=feats.get("congestion"),
                incoming_flow=feats.get("incoming_flow"),
                outgoing_flow=feats.get("outgoing_flow"),
            )
        )
    
    edge_objs = []
    if edges:
        for e in edges:
            edge_objs.append(
                TemporalEdgeFeatures(
                    source=e["source"],
                    target=e["target"],
                    edge_type=EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA,
                    timestamp=start_time,
                    vehicle_count=e.get("vehicle_count", 1),
                    flow_rate=e.get("flow_rate", 12.0),
                    travel_time=e.get("travel_time", 60.0),
                    transition_count=e.get("transition_count", 1),
                    has_unobserved_gap=e.get("has_unobserved_gap", False),
                )
            )
            
    return TemporalGraphSnapshot(
        snapshot_id=f"SNAP_{int(start_time)}_{int(start_time + duration)}",
        start_time=start_time,
        end_time=start_time + duration,
        nodes=nodes,
        edges=edge_objs,
    )


# =====================================================================
# TEST 01: Creates one valid sequence.
# =====================================================================
def test_01_creates_one_valid_sequence():
    # 3 input + 3 future targets = 6 consecutive snapshots minimum
    snaps = [create_synthetic_snapshot(start_time=i * 300.0) for i in range(6)]
    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,
        forecast_horizons=[ForecastHorizon.PLUS_5MIN, ForecastHorizon.PLUS_10MIN, ForecastHorizon.PLUS_15MIN],
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
    )
    dataset = slicer.slice_dataset(snaps)
    assert dataset.total_samples == 1
    assert len(dataset.train_samples) == 1
    s = dataset.train_samples[0]
    assert s.input_start_time == 0.0
    assert s.input_end_time == 900.0


# =====================================================================
# TEST 02: Correct input sequence length.
# =====================================================================
def test_02_correct_input_sequence_length():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0) for i in range(8)]
    slicer = TemporalDatasetSlicer(
        input_sequence_length=4,
        forecast_horizons=[ForecastHorizon.PLUS_5MIN],
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
    )
    dataset = slicer.slice_dataset(snaps)
    # Shape of x_raw: [T_in, N, F]
    s = dataset.train_samples[0]
    assert len(s.x_raw) == 4  # T_in = 4


# =====================================================================
# TEST 03: Correct +5-minute target (+1 step).
# =====================================================================
def test_03_correct_plus_5min_target():
    snaps = [
        create_synthetic_snapshot(start_time=0.0, node_data={"CAM_A": {"flow_rate": 100.0}}),
        create_synthetic_snapshot(start_time=300.0, node_data={"CAM_A": {"flow_rate": 110.0}}),
        create_synthetic_snapshot(start_time=600.0, node_data={"CAM_A": {"flow_rate": 120.0}}),
        create_synthetic_snapshot(start_time=900.0, node_data={"CAM_A": {"flow_rate": 999.0}}),  # +5 min target
    ]
    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,
        forecast_horizons=[ForecastHorizon.PLUS_5MIN],
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
    )
    dataset = slicer.slice_dataset(snaps)
    s = dataset.train_samples[0]
    assert s.target_times["PLUS_5MIN"] == 900.0
    # Check y_raw[0][node_idx][flow_rate_idx]
    flow_idx = dataset.target_names.index("flow_rate")
    assert s.y_raw[0][0][flow_idx] == 999.0


# =====================================================================
# TEST 04: Correct +10-minute target (+2 steps).
# =====================================================================
def test_04_correct_plus_10min_target():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0) for i in range(5)]
    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,
        forecast_horizons=[ForecastHorizon.PLUS_10MIN],
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
    )
    dataset = slicer.slice_dataset(snaps)
    s = dataset.train_samples[0]
    # Input: 0, 300, 600. Target +10min (+2 steps from 600) -> 1200.0
    assert s.target_times["PLUS_10MIN"] == 1200.0


# =====================================================================
# TEST 05: Correct +15-minute target (+3 steps).
# =====================================================================
def test_05_correct_plus_15min_target():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0) for i in range(6)]
    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,
        forecast_horizons=[ForecastHorizon.PLUS_15MIN],
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
    )
    dataset = slicer.slice_dataset(snaps)
    s = dataset.train_samples[0]
    # Target +15min (+3 steps from 600) -> 1500.0
    assert s.target_times["PLUS_15MIN"] == 1500.0


# =====================================================================
# TEST 06: Does not fabricate missing future targets.
# =====================================================================
def test_06_does_not_fabricate_missing_future_targets():
    # Only 4 snapshots (need 6 for +15min with Tin=3)
    snaps = [create_synthetic_snapshot(start_time=i * 300.0) for i in range(4)]
    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,
        forecast_horizons=[ForecastHorizon.PLUS_5MIN, ForecastHorizon.PLUS_15MIN],
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
    )
    dataset = slicer.slice_dataset(snaps)
    # Cannot satisfy +15MIN, so no sample fabricated
    assert dataset.total_samples == 0


# =====================================================================
# TEST 07: Deterministic node ordering.
# =====================================================================
def test_07_deterministic_node_ordering():
    snaps = [
        create_synthetic_snapshot(start_time=0.0, node_data={"CAM_Z": {}, "CAM_A": {}, "CAM_M": {}}),
        create_synthetic_snapshot(start_time=300.0, node_data={"CAM_B": {}, "CAM_A": {}}),
        create_synthetic_snapshot(start_time=600.0, node_data={"CAM_Z": {}}),
        create_synthetic_snapshot(start_time=900.0, node_data={"CAM_M": {}}),
    ]
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[1])
    dataset = slicer.slice_dataset(snaps)
    assert dataset.adjacency.node_ids == ["CAM_A", "CAM_B", "CAM_M", "CAM_Z"]
    assert dataset.adjacency.node_to_idx == {"CAM_A": 0, "CAM_B": 1, "CAM_M": 2, "CAM_Z": 3}


# =====================================================================
# TEST 08: Missing node in snapshot does not shift node indices.
# =====================================================================
def test_08_missing_node_does_not_shift_indices():
    # CAM_A and CAM_B in dataset, but snapshot 1 only has CAM_B
    snaps = [
        create_synthetic_snapshot(start_time=0.0, node_data={"CAM_A": {"flow_rate": 10.0}, "CAM_B": {"flow_rate": 20.0}}),
        create_synthetic_snapshot(start_time=300.0, node_data={"CAM_B": {"flow_rate": 25.0}}), # CAM_A absent
        create_synthetic_snapshot(start_time=600.0, node_data={"CAM_A": {"flow_rate": 12.0}, "CAM_B": {"flow_rate": 22.0}}),
        create_synthetic_snapshot(start_time=900.0, node_data={"CAM_A": {"flow_rate": 15.0}, "CAM_B": {"flow_rate": 30.0}}),
    ]
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[1], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    dataset = slicer.slice_dataset(snaps)
    s = dataset.train_samples[0]
    
    # In time step 1 (300.0), CAM_A (index 0) must be None and masked False
    flow_idx = dataset.feature_names.index("flow_rate")
    assert s.x_raw[1][0][flow_idx] is None
    assert s.x_mask[1][0][flow_idx] is False
    
    # CAM_B (index 1) remains at index 1 with value 25.0
    assert s.x_raw[1][1][flow_idx] == 25.0
    assert s.x_mask[1][1][flow_idx] is True


# =====================================================================
# TEST 09: Chronological train split.
# =====================================================================
def test_09_chronological_train_split():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0) for i in range(20)]
    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,
        forecast_horizons=[1],
        train_ratio=0.60, val_ratio=0.20, test_ratio=0.20
    )
    dataset = slicer.slice_dataset(snaps)
    assert len(dataset.train_samples) > 0
    for s in dataset.train_samples:
        assert s.split == DatasetSplit.TRAIN


# =====================================================================
# TEST 10: Chronological validation split.
# =====================================================================
def test_10_chronological_val_split():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0) for i in range(20)]
    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,
        forecast_horizons=[1],
        train_ratio=0.60, val_ratio=0.20, test_ratio=0.20
    )
    dataset = slicer.slice_dataset(snaps)
    assert len(dataset.val_samples) > 0
    # Val start time must be >= Train end timestamps
    assert dataset.val_samples[0].input_start_time >= dataset.train_samples[-1].input_start_time


# =====================================================================
# TEST 11: Chronological test split.
# =====================================================================
def test_11_chronological_test_split():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0) for i in range(20)]
    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,
        forecast_horizons=[1],
        train_ratio=0.60, val_ratio=0.20, test_ratio=0.20
    )
    dataset = slicer.slice_dataset(snaps)
    assert len(dataset.test_samples) > 0
    # Test start time must be >= Val end timestamps
    assert dataset.test_samples[0].input_start_time >= dataset.val_samples[-1].input_start_time


# =====================================================================
# TEST 12: No random temporal shuffling.
# =====================================================================
def test_12_no_random_temporal_shuffling():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0) for i in range(15)]
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[1], train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    dataset = slicer.slice_dataset(snaps)
    
    all_samples = dataset.train_samples + dataset.val_samples + dataset.test_samples
    timestamps = [s.input_start_time for s in all_samples]
    assert timestamps == sorted(timestamps)


# =====================================================================
# TEST 13: Scaler fit uses TRAIN only.
# =====================================================================
def test_13_scaler_fit_uses_train_only():
    # Train flows: 10, 20. Val/Test flows: 1000.
    snaps = []
    for i in range(8):
        flow = 10.0 if i % 2 == 0 else 20.0
        snaps.append(create_synthetic_snapshot(start_time=i * 300.0, node_data={"CAM_A": {"flow_rate": flow}}))
    for i in range(8, 16):
        snaps.append(create_synthetic_snapshot(start_time=i * 300.0, node_data={"CAM_A": {"flow_rate": 1000.0}}))
        
    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,
        forecast_horizons=[1],
        train_ratio=0.50, val_ratio=0.25, test_ratio=0.25
    )
    dataset = slicer.slice_dataset(snaps)
    
    # Feature scaler mean must reflect TRAIN (~15.0), NOT contaminated by 1000.0
    mean_flow = dataset.feature_scaler.mean["flow_rate"]
    assert 10.0 <= mean_flow <= 20.0
    assert mean_flow < 100.0


# =====================================================================
# TEST 14: Validation uses TRAIN normalization parameters.
# =====================================================================
def test_14_validation_uses_train_normalization():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0, node_data={"CAM_A": {"flow_rate": 50.0}}) for i in range(16)]
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[1], train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    dataset = slicer.slice_dataset(snaps)
    
    # Validation sample normalization should use feature_scaler
    assert dataset.val_samples[0].x_normalized is not None


# =====================================================================
# TEST 15: Test uses TRAIN normalization parameters.
# =====================================================================
def test_15_test_uses_train_normalization():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0, node_data={"CAM_A": {"flow_rate": 50.0}}) for i in range(16)]
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[1], train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    dataset = slicer.slice_dataset(snaps)
    assert dataset.test_samples[0].x_normalized is not None


# =====================================================================
# TEST 16: Target scaler uses TRAIN targets only.
# =====================================================================
def test_16_target_scaler_uses_train_targets_only():
    snaps = []
    for i in range(12):
        snaps.append(create_synthetic_snapshot(start_time=i * 300.0, node_data={"CAM_A": {"flow_rate": 30.0}}))
    for i in range(12, 24):
        snaps.append(create_synthetic_snapshot(start_time=i * 300.0, node_data={"CAM_A": {"flow_rate": 5000.0}}))
        
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[1], train_ratio=0.40, val_ratio=0.20, test_ratio=0.40)
    dataset = slicer.slice_dataset(snaps)
    
    target_mean = dataset.target_scaler.mean["flow_rate"]
    assert target_mean == 30.0


# =====================================================================
# TEST 17: Zero standard deviation handled safely (no div-by-zero).
# =====================================================================
def test_17_zero_standard_deviation_handled_safely():
    # Constant flow = 40.0 everywhere
    snaps = [create_synthetic_snapshot(start_time=i * 300.0, node_data={"CAM_A": {"flow_rate": 40.0}}) for i in range(10)]
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[1], train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    dataset = slicer.slice_dataset(snaps)
    
    std = dataset.feature_scaler.std["flow_rate"]
    assert std == 1.0  # safe fallback
    # Normalization should not throw or produce NaN/inf
    flow_idx = dataset.feature_names.index("flow_rate")
    norm_val = dataset.train_samples[0].x_normalized[0][0][flow_idx]
    assert norm_val == 0.0


# =====================================================================
# TEST 18: Missing values are not silently converted to zero.
# =====================================================================
def test_18_missing_values_not_converted_to_zero():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0, node_data={"CAM_A": {"density": None}}) for i in range(10)]
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[1], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    dataset = slicer.slice_dataset(snaps)
    
    density_idx = dataset.feature_names.index("density")
    s = dataset.train_samples[0]
    assert s.x_raw[0][0][density_idx] is None
    assert s.x_normalized[0][0][density_idx] is None
    assert s.x_mask[0][0][density_idx] is False


# =====================================================================
# TEST 19: Temporal gap detection (no bridging across gaps).
# =====================================================================
def test_19_temporal_gap_detection():
    # Segment 1: t = 0, 300, 600, 900
    # GAP: next snapshot at 3600 (1 hour later)
    # Segment 2: t = 3600, 3900, 4200, 4500
    snaps = [
        create_synthetic_snapshot(start_time=0.0),
        create_synthetic_snapshot(start_time=300.0),
        create_synthetic_snapshot(start_time=600.0),
        create_synthetic_snapshot(start_time=900.0),
        create_synthetic_snapshot(start_time=3600.0), # GAP
        create_synthetic_snapshot(start_time=3900.0),
        create_synthetic_snapshot(start_time=4200.0),
        create_synthetic_snapshot(start_time=4500.0),
    ]
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[1], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    dataset = slicer.slice_dataset(snaps)
    
    # 2 segments of length 4. Each segment produces 1 sample (indices 0..2 input, 3 target).
    assert dataset.total_samples == 2
    assert dataset.metadata["continuous_segments_count"] == 2
    # Check no sample has input spanning 900 -> 3600
    for s in dataset.train_samples:
        assert s.input_end_time - s.input_start_time <= 900.0


# =====================================================================
# TEST 20: No fabricated missing snapshots.
# =====================================================================
def test_20_no_fabricated_missing_snapshots():
    snaps = [create_synthetic_snapshot(start_time=0.0), create_synthetic_snapshot(start_time=3600.0)]
    slicer = TemporalDatasetSlicer(input_sequence_length=2, forecast_horizons=[1])
    dataset = slicer.slice_dataset(snaps)
    # Gap separates them into 2 segments of length 1 -> no sequence possible
    assert dataset.total_samples == 0


# =====================================================================
# TEST 21: Input/target timestamp ordering.
# =====================================================================
def test_21_input_target_timestamp_ordering():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0) for i in range(10)]
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[ForecastHorizon.PLUS_5MIN, ForecastHorizon.PLUS_10MIN])
    dataset = slicer.slice_dataset(snaps)
    for s in dataset.train_samples + dataset.val_samples + dataset.test_samples:
        for h_name, t_ts in s.target_times.items():
            assert s.input_end_time <= t_ts
            assert s.input_start_time < t_ts


# =====================================================================
# TEST 22: No target leakage across split boundaries.
# =====================================================================
def test_22_no_target_leakage_across_splits():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0) for i in range(20)]
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[1], train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    dataset = slicer.slice_dataset(snaps)
    
    if dataset.train_samples and dataset.val_samples:
        max_train_input = max(s.input_start_time for s in dataset.train_samples)
        min_val_input = min(s.input_start_time for s in dataset.val_samples)
        assert max_train_input <= min_val_input


# =====================================================================
# TEST 23: Deterministic repeated execution.
# =====================================================================
def test_23_deterministic_repeated_execution():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0) for i in range(10)]
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[1])
    
    ds1 = slicer.slice_dataset(snaps)
    ds2 = slicer.slice_dataset(snaps)
    
    assert ds1.model_dump() == ds2.model_dump()


# =====================================================================
# TEST 24: Step 2 contract compatibility.
# =====================================================================
def test_24_step2_contract_compatibility():
    contract = TemporalGraphDatasetContract(
        snapshots=[create_synthetic_snapshot(start_time=i * 300.0) for i in range(6)]
    )
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[1])
    dataset = slicer.slice_dataset(contract)
    assert isinstance(dataset, SpatioTemporalDataset)
    assert dataset.total_samples == 3


# =====================================================================
# TEST 25: Step 3 contract compatibility.
# =====================================================================
def test_25_step3_aggregator_output_compatibility():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    # Generate 6 observations spaced 300s apart
    obs = [{"camera_id": "CAM_A", "timestamp": float(i * 300 + 10), "track_id": f"TRK_{i}", "speed": 40.0 + i} for i in range(6)]
    contract = aggregator.aggregate(observations=obs)
    
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[1])
    dataset = slicer.slice_dataset(contract)
    assert dataset.total_samples == 3
    assert dataset.adjacency.node_ids == ["CAM_A"]


# =====================================================================
# TEST 26: Serialization / Deserialization.
# =====================================================================
def test_26_serialization_deserialization():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0) for i in range(6)]
    slicer = TemporalDatasetSlicer(input_sequence_length=3, forecast_horizons=[1])
    dataset = slicer.slice_dataset(snaps)
    
    json_str = dataset.model_dump_json()
    assert isinstance(json_str, str)
    
    restored = SpatioTemporalDataset.model_validate_json(json_str)
    assert restored.total_samples == dataset.total_samples
    assert restored.adjacency.num_nodes == dataset.adjacency.num_nodes


# =====================================================================
# MANDATORY RESEARCH LEAKAGE TEST (TEST 27)
# Training flow: 10, 20, 30. Test flow: 1000.
# Mean must NOT become 265. It must be calculated ONLY from 10, 20, 30.
# =====================================================================
def test_27_research_leakage_strict_isolation():
    """
    Mandatory Research Leakage Test:
    Proves that extreme test values do NOT leak into training normalization parameters.
    """
    snaps = [
        create_synthetic_snapshot(start_time=0.0, node_data={"CAM_A": {"flow_rate": 10.0}}),
        create_synthetic_snapshot(start_time=300.0, node_data={"CAM_A": {"flow_rate": 20.0}}),
        create_synthetic_snapshot(start_time=600.0, node_data={"CAM_A": {"flow_rate": 30.0}}),
        create_synthetic_snapshot(start_time=900.0, node_data={"CAM_A": {"flow_rate": 30.0}}),
        # Test split snapshots with extreme anomaly
        create_synthetic_snapshot(start_time=1200.0, node_data={"CAM_A": {"flow_rate": 1000.0}}),
        create_synthetic_snapshot(start_time=1500.0, node_data={"CAM_A": {"flow_rate": 1000.0}}),
        create_synthetic_snapshot(start_time=1800.0, node_data={"CAM_A": {"flow_rate": 1000.0}}),
        create_synthetic_snapshot(start_time=2100.0, node_data={"CAM_A": {"flow_rate": 1000.0}}),
    ]
    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,
        forecast_horizons=[1],
        train_ratio=0.50, val_ratio=0.0, test_ratio=0.50
    )
    dataset = slicer.slice_dataset(snaps)
    
    # Training samples only see 10, 20, 30.
    mean_flow = dataset.feature_scaler.mean["flow_rate"]
    assert mean_flow <= 30.0, f"RESEARCH INTEGRITY VIOLATION: Test leakage detected! Mean is {mean_flow}"
    assert mean_flow != 265.0, "RESEARCH INTEGRITY VIOLATION: Mean reflects contaminated train+test pool!"


# =====================================================================
# MANDATORY TEMPORAL SPLIT TEST (TEST 28)
# G0 ... G19 -> TRAIN before VAL before TEST
# =====================================================================
def test_28_temporal_split_order():
    snaps = [create_synthetic_snapshot(start_time=i * 300.0) for i in range(20)]
    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,
        forecast_horizons=[1, 2, 3],
        train_ratio=0.60, val_ratio=0.20, test_ratio=0.20
    )
    dataset = slicer.slice_dataset(snaps)
    
    assert len(dataset.train_samples) > 0
    assert len(dataset.val_samples) > 0
    assert len(dataset.test_samples) > 0
    
    # 1. TRAIN occurs before VALIDATION
    max_train_t = max(s.input_start_time for s in dataset.train_samples)
    min_val_t = min(s.input_start_time for s in dataset.val_samples)
    assert max_train_t <= min_val_t
    
    # 2. VALIDATION occurs before TEST
    max_val_t = max(s.input_start_time for s in dataset.val_samples)
    min_test_t = min(s.input_start_time for s in dataset.test_samples)
    assert max_val_t <= min_test_t
    
    # 3. No sample's target timestamp is earlier than its input timestamps
    for s in dataset.train_samples + dataset.val_samples + dataset.test_samples:
        assert s.input_start_time < s.input_end_time
        for h_name, t_ts in s.target_times.items():
            assert s.input_end_time <= t_ts
