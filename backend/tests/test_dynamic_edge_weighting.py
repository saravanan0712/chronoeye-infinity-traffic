"""
Tests for Module 3 Task 2: Dynamic Graph Edge Weighting.
Verifies static-mode identity, dynamic flow/speed/confidence modulation,
boundedness, non-negativity, phantom edge prevention, NaN/Inf protection,
graceful missing-data fallback, degree normalization, and checkpoint compatibility.
"""

import os
import math
import pytest
import torch

from app.models.stgnn.config import STGNNConfig
from app.models.stgnn.layers import GraphSpatialConv, STGNNBlock
from app.models.stgnn.model import SpatioTemporalGNN
from app.forecasting.st_gnn import SpatioTemporalGNNForecaster
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.state.traffic_state_schema import NetworkTrafficSnapshot, RoadSegmentState
from app.forecasting.forecasting_schema import ForecastHorizon


# =====================================================================
# TEST 01: Static Adjacency Identity (No Dynamic Evidence)
# =====================================================================
def test_01_static_adjacency_identity():
    adj_static = torch.tensor([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    # Calling compute_dynamic_adjacency with no dynamic evidence should return exact static matrix
    adj_dyn = GraphSpatialConv.compute_dynamic_adjacency(adj_static)
    assert torch.allclose(adj_dyn, adj_static)

    # Calling normalize_adjacency with static matrix should match standard degree normalization
    norm_direct = GraphSpatialConv.normalize_adjacency(adj_static)
    norm_via_dyn = GraphSpatialConv.normalize_adjacency(adj_static, dynamic_weights=adj_dyn)
    assert torch.allclose(norm_direct, norm_via_dyn)


# =====================================================================
# TEST 02: Zero Transition Count Yields Exactly Static Adjacency
# =====================================================================
def test_02_zero_transition_count_yields_static():
    adj_static = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0],
    ])
    trans_counts = torch.zeros((2, 2))
    adj_dyn = GraphSpatialConv.compute_dynamic_adjacency(
        adj_static, transition_count=trans_counts
    )
    # log1p(0) = 0 -> tanh(0) = 0 -> dynamic_mult = 1.0
    assert torch.allclose(adj_dyn, adj_static)


# =====================================================================
# TEST 03: Increasing Transition Counts Monotonically Modulate Edge Weight
# =====================================================================
def test_03_increasing_transition_counts_monotonic():
    adj_static = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0],
    ])
    counts_low = torch.tensor([[0.0, 5.0], [5.0, 0.0]])
    counts_med = torch.tensor([[0.0, 25.0], [25.0, 0.0]])
    counts_high = torch.tensor([[0.0, 100.0], [100.0, 0.0]])

    adj_low = GraphSpatialConv.compute_dynamic_adjacency(adj_static, transition_count=counts_low)
    adj_med = GraphSpatialConv.compute_dynamic_adjacency(adj_static, transition_count=counts_med)
    adj_high = GraphSpatialConv.compute_dynamic_adjacency(adj_static, transition_count=counts_high)

    w_static = adj_static[0, 1].item()
    w_low = adj_low[0, 1].item()
    w_med = adj_med[0, 1].item()
    w_high = adj_high[0, 1].item()

    assert w_low > w_static
    assert w_med > w_low
    assert w_high > w_med


# =====================================================================
# TEST 04: High Transition Count is Sublinear and Strictly Bounded
# =====================================================================
def test_04_high_transition_count_bounded():
    adj_static = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    counts_extreme = torch.tensor([[0.0, 1e7], [1e7, 0.0]])

    adj_dyn = GraphSpatialConv.compute_dynamic_adjacency(
        adj_static, transition_count=counts_extreme, alpha_scale=1.0
    )
    w_extreme = adj_dyn[0, 1].item()

    # With alpha_scale=1.0, tanh <= 1.0, so dynamic_mult <= 2.0
    assert w_extreme <= 2.0 * adj_static[0, 1].item()
    assert not math.isinf(w_extreme)
    assert not math.isnan(w_extreme)


# =====================================================================
# TEST 05: Speed Variations Appropriately Scale Edge Weight
# =====================================================================
def test_05_speed_variation_scaling():
    adj_static = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    counts = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
    speed_slow = torch.tensor([[0.0, 15.0], [15.0, 0.0]])  # Heavy traffic / congested speed
    speed_fast = torch.tensor([[0.0, 80.0], [80.0, 0.0]])  # Free flow speed

    adj_slow = GraphSpatialConv.compute_dynamic_adjacency(
        adj_static, transition_count=counts, speed=speed_slow, ref_speed=60.0
    )
    adj_fast = GraphSpatialConv.compute_dynamic_adjacency(
        adj_static, transition_count=counts, speed=speed_fast, ref_speed=60.0
    )

    assert adj_fast[0, 1].item() > adj_slow[0, 1].item()


# =====================================================================
# TEST 06: Travel Time Variations (Longer Travel Time -> Reduced Multiplier)
# =====================================================================
def test_06_travel_time_variation():
    adj_static = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    counts = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
    tt_fast = torch.tensor([[0.0, 20.0], [20.0, 0.0]])   # Fast travel time
    tt_delayed = torch.tensor([[0.0, 120.0], [120.0, 0.0]]) # Delayed / congested travel time

    adj_fast = GraphSpatialConv.compute_dynamic_adjacency(
        adj_static, transition_count=counts, travel_time=tt_fast, ref_travel_time=30.0
    )
    adj_delayed = GraphSpatialConv.compute_dynamic_adjacency(
        adj_static, transition_count=counts, travel_time=tt_delayed, ref_travel_time=30.0
    )

    assert adj_fast[0, 1].item() > adj_delayed[0, 1].item()


# =====================================================================
# TEST 07: Transition Confidence Modulation
# =====================================================================
def test_07_transition_confidence_modulation():
    adj_static = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    counts = torch.tensor([[0.0, 20.0], [20.0, 0.0]])
    conf_low = torch.tensor([[0.0, 0.2], [0.2, 0.0]])
    conf_high = torch.tensor([[0.0, 0.95], [0.95, 0.0]])

    adj_low = GraphSpatialConv.compute_dynamic_adjacency(
        adj_static, transition_count=counts, confidence=conf_low
    )
    adj_high = GraphSpatialConv.compute_dynamic_adjacency(
        adj_static, transition_count=counts, confidence=conf_high
    )

    assert adj_high[0, 1].item() > adj_low[0, 1].item()


# =====================================================================
# TEST 08: Unobserved Gap Flag Applies Conservative Discount
# =====================================================================
def test_08_unobserved_gap_discount():
    adj_static = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    counts = torch.tensor([[0.0, 20.0], [20.0, 0.0]])
    gap_present = torch.tensor([[False, True], [True, False]])
    gap_absent = torch.tensor([[False, False], [False, False]])

    adj_with_gap = GraphSpatialConv.compute_dynamic_adjacency(
        adj_static, transition_count=counts, has_unobserved_gap=gap_present
    )
    adj_no_gap = GraphSpatialConv.compute_dynamic_adjacency(
        adj_static, transition_count=counts, has_unobserved_gap=gap_absent
    )

    assert adj_no_gap[0, 1].item() > adj_with_gap[0, 1].item()


# =====================================================================
# TEST 09: No Phantom Edges Created where Static Adjacency is Zero
# =====================================================================
def test_09_no_phantom_edges():
    adj_static = torch.tensor([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    # Transition count present even on disconnected pair (0, 2)
    counts = torch.tensor([
        [0.0, 10.0, 50.0],
        [10.0, 0.0, 10.0],
        [50.0, 10.0, 0.0],
    ])

    adj_dyn = GraphSpatialConv.compute_dynamic_adjacency(adj_static, transition_count=counts)

    # Disconnected pair (0, 2) MUST remain 0.0 (no phantom edge)
    assert adj_dyn[0, 2].item() == 0.0
    assert adj_dyn[2, 0].item() == 0.0
    # Connected edges should be augmented
    assert adj_dyn[0, 1].item() > 1.0
    assert adj_dyn[1, 2].item() > 1.0


# =====================================================================
# TEST 10: Robustness Against NaN, Inf, and Negative Inputs
# =====================================================================
def test_10_nan_inf_negative_protection():
    adj_static = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0],
    ])
    counts_corrupted = torch.tensor([[float("nan"), -10.0], [float("inf"), 5.0]])
    speeds_corrupted = torch.tensor([[float("-inf"), float("nan")], [0.0, 50.0]])

    adj_dyn = GraphSpatialConv.compute_dynamic_adjacency(
        adj_static, transition_count=counts_corrupted, speed=speeds_corrupted
    )

    assert not torch.isnan(adj_dyn).any()
    assert not torch.isinf(adj_dyn).any()
    assert (adj_dyn >= 0.0).all()


# =====================================================================
# TEST 11: Normalized Adjacency Properties (Degree Normalization & Symmetry)
# =====================================================================
def test_11_normalized_adjacency_properties():
    adj_static = torch.tensor([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    counts = torch.tensor([
        [0.0, 15.0, 30.0],
        [15.0, 0.0, 0.0],
        [30.0, 0.0, 0.0],
    ])
    adj_dyn = GraphSpatialConv.compute_dynamic_adjacency(adj_static, transition_count=counts)
    adj_norm = GraphSpatialConv.normalize_adjacency(adj_static, dynamic_weights=adj_dyn)

    # Check shape
    assert adj_norm.shape == (3, 3)
    # Check symmetry
    assert torch.allclose(adj_norm, adj_norm.T, atol=1e-5)
    # Check non-negativity
    assert (adj_norm >= 0.0).all()
    # Check no NaN/Inf
    assert not torch.isnan(adj_norm).any()
    assert not torch.isinf(adj_norm).any()


# =====================================================================
# TEST 12: GraphSpatialConv Forward Pass with Dynamic Adjacency
# =====================================================================
def test_12_graph_spatial_conv_forward_dynamic():
    conv = GraphSpatialConv(in_features=8, out_features=16)
    x = torch.randn(2, 3, 4, 8)  # [B, T, N, F]
    adj_static = torch.eye(4) + torch.ones(4, 4) * 0.2
    
    # Static mode
    norm_static = GraphSpatialConv.normalize_adjacency(adj_static)
    out_static = conv(x, norm_static)
    assert out_static.shape == (2, 3, 4, 16)

    # Dynamic mode
    counts = torch.ones((4, 4)) * 10.0
    adj_dyn = GraphSpatialConv.compute_dynamic_adjacency(adj_static, transition_count=counts)
    norm_dyn = GraphSpatialConv.normalize_adjacency(adj_static, dynamic_weights=adj_dyn)
    out_dyn = conv(x, norm_dyn)
    assert out_dyn.shape == (2, 3, 4, 16)

    # Both modes execute with valid outputs
    assert not torch.isnan(out_dyn).any()


# =====================================================================
# TEST 13: SpatioTemporalGNN Forward Pass Compatibility (Both Static & Dynamic)
# =====================================================================
def test_13_spatio_temporal_gnn_forward_dynamic():
    config = STGNNConfig(
        input_dim=8,
        output_dim=4,
        hidden_dim=32,
        num_spatial_layers=2,
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],
    )
    model = SpatioTemporalGNN(config=config)
    x = torch.randn(2, 3, 4, 8)
    adj_static = torch.eye(4) + torch.ones(4, 4) * 0.1

    # 1. Forward without dynamic_edge_weights (static mode)
    out_static = model(x, adj_static)
    assert out_static.shape == (2, 3, 4, 4)

    # 2. Forward with dynamic_edge_weights
    counts = torch.tensor([
        [0.0, 5.0, 10.0, 0.0],
        [5.0, 0.0, 0.0, 12.0],
        [10.0, 0.0, 0.0, 8.0],
        [0.0, 12.0, 8.0, 0.0],
    ])
    adj_dyn = GraphSpatialConv.compute_dynamic_adjacency(adj_static, transition_count=counts)
    out_dyn = model(x, adj_static, dynamic_edge_weights=adj_dyn)
    assert out_dyn.shape == (2, 3, 4, 4)
    assert not torch.isnan(out_dyn).any()


# =====================================================================
# TEST 14: Research Checkpoint Loading Compatibility
# =====================================================================
def test_14_checkpoint_loading_compatibility():
    ckpt_path_1 = "experiments/module3/research_exp_01/checkpoints/best_stgnn_model.pt"
    ckpt_path_2 = "experiments/module3/research_exp_02/checkpoints/best_stgnn_model.pt"

    for ckpt_path in [ckpt_path_1, ckpt_path_2]:
        if os.path.exists(ckpt_path):
            forecaster = SpatioTemporalGNNForecaster(
                checkpoint_path=ckpt_path,
                use_dynamic_adjacency=True,
            )
            assert forecaster.pytorch_model is not None
            
            # Forward inference pass with loaded checkpoint
            x = torch.randn(1, 3, 4, 8)
            adj = torch.eye(4)
            with torch.no_grad():
                out = forecaster.pytorch_model(x, adj)
            assert out.shape == (1, 3, 4, 4)
            assert not torch.isnan(out).any()


# =====================================================================
# TEST 15: SpatioTemporalGNNForecaster End-to-End Prediction with Dynamic Adjacency
# =====================================================================
def test_15_forecaster_predict_dynamic_adjacency():
    builder = SpatioTemporalGraphBuilder()
    forecaster = SpatioTemporalGNNForecaster(
        checkpoint_path="experiments/module3/research_exp_01/checkpoints/best_stgnn_model.pt"
        if os.path.exists("experiments/module3/research_exp_01/checkpoints/best_stgnn_model.pt")
        else None,
        use_dynamic_adjacency=True,
    )

    history = [
        NetworkTrafficSnapshot(
            timestamp=float(t * 300),
            segment_states={
                "ROAD_R_AB": RoadSegmentState(
                    segment_id="ROAD_R_AB",
                    road_name="R_AB",
                    timestamp=float(t * 300),
                    vehicle_count=10 + t,
                    flow_rate=120.0 + t * 5,
                    density=15.0 + t,
                    average_speed=48.0,
                    queue_length=1,
                    congestion_score=0.20,
                    occupancy=0.15,
                    travel_time=25.0,
                ),
                "ROAD_R_BA": RoadSegmentState(
                    segment_id="ROAD_R_BA",
                    road_name="R_BA",
                    timestamp=float(t * 300),
                    vehicle_count=8 + t,
                    flow_rate=100.0 + t * 5,
                    density=12.0 + t,
                    average_speed=52.0,
                    queue_length=1,
                    congestion_score=0.15,
                    occupancy=0.12,
                    travel_time=22.0,
                ),
            }
        )
        for t in range(5)
    ]

    forecast = forecaster.predict(
        builder=builder,
        history=history,
        horizon=ForecastHorizon.PLUS_5MIN,
        segment_id="ROAD_R_AB",
    )

    assert forecast.segment_id == "ROAD_R_AB"
    assert forecast.predicted_flow >= 0.0
    assert forecast.predicted_density >= 0.0
    assert forecast.predicted_travel_time >= 1.0
