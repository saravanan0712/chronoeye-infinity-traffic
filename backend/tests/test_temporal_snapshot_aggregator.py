"""
Tests for Module 3 Step 3: Temporal Snapshot Window Aggregator.
Verifies sliding window semantics [start, end), feature aggregation, evidence preservation,
missing-data integrity (None != 0), unobserved gap preservation, and no intermediate camera fabrication.
"""

import pytest
import copy
from app.graph.temporal_snapshot_aggregator import TemporalSnapshotAggregator
from app.graph.temporal_snapshot_schema import (
    TemporalNodeFeatures,
    TemporalEdgeFeatures,
    TemporalGraphSnapshot,
    TemporalGraphDatasetContract,
)
from app.schemas.reid import (
    VehicleJourney,
    JourneySegment,
    ReIDScoreBreakdown,
    MatchDecision,
)
from app.schemas.plate import PlateValidationStatus


# =====================================================================
# TEST 01: Single observation produces one snapshot.
# =====================================================================
def test_01_single_observation_produces_one_snapshot():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    obs = [{"camera_id": "CAM_A", "timestamp": 100.0, "track_id": "TRK_1", "speed": 45.0}]
    
    contract = aggregator.aggregate(observations=obs)
    assert len(contract.snapshots) == 1
    snap = contract.snapshots[0]
    assert snap.start_time == 0.0
    assert snap.end_time == 300.0
    assert len(snap.nodes) == 1
    assert snap.nodes[0].node_id == "CAM_A"
    assert snap.nodes[0].vehicle_count == 1
    assert snap.nodes[0].average_speed == 45.0


# =====================================================================
# TEST 02: Multiple observations in the same window aggregate correctly.
# =====================================================================
def test_02_multiple_observations_in_same_window():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    obs = [
        {"camera_id": "CAM_A", "timestamp": 50.0, "global_vehicle_id": "VEH_1", "speed": 40.0},
        {"camera_id": "CAM_A", "timestamp": 150.0, "global_vehicle_id": "VEH_2", "speed": 60.0},
        {"camera_id": "CAM_B", "timestamp": 200.0, "global_vehicle_id": "VEH_3", "speed": 50.0},
    ]
    
    contract = aggregator.aggregate(observations=obs)
    assert len(contract.snapshots) == 1
    snap = contract.snapshots[0]
    assert len(snap.nodes) == 2
    
    node_a = next(n for n in snap.nodes if n.node_id == "CAM_A")
    assert node_a.vehicle_count == 2
    assert node_a.average_speed == 50.0  # (40 + 60) / 2
    
    node_b = next(n for n in snap.nodes if n.node_id == "CAM_B")
    assert node_b.vehicle_count == 1
    assert node_b.average_speed == 50.0


# =====================================================================
# TEST 03: Observation at exact window start belongs to that window.
# =====================================================================
def test_03_observation_at_exact_window_start():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    obs = [{"camera_id": "CAM_A", "timestamp": 300.0, "track_id": "TRK_1"}]
    
    contract = aggregator.aggregate(observations=obs)
    assert len(contract.snapshots) == 1
    snap = contract.snapshots[0]
    assert snap.start_time == 300.0
    assert snap.end_time == 600.0
    assert snap.nodes[0].node_id == "CAM_A"


# =====================================================================
# TEST 04: Observation at exact window end belongs to the next window.
# =====================================================================
def test_04_observation_at_exact_window_end_belongs_to_next():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    # 0.0 -> Window [0, 300)
    # 300.0 -> Window [300, 600) (not [0, 300))
    obs = [
        {"camera_id": "CAM_A", "timestamp": 0.0, "track_id": "TRK_1"},
        {"camera_id": "CAM_A", "timestamp": 300.0, "track_id": "TRK_2"},
    ]
    
    contract = aggregator.aggregate(observations=obs)
    assert len(contract.snapshots) == 2
    
    snap0 = contract.snapshots[0]
    assert snap0.start_time == 0.0
    assert snap0.end_time == 300.0
    assert snap0.nodes[0].vehicle_count == 1
    assert "TRK_1" in snap0.nodes[0].metadata["observed_vehicle_ids"]
    
    snap1 = contract.snapshots[1]
    assert snap1.start_time == 300.0
    assert snap1.end_time == 600.0
    assert snap1.nodes[0].vehicle_count == 1
    assert "TRK_2" in snap1.nodes[0].metadata["observed_vehicle_ids"]


# =====================================================================
# TEST 05: Observations across two windows create two snapshots.
# =====================================================================
def test_05_observations_across_two_windows_create_two_snapshots():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    obs = [
        {"camera_id": "CAM_A", "timestamp": 120.0, "track_id": "TRK_1"},
        {"camera_id": "CAM_B", "timestamp": 450.0, "track_id": "TRK_2"},
    ]
    
    contract = aggregator.aggregate(observations=obs)
    assert len(contract.snapshots) == 2
    assert contract.snapshots[0].start_time == 0.0
    assert contract.snapshots[0].nodes[0].node_id == "CAM_A"
    assert contract.snapshots[1].start_time == 300.0
    assert contract.snapshots[1].nodes[0].node_id == "CAM_B"


# =====================================================================
# TEST 06: Three or more consecutive windows preserve chronological order.
# =====================================================================
def test_06_three_consecutive_windows_preserve_chronological_order():
    aggregator = TemporalSnapshotAggregator(window_seconds=100, stride_seconds=100)
    obs = [
        {"camera_id": "CAM_C", "timestamp": 250.0, "track_id": "TRK_3"},
        {"camera_id": "CAM_A", "timestamp": 50.0, "track_id": "TRK_1"},
        {"camera_id": "CAM_B", "timestamp": 150.0, "track_id": "TRK_2"},
    ]
    
    contract = aggregator.aggregate(observations=obs)
    assert len(contract.snapshots) == 3
    assert [s.start_time for s in contract.snapshots] == [0.0, 100.0, 200.0]
    assert contract.snapshots[0].nodes[0].node_id == "CAM_A"
    assert contract.snapshots[1].nodes[0].node_id == "CAM_B"
    assert contract.snapshots[2].nodes[0].node_id == "CAM_C"


# =====================================================================
# TEST 07: Window size is configurable.
# =====================================================================
def test_07_window_size_is_configurable():
    aggregator = TemporalSnapshotAggregator(window_seconds=60, stride_seconds=60)
    obs = [
        {"camera_id": "CAM_A", "timestamp": 10.0, "track_id": "TRK_1"},
        {"camera_id": "CAM_A", "timestamp": 70.0, "track_id": "TRK_2"},
    ]
    contract = aggregator.aggregate(observations=obs)
    assert contract.window_seconds == 60.0
    assert len(contract.snapshots) == 2
    assert contract.snapshots[0].end_time == 60.0
    assert contract.snapshots[1].end_time == 120.0


# =====================================================================
# TEST 08: Stride is configurable.
# =====================================================================
def test_08_stride_is_configurable_sliding_windows():
    # 50% overlap: window=100, stride=50
    aggregator = TemporalSnapshotAggregator(window_seconds=100, stride_seconds=50)
    obs = [
        {"camera_id": "CAM_A", "timestamp": 60.0, "track_id": "TRK_1"},
    ]
    # Timestamp 60 falls in [0, 100) and [50, 150)
    contract = aggregator.aggregate(observations=obs)
    assert contract.stride_seconds == 50.0
    assert len(contract.snapshots) == 2
    assert contract.snapshots[0].start_time == 0.0
    assert contract.snapshots[0].end_time == 100.0
    assert contract.snapshots[1].start_time == 50.0
    assert contract.snapshots[1].end_time == 150.0


# =====================================================================
# TEST 09: Repeated observation of same vehicle does not inflate unique count.
# =====================================================================
def test_09_repeated_observation_preserves_unique_vehicle_count():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    obs = [
        {"camera_id": "CAM_A", "timestamp": 10.0, "global_vehicle_id": "VEH_GLOBAL_01", "track_id": "TRK_1"},
        {"camera_id": "CAM_A", "timestamp": 20.0, "global_vehicle_id": "VEH_GLOBAL_01", "track_id": "TRK_1"},
        {"camera_id": "CAM_A", "timestamp": 30.0, "global_vehicle_id": "VEH_GLOBAL_01", "track_id": "TRK_1"},
    ]
    contract = aggregator.aggregate(observations=obs)
    assert len(contract.snapshots) == 1
    node = contract.snapshots[0].nodes[0]
    assert node.vehicle_count == 1
    assert node.metadata["observation_count"] == 3


# =====================================================================
# TEST 10: Valid speed values aggregate correctly.
# =====================================================================
def test_10_valid_speed_values_aggregate_correctly():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    obs = [
        {"camera_id": "CAM_A", "timestamp": 10.0, "track_id": "TRK_1", "speed": 30.0},
        {"camera_id": "CAM_A", "timestamp": 20.0, "track_id": "TRK_2", "speed": 50.0},
        {"camera_id": "CAM_A", "timestamp": 30.0, "track_id": "TRK_3", "speed": 70.0},
    ]
    contract = aggregator.aggregate(observations=obs)
    assert contract.snapshots[0].nodes[0].average_speed == 50.0


# =====================================================================
# TEST 11: Missing speed remains None (Missing != 0).
# =====================================================================
def test_11_missing_speed_remains_none():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    obs = [{"camera_id": "CAM_A", "timestamp": 10.0, "track_id": "TRK_1", "speed": None}]
    contract = aggregator.aggregate(observations=obs)
    assert contract.snapshots[0].nodes[0].average_speed is None


# =====================================================================
# TEST 12: Missing density remains None.
# =====================================================================
def test_12_missing_density_remains_none():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    obs = [{"camera_id": "CAM_A", "timestamp": 10.0, "track_id": "TRK_1"}]
    contract = aggregator.aggregate(observations=obs)
    assert contract.snapshots[0].nodes[0].density is None


# =====================================================================
# TEST 13: Missing queue length remains None.
# =====================================================================
def test_13_missing_queue_length_remains_none():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    obs = [{"camera_id": "CAM_A", "timestamp": 10.0, "track_id": "TRK_1"}]
    contract = aggregator.aggregate(observations=obs)
    assert contract.snapshots[0].nodes[0].queue_length is None


# =====================================================================
# TEST 14: Missing congestion remains None (no demo threshold fabrication).
# =====================================================================
def test_14_missing_congestion_remains_none():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    obs = [{"camera_id": "CAM_A", "timestamp": 10.0, "track_id": "TRK_1"}]
    contract = aggregator.aggregate(observations=obs)
    assert contract.snapshots[0].nodes[0].congestion is None


# =====================================================================
# TEST 15: Missing travel time remains None.
# =====================================================================
def test_15_missing_travel_time_remains_none():
    # If a transition has no valid travel time
    journey = VehicleJourney(
        journey_id="JRN_01",
        global_vehicle_id="VEH_01",
        segments=[
            JourneySegment(
                segment_id="SEG_1",
                camera_id="CAM_A",
                timestamp=100.0,
                track_id="TRK_1",
            ),
            JourneySegment(
                segment_id="SEG_2",
                camera_id="CAM_B",
                timestamp=100.0,  # delta_t = 0.0 -> missing/instantaneous
                track_id="TRK_2",
            ),
        ]
    )
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    contract = aggregator.aggregate(journeys=[journey])
    assert len(contract.snapshots) == 1
    edge = contract.snapshots[0].edges[0]
    assert edge.travel_time is None


# =====================================================================
# TEST 16: Valid transition produces an edge.
# =====================================================================
def test_16_valid_transition_produces_edge():
    breakdown = ReIDScoreBreakdown(
        plate_similarity=1.0,
        appearance_similarity=0.92,
        visual_features_similarity=0.88,
        vehicle_type_similarity=1.0,
        temporal_compatibility=0.95,
        spatial_compatibility=1.0,
        direction_similarity=0.90,
    )
    journey = VehicleJourney(
        journey_id="JRN_01",
        global_vehicle_id="VEH_01",
        segments=[
            JourneySegment(
                segment_id="SEG_1",
                camera_id="CAM_A",
                timestamp=50.0,
                track_id="TRK_1",
                direction="EAST",
            ),
            JourneySegment(
                segment_id="SEG_2",
                camera_id="CAM_B",
                timestamp=150.0,
                track_id="TRK_2",
                direction="EAST",
                transition_score=0.94,
                transition_decision=MatchDecision.MATCH_CONFIRMED,
                transition_breakdown=breakdown,
                has_unobserved_gap=False,
            ),
        ]
    )
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    contract = aggregator.aggregate(journeys=[journey])
    assert len(contract.snapshots) == 1
    snap = contract.snapshots[0]
    assert len(snap.edges) == 1
    edge = snap.edges[0]
    assert edge.source == "CAM_A"
    assert edge.target == "CAM_B"
    assert edge.travel_time == 100.0
    assert edge.confidence == 0.94
    assert edge.has_unobserved_gap is False


# =====================================================================
# TEST 17: Multiple transitions aggregate correctly.
# =====================================================================
def test_17_multiple_transitions_aggregate_correctly():
    j1 = VehicleJourney(
        journey_id="JRN_01",
        global_vehicle_id="VEH_01",
        segments=[
            JourneySegment(segment_id="S1", camera_id="CAM_A", timestamp=50.0, track_id="T1"),
            JourneySegment(segment_id="S2", camera_id="CAM_B", timestamp=150.0, track_id="T2", transition_score=0.90),
        ]
    )
    j2 = VehicleJourney(
        journey_id="JRN_02",
        global_vehicle_id="VEH_02",
        segments=[
            JourneySegment(segment_id="S3", camera_id="CAM_A", timestamp=60.0, track_id="T3"),
            JourneySegment(segment_id="S4", camera_id="CAM_B", timestamp=200.0, track_id="T4", transition_score=0.80),
        ]
    )
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    contract = aggregator.aggregate(journeys=[j1, j2])
    edge = contract.snapshots[0].edges[0]
    assert edge.source == "CAM_A"
    assert edge.target == "CAM_B"
    assert edge.transition_count == 2
    assert edge.vehicle_count == 2
    # TT1 = 100s, TT2 = 140s -> Mean = 120s
    assert edge.mean_travel_time == 120.0
    # Mean confidence = (0.90 + 0.80) / 2 = 0.85
    assert edge.confidence == 0.85


# =====================================================================
# TEST 18: Unobserved gap is preserved.
# =====================================================================
def test_18_unobserved_gap_is_preserved():
    journey = VehicleJourney(
        journey_id="JRN_01",
        global_vehicle_id="VEH_01",
        segments=[
            JourneySegment(segment_id="S1", camera_id="CAM_A", timestamp=50.0, track_id="T1"),
            JourneySegment(
                segment_id="S2",
                camera_id="CAM_C",
                timestamp=200.0,
                track_id="T2",
                has_unobserved_gap=True,
            ),
        ]
    )
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    contract = aggregator.aggregate(journeys=[journey])
    edge = contract.snapshots[0].edges[0]
    assert edge.source == "CAM_A"
    assert edge.target == "CAM_C"
    assert edge.has_unobserved_gap is True
    assert edge.unobserved_gap_count == 1
    assert edge.observed_transition_count == 0


# =====================================================================
# TEST 19: No intermediate camera is fabricated.
# =====================================================================
def test_19_no_intermediate_camera_is_fabricated():
    journey = VehicleJourney(
        journey_id="JRN_01",
        global_vehicle_id="VEH_01",
        segments=[
            JourneySegment(segment_id="S1", camera_id="CAM_A", timestamp=50.0, track_id="T1"),
            JourneySegment(
                segment_id="S2",
                camera_id="CAM_Z",
                timestamp=250.0,
                track_id="T2",
                has_unobserved_gap=True,
            ),
        ]
    )
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    contract = aggregator.aggregate(journeys=[journey])
    
    # Check that nodes only contain CAM_A and CAM_Z, no intermediate CAM_B, CAM_C, etc.
    node_ids = {n.node_id for n in contract.snapshots[0].nodes}
    assert node_ids == {"CAM_A", "CAM_Z"}
    
    # Check edges only contain CAM_A -> CAM_Z
    edge_pairs = {(e.source, e.target) for e in contract.snapshots[0].edges}
    assert edge_pairs == {("CAM_A", "CAM_Z")}


# =====================================================================
# TEST 20: Module 2 plate evidence is preserved.
# =====================================================================
def test_20_module2_plate_evidence_is_preserved():
    journey = VehicleJourney(
        journey_id="JRN_01",
        global_vehicle_id="VEH_01",
        plate_number="KA01AB1234",
        segments=[
            JourneySegment(
                segment_id="S1",
                camera_id="CAM_A",
                timestamp=50.0,
                track_id="T1",
                plate_number="KA01AB1234",
                plate_confidence=0.98,
                plate_status="CONFIRMED",
            )
        ]
    )
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    contract = aggregator.aggregate(journeys=[journey])
    node = contract.snapshots[0].nodes[0]
    assert "KA01AB1234" in node.metadata["plates"]


# =====================================================================
# TEST 21: Module 2 transition score/decision/breakdown are preserved.
# =====================================================================
def test_21_module2_transition_breakdown_and_decision_preserved():
    breakdown = ReIDScoreBreakdown(
        plate_similarity=0.95,
        appearance_similarity=0.91,
        visual_features_similarity=0.89,
        vehicle_type_similarity=1.0,
        temporal_compatibility=0.93,
        spatial_compatibility=1.0,
        direction_similarity=0.88,
    )
    journey = VehicleJourney(
        journey_id="JRN_01",
        global_vehicle_id="VEH_01",
        segments=[
            JourneySegment(segment_id="S1", camera_id="CAM_A", timestamp=50.0, track_id="T1"),
            JourneySegment(
                segment_id="S2",
                camera_id="CAM_B",
                timestamp=150.0,
                track_id="T2",
                transition_score=0.925,
                transition_decision=MatchDecision.MATCH_CONFIRMED,
                transition_breakdown=breakdown,
            ),
        ]
    )
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    contract = aggregator.aggregate(journeys=[journey])
    edge = contract.snapshots[0].edges[0]
    assert edge.confidence == 0.925
    assert "MATCH_CONFIRMED" in edge.metadata["transition_decisions"]
    assert len(edge.metadata["transition_breakdowns"]) == 1
    bd_dict = edge.metadata["transition_breakdowns"][0]
    assert bd_dict["plate_similarity"] == 0.95
    assert bd_dict["appearance_similarity"] == 0.91


# =====================================================================
# TEST 22: Timestamp uncertainty is preserved.
# =====================================================================
def test_22_timestamp_uncertainty_is_preserved():
    journey = VehicleJourney(
        journey_id="JRN_01",
        global_vehicle_id="VEH_01",
        segments=[
            JourneySegment(
                segment_id="S1",
                camera_id="CAM_A",
                timestamp=50.0,
                track_id="T1",
                timestamp_uncertainty_seconds=0.0333,
            ),
            JourneySegment(
                segment_id="S2",
                camera_id="CAM_B",
                timestamp=150.0,
                track_id="T2",
                timestamp_uncertainty_seconds=0.0500,
            ),
        ]
    )
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    contract = aggregator.aggregate(journeys=[journey])
    snap = contract.snapshots[0]
    assert snap.nodes[0].metadata["mean_timestamp_uncertainty_seconds"] == 0.0333
    assert snap.edges[0].uncertainty == 0.0500


# =====================================================================
# TEST 23: Empty input behaves deterministically.
# =====================================================================
def test_23_empty_input_behaves_deterministically():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    contract = aggregator.aggregate(observations=[], journeys=[])
    assert isinstance(contract, TemporalGraphDatasetContract)
    assert len(contract.snapshots) == 0
    assert contract.window_seconds == 300.0
    assert contract.stride_seconds == 300.0


# =====================================================================
# TEST 24: Input ordering does not affect output (Input order independence).
# =====================================================================
def test_24_input_order_independence():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    obs1 = [
        {"camera_id": "CAM_A", "timestamp": 50.0, "global_vehicle_id": "VEH_1", "speed": 40.0},
        {"camera_id": "CAM_B", "timestamp": 120.0, "global_vehicle_id": "VEH_2", "speed": 55.0},
        {"camera_id": "CAM_A", "timestamp": 200.0, "global_vehicle_id": "VEH_3", "speed": 45.0},
    ]
    obs2 = [
        {"camera_id": "CAM_A", "timestamp": 200.0, "global_vehicle_id": "VEH_3", "speed": 45.0},
        {"camera_id": "CAM_A", "timestamp": 50.0, "global_vehicle_id": "VEH_1", "speed": 40.0},
        {"camera_id": "CAM_B", "timestamp": 120.0, "global_vehicle_id": "VEH_2", "speed": 55.0},
    ]
    
    contract1 = aggregator.aggregate(observations=obs1)
    contract2 = aggregator.aggregate(observations=obs2)
    
    assert contract1.model_dump() == contract2.model_dump()


# =====================================================================
# TEST 25: Snapshot serialization succeeds.
# =====================================================================
def test_25_snapshot_serialization_succeeds():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    obs = [{"camera_id": "CAM_A", "timestamp": 100.0, "track_id": "TRK_1", "speed": 50.0}]
    contract = aggregator.aggregate(observations=obs)
    
    json_data = contract.model_dump_json()
    assert isinstance(json_data, str)
    assert "CAM_A" in json_data
    
    # Round-trip deserialization
    restored = TemporalGraphDatasetContract.model_validate_json(json_data)
    assert restored.snapshots[0].nodes[0].node_id == "CAM_A"


# =====================================================================
# TEST 26: Step 2 schema compatibility remains intact.
# =====================================================================
def test_26_step2_schema_compatibility():
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    obs = [{"camera_id": "CAM_A", "timestamp": 100.0, "track_id": "TRK_1"}]
    contract = aggregator.aggregate(observations=obs)
    
    snap = contract.snapshots[0]
    assert isinstance(snap, TemporalGraphSnapshot)
    assert isinstance(snap.nodes[0], TemporalNodeFeatures)
    assert snap.nodes[0].density is None
    assert snap.nodes[0].queue_length is None
    assert snap.nodes[0].congestion is None


# =====================================================================
# TEST 27: Existing Module 2 regression remains intact.
# =====================================================================
def test_27_module2_objects_directly_consumed():
    journey = VehicleJourney(
        journey_id="JRN_99",
        global_vehicle_id="VEH_99",
        vehicle_type="truck",
        segments=[
            JourneySegment(
                segment_id="SEG_1",
                camera_id="CAM_X",
                timestamp=500.0,
                track_id="TRK_X",
                speed_estimate=42.0,
                direction="NORTH",
            )
        ]
    )
    aggregator = TemporalSnapshotAggregator(window_seconds=300, stride_seconds=300)
    contract = aggregator.aggregate(journeys=[journey])
    assert len(contract.snapshots) == 1
    snap = contract.snapshots[0]
    assert snap.start_time == 300.0
    assert snap.end_time == 600.0
    assert snap.nodes[0].node_id == "CAM_X"
    assert snap.nodes[0].average_speed == 42.0


# =====================================================================
# MANDATORY RESEARCH INTEGRITY TEST (TEST 28)
# Given: CAM_A -> CAM_C with has_unobserved_gap = true,
# verify the aggregator does NOT create CAM_B.
# =====================================================================
def test_28_research_integrity_no_intermediate_camera_gap_fabrication():
    """
    Mandatory Research Integrity Test:
    CAM_A -> CAM_C transition with unobserved gap MUST NOT fabricate CAM_B
    under any circumstances.
    """
    journey = VehicleJourney(
        journey_id="JRN_GAP_TEST",
        global_vehicle_id="VEH_TEST_GAP",
        segments=[
            JourneySegment(
                segment_id="SEG_A",
                camera_id="CAM_A",
                timestamp=100.0,
                track_id="TRK_A",
            ),
            JourneySegment(
                segment_id="SEG_C",
                camera_id="CAM_C",
                timestamp=400.0,
                track_id="TRK_C",
                has_unobserved_gap=True,
                transition_score=0.85,
            ),
        ]
    )
    aggregator = TemporalSnapshotAggregator(window_seconds=600, stride_seconds=600)
    contract = aggregator.aggregate(journeys=[journey])
    
    assert len(contract.snapshots) == 1
    snap = contract.snapshots[0]
    
    # 1. Verify exact node set
    observed_nodes = [node.node_id for node in snap.nodes]
    assert "CAM_A" in observed_nodes
    assert "CAM_C" in observed_nodes
    assert "CAM_B" not in observed_nodes, "RESEARCH INTEGRITY VIOLATION: Intermediate node CAM_B was fabricated!"
    assert len(observed_nodes) == 2
    
    # 2. Verify exact edge set
    edge_pairs = [(edge.source, edge.target) for edge in snap.edges]
    assert ("CAM_A", "CAM_C") in edge_pairs
    assert ("CAM_A", "CAM_B") not in edge_pairs, "RESEARCH INTEGRITY VIOLATION: Edge CAM_A -> CAM_B fabricated!"
    assert ("CAM_B", "CAM_C") not in edge_pairs, "RESEARCH INTEGRITY VIOLATION: Edge CAM_B -> CAM_C fabricated!"
    
    # 3. Verify unobserved gap flag is strictly preserved
    edge = snap.edges[0]
    assert edge.has_unobserved_gap is True
    assert edge.unobserved_gap_count == 1
    assert edge.observed_transition_count == 0
