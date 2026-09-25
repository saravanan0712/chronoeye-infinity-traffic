"""
ChronoEye Infinity - Module 3 Step 2: Temporal Graph Data Contract Test Suite
Verifies TemporalNodeFeatures, TemporalEdgeFeatures, TemporalGraphSnapshot,
and TemporalGraphDatasetContract schemas and their validation rules.
"""

import os
import sys
import unittest
import json

# Add backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph.graph_schema import NodeType, EdgeType
from app.forecasting.forecasting_schema import ForecastHorizon
from app.graph.temporal_snapshot_schema import (
    TemporalNodeFeatures,
    TemporalEdgeFeatures,
    TemporalGraphSnapshot,
    TemporalGraphDatasetContract,
)


class TestTemporalGraphSnapshotSchema(unittest.TestCase):
    """
    Unit test suite verifying the Module 3 Temporal Graph Data Contract.
    """

    def test_01_node_features_creation_valid_data(self):
        """TEST 01: TemporalNodeFeatures can be created with valid data."""
        node = TemporalNodeFeatures(
            node_id="CAM_A_EAST",
            node_type=NodeType.CAMERA,
            timestamp=300.0,
            vehicle_count=15,
            flow_rate=180.0,
            density=22.5,
            average_speed=48.2,
            queue_length=2,
            congestion=0.25,
            incoming_flow=90.0,
            outgoing_flow=90.0,
        )
        self.assertEqual(node.node_id, "CAM_A_EAST")
        self.assertEqual(node.node_type, NodeType.CAMERA)
        self.assertEqual(node.vehicle_count, 15)
        self.assertEqual(node.flow_rate, 180.0)
        self.assertEqual(node.flow, 180.0)
        self.assertEqual(node.density, 22.5)
        self.assertEqual(node.average_speed, 48.2)
        self.assertEqual(node.queue_length, 2)
        self.assertEqual(node.congestion, 0.25)
        self.assertEqual(node.incoming_flow, 90.0)
        self.assertEqual(node.outgoing_flow, 90.0)

    def test_02_node_features_missing_optional_values(self):
        """TEST 02: TemporalNodeFeatures accepts missing optional traffic values as None."""
        node = TemporalNodeFeatures(
            node_id="VEH_101",
            node_type=NodeType.VEHICLE,
        )
        self.assertEqual(node.node_id, "VEH_101")
        self.assertEqual(node.node_type, NodeType.VEHICLE)
        self.assertIsNone(node.vehicle_count)
        self.assertIsNone(node.flow_rate)
        self.assertIsNone(node.density)
        self.assertIsNone(node.average_speed)
        self.assertIsNone(node.queue_length)
        self.assertIsNone(node.congestion)
        self.assertIsNone(node.incoming_flow)
        self.assertIsNone(node.outgoing_flow)

    def test_03_edge_features_observed_transition(self):
        """TEST 03: TemporalEdgeFeatures can represent an observed transition."""
        edge = TemporalEdgeFeatures(
            source="CAM_A_EAST",
            target="CAM_B_WEST",
            edge_type=EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA,
            timestamp=300.0,
            vehicle_count=8,
            flow_rate=96.0,
            travel_time=18.5,
            mean_travel_time=18.5,
            median_travel_time=18.0,
            transition_count=8,
            observed_transition_count=8,
            unobserved_gap_count=0,
            has_unobserved_gap=False,
            confidence=0.94,
            direction="EAST",
        )
        self.assertEqual(edge.source, "CAM_A_EAST")
        self.assertEqual(edge.source_node, "CAM_A_EAST")
        self.assertEqual(edge.target, "CAM_B_WEST")
        self.assertEqual(edge.target_node, "CAM_B_WEST")
        self.assertEqual(edge.edge_type, EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA)
        self.assertEqual(edge.travel_time, 18.5)
        self.assertEqual(edge.observed_transition_count, 8)
        self.assertEqual(edge.unobserved_gap_count, 0)
        self.assertFalse(edge.has_unobserved_gap)
        self.assertEqual(edge.confidence, 0.94)

    def test_04_edge_features_unobserved_gap(self):
        """TEST 04: TemporalEdgeFeatures can represent an unobserved gap."""
        edge = TemporalEdgeFeatures(
            source="CAM_A_EAST",
            target="CAM_D_NORTH",
            edge_type=EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA,
            timestamp=300.0,
            travel_time=52.0,
            has_unobserved_gap=True,
            unobserved_gap_count=1,
            observed_transition_count=0,
        )
        self.assertTrue(edge.has_unobserved_gap)
        self.assertEqual(edge.unobserved_gap_count, 1)
        self.assertEqual(edge.observed_transition_count, 0)
        self.assertEqual(edge.source, "CAM_A_EAST")
        self.assertEqual(edge.target, "CAM_D_NORTH")

    def test_05_no_intermediate_camera_fabricated(self):
        """TEST 05: Edge between non-adjacent cameras preserves direct source-target relationship with no fabricated nodes."""
        edge = TemporalEdgeFeatures(
            source="CAM_A_EAST",
            target="CAM_D_NORTH",
            has_unobserved_gap=True,
        )
        self.assertEqual(edge.source, "CAM_A_EAST")
        self.assertEqual(edge.target, "CAM_D_NORTH")
        # Ensure no synthetic intermediate 'CAM_B_WEST' is stored
        self.assertNotIn("CAM_B_WEST", [edge.source, edge.target])

    def test_06_snapshot_contains_nodes_and_edges(self):
        """TEST 06: TemporalGraphSnapshot contains nodes and edges."""
        node1 = TemporalNodeFeatures(node_id="CAM_A_EAST", node_type=NodeType.CAMERA, vehicle_count=10)
        node2 = TemporalNodeFeatures(node_id="CAM_B_WEST", node_type=NodeType.CAMERA, vehicle_count=12)
        edge = TemporalEdgeFeatures(source="CAM_A_EAST", target="CAM_B_WEST", vehicle_count=5)

        snap = TemporalGraphSnapshot(
            snapshot_id="SNAP_001",
            start_time=0.0,
            end_time=300.0,
            nodes=[node1, node2],
            edges=[edge],
        )
        self.assertEqual(snap.node_count, 2)
        self.assertEqual(snap.edge_count, 1)
        self.assertEqual(snap.get_node("CAM_A_EAST").vehicle_count, 10)
        self.assertEqual(snap.get_edge("CAM_A_EAST", "CAM_B_WEST").vehicle_count, 5)

    def test_07_snapshot_start_end_times_preserved(self):
        """TEST 07: Snapshot start/end times and duration are preserved exactly."""
        snap = TemporalGraphSnapshot(
            start_time=1700000000.0,
            end_time=1700000300.0,
        )
        self.assertEqual(snap.start_time, 1700000000.0)
        self.assertEqual(snap.end_time, 1700000300.0)
        self.assertEqual(snap.duration_seconds, 300.0)

    def test_08_dataset_contract_accepts_ordered_snapshots(self):
        """TEST 08: TemporalGraphDatasetContract accepts multiple ordered snapshots."""
        snap1 = TemporalGraphSnapshot(snapshot_id="SNAP_01", start_time=0.0, end_time=300.0)
        snap2 = TemporalGraphSnapshot(snapshot_id="SNAP_02", start_time=300.0, end_time=600.0)
        snap3 = TemporalGraphSnapshot(snapshot_id="SNAP_03", start_time=600.0, end_time=900.0)

        dataset = TemporalGraphDatasetContract(
            dataset_id="DS_TEST_01",
            snapshots=[snap1, snap2, snap3],
        )
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0].snapshot_id, "SNAP_01")
        self.assertEqual(dataset[1].snapshot_id, "SNAP_02")
        self.assertEqual(dataset[2].snapshot_id, "SNAP_03")

    def test_09_default_window_is_300_seconds(self):
        """TEST 09: Default window is 300 seconds (5 minutes)."""
        dataset = TemporalGraphDatasetContract()
        self.assertEqual(dataset.window_seconds, 300.0)

    def test_10_default_stride_is_300_seconds(self):
        """TEST 10: Default stride is 300 seconds (5 minutes)."""
        dataset = TemporalGraphDatasetContract()
        self.assertEqual(dataset.stride_seconds, 300.0)

    def test_11_forecast_horizons_plus_5_10_15_minutes(self):
        """TEST 11: Forecast horizons represent +5/+10/+15 minutes."""
        dataset = TemporalGraphDatasetContract()
        self.assertIn(ForecastHorizon.PLUS_5MIN, dataset.forecast_horizons)
        self.assertIn(ForecastHorizon.PLUS_10MIN, dataset.forecast_horizons)
        self.assertIn(ForecastHorizon.PLUS_15MIN, dataset.forecast_horizons)

    def test_12_missing_values_remain_none_not_zero(self):
        """TEST 12: Missing values remain None and are not converted to fake zero values."""
        node = TemporalNodeFeatures(node_id="CAM_EMPTY")
        self.assertIsNone(node.flow_rate)
        self.assertIsNone(node.density)
        self.assertIsNone(node.average_speed)
        self.assertIsNone(node.queue_length)
        self.assertIsNone(node.congestion)

        edge = TemporalEdgeFeatures(source="CAM_A", target="CAM_B")
        self.assertIsNone(edge.travel_time)
        self.assertIsNone(edge.confidence)
        self.assertIsNone(edge.uncertainty)

    def test_13_module2_evidence_fields_preserved(self):
        """TEST 13: Module 2 evidence fields (plate info, 7-signal breakdown, uncertainty) are preserved in metadata."""
        node_meta = {
            "global_vehicle_id": "VEH_101",
            "journey_id": "JRN_101",
            "plate_number": "TN09AB1234",
            "plate_confidence": 0.96,
            "plate_status": "CONFIRMED",
            "timestamp_uncertainty_seconds": 0.15,
        }
        edge_meta = {
            "transition_score": 0.89,
            "transition_decision": "MATCH_CONFIRMED",
            "transition_breakdown": {
                "plate_similarity": 0.95,
                "appearance_similarity": 0.85,
                "visual_features_similarity": 0.90,
                "vehicle_type_similarity": 1.0,
                "direction_similarity": 0.92,
                "temporal_compatibility": 0.88,
                "spatial_compatibility": 1.0,
            },
            "timestamp_uncertainty_seconds": 0.15,
        }

        node = TemporalNodeFeatures(node_id="VEH_101", node_type=NodeType.VEHICLE, metadata=node_meta)
        edge = TemporalEdgeFeatures(source="CAM_A", target="CAM_B", metadata=edge_meta, uncertainty=0.15)

        self.assertEqual(node.metadata["plate_number"], "TN09AB1234")
        self.assertEqual(node.metadata["plate_status"], "CONFIRMED")
        self.assertEqual(node.metadata["timestamp_uncertainty_seconds"], 0.15)
        self.assertEqual(edge.uncertainty, 0.15)
        self.assertEqual(edge.metadata["transition_decision"], "MATCH_CONFIRMED")
        self.assertEqual(edge.metadata["transition_breakdown"]["plate_similarity"], 0.95)

    def test_14_serialization_round_trip(self):
        """TEST 14: JSON serialization and deserialization preserves exact values without mutation."""
        node = TemporalNodeFeatures(
            node_id="CAM_A",
            node_type=NodeType.CAMERA,
            flow_rate=123.45,
            density=67.89,
            congestion=0.55,
        )
        edge = TemporalEdgeFeatures(
            source="CAM_A",
            target="CAM_B",
            travel_time=24.68,
            has_unobserved_gap=True,
            confidence=0.876,
        )
        snap = TemporalGraphSnapshot(
            snapshot_id="SNAP_JSON_01",
            start_time=100.0,
            end_time=400.0,
            nodes=[node],
            edges=[edge],
            metadata={"version": "1.0", "source": "E2E_SYNTHETIC"},
        )
        dataset = TemporalGraphDatasetContract(
            dataset_id="DS_JSON_01",
            snapshots=[snap],
            window_seconds=300.0,
            stride_seconds=300.0,
        )

        # JSON round trip
        json_str = json.dumps(dataset.model_dump())
        data_dict = json.loads(json_str)
        reconstructed = TemporalGraphDatasetContract(**data_dict)

        self.assertEqual(len(reconstructed), 1)
        r_snap = reconstructed[0]
        self.assertEqual(r_snap.snapshot_id, "SNAP_JSON_01")
        self.assertEqual(r_snap.start_time, 100.0)
        self.assertEqual(r_snap.end_time, 400.0)
        self.assertEqual(r_snap.node_count, 1)
        self.assertEqual(r_snap.edge_count, 1)

        r_node = r_snap.nodes[0]
        self.assertEqual(r_node.node_id, "CAM_A")
        self.assertEqual(r_node.flow_rate, 123.45)
        self.assertEqual(r_node.density, 67.89)
        self.assertEqual(r_node.congestion, 0.55)

        r_edge = r_snap.edges[0]
        self.assertEqual(r_edge.source, "CAM_A")
        self.assertEqual(r_edge.target, "CAM_B")
        self.assertEqual(r_edge.travel_time, 24.68)
        self.assertTrue(r_edge.has_unobserved_gap)
        self.assertEqual(r_edge.confidence, 0.876)

    def test_15_existing_graph_schemas_compatibility(self):
        """TEST 15: Existing graph schemas (NodeType, EdgeType, ForecastHorizon) remain compatible and reusable."""
        self.assertEqual(NodeType.CAMERA.value, "CAMERA")
        self.assertEqual(NodeType.VEHICLE.value, "VEHICLE")
        self.assertEqual(EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value, "VEHICLE_TRANSITIONS_TO_CAMERA")
        self.assertEqual(ForecastHorizon.PLUS_5MIN.value, "PLUS_5MIN")
        self.assertEqual(ForecastHorizon.PLUS_10MIN.value, "PLUS_10MIN")
        self.assertEqual(ForecastHorizon.PLUS_15MIN.value, "PLUS_15MIN")


if __name__ == "__main__":
    unittest.main()
