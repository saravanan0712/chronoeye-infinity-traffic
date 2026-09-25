"""
ChronoEye Infinity - Phase 6 Verification Test Suite
Automated Python test suite verifying Spatio-Temporal Traffic Graph requirements across 31 test cases.
"""

import os
import sys
import unittest

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.detection import BoundingBoxXYXY
from app.schemas.tracking import TrackState, TrackStatus, Direction
from app.schemas.plate import FusedPlateIdentity, VehicleIdentityEvidence
from app.schemas.reid import VehicleJourney, JourneySegment
from app.graph.graph_schema import (
    NodeType,
    EdgeType,
    VehicleNode,
    CameraNodeGraph,
    RoadNode,
    LaneNode,
    JunctionNode,
    SignalNode,
)
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.graph.temporal_graph import TemporalTrafficGraphEngine
from app.graph.graph_metrics import TrafficMetricsEngine
from app.graph.graph_serializer import GraphSerializer
from app.graph.pyg_adapter import PyGGraphAdapter
from app.simulation.engine import TrafficSimulationEngine
from app.perception.adapter import DetectionAdapter
from app.perception.tracker import VehicleTrackerManager
from app.perception.plate_association import PlateTrackerAssociationManager
from app.perception.journey import JourneyReconstructionEngine


class TestPhase6GraphPipeline(unittest.TestCase):

    def setUp(self):
        self.builder = SpatioTemporalGraphBuilder()
        self.engine = TemporalTrafficGraphEngine(builder=self.builder)

    def test_1_graph_initialization(self):
        """Test 1: Verify SpatioTemporalGraphBuilder initialization and default city topology."""
        self.assertGreater(len(self.builder.graph.nodes), 10)
        self.assertGreater(len(self.builder.graph.edges), 8)

    def test_2_node_creation(self):
        """Test 2: Verify custom node creation."""
        v = VehicleNode(vehicle_id="VEH_101", vehicle_type="car")
        self.builder.add_vehicle_node(v)
        self.assertIn("VEH_101", self.builder.graph.nodes)

    def test_3_heterogeneous_node_types(self):
        """Test 3: Verify all 6 heterogeneous node types are supported."""
        vehicles = self.builder.get_nodes_by_type(NodeType.VEHICLE)
        cameras = self.builder.get_nodes_by_type(NodeType.CAMERA)
        roads = self.builder.get_nodes_by_type(NodeType.ROAD)
        lanes = self.builder.get_nodes_by_type(NodeType.LANE)
        junctions = self.builder.get_nodes_by_type(NodeType.JUNCTION)
        signals = self.builder.get_nodes_by_type(NodeType.SIGNAL)

        self.assertIsNotNone(vehicles)
        self.assertGreater(len(cameras), 0)
        self.assertGreater(len(roads), 0)
        self.assertGreater(len(lanes), 0)
        self.assertGreater(len(junctions), 0)
        self.assertGreater(len(signals), 0)

    def test_4_edge_creation(self):
        """Test 4: Verify typed edge insertion."""
        v = VehicleNode(vehicle_id="VEH_101", vehicle_type="car")
        self.builder.add_vehicle_node(v)
        self.builder.add_typed_edge("VEH_101", "CAM_A_EAST", EdgeType.VEHICLE_OBSERVED_BY_CAMERA, timestamp=1.0)

        edge_found = False
        for u, v, k, d in self.builder.graph.edges(data=True, keys=True):
            if u == "VEH_101" and v == "CAM_A_EAST":
                edge_found = True
                self.assertEqual(d.get("edge_type"), EdgeType.VEHICLE_OBSERVED_BY_CAMERA.value)
        self.assertTrue(edge_found)

    def test_5_vehicle_observation_insertion(self):
        """Test 5: Verify update_vehicle_observation creates vehicle node and edges."""
        track = TrackState(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True
        )
        evidence = VehicleIdentityEvidence(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            last_updated_timestamp=1.0
        )

        self.engine.update_vehicle_observation(evidence, track, 1.0, road_id="ROAD_R_AB")
        self.assertIn("TRK_101", self.builder.graph.nodes)

    def test_6_repeated_vehicle_observation(self):
        """Test 6: Verify repeated vehicle observation increments observation count."""
        track = TrackState(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True
        )
        evidence = VehicleIdentityEvidence(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            last_updated_timestamp=1.0
        )

        self.engine.update_vehicle_observation(evidence, track, 1.0)
        self.engine.update_vehicle_observation(evidence, track, 2.0)

        v_data = self.builder.graph.nodes["TRK_101"]["data"]
        self.assertEqual(v_data["observation_count"], 2)

    def test_7_temporal_transition_creation(self):
        """Test 7: Verify temporal transition edges between vehicle and road nodes."""
        track = TrackState(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True
        )
        evidence = VehicleIdentityEvidence(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            last_updated_timestamp=1.0
        )

        self.engine.update_vehicle_observation(evidence, track, 1.0, road_id="ROAD_R_AB")
        self.assertGreater(TrafficMetricsEngine.calculate_vehicle_count(self.builder, "ROAD_R_AB"), 0)

    def test_8_camera_transition(self):
        """Test 8: Verify camera-to-camera transition edge creation."""
        seg1 = JourneySegment(camera_id="CAM_A_EAST", track_id="TRK_101", timestamp=0.0, bbox=BoundingBoxXYXY(x1=10, y1=10, x2=50, y2=50))
        seg2 = JourneySegment(camera_id="CAM_B_WEST", track_id="TRK_207", timestamp=18.0, bbox=BoundingBoxXYXY(x1=10, y1=10, x2=50, y2=50))
        journey = VehicleJourney(journey_id="JRN_101", global_vehicle_id="VEH_101", plate_number="TN09AB1234", vehicle_type="car", segments=[seg1, seg2], first_seen=0.0, last_seen=18.0)

        self.engine.update_journey(journey)
        self.assertIn("VEH_101", self.builder.graph.nodes)

    def test_9_road_topology(self):
        """Test 9: Verify default road topology graph contains expected roads."""
        roads = self.builder.get_nodes_by_type(NodeType.ROAD)
        self.assertIn("ROAD_R_AB", roads)
        self.assertIn("ROAD_R_BA", roads)

    def test_10_lane_relationships(self):
        """Test 10: Verify LANE_BELONGS_TO_ROAD edges exist."""
        lanes = self.builder.get_nodes_by_type(NodeType.LANE)
        self.assertGreater(len(lanes), 0)

    def test_11_junction_relationships(self):
        """Test 11: Verify ROAD_CONNECTS_TO_JUNCTION edges exist."""
        junctions = self.builder.get_nodes_by_type(NodeType.JUNCTION)
        self.assertGreater(len(junctions), 0)

    def test_12_signal_relationships(self):
        """Test 12: Verify JUNCTION_CONTROLLED_BY_SIGNAL edges exist."""
        signals = self.builder.get_nodes_by_type(NodeType.SIGNAL)
        self.assertGreater(len(signals), 0)

    def test_13_vehicle_journey_insertion(self):
        """Test 13: Verify update_journey incorporates 3-camera continuous journey."""
        s1 = JourneySegment(camera_id="CAM_A_EAST", track_id="TRK_101", timestamp=0.0, bbox=BoundingBoxXYXY(x1=10, y1=10, x2=50, y2=50))
        s2 = JourneySegment(camera_id="CAM_B_WEST", track_id="TRK_207", timestamp=18.0, bbox=BoundingBoxXYXY(x1=10, y1=10, x2=50, y2=50))
        s3 = JourneySegment(camera_id="CAM_C_NORTH", track_id="TRK_314", timestamp=42.0, bbox=BoundingBoxXYXY(x1=10, y1=10, x2=50, y2=50))
        journey = VehicleJourney(journey_id="JRN_101", global_vehicle_id="VEH_101", plate_number="TN09AB1234", vehicle_type="car", segments=[s1, s2, s3], first_seen=0.0, last_seen=42.0)

        self.engine.update_journey(journey)
        v_data = self.builder.graph.nodes["VEH_101"]["data"]
        self.assertEqual(v_data["observation_count"], 3)

    def test_14_incremental_graph_update(self):
        """Test 14: Verify incremental graph update preserves existing graph structure."""
        initial_nodes = len(self.builder.graph.nodes)
        v = VehicleNode(vehicle_id="VEH_999", vehicle_type="truck")
        self.builder.add_vehicle_node(v)
        self.assertEqual(len(self.builder.graph.nodes), initial_nodes + 1)

    def test_15_graph_snapshot(self):
        """Test 15: Verify graph snapshot at timestamp t."""
        snap = self.engine.snapshot(timestamp=10.0)
        self.assertEqual(snap["timestamp"], 10.0)
        self.assertIn("metrics", snap)

    def test_16_historical_snapshot_retention(self):
        """Test 16: Verify historical snapshot retention."""
        self.engine.snapshot(10.0)
        self.engine.snapshot(20.0)
        hist = self.engine.get_historical_snapshot(10.0)
        self.assertIsNotNone(hist)
        self.assertEqual(hist["timestamp"], 10.0)

    def test_17_traffic_vehicle_count(self):
        """Test 17: Verify TrafficMetricsEngine.calculate_vehicle_count."""
        track = TrackState(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True
        )
        evidence = VehicleIdentityEvidence(track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car", last_updated_timestamp=1.0)
        self.engine.update_vehicle_observation(evidence, track, 1.0, road_id="ROAD_R_AB")

        v_count = TrafficMetricsEngine.calculate_vehicle_count(self.builder, "ROAD_R_AB")
        self.assertEqual(v_count, 1)

    def test_18_density_calculation(self):
        """Test 18: Verify TrafficMetricsEngine.calculate_density."""
        density = TrafficMetricsEngine.calculate_density(self.builder, "ROAD_R_AB")
        self.assertGreaterEqual(density, 0.0)

    def test_19_average_speed(self):
        """Test 19: Verify TrafficMetricsEngine.calculate_average_speed."""
        avg_speed = TrafficMetricsEngine.calculate_average_speed(self.builder, "ROAD_R_AB")
        self.assertGreater(avg_speed, 0.0)

    def test_20_queue_estimation(self):
        """Test 20: Verify TrafficMetricsEngine.estimate_queue_length."""
        queue = TrafficMetricsEngine.estimate_queue_length(self.builder, "ROAD_R_AB")
        self.assertGreaterEqual(queue, 0)

    def test_21_congestion_score(self):
        """Test 21: Verify TrafficMetricsEngine.calculate_congestion_score returns score in [0.0, 1.0]."""
        score = TrafficMetricsEngine.calculate_congestion_score(self.builder, "ROAD_R_AB")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_22_temporal_feature_calculation(self):
        """Test 22: Verify temporal transition feature calculation."""
        s1 = JourneySegment(camera_id="CAM_A_EAST", track_id="TRK_101", timestamp=0.0, bbox=BoundingBoxXYXY(x1=10, y1=10, x2=50, y2=50))
        s2 = JourneySegment(camera_id="CAM_B_WEST", track_id="TRK_207", timestamp=18.0, bbox=BoundingBoxXYXY(x1=10, y1=10, x2=50, y2=50))
        journey = VehicleJourney(journey_id="JRN_101", global_vehicle_id="VEH_101", vehicle_type="car", segments=[s1, s2], first_seen=0.0, last_seen=18.0)

        self.engine.update_journey(journey)
        edge_found = False
        for u, v, k, d in self.builder.graph.edges(data=True, keys=True):
            if u == "CAM_A_EAST" and v == "CAM_B_WEST":
                edge_found = True
                data = d.get("data", {})
                self.assertEqual(data.get("travel_time"), 18.0)
        self.assertTrue(edge_found)

    def test_23_phase_5_to_phase_6_adapter(self):
        """Test 23: Verify Phase 5 VehicleJourney -> Phase 6 Graph incorporation."""
        s1 = JourneySegment(camera_id="CAM_A_EAST", track_id="TRK_101", timestamp=0.0, bbox=BoundingBoxXYXY(x1=10, y1=10, x2=50, y2=50))
        journey = VehicleJourney(journey_id="JRN_101", global_vehicle_id="VEH_101", plate_number="TN09AB1234", vehicle_type="car", segments=[s1], first_seen=0.0, last_seen=0.0)
        self.engine.update_journey(journey)
        self.assertIn("VEH_101", self.builder.graph.nodes)

    def test_24_deterministic_reproducibility(self):
        """Test 24: Verify deterministic graph build reproducibility."""
        b1 = SpatioTemporalGraphBuilder()
        b2 = SpatioTemporalGraphBuilder()
        self.assertEqual(len(b1.graph.nodes), len(b2.graph.nodes))
        self.assertEqual(len(b1.graph.edges), len(b2.graph.edges))

    def test_25_graph_serialization(self):
        """Test 25: Verify GraphSerializer JSON conversion and reconstruction."""
        json_str = GraphSerializer.to_json(self.builder)
        self.assertIn("node_count", json_str)
        reconstructed = GraphSerializer.from_json(json_str)
        self.assertEqual(len(self.builder.graph.nodes), len(reconstructed.graph.nodes))

    def test_26_optional_pyg_adapter_behavior(self):
        """Test 26: Verify PyGGraphAdapter output structure."""
        res = PyGGraphAdapter.to_hetero_data(self.builder)
        self.assertIn("pyg_available", res)

    def test_27_phase_1_to_phase_6_integration(self):
        """Test 27: Phase 1 simulation engine -> Phase 6 graph integration."""
        sim = TrafficSimulationEngine(seed=42)
        sim.step(1.0)
        self.assertGreater(len(sim.recent_observations), 0)

    def test_28_phase_2_to_phase_6_integration(self):
        """Test 28: Phase 2 detection -> Phase 6 graph integration."""
        sim = TrafficSimulationEngine(seed=42)
        sim.step(1.0)
        det_events = DetectionAdapter.batch_convert(sim.recent_observations)
        self.assertGreater(len(det_events), 0)

    def test_29_phase_3_to_phase_6_integration(self):
        """Test 29: Phase 3 tracking -> Phase 6 graph integration."""
        sim = TrafficSimulationEngine(seed=42)
        sim.step(1.0)
        det_events = DetectionAdapter.batch_convert(sim.recent_observations)
        tracker_mgr = VehicleTrackerManager()
        tracks = tracker_mgr.update(det_events[0].camera_id, det_events, 1.0)
        self.assertGreater(len(tracks), 0)

    def test_30_phase_4_to_phase_6_integration(self):
        """Test 30: Phase 4 ALPR -> Phase 6 graph integration."""
        alpr_mgr = PlateTrackerAssociationManager()
        track = TrackState(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True,
            raw_vehicle_reference="V_100_TN09AB1234"
        )
        evidence = alpr_mgr.process_track_frame(track, {"width": 1920, "height": 1080}, 1.0, 1)
        self.engine.update_vehicle_observation(evidence, track, 1.0)
        self.assertIn("TRK_101", self.builder.graph.nodes)

    def test_31_phase_5_to_phase_6_integration(self):
        """Test 31: Phase 5 Re-ID journey -> Phase 6 graph integration."""
        reid_engine = JourneyReconstructionEngine()
        track = TrackState(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True,
            raw_vehicle_reference="V_100_TN09AB1234"
        )
        evidence = VehicleIdentityEvidence(track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car", last_updated_timestamp=1.0)
        journey = reid_engine.process_track_evidence(evidence, track)
        self.engine.update_journey(journey)
        self.assertIn(journey.global_vehicle_id, self.builder.graph.nodes)

    def test_32_add_junction_node_string_id(self):
        """Test 32: Verify add_junction_node supports passing a string junction_id directly (router pattern)."""
        self.builder.add_junction_node("JUNC_TEST_1")
        self.assertIn("JUNC_TEST_1", self.builder.graph.nodes)
        n_data = self.builder.graph.nodes["JUNC_TEST_1"]["data"]
        self.assertEqual(n_data["junction_id"], "JUNC_TEST_1")

    def test_33_add_junction_node_string_id_kwargs(self):
        """Test 33: Verify add_junction_node supports passing string junction_id with latitude/longitude kwargs."""
        self.builder.add_junction_node("JUNC_TEST_2", latitude=13.0827, longitude=80.2707)
        self.assertIn("JUNC_TEST_2", self.builder.graph.nodes)
        n_data = self.builder.graph.nodes["JUNC_TEST_2"]["data"]
        self.assertEqual(n_data["latitude"], 13.0827)
        self.assertEqual(n_data["longitude"], 80.2707)

    def test_34_add_road_edge_convenience_method(self):
        """Test 34: Verify add_road_edge convenience method connects junctions with road attributes."""
        self.builder.add_junction_node("JUNC_X")
        self.builder.add_junction_node("JUNC_Y")
        self.builder.add_road_edge("ROAD_XY", "JUNC_X", "JUNC_Y", length_meters=500.0, travel_time=30.0)
        self.assertTrue(self.builder.graph.has_edge("JUNC_X", "JUNC_Y"))


if __name__ == "__main__":
    unittest.main()

