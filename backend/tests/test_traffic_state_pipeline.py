"""
ChronoEye Infinity - Phase 7 Verification Test Suite
Automated Python test suite verifying Dynamic Traffic State Engine requirements across 18 test cases.
"""

import os
import sys
import unittest

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.detection import BoundingBoxXYXY
from app.schemas.tracking import TrackState, TrackStatus, Direction
from app.schemas.plate import VehicleIdentityEvidence
from app.graph.graph_schema import NodeType, EdgeType, VehicleNode
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.graph.temporal_graph import TemporalTrafficGraphEngine
from app.state.traffic_state_schema import (
    CongestionLevel,
    RoadSegmentState,
    NetworkTrafficSnapshot,
)
from app.state.flow_calculator import FlowCalculator
from app.state.density_calculator import DensityCalculator
from app.state.queue_estimator import QueueEstimator
from app.state.travel_time import TravelTimeEstimator
from app.state.congestion import CongestionAnalyzer
from app.state.state_history import StateHistoryManager
from app.state.traffic_state_engine import DynamicTrafficStateEngine


class TestPhase7TrafficStatePipeline(unittest.TestCase):

    def setUp(self):
        self.builder = SpatioTemporalGraphBuilder()
        self.temporal_graph = TemporalTrafficGraphEngine(builder=self.builder)
        self.state_engine = DynamicTrafficStateEngine(max_history=60)

    def test_1_initialization(self):
        """Test 1: Verify DynamicTrafficStateEngine initialization."""
        self.assertIsNotNone(self.state_engine.history_manager)
        self.assertEqual(self.state_engine.history_manager.max_history_length, 60)

    def test_2_empty_traffic_state(self):
        """Test 2: Verify empty road traffic state calculation."""
        snapshot = self.state_engine.compute_network_state(self.builder, timestamp=0.0)
        self.assertIn("ROAD_R_AB", snapshot.segment_states)
        seg = snapshot.segment_states["ROAD_R_AB"]
        self.assertEqual(seg.vehicle_count, 0)
        self.assertEqual(seg.flow_rate, 0.0)
        self.assertEqual(seg.density, 0.0)
        self.assertEqual(seg.congestion_level, CongestionLevel.FREE_FLOW)

    def test_3_vehicle_counting(self):
        """Test 3: Verify vehicle counting on road segment."""
        track = TrackState(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True
        )
        evidence = VehicleIdentityEvidence(track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car", last_updated_timestamp=1.0)
        self.temporal_graph.update_vehicle_observation(evidence, track, 1.0, road_id="ROAD_R_AB")

        snapshot = self.state_engine.compute_network_state(self.builder, timestamp=1.0)
        seg = snapshot.segment_states["ROAD_R_AB"]
        self.assertEqual(seg.vehicle_count, 1)

    def test_4_flow_calculation(self):
        """Test 4: Verify FlowCalculator hourly flow rate calculation."""
        flow = FlowCalculator.calculate_flow_rate(vehicle_count=10, window_seconds=60.0)
        self.assertEqual(flow, 600.0)

    def test_5_density_calculation(self):
        """Test 5: Verify DensityCalculator density (vehicles / km)."""
        density = DensityCalculator.calculate_density(vehicle_count=5, length_meters=500.0)
        self.assertEqual(density, 10.0)

    def test_6_average_speed(self):
        """Test 6: Verify average vehicle speed calculation."""
        track = TrackState(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True,
            speed_estimate=40.0
        )
        evidence = VehicleIdentityEvidence(track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car", last_updated_timestamp=1.0)
        self.temporal_graph.update_vehicle_observation(evidence, track, 1.0, road_id="ROAD_R_AB")

        snapshot = self.state_engine.compute_network_state(self.builder, timestamp=1.0)
        seg = snapshot.segment_states["ROAD_R_AB"]
        self.assertEqual(seg.average_speed, 40.0)

    def test_7_occupancy(self):
        """Test 7: Verify spatial occupancy calculation ratio."""
        occ = DensityCalculator.calculate_occupancy(vehicle_count=10, length_meters=500.0, avg_vehicle_length_m=5.0)
        self.assertEqual(occ, 0.10)

    def test_8_queue_estimation(self):
        """Test 8: Verify QueueEstimator queued vehicle count and spatial length."""
        speeds = [2.0, 3.0, 50.0, 60.0]  # 2 queued (< 5.0 km/h)
        q_cnt, q_len_m = QueueEstimator.estimate_queue(speeds)
        self.assertEqual(q_cnt, 2)
        self.assertEqual(q_len_m, 14.0)

    def test_9_travel_time_estimation(self):
        """Test 9: Verify TravelTimeEstimator traversal time estimation."""
        t_time = TravelTimeEstimator.estimate_travel_time(length_meters=500.0, avg_speed_kmh=60.0, speed_limit_kmh=60.0)
        self.assertEqual(t_time, 30.0)

    def test_10_congestion_scoring(self):
        """Test 10: Verify CongestionAnalyzer scoring and CongestionLevel enum classification."""
        score = CongestionAnalyzer.calculate_congestion_score(
            density=200.0, capacity_density=200.0, avg_speed=5.0, speed_limit=60.0, queue_count=90, capacity=100
        )
        level = CongestionAnalyzer.classify_congestion_level(score)
        self.assertGreaterEqual(score, 0.85)
        self.assertIn(level, [CongestionLevel.SEVERELY_CONGESTED, CongestionLevel.STATIONARY_GRIDLOCK])

    def test_11_temporal_state_update(self):
        """Test 11: Verify dynamic temporal state update across t0 -> t1 -> t2."""
        s1 = self.state_engine.compute_network_state(self.builder, timestamp=0.0)
        s2 = self.state_engine.compute_network_state(self.builder, timestamp=5.0)
        s3 = self.state_engine.compute_network_state(self.builder, timestamp=10.0)
        self.assertEqual(len(self.state_engine.history_manager.history), 3)

    def test_12_history_retention(self):
        """Test 12: Verify bounded sliding window history retention."""
        mgr = StateHistoryManager(max_history_length=3)
        for t in range(5):
            mgr.add_snapshot(NetworkTrafficSnapshot(timestamp=float(t)))
        self.assertEqual(len(mgr.history), 3)
        self.assertEqual(mgr.history[0].timestamp, 2.0)

    def test_13_missing_observation_handling(self):
        """Test 13: Verify handling missing observations gracefully."""
        snapshot = self.state_engine.compute_network_state(self.builder, timestamp=100.0)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.total_active_vehicles, 0)

    def test_14_zero_speed_handling(self):
        """Test 14: Verify zero-speed handling (stationary gridlock) without division by zero."""
        t_time = TravelTimeEstimator.estimate_travel_time(length_meters=500.0, avg_speed_kmh=0.0, speed_limit_kmh=60.0)
        self.assertGreater(t_time, 0.0)

    def test_15_deterministic_reproducibility(self):
        """Test 15: Verify deterministic state calculation reproducibility."""
        s1 = self.state_engine.compute_network_state(self.builder, timestamp=10.0)
        s2 = self.state_engine.compute_network_state(self.builder, timestamp=10.0)
        self.assertEqual(s1.network_congestion_score, s2.network_congestion_score)

    def test_16_phase_6_to_phase_7_integration(self):
        """Test 16: Verify Phase 6 Spatio-Temporal Graph -> Phase 7 Traffic State integration."""
        self.builder.add_typed_edge("VEH_101", "ROAD_R_AB", EdgeType.VEHICLE_TRANSITIONS_TO_ROAD, timestamp=1.0, speed=50.0)
        snapshot = self.state_engine.compute_network_state(self.builder, timestamp=1.0)
        self.assertGreater(snapshot.segment_states["ROAD_R_AB"].vehicle_count, 0)

    def test_17_historical_sequence_generation(self):
        """Test 17: Verify historical sequence generation for Phase 8 ST-GNN forecasting."""
        for t in range(5):
            self.state_engine.compute_network_state(self.builder, timestamp=float(t))
        seq = self.state_engine.history_manager.get_recent_sequence(window_size=3, segment_id="ROAD_R_AB")
        self.assertEqual(len(seq), 3)
        self.assertEqual(len(seq[0]), 6)  # ML feature vector dimension

    def test_18_no_future_leakage(self):
        """Test 18: Verify enforcement of temporal ordering preventing future data leakage."""
        self.state_engine.history_manager.add_snapshot(NetworkTrafficSnapshot(timestamp=10.0))
        with self.assertRaises(ValueError):
            self.state_engine.history_manager.add_snapshot(NetworkTrafficSnapshot(timestamp=5.0))


if __name__ == "__main__":
    unittest.main()
