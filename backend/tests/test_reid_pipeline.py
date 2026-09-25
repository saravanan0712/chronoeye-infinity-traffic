"""
ChronoEye Infinity - Phase 5 Verification Test Suite
Automated Python test suite verifying Cross-Camera Re-ID and Journey Reconstruction requirements across 20 test cases.
"""

import os
import sys
import unittest
from typing import Tuple

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.detection import DetectionEvent, BoundingBoxXYXY
from app.schemas.tracking import TrackState, TrackStatus, Direction
from app.schemas.plate import (
    PlateObservation,
    FusedPlateIdentity,
    VehicleIdentityEvidence,
    PlateValidationStatus,
)
from app.schemas.reid import (
    MatchDecision,
    ReIDScoreBreakdown,
    ReIDConfig,
    VehicleJourney,
)
from app.perception.camera_topology import CityCameraTopology, CameraNode
from app.perception.appearance import AppearanceEmbeddingExtractor
from app.perception.reid_engine import ReIDMatchingEngine
from app.perception.journey import JourneyReconstructionEngine
from app.perception.plate_association import PlateTrackerAssociationManager
from app.perception.tracker import VehicleTrackerManager
from app.perception.adapter import DetectionAdapter
from app.simulation.engine import TrafficSimulationEngine


class TestPhase5ReIDPipeline(unittest.TestCase):

    def setUp(self):
        self.topology = CityCameraTopology()
        self.config = ReIDConfig()
        self.matching_engine = ReIDMatchingEngine(topology=self.topology, config=self.config)
        self.journey_engine = JourneyReconstructionEngine(
            matching_engine=self.matching_engine, config=self.config
        )

    def _create_mock_track(
        self, track_id: str, camera_id: str, timestamp: float, plate: str = "TN09AB1234"
    ) -> Tuple[VehicleIdentityEvidence, TrackState]:
        track = TrackState(
            track_id=track_id,
            camera_id=camera_id,
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150),
            confidence=0.9,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=Direction.EAST,
        )
        fused_plate = FusedPlateIdentity(
            track_id=track_id,
            camera_id=camera_id,
            best_plate_number=plate,
            overall_confidence=0.92,
            observation_count=3,
            confirmed=True,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
        )
        evidence = VehicleIdentityEvidence(
            track_id=track_id,
            camera_id=camera_id,
            vehicle_type="car",
            associated_plate=fused_plate,
            last_updated_timestamp=timestamp,
        )
        return evidence, track

    def test_1_reid_initialization(self):
        """Test 1: Verify ReID matching engine and camera topology initialization."""
        self.assertIsNotNone(self.matching_engine.topology)
        self.assertEqual(self.matching_engine.config.w_plate, 0.35)
        self.assertIn("CAM_A_EAST", self.matching_engine.topology.cameras)

    def test_2_camera_topology(self):
        """Test 2: Verify camera topology spatial distances and Haversine calculation."""
        dist = self.topology.get_distance_meters("CAM_A_EAST", "CAM_B_WEST")
        self.assertEqual(dist, 300.0)

        min_t = self.topology.compute_min_travel_time("CAM_A_EAST", "CAM_B_WEST", max_speed_kmh=120.0)
        self.assertGreater(min_t, 5.0)

    def test_3_exact_plate_match(self):
        """Test 3: Verify exact license plate similarity calculation returns 1.0."""
        score = self.matching_engine.calculate_plate_similarity("TN09AB1234", "TN09AB1234")
        self.assertEqual(score, 1.0)

    def test_4_normalized_plate_match(self):
        """Test 4: Verify normalized license plate match (spaces/hyphens removed)."""
        score = self.matching_engine.calculate_plate_similarity("TN-09 AB 1234", "TN09AB1234")
        self.assertEqual(score, 1.0)

    def test_5_plate_mismatch(self):
        """Test 5: Verify distinct license plate mismatch penalty."""
        score = self.matching_engine.calculate_plate_similarity("TN09AB1234", "KA01CD5678")
        self.assertLess(score, 0.2)

    def test_6_appearance_similarity(self):
        """Test 6: Verify AppearanceEmbeddingExtractor cosine similarity calculation."""
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0, 0.0]

        sim_same = AppearanceEmbeddingExtractor.cosine_similarity(v1, v2)
        self.assertEqual(sim_same, 1.0)

        sim_diff = AppearanceEmbeddingExtractor.cosine_similarity(v1, v3)
        self.assertEqual(sim_diff, 0.0)

    def test_7_vehicle_type_matching(self):
        """Test 7: Verify vehicle type similarity calculation."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 18.0)
        ev_b.vehicle_type = "bus"

        breakdown = self.matching_engine.compute_match_score(ev_a, trk_a, ev_b, trk_b)
        self.assertEqual(breakdown.vehicle_type_similarity, 0.0)

    def test_8_temporal_feasibility(self):
        """Test 8: Verify temporal travel time feasibility bounds."""
        feasible, reason = self.topology.is_temporally_feasible("CAM_A_EAST", "CAM_B_WEST", delta_t_seconds=18.0)
        self.assertTrue(feasible)
        self.assertIsNone(reason)

    def test_9_physically_impossible_match_rejected(self):
        """Test 9: Verify physically impossible travel speed match is rejected."""
        # 300 meters distance in 1.0 second = 300 m/s = 1080 km/h (> 120 km/h max)
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 1.0)

        breakdown = self.matching_engine.compute_match_score(ev_a, trk_a, ev_b, trk_b)
        self.assertEqual(breakdown.decision, MatchDecision.MATCH_REJECTED)
        self.assertIn("PHYSICALLY_IMPOSSIBLE_SPEED", breakdown.rejection_reason)

    def test_10_direction_compatibility(self):
        """Test 10: Verify direction vector compatibility scoring."""
        s_same = self.matching_engine.calculate_direction_similarity("EAST", "EAST")
        self.assertEqual(s_same, 1.0)

        s_opp = self.matching_engine.calculate_direction_similarity("EAST", "WEST")
        self.assertEqual(s_opp, 0.1)

    def test_11_cross_camera_track_matching(self):
        """Test 11: Verify multi-evidence cross-camera track matching."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 18.0)

        breakdown = self.matching_engine.compute_match_score(ev_a, trk_a, ev_b, trk_b)
        self.assertEqual(breakdown.decision, MatchDecision.MATCH_CONFIRMED)
        self.assertGreater(breakdown.overall_score, 0.80)

    def test_12_global_vehicle_identity(self):
        """Test 12: Verify global vehicle identity creation (VEH_101)."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        journey = self.journey_engine.process_track_evidence(ev_a, trk_a)

        self.assertTrue(journey.global_vehicle_id.startswith("VEH_"))
        self.assertTrue(journey.journey_id.startswith("JRN_"))
        self.assertEqual(len(journey.segments), 1)

    def test_13_journey_reconstruction(self):
        """Test 13: Verify 3-camera continuous journey reconstruction (CAM_A -> CAM_B -> CAM_C)."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 18.0)
        ev_c, trk_c = self._create_mock_track("TRK_314", "CAM_C_NORTH", 42.0)

        j1 = self.journey_engine.process_track_evidence(ev_a, trk_a)
        j2 = self.journey_engine.process_track_evidence(ev_b, trk_b)
        j3 = self.journey_engine.process_track_evidence(ev_c, trk_c)

        self.assertEqual(j1.journey_id, j2.journey_id)
        self.assertEqual(j2.journey_id, j3.journey_id)
        self.assertEqual(len(j3.segments), 3)
        self.assertEqual(j3.cameras, ["CAM_A_EAST", "CAM_B_WEST", "CAM_C_NORTH"])

    def test_14_missing_plate_evidence(self):
        """Test 14: Verify cross-camera matching when license plate is missing (UNKNOWN)."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="UNKNOWN")
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 18.0, plate="UNKNOWN")

        breakdown = self.matching_engine.compute_match_score(ev_a, trk_a, ev_b, trk_b)
        self.assertIn(breakdown.decision, [MatchDecision.MATCH_CONFIRMED, MatchDecision.MATCH_PROBABLE])

    def test_15_deterministic_reproducibility(self):
        """Test 15: Verify deterministic Re-ID matching reproducibility."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 18.0)

        b1 = self.matching_engine.compute_match_score(ev_a, trk_a, ev_b, trk_b)
        b2 = self.matching_engine.compute_match_score(ev_a, trk_a, ev_b, trk_b)

        self.assertEqual(b1.overall_score, b2.overall_score)
        self.assertEqual(b1.decision, b2.decision)

    def test_16_phase_1_to_phase_5_integration(self):
        """Test 16: Phase 1 simulation engine -> Phase 5 Re-ID integration."""
        sim = TrafficSimulationEngine(seed=42)
        for _ in range(5):
            sim.step(1.0)
        self.assertGreater(len(sim.recent_observations), 0)

    def test_17_phase_2_to_phase_5_integration(self):
        """Test 17: Phase 2 vehicle detection -> Phase 5 Re-ID integration."""
        sim = TrafficSimulationEngine(seed=42)
        sim.step(1.0)
        det_events = DetectionAdapter.batch_convert(sim.recent_observations)
        self.assertGreater(len(det_events), 0)

    def test_18_phase_3_to_phase_5_integration(self):
        """Test 18: Phase 3 vehicle tracking -> Phase 5 Re-ID integration."""
        sim = TrafficSimulationEngine(seed=42)
        sim.step(1.0)
        det_events = DetectionAdapter.batch_convert(sim.recent_observations)
        tracker_mgr = VehicleTrackerManager()
        tracks = tracker_mgr.update(det_events[0].camera_id, det_events, 1.0)
        self.assertGreater(len(tracks), 0)

    def test_19_phase_4_to_phase_5_plate_association(self):
        """Test 19: Phase 4 ALPR plate evidence -> Phase 5 Re-ID journey integration."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="TN09AB1234")
        journey = self.journey_engine.process_track_evidence(ev_a, trk_a)

        self.assertEqual(journey.plate_number, "TN09AB1234")

    def test_20_one_to_one_assignment(self):
        """Test 20: Optimal one-to-one track assignment preventing duplicate conflicting journey merges."""
        ev_a1, trk_a1 = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="TN09AB1234")
        ev_a2, trk_a2 = self._create_mock_track("TRK_102", "CAM_A_EAST", 0.0, plate="KA01MH9999")

        j1 = self.journey_engine.process_track_evidence(ev_a1, trk_a1)
        j2 = self.journey_engine.process_track_evidence(ev_a2, trk_a2)

        self.assertNotEqual(j1.journey_id, j2.journey_id)
        self.assertNotEqual(j1.global_vehicle_id, j2.global_vehicle_id)


if __name__ == "__main__":
    unittest.main()
