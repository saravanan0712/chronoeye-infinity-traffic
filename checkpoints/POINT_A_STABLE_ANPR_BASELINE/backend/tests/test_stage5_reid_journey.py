"""
ChronoEye Infinity - Stage 5 Cross-Camera Re-ID & Journey Reconstruction Verification Test Suite
Automated Python test suite verifying Stage 5 requirements across 20 explicit test cases.
"""

import os
import sys
import unittest
import json
from typing import Tuple

# Add backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.detection import BoundingBoxXYXY
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
    JourneySegment,
)
from app.perception.camera_topology import CityCameraTopology, CameraNode
from app.perception.appearance import AppearanceEmbeddingExtractor
from app.perception.reid_engine import ReIDMatchingEngine
from app.perception.journey import JourneyReconstructionEngine
from app.perception.plate_association import PlateTrackerAssociationManager


class TestStage5ReIDJourneyPipeline(unittest.TestCase):
    """Stage 5: Cross-Camera Re-ID & Vehicle Journey Reconstruction Verification Tests."""

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

    def test_1_configuration(self):
        """Test 5.1: Verify Re-ID matching engine configuration parameters."""
        cfg = ReIDConfig(w_plate=0.40, w_appearance=0.30, confirm_threshold=0.80)
        self.assertEqual(cfg.w_plate, 0.40)
        self.assertEqual(cfg.confirm_threshold, 0.80)

    def test_2_identity_schema(self):
        """Test 5.2: Verify VehicleJourney and ReIDScoreBreakdown schema instantiation."""
        breakdown = ReIDScoreBreakdown(
            plate_similarity=0.9, appearance_similarity=0.85, overall_score=0.88, decision=MatchDecision.MATCH_CONFIRMED
        )
        self.assertEqual(breakdown.decision, MatchDecision.MATCH_CONFIRMED)
        self.assertEqual(breakdown.overall_score, 0.88)

    def test_3_camera_topology(self):
        """Test 5.3: Verify camera topology distances and spatial connectivity."""
        dist = self.topology.get_distance_meters("CAM_A_EAST", "CAM_B_WEST")
        self.assertEqual(dist, 300.0)

    def test_4_plate_matching(self):
        """Test 5.4: Verify exact and normalized license plate similarity calculation."""
        self.assertEqual(self.matching_engine.calculate_plate_similarity("TN09AB1234", "TN09AB1234"), 1.0)
        self.assertEqual(self.matching_engine.calculate_plate_similarity("TN-09 AB 1234", "TN09AB1234"), 1.0)

    def test_5_ocr_confidence_handling(self):
        """Test 5.5: Verify OCR plate similarity gracefully handles neutral fallback for unknown plates."""
        score = self.matching_engine.calculate_plate_similarity("UNKNOWN", "TN09AB1234")
        self.assertEqual(score, 0.50)

    def test_6_vehicle_class_compatibility(self):
        """Test 5.6: Verify distinct vehicle classes mismatch score."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 18.0)
        ev_b.vehicle_type = "bus"

        breakdown = self.matching_engine.compute_match_score(ev_a, trk_a, ev_b, trk_b)
        self.assertEqual(breakdown.vehicle_type_similarity, 0.0)

    def test_7_appearance_similarity(self):
        """Test 5.7: Verify AppearanceEmbeddingExtractor cosine similarity calculation."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]

        self.assertEqual(AppearanceEmbeddingExtractor.cosine_similarity(v1, v2), 1.0)
        self.assertEqual(AppearanceEmbeddingExtractor.cosine_similarity(v1, v3), 0.0)

    def test_8_temporal_feasibility(self):
        """Test 5.8: Verify temporal travel time feasibility bounds."""
        feasible, reason = self.topology.is_temporally_feasible("CAM_A_EAST", "CAM_B_WEST", delta_t_seconds=18.0)
        self.assertTrue(feasible)
        self.assertIsNone(reason)

    def test_9_impossible_transition_rejection(self):
        """Test 5.9: Verify physically impossible travel speed (e.g., 300m in 1s) is explicitly rejected."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 1.0)

        breakdown = self.matching_engine.compute_match_score(ev_a, trk_a, ev_b, trk_b)
        self.assertEqual(breakdown.decision, MatchDecision.MATCH_REJECTED)
        self.assertIn("PHYSICALLY_IMPOSSIBLE_SPEED", breakdown.rejection_reason)

    def test_10_cross_camera_matching(self):
        """Test 5.10: Verify multi-evidence cross-camera track matching calculation."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 18.0)

        breakdown = self.matching_engine.compute_match_score(ev_a, trk_a, ev_b, trk_b)
        self.assertEqual(breakdown.decision, MatchDecision.MATCH_CONFIRMED)
        self.assertGreater(breakdown.overall_score, 0.80)

    def test_11_persistent_vehicle_id_creation(self):
        """Test 5.11: Verify global persistent vehicle ID creation (VEH_101)."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        journey = self.journey_engine.process_track_evidence(ev_a, trk_a)

        self.assertTrue(journey.global_vehicle_id.startswith("VEH_"))

    def test_12_journey_creation(self):
        """Test 5.12: Verify vehicle journey creation (JRN_101)."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        journey = self.journey_engine.process_track_evidence(ev_a, trk_a)

        self.assertTrue(journey.journey_id.startswith("JRN_"))
        self.assertEqual(len(journey.segments), 1)

    def test_13_journey_event_ordering(self):
        """Test 5.13: Verify chronological segment ordering in journey timeline."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 18.0)

        self.journey_engine.process_track_evidence(ev_a, trk_a)
        journey = self.journey_engine.process_track_evidence(ev_b, trk_b)

        self.assertEqual(journey.segments[0].timestamp, 0.0)
        self.assertEqual(journey.segments[1].timestamp, 18.0)

    def test_14_uncertain_match_handling(self):
        """Test 5.14: Verify weak evidence yields INSUFFICIENT_EVIDENCE without false confirmation."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="TN09AB1234")
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 18.0, plate="KA01CD5678")
        ev_b.vehicle_type = "truck"

        breakdown = self.matching_engine.compute_match_score(ev_a, trk_a, ev_b, trk_b)
        self.assertEqual(breakdown.decision, MatchDecision.INSUFFICIENT_EVIDENCE)

    def test_15_unknown_plate_handling(self):
        """Test 5.15: Verify missing/unknown license plate does not fabricate text."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="UNKNOWN")
        journey = self.journey_engine.process_track_evidence(ev_a, trk_a)

        self.assertEqual(journey.plate_number, "UNKNOWN")

    def test_16_serialization(self):
        """Test 5.16: Verify Pydantic schema serialization for VehicleJourney."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        journey = self.journey_engine.process_track_evidence(ev_a, trk_a)

        data = journey.model_dump()
        self.assertIn("journey_id", data)
        self.assertIn("global_vehicle_id", data)
        dump_json = json.dumps(data)
        self.assertIn(journey.global_vehicle_id, dump_json)

    def test_17_stage4_to_stage5_integration(self):
        """Test 5.17: Verify Stage 4 ALPR plate evidence feeds directly into Stage 5 Journey engine."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="TN09AB1234")
        journey = self.journey_engine.process_track_evidence(ev_a, trk_a)

        self.assertEqual(journey.plate_number, "TN09AB1234")

    def test_18_multi_camera_integration(self):
        """Test 5.18: Verify 3-camera continuous journey reconstruction (CAM_A -> CAM_B -> CAM_C)."""
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

    def test_19_deterministic_behavior(self):
        """Test 5.19: Verify deterministic Re-ID matching reproducibility."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 18.0)

        b1 = self.matching_engine.compute_match_score(ev_a, trk_a, ev_b, trk_b)
        b2 = self.matching_engine.compute_match_score(ev_a, trk_a, ev_b, trk_b)

        self.assertEqual(b1.overall_score, b2.overall_score)
        self.assertEqual(b1.decision, b2.decision)

    def test_20_regression_compatibility(self):
        """Test 5.20: Verify non-conflicting journey creation for distinct vehicles."""
        ev_a1, trk_a1 = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="TN09AB1234")
        ev_a2, trk_a2 = self._create_mock_track("TRK_102", "CAM_A_EAST", 0.0, plate="KA01MH9999")

        j1 = self.journey_engine.process_track_evidence(ev_a1, trk_a1)
        j2 = self.journey_engine.process_track_evidence(ev_a2, trk_a2)

        self.assertNotEqual(j1.journey_id, j2.journey_id)
        self.assertNotEqual(j1.global_vehicle_id, j2.global_vehicle_id)


if __name__ == "__main__":
    unittest.main()
