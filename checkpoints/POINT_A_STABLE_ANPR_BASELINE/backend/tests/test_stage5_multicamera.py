"""
ChronoEye Infinity - Stage 5: Multi-Camera Cross-ReID Test Harness
Deterministic test harness verifying cross-camera track association,
global vehicle identity assignment, and journey reconstruction across multiple CCTV cameras.
"""

import math
import os
import sys
import unittest
from typing import List, Tuple

# Add backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.detection import BoundingBoxXYXY
from app.schemas.tracking import TrackState, TrackStatus, Direction
from app.schemas.plate import FusedPlateIdentity, VehicleIdentityEvidence
from app.schemas.reid import MatchDecision, ReIDConfig, VehicleJourney
from app.perception.camera_topology import CityCameraTopology
from app.perception.reid_engine import ReIDMatchingEngine
from app.perception.journey import JourneyReconstructionEngine


class TestStage5MultiCameraHarness(unittest.TestCase):
    """
    Deterministic multi-camera test harness for Stage 5 Re-ID and Journey Reconstruction.
    """

    def setUp(self):
        self.topology = CityCameraTopology()
        self.config = ReIDConfig()
        self.matching_engine = ReIDMatchingEngine(topology=self.topology, config=self.config)
        self.journey_engine = JourneyReconstructionEngine(
            matching_engine=self.matching_engine, config=self.config
        )

        # Fixed deterministic 128-D normalized embedding vector
        val = 1.0 / math.sqrt(128)
        self.fixed_embedding: List[float] = [val] * 128

    def _create_synthetic_observation(
        self,
        camera_id: str,
        track_id: str,
        timestamp: float,
        plate: str = "KW527",
        vehicle_type: str = "car",
        direction: Direction = Direction.EAST,
        embedding: List[float] = None,
    ) -> Tuple[VehicleIdentityEvidence, TrackState]:
        """Creates a synthetic track and associated plate evidence."""
        track = TrackState(
            track_id=track_id,
            camera_id=camera_id,
            vehicle_type=vehicle_type,
            current_bbox=BoundingBoxXYXY(x1=150.0, y1=200.0, x2=350.0, y2=400.0),
            current_center=(250.0, 300.0),
            confidence=0.95,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=direction,
        )

        fused_plate = FusedPlateIdentity(
            track_id=track_id,
            camera_id=camera_id,
            best_plate_number=plate,
            overall_confidence=0.95,
            observation_count=5,
            confirmed=True,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
        )

        evidence = VehicleIdentityEvidence(
            track_id=track_id,
            camera_id=camera_id,
            vehicle_type=vehicle_type,
            associated_plate=fused_plate,
            last_updated_timestamp=timestamp,
        )

        # Inject fixed appearance embedding into journey engine cache for this track
        emb = embedding if embedding is not None else self.fixed_embedding
        self.journey_engine.track_embeddings[track_id] = (timestamp, emb)

        return evidence, track

    def test_cross_camera_same_vehicle_merges(self):
        """
        Verify that sequential observations of vehicle KW527 on Camera A and Camera B
        merge into a single global vehicle ID with MATCH_CONFIRMED.
        """
        # Camera A: CAM_A_EAST, TRK_A_001, t=0.0s, plate=KW527
        ev_a, trk_a = self._create_synthetic_observation(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_001",
            timestamp=0.0,
            plate="KW527",
            direction=Direction.EAST,
        )
        journey_a = self.journey_engine.process_track_evidence(ev_a, trk_a)

        # Explicitly verify matching engine decision
        ev_b, trk_b = self._create_synthetic_observation(
            camera_id="CAM_B_WEST",
            track_id="TRK_B_001",
            timestamp=18.0,  # 300m at 16.67 m/s (60 km/h) — physically feasible
            plate="KW527",
            direction=Direction.EAST,
        )
        breakdown = self.matching_engine.compute_match_score(
            ev_a, trk_a, ev_b, trk_b, self.fixed_embedding, self.fixed_embedding
        )
        self.assertEqual(breakdown.decision, MatchDecision.MATCH_CONFIRMED)
        self.assertGreaterEqual(breakdown.overall_score, 0.75)

        # Process Camera B through Journey Engine
        journey_b = self.journey_engine.process_track_evidence(ev_b, trk_b)

        # 1. Two local tracks must merge into ONE global vehicle ID
        self.assertEqual(journey_a.global_vehicle_id, journey_b.global_vehicle_id)
        # 2. Both segments belong to the same journey
        self.assertEqual(journey_a.journey_id, journey_b.journey_id)
        # 3. Pooled plate identity must remain KW527
        self.assertEqual(journey_b.plate_number, "KW527")

    def test_cross_camera_journey_has_two_segments(self):
        """
        Verify that the merged journey contains exactly two chronological JourneySegments
        representing CAM_A_EAST and CAM_B_WEST.
        """
        ev_a, trk_a = self._create_synthetic_observation(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_001",
            timestamp=0.0,
            plate="KW527",
        )
        self.journey_engine.process_track_evidence(ev_a, trk_a)

        ev_b, trk_b = self._create_synthetic_observation(
            camera_id="CAM_B_WEST",
            track_id="TRK_B_001",
            timestamp=18.0,
            plate="KW527",
        )
        journey = self.journey_engine.process_track_evidence(ev_b, trk_b)

        # Journey must contain two segments
        self.assertEqual(len(journey.segments), 2)
        self.assertEqual(journey.segments[0].camera_id, "CAM_A_EAST")
        self.assertEqual(journey.segments[0].track_id, "TRK_A_001")
        self.assertEqual(journey.segments[0].timestamp, 0.0)

        self.assertEqual(journey.segments[1].camera_id, "CAM_B_WEST")
        self.assertEqual(journey.segments[1].track_id, "TRK_B_001")
        self.assertEqual(journey.segments[1].timestamp, 18.0)

        # Cameras list must contain both cameras in order
        self.assertEqual(journey.cameras, ["CAM_A_EAST", "CAM_B_WEST"])

    def test_impossible_travel_is_rejected(self):
        """
        Verify that physically impossible travel time (e.g. 300m in 1.0s = 1080 km/h)
        is rejected and produces two separate global vehicle IDs.
        """
        ev_a, trk_a = self._create_synthetic_observation(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_001",
            timestamp=0.0,
            plate="KW527",
        )
        journey_a = self.journey_engine.process_track_evidence(ev_a, trk_a)

        # Camera B observation after only 1.0s (300m / 1s = 1080 km/h > 120 km/h max)
        ev_b_fast, trk_b_fast = self._create_synthetic_observation(
            camera_id="CAM_B_WEST",
            track_id="TRK_B_FAST",
            timestamp=1.0,
            plate="KW527",
        )

        breakdown = self.matching_engine.compute_match_score(
            ev_a, trk_a, ev_b_fast, trk_b_fast, self.fixed_embedding, self.fixed_embedding
        )
        self.assertEqual(breakdown.decision, MatchDecision.MATCH_REJECTED)
        self.assertIn("PHYSICALLY_IMPOSSIBLE_SPEED", breakdown.rejection_reason)

        journey_b_fast = self.journey_engine.process_track_evidence(ev_b_fast, trk_b_fast)

        # Must not merge into the same journey or global vehicle ID
        self.assertNotEqual(journey_a.global_vehicle_id, journey_b_fast.global_vehicle_id)
        self.assertNotEqual(journey_a.journey_id, journey_b_fast.journey_id)
        self.assertEqual(len(journey_a.segments), 1)
        self.assertEqual(len(journey_b_fast.segments), 1)


if __name__ == "__main__":
    unittest.main()
