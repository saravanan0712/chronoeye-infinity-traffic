"""
ChronoEye Infinity - Stage 5: Journey Completion Lifecycle Verification Test Suite
Deterministic test harness verifying that active vehicle journeys transition to COMPLETED
after a configured inactivity timeout, completed journeys are excluded from future matching,
and new observations create fresh global vehicle identities while cross-camera Re-ID is preserved.
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


class TestStage5JourneyLifecycle(unittest.TestCase):
    """
    Stage 5 Journey Retirement and Completion Lifecycle Verification Tests.
    """

    def setUp(self):
        self.topology = CityCameraTopology()
        # Explicit deterministic inactivity timeout: 60.0 seconds
        self.config = ReIDConfig(journey_timeout_seconds=60.0)
        self.matching_engine = ReIDMatchingEngine(topology=self.topology, config=self.config)
        self.journey_engine = JourneyReconstructionEngine(
            matching_engine=self.matching_engine,
            config=self.config,
            journey_timeout_seconds=60.0,
        )

        # Fixed deterministic 128-D normalized embedding vector
        val = 1.0 / math.sqrt(128)
        self.fixed_embedding: List[float] = [val] * 128

    def _create_synthetic_track(
        self,
        camera_id: str,
        track_id: str,
        timestamp: float,
        plate: str = "KW527",
        vehicle_type: str = "car",
        direction: Direction = Direction.EAST,
        embedding: List[float] = None,
    ) -> Tuple[VehicleIdentityEvidence, TrackState]:
        track = TrackState(
            track_id=track_id,
            camera_id=camera_id,
            vehicle_type=vehicle_type,
            current_bbox=BoundingBoxXYXY(x1=100.0, y1=150.0, x2=300.0, y2=350.0),
            current_center=(200.0, 250.0),
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
            observation_count=3,
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

        emb = embedding if embedding is not None else self.fixed_embedding
        self.journey_engine.track_embeddings[track_id] = (timestamp, emb)

        return evidence, track

    def test_active_journey_completes_after_timeout(self):
        """
        Verify that an active journey transitions from ACTIVE to COMPLETED
        when time advances beyond the configured inactivity timeout (60s).
        """
        # Observation at t=0.0s
        ev1, trk1 = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_001",
            timestamp=0.0,
            plate="KW527",
        )
        journey = self.journey_engine.process_track_evidence(ev1, trk1)
        self.assertEqual(journey.status, "ACTIVE")
        self.assertEqual(len(self.journey_engine.get_active_journeys()), 1)
        self.assertEqual(len(self.journey_engine.get_completed_journeys()), 0)

        # Advance deterministic time to t=65.0s (> 60.0s timeout) via retirement sweep
        completed_ids = self.journey_engine.retire_completed_journeys(current_timestamp=65.0)

        self.assertIn(journey.journey_id, completed_ids)
        self.assertEqual(journey.status, "COMPLETED")
        self.assertEqual(len(self.journey_engine.get_active_journeys()), 0)
        self.assertEqual(len(self.journey_engine.get_completed_journeys()), 1)

    def test_active_journey_remains_active_before_timeout(self):
        """
        Verify that an active journey remains ACTIVE while observations
        continue to arrive within the inactivity timeout window (< 60s).
        """
        ev1, trk1 = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_001",
            timestamp=0.0,
            plate="KW527",
        )
        journey = self.journey_engine.process_track_evidence(ev1, trk1)
        self.assertEqual(journey.status, "ACTIVE")

        # Advance deterministic time to t=25.0s (< 60.0s timeout)
        completed_ids = self.journey_engine.retire_completed_journeys(current_timestamp=25.0)
        self.assertEqual(len(completed_ids), 0)
        self.assertEqual(journey.status, "ACTIVE")

        # Second observation within timeout window (t=30.0s)
        ev2, trk2 = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_002",
            timestamp=30.0,
            plate="KW527",
        )
        updated_journey = self.journey_engine.process_track_evidence(ev2, trk2)

        self.assertEqual(updated_journey.status, "ACTIVE")
        self.assertEqual(updated_journey.last_seen, 30.0)
        self.assertEqual(len(self.journey_engine.get_active_journeys()), 1)

    def test_completed_journey_not_used_for_matching(self):
        """
        Verify that once a journey is COMPLETED, it is excluded from future
        active matching, and a new observation spawns a separate global vehicle ID.
        """
        # Journey 1 created at t=0.0s
        ev1, trk1 = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_001",
            timestamp=0.0,
            plate="KW527",
        )
        j1 = self.journey_engine.process_track_evidence(ev1, trk1)
        j1_id = j1.journey_id
        j1_veh_id = j1.global_vehicle_id

        # New observation arrives at t=120.0s (> 60s after t=0.0s)
        # process_track_evidence triggers retirement sweep at t=120.0s, completing j1
        ev2, trk2 = self._create_synthetic_track(
            camera_id="CAM_B_WEST",
            track_id="TRK_B_001",
            timestamp=120.0,
            plate="KW527",
        )
        j2 = self.journey_engine.process_track_evidence(ev2, trk2)

        # J1 must now be COMPLETED
        self.assertEqual(self.journey_engine.journeys[j1_id].status, "COMPLETED")
        # J2 is a new ACTIVE journey with a distinct global vehicle ID
        self.assertEqual(j2.status, "ACTIVE")
        self.assertNotEqual(j1_id, j2.journey_id)
        self.assertNotEqual(j1_veh_id, j2.global_vehicle_id)
        self.assertEqual(len(j2.segments), 1)

    def test_existing_cross_camera_matching_still_works(self):
        """
        Verify that active cross-camera matching (CAM_A_EAST -> CAM_B_WEST within timeout)
        continues to merge smoothly under the lifecycle system.
        """
        # CAM_A_EAST at t=0.0s
        ev_a, trk_a = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_001",
            timestamp=0.0,
            plate="KW527",
        )
        j_a = self.journey_engine.process_track_evidence(ev_a, trk_a)

        # CAM_B_WEST at t=18.0s (well within 60s timeout)
        ev_b, trk_b = self._create_synthetic_track(
            camera_id="CAM_B_WEST",
            track_id="TRK_B_001",
            timestamp=18.0,
            plate="KW527",
        )
        j_b = self.journey_engine.process_track_evidence(ev_b, trk_b)

        # Successfully merged into single active journey
        self.assertEqual(j_a.global_vehicle_id, j_b.global_vehicle_id)
        self.assertEqual(j_a.journey_id, j_b.journey_id)
        self.assertEqual(j_b.status, "ACTIVE")
        self.assertEqual(len(j_b.segments), 2)
        self.assertEqual(j_b.plate_number, "KW527")
        self.assertEqual(j_b.cameras, ["CAM_A_EAST", "CAM_B_WEST"])


if __name__ == "__main__":
    unittest.main()
