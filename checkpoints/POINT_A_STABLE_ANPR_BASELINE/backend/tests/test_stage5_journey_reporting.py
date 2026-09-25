"""
ChronoEye Infinity - Stage 5: Journey Observability & Reporting Verification Test Suite
Deterministic test harness verifying that the Stage 5 Journey Reconstruction Engine
provides structured, accurate reporting of reconstructed vehicle journeys, including
global vehicle IDs, journey IDs, lifecycle status, pooled plates, segment counts,
camera sequences, local track IDs, and temporal boundaries.
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


class TestStage5JourneyReporting(unittest.TestCase):
    """
    Stage 5 Journey Observability & Reporting Verification Tests.
    """

    def setUp(self):
        self.topology = CityCameraTopology()
        self.config = ReIDConfig(journey_timeout_seconds=60.0)
        self.matching_engine = ReIDMatchingEngine(topology=self.topology, config=self.config)
        self.journey_engine = JourneyReconstructionEngine(
            matching_engine=self.matching_engine,
            config=self.config,
            journey_timeout_seconds=60.0,
        )

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
            confirmed=(plate != "UNKNOWN"),
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

    def test_journey_summary_reporting_fields(self):
        """
        Verify that get_journey_summary_report returns all required fields
        with correct types and values for cross-camera and single-camera journeys.
        """
        # Vehicle 1 Segment 1 (CAM_A_EAST at t=0.0s)
        ev_a, trk_a = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_001",
            timestamp=0.0,
            plate="KW527",
        )
        self.journey_engine.process_track_evidence(ev_a, trk_a)

        # Vehicle 2: Single camera (CAM_C_NORTH at t=5.0s), plate UNKNOWN
        ev_c, trk_c = self._create_synthetic_track(
            camera_id="CAM_C_NORTH",
            track_id="TRK_C_001",
            timestamp=5.0,
            plate="UNKNOWN",
        )
        self.journey_engine.process_track_evidence(ev_c, trk_c)

        # Vehicle 1 Segment 2 (CAM_B_WEST at t=18.0s)
        ev_b, trk_b = self._create_synthetic_track(
            camera_id="CAM_B_WEST",
            track_id="TRK_B_001",
            timestamp=18.0,
            plate="KW527",
        )
        self.journey_engine.process_track_evidence(ev_b, trk_b)

        # Generate summary report
        report = self.journey_engine.get_journey_summary_report()

        self.assertEqual(len(report), 2)

        # Inspect Vehicle 1 (Cross-camera journey)
        v1_rep = next(r for r in report if r["plate_number"] == "KW527")
        self.assertEqual(v1_rep["global_vehicle_id"], "VEH_101")
        self.assertEqual(v1_rep["journey_id"], "JRN_101")
        self.assertEqual(v1_rep["status"], "ACTIVE")
        self.assertEqual(v1_rep["plate_number"], "KW527")
        self.assertEqual(v1_rep["number_of_segments"], 2)
        self.assertEqual(v1_rep["camera_sequence"], ["CAM_A_EAST", "CAM_B_WEST"])
        self.assertEqual(v1_rep["local_track_ids"], ["TRK_A_001", "TRK_B_001"])
        self.assertEqual(v1_rep["first_timestamp"], 0.0)
        self.assertEqual(v1_rep["last_timestamp"], 18.0)

        # Inspect Vehicle 2 (Single camera journey)
        v2_rep = next(r for r in report if r["plate_number"] == "UNKNOWN")
        self.assertEqual(v2_rep["global_vehicle_id"], "VEH_102")
        self.assertEqual(v2_rep["journey_id"], "JRN_102")
        self.assertEqual(v2_rep["status"], "ACTIVE")
        self.assertEqual(v2_rep["number_of_segments"], 1)
        self.assertEqual(v2_rep["camera_sequence"], ["CAM_C_NORTH"])
        self.assertEqual(v2_rep["local_track_ids"], ["TRK_C_001"])
        self.assertEqual(v2_rep["first_timestamp"], 5.0)
        self.assertEqual(v2_rep["last_timestamp"], 5.0)

    def test_reporting_reflects_completed_lifecycle_status(self):
        """
        Verify that the journey summary report accurately reflects COMPLETED status
        after the inactivity timeout is exceeded.
        """
        ev, trk = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_001",
            timestamp=0.0,
            plate="KW527",
        )
        self.journey_engine.process_track_evidence(ev, trk)

        # Before timeout: ACTIVE
        rep_active = self.journey_engine.get_journey_summary_report()
        self.assertEqual(rep_active[0]["status"], "ACTIVE")

        # Advance time to t=70.0s (> 60s)
        self.journey_engine.retire_completed_journeys(70.0)

        # After timeout: COMPLETED
        rep_completed = self.journey_engine.get_journey_summary_report()
        self.assertEqual(rep_completed[0]["status"], "COMPLETED")
        self.assertEqual(rep_completed[0]["global_vehicle_id"], "VEH_101")


if __name__ == "__main__":
    unittest.main()
