"""
ChronoEye Infinity - Stage 5: Intra-Camera Track Recovery Verification Test Suite
Deterministic test harness verifying that broken/lost tracks within the SAME camera
(e.g. temporary ByteTrack loss or occlusion) are correctly re-identified and merged
into a single continuous global vehicle journey when evidence matches, while genuinely
distinct vehicles remain strictly separate.
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


class TestStage5IntraCameraRecovery(unittest.TestCase):
    """
    Stage 5 Intra-Camera Track Recovery & Re-ID Verification Tests.
    """

    def setUp(self):
        self.topology = CityCameraTopology()
        self.config = ReIDConfig()
        self.matching_engine = ReIDMatchingEngine(topology=self.topology, config=self.config)
        self.journey_engine = JourneyReconstructionEngine(
            matching_engine=self.matching_engine, config=self.config
        )

        # Fixed deterministic 128-D normalized embedding vectors
        val = 1.0 / math.sqrt(128)
        self.fixed_embedding_a: List[float] = [val] * 128
        # Orthogonal embedding for distinct vehicle
        self.fixed_embedding_b: List[float] = [-val if i % 2 == 0 else val for i in range(128)]

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
            confidence=0.92,
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

        emb = embedding if embedding is not None else self.fixed_embedding_a
        self.journey_engine.track_embeddings[track_id] = (timestamp, emb)

        return evidence, track

    def test_intra_camera_same_vehicle_merges_after_track_loss(self):
        """
        Verify that a vehicle observed as TRK_A_001 at t=0s on CAM_A_EAST
        and re-appearing as TRK_A_002 at t=5s on the SAME camera CAM_A_EAST
        with matching evidence (plate KW527 and matching appearance) merges into
        ONE global vehicle ID and ONE journey with TWO segments.
        """
        # Observation 1: TRK_A_001 on CAM_A_EAST at t=0.0s
        ev1, trk1 = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_001",
            timestamp=0.0,
            plate="KW527",
            embedding=self.fixed_embedding_a,
        )
        j1 = self.journey_engine.process_track_evidence(ev1, trk1)

        # Observation 2: TRK_A_002 on SAME camera CAM_A_EAST at t=5.0s (after temporary ByteTrack loss)
        ev2, trk2 = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_002",
            timestamp=5.0,
            plate="KW527",
            embedding=self.fixed_embedding_a,
        )
        j2 = self.journey_engine.process_track_evidence(ev2, trk2)

        # 1. TRK_A_001 and TRK_A_002 map to ONE global vehicle ID
        self.assertEqual(j1.global_vehicle_id, j2.global_vehicle_id)
        # 2. They belong to ONE journey
        self.assertEqual(j1.journey_id, j2.journey_id)
        # 3. The journey contains TWO segments
        self.assertEqual(len(j2.segments), 2)
        self.assertEqual(j2.segments[0].track_id, "TRK_A_001")
        self.assertEqual(j2.segments[0].timestamp, 0.0)
        self.assertEqual(j2.segments[1].track_id, "TRK_A_002")
        self.assertEqual(j2.segments[1].timestamp, 5.0)
        # 4. KW527 remains pooled
        self.assertEqual(j2.plate_number, "KW527")

    def test_intra_camera_distinct_vehicles_do_not_merge(self):
        """
        Negative case: Two different vehicles observed on the SAME camera CAM_A_EAST
        (TRK_A_001 with plate KW527 vs TRK_A_002 with plate MH12DE1234 and distinct appearance)
        must NOT merge and must create TWO distinct global vehicle IDs and journeys.
        """
        # Vehicle 1: TRK_A_001 on CAM_A_EAST at t=0.0s, plate KW527
        ev1, trk1 = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_001",
            timestamp=0.0,
            plate="KW527",
            vehicle_type="car",
            embedding=self.fixed_embedding_a,
        )
        j1 = self.journey_engine.process_track_evidence(ev1, trk1)

        # Vehicle 2: TRK_A_002 on SAME camera CAM_A_EAST at t=5.0s, plate MH12DE1234, distinct embedding
        ev2, trk2 = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_002",
            timestamp=5.0,
            plate="MH12DE1234",
            vehicle_type="car",
            embedding=self.fixed_embedding_b,
        )
        j2 = self.journey_engine.process_track_evidence(ev2, trk2)

        # Must NOT merge
        self.assertNotEqual(j1.global_vehicle_id, j2.global_vehicle_id)
        self.assertNotEqual(j1.journey_id, j2.journey_id)
        self.assertEqual(len(j1.segments), 1)
        self.assertEqual(len(j2.segments), 1)
        self.assertEqual(j1.plate_number, "KW527")
        self.assertEqual(j2.plate_number, "MH12DE1234")

    def test_intra_camera_simultaneous_tracks_do_not_merge(self):
        """
        Negative case: Two tracks co-present in the SAME camera frame at the exact same timestamp (t=0.0s)
        represent two simultaneous physical objects and must NOT merge.
        """
        ev1, trk1 = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_001",
            timestamp=0.0,
            plate="KW527",
            embedding=self.fixed_embedding_a,
        )
        j1 = self.journey_engine.process_track_evidence(ev1, trk1)

        # Another track at the exact same timestamp t=0.0s
        ev2, trk2 = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_002",
            timestamp=0.0,
            plate="KW527",
            embedding=self.fixed_embedding_a,
        )
        j2 = self.journey_engine.process_track_evidence(ev2, trk2)

        # Must NOT merge when co-present at the same instant
        self.assertNotEqual(j1.global_vehicle_id, j2.global_vehicle_id)
        self.assertNotEqual(j1.journey_id, j2.journey_id)


if __name__ == "__main__":
    unittest.main()
