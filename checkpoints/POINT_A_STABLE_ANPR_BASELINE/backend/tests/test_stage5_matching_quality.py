"""
ChronoEye Infinity - Stage 5: Matching Quality & Competing Candidate Association Tests
Deterministic test harness verifying that when multiple active candidate journeys exist,
the Re-ID engine correctly scores all candidates and assigns the observation to the
highest-scoring candidate, and rejects ambiguous/insufficient evidence without forcing a false match.
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


class TestStage5MatchingQuality(unittest.TestCase):
    """
    Stage 5 Matching Quality & Competing Candidate Association Tests.
    """

    def setUp(self):
        self.topology = CityCameraTopology()
        self.config = ReIDConfig()
        self.matching_engine = ReIDMatchingEngine(topology=self.topology, config=self.config)
        self.journey_engine = JourneyReconstructionEngine(
            matching_engine=self.matching_engine, config=self.config
        )

        # Embedding vectors:
        val = 1.0 / math.sqrt(128)
        # Vector A (e.g. for KW527)
        self.emb_a: List[float] = [val] * 128
        # Vector B (partially correlated with A, dot product = 0.5)
        # 96 components identical (+val), 32 inverted (-val) -> (96 - 32)/128 = 64/128 = 0.50
        self.emb_b: List[float] = [val if i < 96 else -val for i in range(128)]
        # Vector C (orthogonal to A, dot product = 0.0)
        self.emb_c: List[float] = [val if i % 2 == 0 else -val for i in range(128)]

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

        emb = embedding if embedding is not None else self.emb_a
        self.journey_engine.track_embeddings[track_id] = (timestamp, emb)

        return evidence, track

    def test_competing_candidates_selects_highest_scoring_journey(self):
        """
        Scenario:
        - Journey 1 (VEH_101): CAM_A_EAST at t=0.0s, plate=UNKNOWN, embedding=emb_b (partial match).
        - Journey 2 (VEH_102): CAM_C_NORTH at t=0.0s, plate=KW527, embedding=emb_a (exact match).
        - New Observation: CAM_B_WEST at t=18.0s, plate=KW527, embedding=emb_a.

        Both J1 and J2 are active candidates connected to CAM_B_WEST.
        J2 has a clearly higher Re-ID score than J1.
        Verify that the engine evaluates both and assigns the track to J2 (VEH_102).
        """
        # 1. Create Journey 1 (Weak/Plausible Candidate)
        ev_1, trk_1 = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_001",
            timestamp=0.0,
            plate="UNKNOWN",
            embedding=self.emb_b,
        )
        j1 = self.journey_engine.process_track_evidence(ev_1, trk_1)
        j1_veh_id = j1.global_vehicle_id

        # 2. Create Journey 2 (Strong/Exact Candidate)
        ev_2, trk_2 = self._create_synthetic_track(
            camera_id="CAM_C_NORTH",
            track_id="TRK_C_001",
            timestamp=0.0,
            plate="KW527",
            embedding=self.emb_a,
        )
        j2 = self.journey_engine.process_track_evidence(ev_2, trk_2)
        j2_veh_id = j2.global_vehicle_id

        self.assertNotEqual(j1_veh_id, j2_veh_id)

        # 3. New observation on CAM_B_WEST at t=18.0s
        ev_new, trk_new = self._create_synthetic_track(
            camera_id="CAM_B_WEST",
            track_id="TRK_B_001",
            timestamp=18.0,
            plate="KW527",
            embedding=self.emb_a,
        )

        # Directly compute and inspect individual candidate Re-ID scores
        score_vs_j1 = self.matching_engine.compute_match_score(
            ev_new, trk_new, ev_1, trk_1, self.emb_a, self.emb_b
        )
        score_vs_j2 = self.matching_engine.compute_match_score(
            ev_new, trk_new, ev_2, trk_2, self.emb_a, self.emb_a
        )

        # Candidate 1 Score (Plate=0.50, App=0.50, Type=1.0, Dir=1.0, Time=1.0, Space=1.0)
        # Score = 0.35(0.5) + 0.20(0.5) + 0.10(1.0) + 0.10(1.0) + 0.10(1.0) + 0.15(1.0) = 0.725
        self.assertGreaterEqual(score_vs_j1.overall_score, 0.50)  # Plausible (MATCH_PROBABLE)
        self.assertLess(score_vs_j1.overall_score, score_vs_j2.overall_score)

        # Candidate 2 Score (Plate=1.0, App=1.0, Type=1.0, Dir=1.0, Time=1.0, Space=1.0)
        # Score = 1.000 (MATCH_CONFIRMED)
        self.assertEqual(score_vs_j2.decision, MatchDecision.MATCH_CONFIRMED)
        self.assertGreaterEqual(score_vs_j2.overall_score, 0.95)

        # 4. Process new track through Journey Engine
        assigned_journey = self.journey_engine.process_track_evidence(ev_new, trk_new)

        # Verify that the selected candidate is the higher-scoring Candidate 2 (VEH_102)
        self.assertEqual(assigned_journey.global_vehicle_id, j2_veh_id)
        self.assertEqual(assigned_journey.journey_id, j2.journey_id)
        self.assertEqual(len(assigned_journey.segments), 2)
        self.assertEqual(assigned_journey.plate_number, "KW527")

        # Verify Candidate 1 remains untouched with 1 segment
        self.assertEqual(len(self.journey_engine.journeys[j1.journey_id].segments), 1)

    def test_ambiguous_evidence_yields_insufficient_evidence_and_no_merge(self):
        """
        Verify that when evidence against existing active journeys is insufficient
        (low score below probable_threshold 0.50), the system does NOT force a match
        and correctly creates a fresh global vehicle identity.
        """
        # Active Journey 1: CAM_A_EAST, plate=TN09AB1234, car
        ev_1, trk_1 = self._create_synthetic_track(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_001",
            timestamp=0.0,
            plate="TN09AB1234",
            vehicle_type="car",
            embedding=self.emb_a,
        )
        j1 = self.journey_engine.process_track_evidence(ev_1, trk_1)

        # New Observation: CAM_B_WEST at t=18.0s, plate=KA01CD5678, truck, orthogonal embedding
        ev_new, trk_new = self._create_synthetic_track(
            camera_id="CAM_B_WEST",
            track_id="TRK_B_001",
            timestamp=18.0,
            plate="KA01CD5678",
            vehicle_type="truck",
            embedding=self.emb_c,
        )

        breakdown = self.matching_engine.compute_match_score(
            ev_new, trk_new, ev_1, trk_1, self.emb_c, self.emb_a
        )
        self.assertEqual(breakdown.decision, MatchDecision.INSUFFICIENT_EVIDENCE)
        self.assertLess(breakdown.overall_score, 0.50)

        # Process through Journey Engine
        j_new = self.journey_engine.process_track_evidence(ev_new, trk_new)

        # Must not merge into J1
        self.assertNotEqual(j1.global_vehicle_id, j_new.global_vehicle_id)
        self.assertNotEqual(j1.journey_id, j_new.journey_id)
        self.assertEqual(len(j1.segments), 1)
        self.assertEqual(len(j_new.segments), 1)
        self.assertEqual(j_new.plate_number, "KA01CD5678")


if __name__ == "__main__":
    unittest.main()
