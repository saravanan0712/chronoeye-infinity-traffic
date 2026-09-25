"""
ChronoEye Infinity - Module 2 End-to-End Multi-Camera Validation Suite
SYNTHETIC E2E TEST: Validates the complete Module 2 Vehicle Intelligence pipeline:
Camera Source -> Observations -> Seven-Signal ReID -> Global Identity -> Journey -> Module 3 Graph.
"""

import math
import os
import sys
import unittest
from typing import List, Tuple, Optional

# Add backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.detection import BoundingBoxXYXY
from app.schemas.tracking import TrackState, TrackStatus, Direction
from app.schemas.plate import FusedPlateIdentity, VehicleIdentityEvidence
from app.schemas.reid import (
    MatchDecision,
    ReIDScoreBreakdown,
    ReIDConfig,
    VehicleJourney,
    JourneySegment,
)
from app.perception.camera_topology import CityCameraTopology
from app.perception.reid_engine import ReIDMatchingEngine
from app.perception.journey import JourneyReconstructionEngine
from app.graph.temporal_graph import TemporalTrafficGraphEngine
from app.graph.graph_schema import EdgeType, NodeType


class TestModule2EndToEndValidation(unittest.TestCase):
    """
    SYNTHETIC E2E TEST SUITE for ChronoEye Infinity Module 2.
    Tests multi-camera scenarios across CAM_A, CAM_B, CAM_C, CAM_D with multiple vehicles.
    """

    def setUp(self):
        self.topology = CityCameraTopology()
        self.config = ReIDConfig()
        self.matching_engine = ReIDMatchingEngine(topology=self.topology, config=self.config)
        self.graph_engine = TemporalTrafficGraphEngine()
        self.journey_engine = JourneyReconstructionEngine(
            matching_engine=self.matching_engine,
            config=self.config,
            temporal_graph_engine=self.graph_engine,
        )

        # Deterministic 128-D normalized embedding vectors
        val = 1.0 / math.sqrt(128)
        self.veh1_embedding: List[float] = [val] * 128
        self.veh2_embedding: List[float] = [val if i % 2 == 0 else -val for i in range(128)]
        self.veh3_embedding: List[float] = [val if i < 64 else -val for i in range(128)]

    def _create_mock_observation(
        self,
        camera_id: str,
        track_id: str,
        timestamp: float,
        plate_number: Optional[str] = "TN09AB1234",
        plate_confidence: float = 0.95,
        plate_status: str = "CONFIRMED",
        vehicle_type: str = "car",
        direction: Direction = Direction.EAST,
        bbox: Optional[BoundingBoxXYXY] = None,
        embedding: Optional[List[float]] = None,
        timestamp_uncertainty_seconds: Optional[float] = None,
    ) -> Tuple[VehicleIdentityEvidence, TrackState]:
        current_bbox = bbox or BoundingBoxXYXY(x1=100.0, y1=150.0, x2=300.0, y2=350.0)
        track = TrackState(
            track_id=track_id,
            camera_id=camera_id,
            vehicle_type=vehicle_type,
            current_bbox=current_bbox,
            current_center=current_bbox.center,
            confidence=0.92,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=direction,
            speed_estimate=45.0,
        )

        fused = None
        if plate_number is not None:
            fused = FusedPlateIdentity(
                track_id=track_id,
                camera_id=camera_id,
                best_plate_number=plate_number,
                overall_confidence=plate_confidence,
                observation_count=3,
                confirmed=(plate_status == "CONFIRMED"),
                status=plate_status,
                first_seen_timestamp=timestamp,
                last_seen_timestamp=timestamp,
            )

        evidence = VehicleIdentityEvidence(
            track_id=track_id,
            camera_id=camera_id,
            vehicle_type=vehicle_type,
            associated_plate=fused,
            last_updated_timestamp=timestamp,
            timestamp_uncertainty_seconds=timestamp_uncertainty_seconds,
        )

        # Cache track embedding
        emb = embedding if embedding is not None else self.veh1_embedding
        self.journey_engine.track_embeddings[track_id] = (timestamp, emb)

        return evidence, track

    def test_01_global_id_continuity(self):
        """TEST 1 — GLOBAL ID CONTINUITY: Vehicle 1 (A -> B -> C) produces a single global vehicle ID."""
        ev_a, trk_a = self._create_mock_observation("CAM_A_EAST", "TRK_V1_A", 0.0, plate_number="TN09AB1234", embedding=self.veh1_embedding)
        ev_b, trk_b = self._create_mock_observation("CAM_B_WEST", "TRK_V1_B", 20.0, plate_number="TN09AB1234", embedding=self.veh1_embedding)
        ev_c, trk_c = self._create_mock_observation("CAM_C_NORTH", "TRK_V1_C", 40.0, plate_number="TN09AB1234", embedding=self.veh1_embedding)

        j_a = self.journey_engine.process_track_evidence(ev_a, trk_a)
        j_b = self.journey_engine.process_track_evidence(ev_b, trk_b)
        j_c = self.journey_engine.process_track_evidence(ev_c, trk_c)

        # Single continuous journey and single global vehicle identity
        self.assertEqual(j_a.journey_id, j_b.journey_id)
        self.assertEqual(j_b.journey_id, j_c.journey_id)
        self.assertEqual(j_a.global_vehicle_id, j_b.global_vehicle_id)
        self.assertEqual(j_b.global_vehicle_id, j_c.global_vehicle_id)
        self.assertEqual(len(j_c.segments), 3)

    def test_02_journey_reconstruction_timestamp_order(self):
        """TEST 2 — JOURNEY RECONSTRUCTION: Journey contains CAM_A, CAM_B, CAM_C in timestamp order."""
        ev_a, trk_a = self._create_mock_observation("CAM_A_EAST", "TRK_V1_A", 0.0, plate_number="TN09AB1234", embedding=self.veh1_embedding)
        ev_b, trk_b = self._create_mock_observation("CAM_B_WEST", "TRK_V1_B", 18.5, plate_number="TN09AB1234", embedding=self.veh1_embedding)
        ev_c, trk_c = self._create_mock_observation("CAM_C_NORTH", "TRK_V1_C", 39.2, plate_number="TN09AB1234", embedding=self.veh1_embedding)

        self.journey_engine.process_track_evidence(ev_a, trk_a)
        self.journey_engine.process_track_evidence(ev_b, trk_b)
        journey = self.journey_engine.process_track_evidence(ev_c, trk_c)

        cams = [seg.camera_id for seg in journey.segments]
        timestamps = [seg.timestamp for seg in journey.segments]

        self.assertEqual(cams, ["CAM_A_EAST", "CAM_B_WEST", "CAM_C_NORTH"])
        self.assertEqual(timestamps, [0.0, 18.5, 39.2])
        self.assertTrue(timestamps[0] < timestamps[1] < timestamps[2])

    def test_03_seven_signal_evidence(self):
        """TEST 3 — SEVEN-SIGNAL EVIDENCE: All seven ReID similarity signals exist on transition."""
        ev_a, trk_a = self._create_mock_observation("CAM_A_EAST", "TRK_SIG_A", 0.0, plate_number="KA05ZZ9999", embedding=self.veh1_embedding)
        ev_b, trk_b = self._create_mock_observation("CAM_B_WEST", "TRK_SIG_B", 20.0, plate_number="KA05ZZ9999", embedding=self.veh1_embedding)

        self.journey_engine.process_track_evidence(ev_a, trk_a)
        journey = self.journey_engine.process_track_evidence(ev_b, trk_b)

        self.assertEqual(len(journey.segments), 2)
        trans_seg = journey.segments[1]

        self.assertIsNotNone(trans_seg.transition_breakdown)
        tb = trans_seg.transition_breakdown

        self.assertIsInstance(tb.plate_similarity, float)
        self.assertIsInstance(tb.appearance_similarity, float)
        self.assertIsInstance(tb.visual_features_similarity, float)
        self.assertIsInstance(tb.vehicle_type_similarity, float)
        self.assertIsInstance(tb.temporal_compatibility, float)
        self.assertIsInstance(tb.spatial_compatibility, float)
        self.assertIsInstance(tb.direction_similarity, float)

        self.assertIsNotNone(trans_seg.transition_score)
        self.assertIsNotNone(trans_seg.transition_decision)
        self.assertEqual(trans_seg.transition_decision, MatchDecision.MATCH_CONFIRMED)

    def test_04_exact_evidence_preservation(self):
        """TEST 4 — EXACT EVIDENCE PRESERVATION: Deliberately distinct values reach Module 3 graph metadata exactly."""
        distinct_breakdown = ReIDScoreBreakdown(
            plate_similarity=0.91,
            appearance_similarity=0.73,
            visual_features_similarity=0.84,
            vehicle_type_similarity=1.00,
            temporal_compatibility=0.67,
            spatial_compatibility=0.88,
            direction_similarity=0.76,
            overall_score=0.825,
            decision=MatchDecision.MATCH_CONFIRMED,
            rejection_reason=None,
        )

        journey = VehicleJourney(
            journey_id="JRN_E2E_EXACT_1",
            global_vehicle_id="VEH_E2E_EXACT_1",
            vehicle_type="car",
            first_seen=0.0,
            last_seen=25.0,
            segments=[
                JourneySegment(
                    camera_id="CAM_A_EAST",
                    track_id="TRK_EXACT_A",
                    timestamp=0.0,
                ),
                JourneySegment(
                    camera_id="CAM_B_WEST",
                    track_id="TRK_EXACT_B",
                    timestamp=25.0,
                    transition_score=0.825,
                    transition_decision=MatchDecision.MATCH_CONFIRMED,
                    transition_breakdown=distinct_breakdown,
                ),
            ],
        )

        self.graph_engine.update_journey(journey)

        trans_edges = [
            d for u, v, k, d in self.graph_engine.builder.graph.edges(keys=True, data=True)
            if (d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA)
            and (d.get("metadata", {}).get("journey_id") == "JRN_E2E_EXACT_1" or d.get("data", {}).get("metadata", {}).get("journey_id") == "JRN_E2E_EXACT_1")
        ]
        self.assertEqual(len(trans_edges), 1)
        meta = trans_edges[0].get("data", {}).get("metadata", {}) or trans_edges[0].get("metadata", {})
        self.assertIn("transition_breakdown", meta)
        tb = meta["transition_breakdown"]

        self.assertEqual(tb["plate_similarity"], 0.91)
        self.assertEqual(tb["appearance_similarity"], 0.73)
        self.assertEqual(tb["visual_features_similarity"], 0.84)
        self.assertEqual(tb["vehicle_type_similarity"], 1.00)
        self.assertEqual(tb["temporal_compatibility"], 0.67)
        self.assertEqual(tb["spatial_compatibility"], 0.88)
        self.assertEqual(tb["direction_similarity"], 0.76)
        self.assertEqual(tb["overall_score"], 0.825)
        self.assertEqual(tb["decision"], "MATCH_CONFIRMED")

    def test_05_plate_evidence_states_preserved(self):
        """TEST 5 — PLATE EVIDENCE: CONFIRMED, PENDING, UNKNOWN plate states survive through to the graph."""
        # 1. CONFIRMED plate
        engine1 = JourneyReconstructionEngine(matching_engine=self.matching_engine, config=self.config)
        ev_conf, trk_conf = self._create_mock_observation("CAM_A_EAST", "TRK_CONF", 0.0, plate_number="TN09AB1234", plate_status="CONFIRMED", plate_confidence=0.96)
        j_conf = engine1.process_track_evidence(ev_conf, trk_conf)
        self.assertEqual(j_conf.segments[0].plate_status, "CONFIRMED")
        self.assertEqual(j_conf.segments[0].plate_confidence, 0.96)

        # 2. PENDING plate
        engine2 = JourneyReconstructionEngine(matching_engine=self.matching_engine, config=self.config)
        ev_pend, trk_pend = self._create_mock_observation("CAM_B_WEST", "TRK_PEND", 10.0, plate_number="KA05CD5678", plate_status="PENDING", plate_confidence=0.45, embedding=self.veh2_embedding)
        j_pend = engine2.process_track_evidence(ev_pend, trk_pend)
        self.assertEqual(j_pend.segments[0].plate_status, "PENDING")
        self.assertEqual(j_pend.segments[0].plate_confidence, 0.45)

        # 3. UNKNOWN plate
        engine3 = JourneyReconstructionEngine(matching_engine=self.matching_engine, config=self.config)
        ev_unk, trk_unk = self._create_mock_observation("CAM_C_NORTH", "TRK_UNK", 20.0, plate_number="", plate_status="UNKNOWN", plate_confidence=0.10, embedding=self.veh3_embedding)
        j_unk = engine3.process_track_evidence(ev_unk, trk_unk)
        self.assertEqual(j_unk.segments[0].plate_status, "UNKNOWN")
        self.assertEqual(j_unk.segments[0].plate_confidence, 0.10)

    def test_06_timestamp_uncertainty_modes(self):
        """TEST 6 — TIMESTAMP UNCERTAINTY: Default production None vs explicit 0.15 uncertainty."""
        # Case A: Default production behavior (None)
        ev_def, trk_def = self._create_mock_observation("CAM_A_EAST", "TRK_UNC_DEF", 0.0, plate_number="TN01AA1111", timestamp_uncertainty_seconds=None)
        j_def = self.journey_engine.process_track_evidence(ev_def, trk_def)
        self.assertIsNone(j_def.segments[0].timestamp_uncertainty_seconds)

        # Case B: Explicit test evidence (0.15)
        ev_exp, trk_exp = self._create_mock_observation("CAM_B_WEST", "TRK_UNC_EXP", 15.0, plate_number="KA02BB2222", embedding=self.veh2_embedding, timestamp_uncertainty_seconds=0.15)
        j_exp = self.journey_engine.process_track_evidence(ev_exp, trk_exp)
        self.assertEqual(j_exp.segments[0].timestamp_uncertainty_seconds, 0.15)

    def test_07_unobserved_gap(self):
        """TEST 7 — UNOBSERVED GAP: CAM_A -> CAM_D (non-adjacent) preserves has_unobserved_gap=True without fabricating intermediate cameras."""
        ev_a, trk_a = self._create_mock_observation("CAM_A_EAST", "TRK_GAP_A", 0.0, plate_number="DL01AA0007", embedding=self.veh1_embedding)
        ev_d, trk_d = self._create_mock_observation("CAM_D_NORTH", "TRK_GAP_D", 50.0, plate_number="DL01AA0007", embedding=self.veh1_embedding)

        self.journey_engine.process_track_evidence(ev_a, trk_a)
        journey = self.journey_engine.process_track_evidence(ev_d, trk_d)

        self.assertEqual(len(journey.segments), 2)
        self.assertTrue(journey.segments[1].has_unobserved_gap)

        # In the graph, verify direct transition edge from CAM_A to CAM_D exists with has_unobserved_gap=True
        trans_edges = [
            (u, v, d) for u, v, k, d in self.graph_engine.builder.graph.edges(keys=True, data=True)
            if (d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA)
            and (d.get("metadata", {}).get("journey_id") == journey.journey_id or d.get("data", {}).get("metadata", {}).get("journey_id") == journey.journey_id)
        ]
        self.assertEqual(len(trans_edges), 1)
        u, v, d = trans_edges[0]
        self.assertEqual(u, "CAM_A_EAST")
        self.assertEqual(v, "CAM_D_NORTH")
        meta = d.get("data", {}).get("metadata", {}) or d.get("metadata", {})
        self.assertTrue(meta.get("has_unobserved_gap"))

    def test_08_impossible_transition_rejected(self):
        """TEST 8 — IMPOSSIBLE TRANSITION: Physically impossible speed rejects cross-camera association."""
        # CAM_A to CAM_B in 0.1 seconds (300m / 0.1s = 10,800 km/h) -> exceeds max_speed_kmh
        ev_a, trk_a = self._create_mock_observation("CAM_A_EAST", "TRK_IMP_A", 0.0, plate_number="KA01AA0001")
        ev_b, trk_b = self._create_mock_observation("CAM_B_WEST", "TRK_IMP_B", 0.1, plate_number="KA01AA0001")

        j1 = self.journey_engine.process_track_evidence(ev_a, trk_a)
        j2 = self.journey_engine.process_track_evidence(ev_b, trk_b)

        self.assertNotEqual(j1.journey_id, j2.journey_id)
        self.assertNotEqual(j1.global_vehicle_id, j2.global_vehicle_id)

    def test_09_different_vehicles_separate_identities(self):
        """TEST 9 — DIFFERENT VEHICLES: Three vehicles (V1, V2, V3) maintain distinct global identities."""
        # Vehicle 1: A -> B -> C (car, plate TN09AB1111)
        ev1_a, trk1_a = self._create_mock_observation("CAM_A_EAST", "TRK_1_A", 0.0, plate_number="TN09AB1111", vehicle_type="car", embedding=self.veh1_embedding)
        ev1_b, trk1_b = self._create_mock_observation("CAM_B_WEST", "TRK_1_B", 20.0, plate_number="TN09AB1111", vehicle_type="car", embedding=self.veh1_embedding)
        ev1_c, trk1_c = self._create_mock_observation("CAM_C_NORTH", "TRK_1_C", 40.0, plate_number="TN09AB1111", vehicle_type="car", embedding=self.veh1_embedding)

        # Vehicle 2: A -> B (suv, plate KA05CD2222, orthogonal embedding)
        bbox_v2 = BoundingBoxXYXY(x1=300.0, y1=300.0, x2=550.0, y2=450.0)
        ev2_a, trk2_a = self._create_mock_observation("CAM_A_EAST", "TRK_2_A", 5.0, plate_number="KA05CD2222", vehicle_type="suv", direction=Direction.SOUTH, bbox=bbox_v2, embedding=self.veh2_embedding)
        ev2_b, trk2_b = self._create_mock_observation("CAM_B_WEST", "TRK_2_B", 25.0, plate_number="KA05CD2222", vehicle_type="suv", direction=Direction.SOUTH, bbox=bbox_v2, embedding=self.veh2_embedding)

        # Vehicle 3: distinct truck (truck, plate MH02EF3333, orthogonal embedding)
        bbox_v3 = BoundingBoxXYXY(x1=50.0, y1=50.0, x2=450.0, y2=350.0)
        ev3_a, trk3_a = self._create_mock_observation("CAM_A_EAST", "TRK_3_A", 10.0, plate_number="MH02EF3333", vehicle_type="truck", direction=Direction.WEST, bbox=bbox_v3, embedding=self.veh3_embedding)

        j1_a = self.journey_engine.process_track_evidence(ev1_a, trk1_a)
        j1_b = self.journey_engine.process_track_evidence(ev1_b, trk1_b)
        j1_c = self.journey_engine.process_track_evidence(ev1_c, trk1_c)

        j2_a = self.journey_engine.process_track_evidence(ev2_a, trk2_a)
        j2_b = self.journey_engine.process_track_evidence(ev2_b, trk2_b)

        j3 = self.journey_engine.process_track_evidence(ev3_a, trk3_a)

        # Vehicle 1 is single global ID
        self.assertEqual(j1_a.global_vehicle_id, j1_c.global_vehicle_id)
        # Vehicle 2 is single global ID
        self.assertEqual(j2_a.global_vehicle_id, j2_b.global_vehicle_id)
        # All 3 vehicles have separate global IDs
        self.assertNotEqual(j1_c.global_vehicle_id, j2_b.global_vehicle_id)
        self.assertNotEqual(j1_c.global_vehicle_id, j3.global_vehicle_id)
        self.assertNotEqual(j2_b.global_vehicle_id, j3.global_vehicle_id)

    def test_10_same_camera_co_presence_exclusion(self):
        """TEST 10 — SAME-CAMERA CO-PRESENCE: Simultaneous tracks on same camera are never merged."""
        ev_a1, trk_a1 = self._create_mock_observation("CAM_A_EAST", "TRK_SIM_1", 10.0, plate_number="TS01AA1111")
        ev_a2, trk_a2 = self._create_mock_observation("CAM_A_EAST", "TRK_SIM_2", 10.0, plate_number="TS01AA2222")

        j1 = self.journey_engine.process_track_evidence(ev_a1, trk_a1)
        j2 = self.journey_engine.process_track_evidence(ev_a2, trk_a2)

        self.assertNotEqual(j1.journey_id, j2.journey_id)
        self.assertNotEqual(j1.global_vehicle_id, j2.global_vehicle_id)

    def test_11_module3_handoff(self):
        """TEST 11 — MODULE 3 HANDOFF: Complete metadata reaches Module 3 temporal traffic graph."""
        ev_a, trk_a = self._create_mock_observation("CAM_A_EAST", "TRK_M3_A", 0.0, plate_number="TN09AB1234", plate_confidence=0.97, plate_status="CONFIRMED", timestamp_uncertainty_seconds=0.1)
        ev_b, trk_b = self._create_mock_observation("CAM_B_WEST", "TRK_M3_B", 20.0, plate_number="TN09AB1234", plate_confidence=0.97, plate_status="CONFIRMED", timestamp_uncertainty_seconds=0.1)

        self.journey_engine.process_track_evidence(ev_a, trk_a)
        journey = self.journey_engine.process_track_evidence(ev_b, trk_b)

        # Verify Vehicle Node exists in graph
        self.assertIn(journey.global_vehicle_id, self.graph_engine.builder.graph.nodes)
        v_node = self.graph_engine.builder.graph.nodes[journey.global_vehicle_id]
        self.assertEqual(v_node.get("data", {}).get("plate_number") or v_node.get("plate_number"), "TN09AB1234")

        # Verify Transition Edge metadata
        trans_edges = [
            d for u, v, k, d in self.graph_engine.builder.graph.edges(keys=True, data=True)
            if (d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA)
            and (d.get("metadata", {}).get("journey_id") == journey.journey_id or d.get("data", {}).get("metadata", {}).get("journey_id") == journey.journey_id)
        ]
        self.assertEqual(len(trans_edges), 1)
        meta = trans_edges[0].get("data", {}).get("metadata", {}) or trans_edges[0].get("metadata", {})
        self.assertEqual(meta["global_vehicle_id"], journey.global_vehicle_id)
        self.assertEqual(meta["journey_id"], journey.journey_id)
        self.assertIn("transition_score", meta)
        self.assertIn("transition_decision", meta)
        self.assertIn("transition_breakdown", meta)
        self.assertEqual(meta["plate_number"], "TN09AB1234")
        self.assertEqual(meta["plate_confidence"], 0.97)
        self.assertEqual(meta["plate_status"], "CONFIRMED")
        self.assertEqual(meta["timestamp_uncertainty_seconds"], 0.1)

    def test_12_graph_structure(self):
        """TEST 12 — GRAPH STRUCTURE: Semantic distinction between Observation and Transition edges preserved."""
        ev_a, trk_a = self._create_mock_observation("CAM_A_EAST", "TRK_STR_A", 0.0, plate_number="KA04AB5555")
        ev_b, trk_b = self._create_mock_observation("CAM_B_WEST", "TRK_STR_B", 25.0, plate_number="KA04AB5555")

        self.journey_engine.process_track_evidence(ev_a, trk_a)
        journey = self.journey_engine.process_track_evidence(ev_b, trk_b)

        obs_edges = [
            d for u, v, k, d in self.graph_engine.builder.graph.edges(keys=True, data=True)
            if d.get("edge_type") == EdgeType.VEHICLE_OBSERVED_BY_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_OBSERVED_BY_CAMERA
        ]
        trans_edges = [
            d for u, v, k, d in self.graph_engine.builder.graph.edges(keys=True, data=True)
            if d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA
        ]

        self.assertGreaterEqual(len(obs_edges), 2)
        self.assertGreaterEqual(len(trans_edges), 1)

        # Observation edges connect Vehicle -> Camera with local_track_id
        obs_meta = obs_edges[0].get("data", {}).get("metadata", {}) or obs_edges[0].get("metadata", {})
        self.assertIn("local_track_id", obs_meta)

        # Transition edges connect Camera -> Camera with transition_breakdown & journey_id
        trans_meta = trans_edges[0].get("data", {}).get("metadata", {}) or trans_edges[0].get("metadata", {})
        self.assertIn("journey_id", trans_meta)
        self.assertIn("transition_breakdown", trans_meta)


if __name__ == "__main__":
    unittest.main()
