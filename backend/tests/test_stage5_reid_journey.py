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

    def test_21_seven_independent_signals_exist(self):
        """TEST 1: Verify all 7 independent signal components exist in ReIDConfig and ReIDScoreBreakdown."""
        cfg = self.matching_engine.config
        self.assertGreater(cfg.w_plate, 0.0)
        self.assertGreater(cfg.w_appearance, 0.0)
        self.assertGreater(cfg.w_visual_features, 0.0)
        self.assertGreater(cfg.w_type, 0.0)
        self.assertGreater(cfg.w_direction, 0.0)
        self.assertGreater(cfg.w_time, 0.0)
        self.assertGreater(cfg.w_space, 0.0)
        total_w = (
            cfg.w_plate
            + cfg.w_appearance
            + cfg.w_visual_features
            + cfg.w_type
            + cfg.w_direction
            + cfg.w_time
            + cfg.w_space
        )
        self.assertAlmostEqual(total_w, 1.0, places=4)

    def test_22_visual_feature_similarity_independent(self):
        """TEST 2: Verify visual-feature similarity is independently calculated and bounded [0, 1]."""
        bbox1 = BoundingBoxXYXY(x1=100.0, y1=100.0, x2=250.0, y2=200.0)  # w=150, h=100, ar=1.5
        bbox2 = BoundingBoxXYXY(x1=0.0, y1=0.0, x2=300.0, y2=200.0)      # w=300, h=200, ar=1.5
        bbox3 = BoundingBoxXYXY(x1=0.0, y1=0.0, x2=100.0, y2=300.0)      # w=100, h=300, ar=0.333

        vf1 = self.matching_engine.extract_visual_features(bbox1)
        vf2 = self.matching_engine.extract_visual_features(bbox2)
        vf3 = self.matching_engine.extract_visual_features(bbox3)

        sim_same = self.matching_engine.calculate_visual_feature_similarity(vf1, vf2)
        sim_diff = self.matching_engine.calculate_visual_feature_similarity(vf1, vf3)

        self.assertAlmostEqual(sim_same, 1.0, places=2)
        self.assertLess(sim_diff, 0.60)
        self.assertGreaterEqual(sim_diff, 0.0)

    def test_23_appearance_and_visual_features_differ(self):
        """TEST 3: Verify appearance (color) and visual features (geometry) can produce distinct scores."""
        # Same color embedding, different geometry
        emb_a = [1.0] + [0.0] * 127
        emb_b = [1.0] + [0.0] * 127

        vf_a = {"aspect_ratio": 1.5, "elongation": 0.2, "vehicle_type": "car"}
        vf_b = {"aspect_ratio": 0.4, "elongation": 0.6, "vehicle_type": "car"}

        s_app = AppearanceEmbeddingExtractor.cosine_similarity(emb_a, emb_b)
        s_vf = self.matching_engine.calculate_visual_feature_similarity(vf_a, vf_b)

        self.assertEqual(s_app, 1.0)
        self.assertLess(s_vf, 0.6)
        self.assertNotEqual(s_app, s_vf)

    def test_24_breakdown_retains_visual_features(self):
        """TEST 4: Verify Re-ID score breakdown retains visual_features_similarity."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0)
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 18.0)

        breakdown = self.matching_engine.compute_match_score(ev_a, trk_a, ev_b, trk_b)
        self.assertIn("visual_features_similarity", breakdown.model_dump())
        self.assertGreater(breakdown.visual_features_similarity, 0.0)

    def test_25_journey_segment_stores_transition_evidence(self):
        """TEST 5: Verify JourneySegment retains transition breakdown evidence."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="KA05XY9999")
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 20.0, plate="KA05XY9999")

        self.journey_engine.process_track_evidence(ev_a, trk_a)
        journey = self.journey_engine.process_track_evidence(ev_b, trk_b)

        self.assertEqual(len(journey.segments), 2)
        seg2 = journey.segments[1]
        self.assertIsNotNone(seg2.transition_breakdown)
        self.assertIsNotNone(seg2.transition_score)
        self.assertGreater(seg2.transition_score, 0.70)
        self.assertEqual(seg2.transition_breakdown.plate_similarity, 1.0)

    def test_26_journey_segment_stores_transition_decision(self):
        """TEST 6: Verify JourneySegment stores explicit transition decision."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="MH12AB1111")
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 20.0, plate="MH12AB1111")

        self.journey_engine.process_track_evidence(ev_a, trk_a)
        journey = self.journey_engine.process_track_evidence(ev_b, trk_b)

        seg2 = journey.segments[1]
        self.assertEqual(seg2.transition_decision, MatchDecision.MATCH_CONFIRMED)

    def test_27_confirmed_transition_marked(self):
        """TEST 7: Verify high-confidence match is marked MATCH_CONFIRMED."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="DL01AA1234")
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 18.0, plate="DL01AA1234")

        self.journey_engine.process_track_evidence(ev_a, trk_a)
        journey = self.journey_engine.process_track_evidence(ev_b, trk_b)

        self.assertEqual(journey.segments[1].transition_decision, MatchDecision.MATCH_CONFIRMED)

    def test_28_probable_transition_marked(self):
        """TEST 8: Verify probable match (missing plate with partial appearance match) is marked MATCH_PROBABLE."""
        # Plate is UNKNOWN on both cameras, and appearance embedding has partial correlation (cosine similarity 0.0)
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="UNKNOWN")
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 20.0, plate="UNKNOWN")

        # Set orthogonal appearance embeddings
        self.journey_engine.track_embeddings["TRK_101"] = (0.0, [1.0] + [0.0] * 127)
        self.journey_engine.track_embeddings["TRK_207"] = (20.0, [0.0, 1.0] + [0.0] * 126)

        self.journey_engine.process_track_evidence(ev_a, trk_a)
        journey = self.journey_engine.process_track_evidence(ev_b, trk_b)

        self.assertEqual(len(journey.segments), 2)
        # Score is ~0.675 which is >= probable_threshold (0.50) but < confirm_threshold (0.75)
        self.assertEqual(journey.segments[1].transition_decision, MatchDecision.MATCH_PROBABLE)

    def test_29_unobserved_gap_explicitly_represented(self):
        """TEST 9: Verify unobserved gap is True when jumping across non-adjacent cameras."""
        # CAM_A_EAST is directly connected to CAM_B_WEST and CAM_C_NORTH, but NOT CAM_D_NORTH
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="TS09ZZ8888")
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 20.0, plate="TS09ZZ8888")

        self.journey_engine.process_track_evidence(ev_a, trk_a)
        j1 = self.journey_engine.process_track_evidence(ev_b, trk_b)
        # Direct hop CAM_A -> CAM_B: has_unobserved_gap must be False
        self.assertFalse(j1.segments[1].has_unobserved_gap)

        # Now test non-adjacent hop CAM_A_EAST -> CAM_D_NORTH (distance 500m, time 40s)
        engine2 = JourneyReconstructionEngine(matching_engine=self.matching_engine, config=self.config)
        ev_a2, trk_a2 = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="TS09ZZ7777")
        ev_d2, trk_d2 = self._create_mock_track("TRK_404", "CAM_D_NORTH", 40.0, plate="TS09ZZ7777")

        engine2.process_track_evidence(ev_a2, trk_a2)
        j2 = engine2.process_track_evidence(ev_d2, trk_d2)

        self.assertEqual(len(j2.segments), 2)
        # Non-adjacent hop CAM_A -> CAM_D: has_unobserved_gap must be True
        self.assertTrue(j2.segments[1].has_unobserved_gap)

    def test_30_no_intermediate_camera_fabricated(self):
        """TEST 10: Verify no intermediate cameras are fabricated when an unobserved gap occurs."""
        engine = JourneyReconstructionEngine(matching_engine=self.matching_engine, config=self.config)
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="AP01AA5555")
        ev_d, trk_d = self._create_mock_track("TRK_404", "CAM_D_NORTH", 40.0, plate="AP01AA5555")

        engine.process_track_evidence(ev_a, trk_a)
        j = engine.process_track_evidence(ev_d, trk_d)

        # Must strictly contain observed cameras only
        self.assertEqual(j.cameras, ["CAM_A_EAST", "CAM_D_NORTH"])
        self.assertEqual(len(j.segments), 2)
        self.assertNotIn("CAM_B_WEST", j.cameras)
        self.assertNotIn("CAM_C_NORTH", j.cameras)

    def test_31_impossible_travel_rejects_association(self):
        """TEST 11: Verify impossible physical travel rejects association."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="KA01AA0001")
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 0.5, plate="KA01AA0001")

        j1 = self.journey_engine.process_track_evidence(ev_a, trk_a)
        j2 = self.journey_engine.process_track_evidence(ev_b, trk_b)

        self.assertNotEqual(j1.journey_id, j2.journey_id)
        self.assertNotEqual(j1.global_vehicle_id, j2.global_vehicle_id)

    def test_32_same_camera_co_presence_exclusion(self):
        """TEST 12: Verify co-present simultaneous tracks on same camera create distinct journeys."""
        ev_a1, trk_a1 = self._create_mock_track("TRK_101", "CAM_A_EAST", 10.0, plate="TN01AA1111")
        ev_a2, trk_a2 = self._create_mock_track("TRK_102", "CAM_A_EAST", 10.0, plate="TN01AA2222")

        j1 = self.journey_engine.process_track_evidence(ev_a1, trk_a1)
        j2 = self.journey_engine.process_track_evidence(ev_a2, trk_a2)

        self.assertNotEqual(j1.journey_id, j2.journey_id)
        self.assertNotEqual(j1.global_vehicle_id, j2.global_vehicle_id)

    def test_33_module3_temporal_graph_handoff(self):
        """TEST 13: Verify Module 3 TemporalTrafficGraphEngine consumes journey with transition metadata."""
        from app.graph.temporal_graph import TemporalTrafficGraphEngine
        graph_engine = TemporalTrafficGraphEngine()

        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 0.0, plate="KA05ZZ9999")
        ev_b, trk_b = self._create_mock_track("TRK_207", "CAM_B_WEST", 20.0, plate="KA05ZZ9999")

        self.journey_engine.temporal_graph_engine = graph_engine
        self.journey_engine.process_track_evidence(ev_a, trk_a)
        journey = self.journey_engine.process_track_evidence(ev_b, trk_b)

        # Graph should contain global vehicle node and transition edges
        self.assertIn(journey.global_vehicle_id, graph_engine.builder.graph.nodes)
        self.assertGreaterEqual(len(graph_engine.builder.graph.edges), 3)

    def test_34_step3a_confirmed_plate_confidence_and_status(self):
        """Step 3A Test 1: CONFIRMED plate status and overall_confidence are preserved on JourneySegment."""
        track = TrackState(
            track_id="TRK_CONF_1",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150),
            confidence=0.9,
            first_seen_timestamp=10.0,
            last_seen_timestamp=10.0,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=Direction.EAST,
        )
        fused_plate = FusedPlateIdentity(
            track_id="TRK_CONF_1",
            camera_id="CAM_A_EAST",
            best_plate_number="TN09AB1234",
            overall_confidence=0.91,
            observation_count=5,
            confirmed=True,
            status="CONFIRMED",
            first_seen_timestamp=10.0,
            last_seen_timestamp=10.0,
        )
        evidence = VehicleIdentityEvidence(
            track_id="TRK_CONF_1",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            associated_plate=fused_plate,
            last_updated_timestamp=10.0,
        )

        journey = self.journey_engine.process_track_evidence(evidence, track)
        self.assertEqual(len(journey.segments), 1)
        seg = journey.segments[0]
        self.assertEqual(seg.plate_number, "TN09AB1234")
        self.assertEqual(seg.plate_confidence, 0.91)
        self.assertEqual(seg.plate_status, "CONFIRMED")

    def test_35_step3a_pending_plate_confidence_and_status(self):
        """Step 3A Test 2: PENDING plate status and overall_confidence are preserved on JourneySegment."""
        track = TrackState(
            track_id="TRK_PEND_1",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150),
            confidence=0.9,
            first_seen_timestamp=15.0,
            last_seen_timestamp=15.0,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=Direction.EAST,
        )
        fused_plate = FusedPlateIdentity(
            track_id="TRK_PEND_1",
            camera_id="CAM_A_EAST",
            best_plate_number="TN09AB1234",
            overall_confidence=0.42,
            observation_count=2,
            confirmed=False,
            status="PENDING",
            first_seen_timestamp=15.0,
            last_seen_timestamp=15.0,
        )
        evidence = VehicleIdentityEvidence(
            track_id="TRK_PEND_1",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            associated_plate=fused_plate,
            last_updated_timestamp=15.0,
        )

        journey = self.journey_engine.process_track_evidence(evidence, track)
        self.assertEqual(len(journey.segments), 1)
        seg = journey.segments[0]
        self.assertEqual(seg.plate_confidence, 0.42)
        self.assertEqual(seg.plate_status, "PENDING")

    def test_36_step3a_unknown_plate_status(self):
        """Step 3A Test 3: UNKNOWN plate status is preserved as 'UNKNOWN' (not renamed)."""
        track = TrackState(
            track_id="TRK_UNK_1",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150),
            confidence=0.9,
            first_seen_timestamp=20.0,
            last_seen_timestamp=20.0,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=Direction.EAST,
        )
        fused_plate = FusedPlateIdentity(
            track_id="TRK_UNK_1",
            camera_id="CAM_A_EAST",
            best_plate_number="",
            overall_confidence=0.15,
            observation_count=1,
            confirmed=False,
            status="UNKNOWN",
            first_seen_timestamp=20.0,
            last_seen_timestamp=20.0,
        )
        evidence = VehicleIdentityEvidence(
            track_id="TRK_UNK_1",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            associated_plate=fused_plate,
            last_updated_timestamp=20.0,
        )

        journey = self.journey_engine.process_track_evidence(evidence, track)
        self.assertEqual(len(journey.segments), 1)
        seg = journey.segments[0]
        self.assertEqual(seg.plate_status, "UNKNOWN")
        self.assertEqual(seg.plate_confidence, 0.15)

    def test_37_step3a_missing_plate_evidence(self):
        """Step 3A Test 4: Missing plate evidence results in plate_confidence=None and plate_status=None."""
        track = TrackState(
            track_id="TRK_NOPLT_1",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150),
            confidence=0.9,
            first_seen_timestamp=25.0,
            last_seen_timestamp=25.0,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=Direction.EAST,
        )
        evidence = VehicleIdentityEvidence(
            track_id="TRK_NOPLT_1",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            associated_plate=None,
            last_updated_timestamp=25.0,
        )

        journey = self.journey_engine.process_track_evidence(evidence, track)
        self.assertEqual(len(journey.segments), 1)
        seg = journey.segments[0]
        self.assertIsNone(seg.plate_number)
        self.assertIsNone(seg.plate_confidence)
        self.assertIsNone(seg.plate_status)

    def test_38_step3a_module3_temporal_graph_plate_metadata(self):
        """Step 3A Test 5: Verify Module 3 temporal graph metadata contains plate_confidence and plate_status while preserving existing fields."""
        from app.graph.temporal_graph import TemporalTrafficGraphEngine
        from app.graph.graph_schema import EdgeType
        graph_engine = TemporalTrafficGraphEngine()

        track_a = TrackState(
            track_id="TRK_M3_A",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150),
            confidence=0.9,
            first_seen_timestamp=0.0,
            last_seen_timestamp=0.0,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=Direction.EAST,
        )
        fused_a = FusedPlateIdentity(
            track_id="TRK_M3_A",
            camera_id="CAM_A_EAST",
            best_plate_number="KA05ZZ8888",
            overall_confidence=0.95,
            observation_count=4,
            confirmed=True,
            status="CONFIRMED",
            first_seen_timestamp=0.0,
            last_seen_timestamp=0.0,
        )
        ev_a = VehicleIdentityEvidence(
            track_id="TRK_M3_A",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            associated_plate=fused_a,
            last_updated_timestamp=0.0,
        )

        track_b = TrackState(
            track_id="TRK_M3_B",
            camera_id="CAM_B_WEST",
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150),
            confidence=0.9,
            first_seen_timestamp=20.0,
            last_seen_timestamp=20.0,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=Direction.WEST,
        )
        fused_b = FusedPlateIdentity(
            track_id="TRK_M3_B",
            camera_id="CAM_B_WEST",
            best_plate_number="KA05ZZ8888",
            overall_confidence=0.93,
            observation_count=3,
            confirmed=True,
            status="CONFIRMED",
            first_seen_timestamp=20.0,
            last_seen_timestamp=20.0,
        )
        ev_b = VehicleIdentityEvidence(
            track_id="TRK_M3_B",
            camera_id="CAM_B_WEST",
            vehicle_type="car",
            associated_plate=fused_b,
            last_updated_timestamp=20.0,
        )

        self.journey_engine.temporal_graph_engine = graph_engine
        self.journey_engine.process_track_evidence(ev_a, track_a)
        journey = self.journey_engine.process_track_evidence(ev_b, track_b)

        # 1. Check Observation Edge metadata
        obs_edges = [
            d for u, v, k, d in graph_engine.builder.graph.edges(keys=True, data=True)
            if d.get("edge_type") == EdgeType.VEHICLE_OBSERVED_BY_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_OBSERVED_BY_CAMERA
        ]
        self.assertGreaterEqual(len(obs_edges), 2)
        for edge_data in obs_edges:
            meta = edge_data.get("data", {}).get("metadata", {}) or edge_data.get("metadata", {})
            self.assertIn("local_track_id", meta)
            self.assertIn("plate_confidence", meta)
            self.assertIn("plate_status", meta)
            self.assertEqual(meta["plate_status"], "CONFIRMED")
            self.assertGreater(meta["plate_confidence"], 0.9)

        # 2. Check Transition Edge metadata
        trans_edges = [
            d for u, v, k, d in graph_engine.builder.graph.edges(keys=True, data=True)
            if d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA
        ]
        self.assertGreaterEqual(len(trans_edges), 1)
        trans_meta = trans_edges[0].get("data", {}).get("metadata", {}) or trans_edges[0].get("metadata", {})
        self.assertEqual(trans_meta.get("global_vehicle_id"), journey.global_vehicle_id)
        self.assertEqual(trans_meta.get("journey_id"), journey.journey_id)
        self.assertIn("transition_score", trans_meta)
        self.assertIn("transition_decision", trans_meta)
        self.assertIn("plate_confidence", trans_meta)
        self.assertIn("plate_status", trans_meta)
        self.assertEqual(trans_meta["plate_confidence"], 0.93)
        self.assertEqual(trans_meta["plate_status"], "CONFIRMED")

    def test_39_step3b_journey_segment_supports_timestamp_uncertainty(self):
        """TEST 39: JourneySegment supports timestamp_uncertainty_seconds field."""
        seg = JourneySegment(
            camera_id="CAM_A_EAST",
            track_id="TRK_UNC_1",
            timestamp=50.0,
            timestamp_uncertainty_seconds=0.5,
        )
        self.assertEqual(seg.timestamp_uncertainty_seconds, 0.5)

    def test_40_step3b_default_uncertainty_is_none(self):
        """TEST 40: Default value of timestamp_uncertainty_seconds is None when not supplied."""
        seg = JourneySegment(
            camera_id="CAM_A_EAST",
            track_id="TRK_UNC_2",
            timestamp=50.0,
        )
        self.assertIsNone(seg.timestamp_uncertainty_seconds)

    def test_41_step3b_supplied_uncertainty_preserved_exactly(self):
        """TEST 41: A supplied defensible uncertainty value is preserved exactly with no transformation."""
        input_val = 0.5
        seg = JourneySegment(
            camera_id="CAM_A_EAST",
            track_id="TRK_UNC_3",
            timestamp=12.34,
            timestamp_uncertainty_seconds=input_val,
        )
        self.assertEqual(seg.timestamp_uncertainty_seconds, input_val)
        self.assertIsInstance(seg.timestamp_uncertainty_seconds, float)

    def test_42_step3b_existing_timestamp_unchanged(self):
        """TEST 42: Existing timestamp calculation and properties remain unchanged when uncertainty metadata is added."""
        ts = 150.25
        seg = JourneySegment(
            camera_id="CAM_A_EAST",
            track_id="TRK_UNC_4",
            timestamp=ts,
            timestamp_uncertainty_seconds=0.2,
        )
        self.assertEqual(seg.timestamp, ts)
        self.assertEqual(seg.enter_timestamp, ts)
        self.assertEqual(seg.exit_timestamp, ts)

    def test_43_step3b_camera_timestamp_offset_behavior_unchanged(self):
        """TEST 43: Camera timestamp_offset_seconds behavior remains unchanged."""
        frame_number = 150
        fps = 30.0
        offset = 10.0
        expected_ts = (frame_number / fps) + offset

        evidence = VehicleIdentityEvidence(
            track_id="TRK_OFFSET_1",
            camera_id="CAM_B_WEST",
            vehicle_type="car",
            last_updated_timestamp=expected_ts,
            timestamp_uncertainty_seconds=None,
        )
        track = TrackState(
            track_id="TRK_OFFSET_1",
            camera_id="CAM_B_WEST",
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150),
            confidence=0.9,
            first_seen_timestamp=expected_ts,
            last_seen_timestamp=expected_ts,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=Direction.EAST,
        )
        journey = self.journey_engine.process_track_evidence(evidence, track)
        self.assertEqual(len(journey.segments), 1)
        self.assertEqual(journey.segments[0].timestamp, expected_ts)
        self.assertIsNone(journey.segments[0].timestamp_uncertainty_seconds)

    def test_44_step3b_reaches_module3_temporal_graph_metadata(self):
        """TEST 44: timestamp_uncertainty_seconds reaches Module 3 temporal graph metadata when supplied."""
        from app.graph.temporal_graph import TemporalTrafficGraphEngine
        from app.graph.graph_schema import EdgeType
        graph_engine = TemporalTrafficGraphEngine()

        track_a = TrackState(
            track_id="TRK_UNC_M3_A",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150),
            confidence=0.9,
            first_seen_timestamp=10.0,
            last_seen_timestamp=10.0,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=Direction.EAST,
        )
        ev_a = VehicleIdentityEvidence(
            track_id="TRK_UNC_M3_A",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            last_updated_timestamp=10.0,
            timestamp_uncertainty_seconds=0.35,
        )

        track_b = TrackState(
            track_id="TRK_UNC_M3_B",
            camera_id="CAM_B_WEST",
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150),
            confidence=0.9,
            first_seen_timestamp=25.0,
            last_seen_timestamp=25.0,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=Direction.WEST,
        )
        ev_b = VehicleIdentityEvidence(
            track_id="TRK_UNC_M3_B",
            camera_id="CAM_B_WEST",
            vehicle_type="car",
            last_updated_timestamp=25.0,
            timestamp_uncertainty_seconds=0.35,
        )

        self.journey_engine.temporal_graph_engine = graph_engine
        self.journey_engine.process_track_evidence(ev_a, track_a)
        self.journey_engine.process_track_evidence(ev_b, track_b)

        # 1. Observation Edge check
        obs_edges = [
            d for u, v, k, d in graph_engine.builder.graph.edges(keys=True, data=True)
            if d.get("edge_type") == EdgeType.VEHICLE_OBSERVED_BY_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_OBSERVED_BY_CAMERA
        ]
        for edge_data in obs_edges:
            meta = edge_data.get("data", {}).get("metadata", {}) or edge_data.get("metadata", {})
            self.assertIn("timestamp_uncertainty_seconds", meta)
            self.assertEqual(meta["timestamp_uncertainty_seconds"], 0.35)

        # 2. Transition Edge check
        trans_edges = [
            d for u, v, k, d in graph_engine.builder.graph.edges(keys=True, data=True)
            if d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA
        ]
        self.assertGreaterEqual(len(trans_edges), 1)
        trans_meta = trans_edges[0].get("data", {}).get("metadata", {}) or trans_edges[0].get("metadata", {})
        self.assertIn("timestamp_uncertainty_seconds", trans_meta)
        self.assertEqual(trans_meta["timestamp_uncertainty_seconds"], 0.35)

    def test_45_step3b_none_uncertainty_no_fabrication(self):
        """TEST 45: None uncertainty does not result in a fabricated value in graph metadata."""
        from app.graph.temporal_graph import TemporalTrafficGraphEngine
        from app.graph.graph_schema import EdgeType
        graph_engine = TemporalTrafficGraphEngine()

        track = TrackState(
            track_id="TRK_NONE_UNC_1",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150),
            confidence=0.9,
            first_seen_timestamp=5.0,
            last_seen_timestamp=5.0,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=Direction.EAST,
        )
        evidence = VehicleIdentityEvidence(
            track_id="TRK_NONE_UNC_1",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            last_updated_timestamp=5.0,
            timestamp_uncertainty_seconds=None,
        )

        self.journey_engine.temporal_graph_engine = graph_engine
        self.journey_engine.process_track_evidence(evidence, track)

        obs_edges = [
            d for u, v, k, d in graph_engine.builder.graph.edges(keys=True, data=True)
            if d.get("edge_type") == EdgeType.VEHICLE_OBSERVED_BY_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_OBSERVED_BY_CAMERA
        ]
        self.assertGreaterEqual(len(obs_edges), 1)
        meta = obs_edges[0].get("data", {}).get("metadata", {}) or obs_edges[0].get("metadata", {})
        self.assertNotIn("timestamp_uncertainty_seconds", meta)

    def test_46_step3b_step3a_plate_metadata_remains_intact(self):
        """TEST 46: Step 3A plate metadata remains intact after Step 3B."""
        track = TrackState(
            track_id="TRK_STEP3A_VERIF_1",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150),
            confidence=0.9,
            first_seen_timestamp=10.0,
            last_seen_timestamp=10.0,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=Direction.EAST,
        )
        fused = FusedPlateIdentity(
            track_id="TRK_STEP3A_VERIF_1",
            camera_id="CAM_A_EAST",
            best_plate_number="TN09AB1234",
            overall_confidence=0.94,
            observation_count=5,
            confirmed=True,
            status="CONFIRMED",
            first_seen_timestamp=10.0,
            last_seen_timestamp=10.0,
        )
        evidence = VehicleIdentityEvidence(
            track_id="TRK_STEP3A_VERIF_1",
            camera_id="CAM_A_EAST",
            vehicle_type="car",
            associated_plate=fused,
            last_updated_timestamp=10.0,
            timestamp_uncertainty_seconds=0.25,
        )

        journey = self.journey_engine.process_track_evidence(evidence, track)
        self.assertEqual(len(journey.segments), 1)
        seg = journey.segments[0]
        self.assertEqual(seg.plate_number, "TN09AB1234")
        self.assertEqual(seg.plate_confidence, 0.94)
        self.assertEqual(seg.plate_status, "CONFIRMED")
        self.assertEqual(seg.timestamp_uncertainty_seconds, 0.25)

    def test_47_step3c_full_seven_signal_breakdown_exists(self):
        """TEST 47: Full seven-signal transition breakdown exists on ReIDScoreBreakdown and JourneySegment."""
        breakdown = ReIDScoreBreakdown(
            plate_similarity=0.95,
            appearance_similarity=0.85,
            visual_features_similarity=0.90,
            vehicle_type_similarity=1.0,
            direction_similarity=0.92,
            temporal_compatibility=0.88,
            spatial_compatibility=1.0,
            overall_score=0.91,
            decision=MatchDecision.MATCH_CONFIRMED,
            rejection_reason=None,
        )
        seg = JourneySegment(
            camera_id="CAM_A_EAST",
            track_id="TRK_TB_1",
            timestamp=100.0,
            transition_breakdown=breakdown,
            transition_score=0.91,
            transition_decision=MatchDecision.MATCH_CONFIRMED,
        )
        self.assertIsNotNone(seg.transition_breakdown)
        self.assertEqual(seg.transition_breakdown.plate_similarity, 0.95)
        self.assertEqual(seg.transition_breakdown.appearance_similarity, 0.85)
        self.assertEqual(seg.transition_breakdown.visual_features_similarity, 0.90)
        self.assertEqual(seg.transition_breakdown.vehicle_type_similarity, 1.0)
        self.assertEqual(seg.transition_breakdown.direction_similarity, 0.92)
        self.assertEqual(seg.transition_breakdown.temporal_compatibility, 0.88)
        self.assertEqual(seg.transition_breakdown.spatial_compatibility, 1.0)

    def test_48_step3c_all_seven_signal_values_preserved_exact(self):
        """TEST 48: All seven signal values are preserved exactly on JourneySegment."""
        breakdown = ReIDScoreBreakdown(
            plate_similarity=0.812,
            appearance_similarity=0.745,
            visual_features_similarity=0.678,
            vehicle_type_similarity=1.0,
            direction_similarity=0.950,
            temporal_compatibility=0.880,
            spatial_compatibility=0.750,
            overall_score=0.815,
            decision=MatchDecision.MATCH_CONFIRMED,
        )
        seg = JourneySegment(
            camera_id="CAM_B_WEST",
            track_id="TRK_TB_2",
            timestamp=120.0,
            transition_breakdown=breakdown,
        )
        tb = seg.transition_breakdown
        self.assertEqual(tb.plate_similarity, 0.812)
        self.assertEqual(tb.appearance_similarity, 0.745)
        self.assertEqual(tb.visual_features_similarity, 0.678)
        self.assertEqual(tb.vehicle_type_similarity, 1.0)
        self.assertEqual(tb.direction_similarity, 0.950)
        self.assertEqual(tb.temporal_compatibility, 0.880)
        self.assertEqual(tb.spatial_compatibility, 0.750)

    def test_49_step3c_transition_score_preserved(self):
        """TEST 49: transition_score is preserved on JourneySegment."""
        seg = JourneySegment(
            camera_id="CAM_A_EAST",
            track_id="TRK_TS_1",
            timestamp=10.0,
            transition_score=0.876,
        )
        self.assertEqual(seg.transition_score, 0.876)

    def test_50_step3c_transition_decision_preserved(self):
        """TEST 50: transition_decision is preserved on JourneySegment."""
        seg = JourneySegment(
            camera_id="CAM_A_EAST",
            track_id="TRK_TD_1",
            timestamp=10.0,
            transition_decision=MatchDecision.MATCH_PROBABLE,
        )
        self.assertEqual(seg.transition_decision, MatchDecision.MATCH_PROBABLE)

    def test_51_step3c_rejection_reason_preserved(self):
        """TEST 51: rejection_reason is preserved when present."""
        breakdown = ReIDScoreBreakdown(
            overall_score=0.0,
            decision=MatchDecision.MATCH_REJECTED,
            rejection_reason="impossible travel speed: 250 km/h exceeds max 120 km/h",
        )
        seg = JourneySegment(
            camera_id="CAM_C_NORTH",
            track_id="TRK_REJ_1",
            timestamp=15.0,
            transition_breakdown=breakdown,
            transition_decision=MatchDecision.MATCH_REJECTED,
        )
        self.assertEqual(seg.transition_breakdown.rejection_reason, "impossible travel speed: 250 km/h exceeds max 120 km/h")

    def test_52_step3c_full_transition_breakdown_reaches_module3_edge_metadata(self):
        """TEST 52: Full transition_breakdown reaches Module 3 transition edge metadata."""
        from app.graph.temporal_graph import TemporalTrafficGraphEngine
        from app.graph.graph_schema import EdgeType
        graph_engine = TemporalTrafficGraphEngine()

        breakdown = ReIDScoreBreakdown(
            plate_similarity=0.92,
            appearance_similarity=0.88,
            visual_features_similarity=0.85,
            vehicle_type_similarity=1.0,
            direction_similarity=0.90,
            temporal_compatibility=0.80,
            spatial_compatibility=1.0,
            overall_score=0.89,
            decision=MatchDecision.MATCH_CONFIRMED,
            rejection_reason=None,
        )

        journey = VehicleJourney(
            journey_id="JRN_STEP3C_1",
            global_vehicle_id="VEH_STEP3C_1",
            vehicle_type="car",
            first_seen=0.0,
            last_seen=25.0,
            segments=[
                JourneySegment(
                    camera_id="CAM_A_EAST",
                    track_id="TRK_1",
                    timestamp=0.0,
                ),
                JourneySegment(
                    camera_id="CAM_B_WEST",
                    track_id="TRK_2",
                    timestamp=25.0,
                    transition_score=0.89,
                    transition_decision=MatchDecision.MATCH_CONFIRMED,
                    transition_breakdown=breakdown,
                ),
            ],
        )

        graph_engine.update_journey(journey)

        trans_edges = [
            d for u, v, k, d in graph_engine.builder.graph.edges(keys=True, data=True)
            if d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA
        ]
        self.assertEqual(len(trans_edges), 1)
        meta = trans_edges[0].get("data", {}).get("metadata", {}) or trans_edges[0].get("metadata", {})
        self.assertIn("transition_breakdown", meta)
        tb = meta["transition_breakdown"]
        self.assertEqual(tb["plate_similarity"], 0.92)
        self.assertEqual(tb["appearance_similarity"], 0.88)
        self.assertEqual(tb["visual_features_similarity"], 0.85)
        self.assertEqual(tb["vehicle_type_similarity"], 1.0)
        self.assertEqual(tb["direction_similarity"], 0.90)
        self.assertEqual(tb["temporal_compatibility"], 0.80)
        self.assertEqual(tb["spatial_compatibility"], 1.0)
        self.assertEqual(tb["overall_score"], 0.89)
        self.assertEqual(tb["decision"], "MATCH_CONFIRMED")

    def test_53_step3c_step3a_plate_metadata_remains_intact(self):
        """TEST 53: Step 3A plate metadata remains intact alongside transition breakdown."""
        from app.graph.temporal_graph import TemporalTrafficGraphEngine
        from app.graph.graph_schema import EdgeType
        graph_engine = TemporalTrafficGraphEngine()

        breakdown = ReIDScoreBreakdown(
            plate_similarity=1.0,
            overall_score=0.95,
            decision=MatchDecision.MATCH_CONFIRMED,
        )

        journey = VehicleJourney(
            journey_id="JRN_STEP3C_2",
            global_vehicle_id="VEH_STEP3C_2",
            vehicle_type="car",
            first_seen=0.0,
            last_seen=30.0,
            segments=[
                JourneySegment(
                    camera_id="CAM_A_EAST",
                    track_id="TRK_A",
                    timestamp=0.0,
                    plate_number="TN09AB1234",
                    plate_confidence=0.96,
                    plate_status="CONFIRMED",
                ),
                JourneySegment(
                    camera_id="CAM_B_WEST",
                    track_id="TRK_B",
                    timestamp=30.0,
                    plate_number="TN09AB1234",
                    plate_confidence=0.96,
                    plate_status="CONFIRMED",
                    transition_score=0.95,
                    transition_decision=MatchDecision.MATCH_CONFIRMED,
                    transition_breakdown=breakdown,
                ),
            ],
        )

        graph_engine.update_journey(journey)

        trans_edges = [
            d for u, v, k, d in graph_engine.builder.graph.edges(keys=True, data=True)
            if d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA
        ]
        self.assertEqual(len(trans_edges), 1)
        meta = trans_edges[0].get("data", {}).get("metadata", {}) or trans_edges[0].get("metadata", {})
        self.assertEqual(meta.get("plate_number"), "TN09AB1234")
        self.assertEqual(meta.get("plate_confidence"), 0.96)
        self.assertEqual(meta.get("plate_status"), "CONFIRMED")
        self.assertIn("transition_breakdown", meta)

    def test_54_step3c_step3b_timestamp_uncertainty_remains_intact(self):
        """TEST 54: Step 3B timestamp uncertainty remains intact alongside transition breakdown."""
        from app.graph.temporal_graph import TemporalTrafficGraphEngine
        from app.graph.graph_schema import EdgeType
        graph_engine = TemporalTrafficGraphEngine()

        breakdown = ReIDScoreBreakdown(
            plate_similarity=0.90,
            overall_score=0.88,
            decision=MatchDecision.MATCH_CONFIRMED,
        )

        journey = VehicleJourney(
            journey_id="JRN_STEP3C_3",
            global_vehicle_id="VEH_STEP3C_3",
            vehicle_type="car",
            first_seen=0.0,
            last_seen=20.0,
            segments=[
                JourneySegment(
                    camera_id="CAM_A_EAST",
                    track_id="TRK_A",
                    timestamp=0.0,
                    timestamp_uncertainty_seconds=0.15,
                ),
                JourneySegment(
                    camera_id="CAM_B_WEST",
                    track_id="TRK_B",
                    timestamp=20.0,
                    timestamp_uncertainty_seconds=0.15,
                    transition_score=0.88,
                    transition_decision=MatchDecision.MATCH_CONFIRMED,
                    transition_breakdown=breakdown,
                ),
            ],
        )

        graph_engine.update_journey(journey)

        trans_edges = [
            d for u, v, k, d in graph_engine.builder.graph.edges(keys=True, data=True)
            if d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA
        ]
        self.assertEqual(len(trans_edges), 1)
        meta = trans_edges[0].get("data", {}).get("metadata", {}) or trans_edges[0].get("metadata", {})
        self.assertEqual(meta.get("timestamp_uncertainty_seconds"), 0.15)
        self.assertIn("transition_breakdown", meta)

    def test_55_step3c_has_unobserved_gap_remains_intact(self):
        """TEST 55: has_unobserved_gap remains intact when transition breakdown is present."""
        from app.graph.temporal_graph import TemporalTrafficGraphEngine
        from app.graph.graph_schema import EdgeType
        graph_engine = TemporalTrafficGraphEngine()

        breakdown = ReIDScoreBreakdown(
            plate_similarity=0.90,
            overall_score=0.85,
            decision=MatchDecision.MATCH_CONFIRMED,
        )

        journey = VehicleJourney(
            journey_id="JRN_GAP_1",
            global_vehicle_id="VEH_GAP_1",
            vehicle_type="car",
            first_seen=0.0,
            last_seen=50.0,
            segments=[
                JourneySegment(
                    camera_id="CAM_A_EAST",
                    track_id="TRK_A",
                    timestamp=0.0,
                ),
                JourneySegment(
                    camera_id="CAM_C_NORTH",
                    track_id="TRK_C",
                    timestamp=50.0,
                    transition_score=0.85,
                    transition_decision=MatchDecision.MATCH_CONFIRMED,
                    transition_breakdown=breakdown,
                    has_unobserved_gap=True,
                ),
            ],
        )

        graph_engine.update_journey(journey)

        trans_edges = [
            d for u, v, k, d in graph_engine.builder.graph.edges(keys=True, data=True)
            if d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA
        ]
        self.assertEqual(len(trans_edges), 1)
        meta = trans_edges[0].get("data", {}).get("metadata", {}) or trans_edges[0].get("metadata", {})
        self.assertTrue(meta.get("has_unobserved_gap"))

    def test_56_step3c_no_intermediate_camera_fabricated_for_unobserved_gap(self):
        """TEST 56: No intermediate camera node/edge is fabricated for an unobserved gap."""
        from app.graph.temporal_graph import TemporalTrafficGraphEngine
        from app.graph.graph_schema import EdgeType
        graph_engine = TemporalTrafficGraphEngine()

        breakdown = ReIDScoreBreakdown(
            plate_similarity=0.95,
            overall_score=0.90,
            decision=MatchDecision.MATCH_CONFIRMED,
        )

        journey = VehicleJourney(
            journey_id="JRN_NO_FAB_1",
            global_vehicle_id="VEH_NO_FAB_1",
            vehicle_type="car",
            first_seen=0.0,
            last_seen=60.0,
            segments=[
                JourneySegment(
                    camera_id="CAM_A_EAST",
                    track_id="TRK_A",
                    timestamp=0.0,
                ),
                JourneySegment(
                    camera_id="CAM_C_NORTH",
                    track_id="TRK_C",
                    timestamp=60.0,
                    transition_score=0.90,
                    transition_decision=MatchDecision.MATCH_CONFIRMED,
                    transition_breakdown=breakdown,
                    has_unobserved_gap=True,
                ),
            ],
        )

        graph_engine.update_journey(journey)

        # Verify only direct transition edge from CAM_A_EAST to CAM_C_NORTH exists for this journey
        trans_edges = [
            (u, v, d) for u, v, k, d in graph_engine.builder.graph.edges(keys=True, data=True)
            if (d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA)
            and (d.get("metadata", {}).get("journey_id") == journey.journey_id or d.get("data", {}).get("metadata", {}).get("journey_id") == journey.journey_id)
        ]
        self.assertEqual(len(trans_edges), 1)
        u, v, d = trans_edges[0]
        self.assertEqual(u, "CAM_A_EAST")
        self.assertEqual(v, "CAM_C_NORTH")
        meta = d.get("data", {}).get("metadata", {}) or d.get("metadata", {})
        self.assertTrue(meta.get("has_unobserved_gap"))

    def test_57_step3c_existing_transition_metadata_not_lost(self):
        """TEST 57: Existing transition metadata is not lost when breakdown is added."""
        from app.graph.temporal_graph import TemporalTrafficGraphEngine
        from app.graph.graph_schema import EdgeType
        graph_engine = TemporalTrafficGraphEngine()

        breakdown = ReIDScoreBreakdown(
            plate_similarity=0.88,
            overall_score=0.85,
            decision=MatchDecision.MATCH_CONFIRMED,
            rejection_reason="none",
        )

        journey = VehicleJourney(
            journey_id="JRN_META_1",
            global_vehicle_id="VEH_META_1",
            vehicle_type="car",
            first_seen=0.0,
            last_seen=25.0,
            segments=[
                JourneySegment(
                    camera_id="CAM_A_EAST",
                    track_id="TRK_A",
                    timestamp=0.0,
                ),
                JourneySegment(
                    camera_id="CAM_B_WEST",
                    track_id="TRK_B",
                    timestamp=25.0,
                    transition_score=0.85,
                    transition_decision=MatchDecision.MATCH_CONFIRMED,
                    transition_breakdown=breakdown,
                    has_unobserved_gap=False,
                ),
            ],
        )

        graph_engine.update_journey(journey)

        trans_edges = [
            d for u, v, k, d in graph_engine.builder.graph.edges(keys=True, data=True)
            if d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA
        ]
        self.assertEqual(len(trans_edges), 1)
        meta = trans_edges[0].get("data", {}).get("metadata", {}) or trans_edges[0].get("metadata", {})
        self.assertEqual(meta["global_vehicle_id"], "VEH_META_1")
        self.assertEqual(meta["journey_id"], "JRN_META_1")
        self.assertEqual(meta["transition_score"], 0.85)
        self.assertEqual(meta["transition_decision"], "MATCH_CONFIRMED")
        self.assertIn("transition_breakdown", meta)

    def test_58_step3c_exact_value_integrity_test(self):
        """TEST 58: Deliberately distinct seven-signal values are forwarded to Module 3 graph exactly without recalculation."""
        from app.graph.temporal_graph import TemporalTrafficGraphEngine
        from app.graph.graph_schema import EdgeType
        graph_engine = TemporalTrafficGraphEngine()

        distinct_breakdown = ReIDScoreBreakdown(
            plate_similarity=0.91,
            appearance_similarity=0.73,
            visual_features_similarity=0.84,
            vehicle_type_similarity=1.0,
            temporal_compatibility=0.67,
            spatial_compatibility=0.88,
            direction_similarity=0.76,
            overall_score=0.825,
            decision=MatchDecision.MATCH_CONFIRMED,
            rejection_reason=None,
        )

        journey = VehicleJourney(
            journey_id="JRN_EXACT_1",
            global_vehicle_id="VEH_EXACT_1",
            vehicle_type="car",
            first_seen=0.0,
            last_seen=30.0,
            segments=[
                JourneySegment(
                    camera_id="CAM_A_EAST",
                    track_id="TRK_1",
                    timestamp=0.0,
                ),
                JourneySegment(
                    camera_id="CAM_B_WEST",
                    track_id="TRK_2",
                    timestamp=30.0,
                    transition_score=0.825,
                    transition_decision=MatchDecision.MATCH_CONFIRMED,
                    transition_breakdown=distinct_breakdown,
                ),
            ],
        )

        graph_engine.update_journey(journey)

        trans_edges = [
            d for u, v, k, d in graph_engine.builder.graph.edges(keys=True, data=True)
            if d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value or d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA
        ]
        self.assertEqual(len(trans_edges), 1)
        meta = trans_edges[0].get("data", {}).get("metadata", {}) or trans_edges[0].get("metadata", {})
        self.assertIn("transition_breakdown", meta)
        tb = meta["transition_breakdown"]

        self.assertEqual(tb["plate_similarity"], 0.91)
        self.assertEqual(tb["appearance_similarity"], 0.73)
        self.assertEqual(tb["visual_features_similarity"], 0.84)
        self.assertEqual(tb["vehicle_type_similarity"], 1.0)
        self.assertEqual(tb["temporal_compatibility"], 0.67)
        self.assertEqual(tb["spatial_compatibility"], 0.88)
        self.assertEqual(tb["direction_similarity"], 0.76)
        self.assertEqual(tb["overall_score"], 0.825)
        self.assertEqual(tb["decision"], "MATCH_CONFIRMED")


if __name__ == "__main__":
    unittest.main()


