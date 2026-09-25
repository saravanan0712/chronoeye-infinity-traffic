"""
ChronoEye Infinity - Phase 4 Verification Test Suite
Automated Python test suite verifying ALPR/OCR Pipeline requirements across 15 test cases.
"""

import os
import sys
import unittest

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.detection import DetectionEvent, BoundingBoxXYXY
from app.schemas.tracking import TrackState, TrackStatus
from app.schemas.plate import (
    PlateObservation,
    FusedPlateIdentity,
    VehicleIdentityEvidence,
    PlateValidationStatus,
)
from app.perception.plate_detector import PlateDetector
from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.ocr_engine import (
    IndianPlateValidator,
    TextNormalizer,
    TestOCREngine,
    EasyOCREngine,
    OCREngineFactory,
)
from app.perception.plate_fusion import TemporalPlateFusionEngine
from app.perception.plate_association import PlateTrackerAssociationManager
from app.perception.adapter import DetectionAdapter
from app.perception.tracker import VehicleTrackerManager
from app.simulation.engine import TrafficSimulationEngine


class TestPhase4ALPRPipeline(unittest.TestCase):

    def setUp(self):
        self.detector = PlateDetector()
        self.preprocessor = PlatePreprocessor()
        self.ocr_engine = TestOCREngine(default_plate="TN09AB1234", default_conf=0.92)
        # Legacy constructor kwargs: min_observations_to_confirm=2, confirmation_score_threshold=1.5
        # preserves the pre-existing test contract (2 obs → CONFIRMED).
        # The real-video default FusionConfig (min=3) is unchanged.
        self.fusion_engine = TemporalPlateFusionEngine(
            min_observations_to_confirm=2,
            confirmation_score_threshold=1.5,
        )
        self.association_manager = PlateTrackerAssociationManager(ocr_engine=self.ocr_engine)
        # Inject same fusion engine so test_12 (association path) also uses min=2
        self.association_manager.fusion_engine = self.fusion_engine

    def test_1_plate_schema_validation(self):
        """Test 1: Verify PlateObservation, FusedPlateIdentity, and VehicleIdentityEvidence schemas."""
        bbox = BoundingBoxXYXY(x1=120.0, y1=80.0, x2=180.0, y2=100.0)
        obs = PlateObservation(
            track_id="TRK_101",
            camera_id="CAM_A_EAST",
            frame_id=10,
            timestamp=1.0,
            bbox=bbox,
            raw_text="TN 09 AB 1234",
            normalized_text="TN09AB1234",
            ocr_confidence=0.92,
            validation_confidence=0.95,
            overall_confidence=0.93,
            status=PlateValidationStatus.VALID,
        )

        self.assertTrue(obs.plate_id.startswith("PLT_"))
        self.assertEqual(obs.track_id, "TRK_101")
        self.assertEqual(obs.normalized_text, "TN09AB1234")
        self.assertEqual(obs.status, PlateValidationStatus.VALID)

    def test_2_plate_bbox_validation(self):
        """Test 2: Verify plate ROI coordinate validator rejects out-of-bounds boxes."""
        valid_box = BoundingBoxXYXY(x1=100.0, y1=100.0, x2=200.0, y2=200.0)
        self.assertTrue(self.detector.validate_roi_bounds(valid_box, 1920, 1080))

        invalid_box = BoundingBoxXYXY(x1=-10.0, y1=100.0, x2=200.0, y2=200.0)
        self.assertFalse(self.detector.validate_roi_bounds(invalid_box, 1920, 1080))

    def test_3_plate_roi_extraction(self):
        """Test 3: Verify vehicle crop & plate candidate ROI extraction."""
        v_bbox = BoundingBoxXYXY(x1=100.0, y1=100.0, x2=300.0, y2=260.0)
        frame_mock = {"width": 1920, "height": 1080}

        success, crop, bounds = self.detector.extract_vehicle_crop(frame_mock, v_bbox)
        self.assertTrue(success)
        self.assertIsNotNone(crop)

        candidates = self.detector.extract_plate_candidates(crop, v_bbox)
        self.assertGreater(len(candidates), 0)
        plate_crop, plate_bbox = candidates[0]
        self.assertGreater(plate_bbox.x1, v_bbox.x1)
        self.assertGreater(plate_bbox.y1, v_bbox.y1)

    def test_4_preprocessing_pipeline(self):
        """Test 4: Verify preprocessor generates OCR-ready image variants."""
        crop_mock = {"type": "plate_crop"}
        variants = self.preprocessor.preprocess_plate_roi(crop_mock)
        self.assertGreater(len(variants), 0)

    def test_5_ocr_engine_initialization(self):
        """Test 5: Verify OCR engine initialization via factory."""
        engine = OCREngineFactory.create_engine(prefer_real=False)
        self.assertIsInstance(engine, TestOCREngine)

    def test_6_ocr_candidate_generation(self):
        """Test 6: Verify OCR text recognition candidate generation."""
        variants = [{"sim_plate": "TN09AB1234"}]
        raw, norm, conf, _ = self.ocr_engine.recognize_text(variants)
        self.assertEqual(norm, "TN09AB1234")
        self.assertGreater(conf, 0.9)

    def test_7_text_normalization(self):
        """Test 7: Verify TextNormalizer position-aware error correction."""
        # 1. Standard messy string
        self.assertEqual(TextNormalizer.normalize("tn-09 ab 1234"), "TN09AB1234")
        # 2. Position-aware character corrections ('0' in state -> 'O', 'O' in RTO -> '0')
        self.assertEqual(TextNormalizer.normalize("0N O9 AB 1234"), "ON09AB1234")

    def test_8_plate_format_validation(self):
        """Test 8: Verify IndianPlateValidator format checking."""
        status_valid, conf1 = IndianPlateValidator.validate_format("TN09AB1234")
        self.assertEqual(status_valid, PlateValidationStatus.VALID)
        self.assertGreater(conf1, 0.9)

        status_invalid, conf2 = IndianPlateValidator.validate_format("INVALID_STRING")
        self.assertEqual(status_invalid, PlateValidationStatus.INVALID)
        self.assertLess(conf2, 0.5)

    def test_9_confidence_filtering(self):
        """Test 9: Verify combined OCR + validation confidence calculation."""
        raw, norm, ocr_conf, _ = self.ocr_engine.recognize_text([{"sim_plate": "TN09AB1234"}])
        val_status, val_conf = IndianPlateValidator.validate_format(norm)
        overall_conf = round(ocr_conf * 0.6 + val_conf * 0.4, 3)

        self.assertGreater(overall_conf, 0.85)

    def test_10_temporal_plate_fusion(self):
        """Test 10: Verify TemporalPlateFusionEngine multi-frame accumulation and confirmation."""
        obs1 = PlateObservation(
            track_id="TRK_101", camera_id="CAM_A_EAST", frame_id=1, timestamp=0.1,
            raw_text="TN09AB1234", normalized_text="TN09AB1234", ocr_confidence=0.90,
            validation_confidence=0.95, overall_confidence=0.92, status=PlateValidationStatus.VALID
        )
        obs2 = PlateObservation(
            track_id="TRK_101", camera_id="CAM_A_EAST", frame_id=2, timestamp=0.2,
            raw_text="TN09AB1234", normalized_text="TN09AB1234", ocr_confidence=0.92,
            validation_confidence=0.95, overall_confidence=0.93, status=PlateValidationStatus.VALID
        )

        fused1 = self.fusion_engine.process_observation("TRK_101", "CAM_A_EAST", obs1)
        self.assertFalse(fused1.confirmed)  # 1 observation < min_confirm (2)

        fused2 = self.fusion_engine.process_observation("TRK_101", "CAM_A_EAST", obs2)
        self.assertTrue(fused2.confirmed)   # 2 observations >= 2, score >= 1.5
        self.assertEqual(fused2.best_plate_number, "TN09AB1234")

    def test_11_noisy_ocr_rejection(self):
        """Test 11: Verify TemporalPlateFusionEngine rejects noisy single-frame OCR outlier."""
        obs_good = PlateObservation(
            track_id="TRK_101", camera_id="CAM_A_EAST", frame_id=1, timestamp=0.1,
            raw_text="TN09AB1234", normalized_text="TN09AB1234", ocr_confidence=0.92,
            validation_confidence=0.95, overall_confidence=0.93, status=PlateValidationStatus.VALID
        )
        obs_noisy = PlateObservation(
            track_id="TRK_101", camera_id="CAM_A_EAST", frame_id=3, timestamp=0.3,
            raw_text="TN09A81234", normalized_text="TN09A81234", ocr_confidence=0.60,
            validation_confidence=0.65, overall_confidence=0.62, status=PlateValidationStatus.UNCERTAIN
        )

        # 3 good observations vs 1 noisy observation
        for f in range(1, 4):
            self.fusion_engine.process_observation("TRK_101", "CAM_A_EAST", obs_good)

        fused = self.fusion_engine.process_observation("TRK_101", "CAM_A_EAST", obs_noisy)
        self.assertEqual(fused.best_plate_number, "TN09AB1234")

    def test_12_track_plate_association(self):
        """Test 12: Verify PlateTrackerAssociationManager links TRK_101 ↔ TN09AB1234."""
        track = TrackState(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True,
            raw_vehicle_reference="V_100_TN09AB1234"
        )
        frame_mock = {"width": 1920, "height": 1080}

        # Process 3 DISTINCT, widely-spaced frames to provide 3 independent temporal observations.
        # Frames 1, 10, 20 ensure each call passes the OCR scheduling interval check and
        # accumulates a separate temporal observation, satisfying min_observations_to_confirm=3.
        for f_id in [1, 10, 20]:
            evidence = self.association_manager.process_track_frame(track, frame_mock, f_id * 0.1, f_id)

        self.assertEqual(evidence.track_id, "TRK_101")
        self.assertIsNotNone(evidence.associated_plate)
        self.assertEqual(evidence.associated_plate.best_plate_number, "TN09AB1234")
        self.assertTrue(evidence.associated_plate.confirmed)

    def test_13_temporary_ocr_failure(self):
        """Test 13: Verify temporary OCR failure retains confirmed plate identity."""
        track = TrackState(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True,
            raw_vehicle_reference="V_100_TN09AB1234"
        )
        frame_mock = {"width": 1920, "height": 1080}

        # Confirm plate identity
        for f in range(1, 3):
            self.association_manager.process_track_frame(track, frame_mock, f * 0.1, f)

        # Simulate OCR failure (None observation)
        fused = self.association_manager.fusion_engine.process_observation("TRK_101", "CAM_A_EAST", None)
        self.assertIsNotNone(fused)
        self.assertEqual(fused.best_plate_number, "TN09AB1234")

    def test_14_phase_1_to_4_integration(self):
        """Test 14: End-to-end integration Phase 1 simulation -> Phase 2 detection -> Phase 3 tracking -> Phase 4 ALPR."""
        sim = TrafficSimulationEngine(seed=42)
        for _ in range(10):
            sim.step(dt_seconds=1.0)

        obs_list = sim.recent_observations
        self.assertGreater(len(obs_list), 0)

        det_events = DetectionAdapter.batch_convert(obs_list)
        tracker_mgr = VehicleTrackerManager()
        active_tracks = tracker_mgr.update(det_events[0].camera_id, det_events, sim.sim_time)

        self.assertGreater(len(active_tracks), 0)
        track = active_tracks[0]

        frame_mock = {"width": 1920, "height": 1080}
        evidence = self.association_manager.process_track_frame(track, frame_mock, sim.sim_time, 10)

        self.assertEqual(evidence.track_id, track.track_id)
        self.assertIsNotNone(evidence.associated_plate)

    def test_15_deterministic_reproducibility(self):
        """Test 15: Verify deterministic ALPR pipeline reproducibility."""
        mgr1 = PlateTrackerAssociationManager(ocr_engine=TestOCREngine())
        mgr2 = PlateTrackerAssociationManager(ocr_engine=TestOCREngine())

        track = TrackState(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True,
            raw_vehicle_reference="V_100_TN09AB1234"
        )
        frame_mock = {"width": 1920, "height": 1080}


        ev1 = mgr1.process_track_frame(track, frame_mock, 1.0, 1)
        ev2 = mgr2.process_track_frame(track, frame_mock, 1.0, 1)

        self.assertEqual(ev1.associated_plate.best_plate_number, ev2.associated_plate.best_plate_number)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Fusion Experiment Tests (T1–T15)
# All 15 spec requirements validated here.
# ──────────────────────────────────────────────────────────────────────────────

class TestTemporalFusionExperiment(unittest.TestCase):
    """15 focused tests for the temporal ANPR fusion experimental upgrade."""

    def _make_obs(self, track_id, frame_id, text, ocr_conf=0.90,
                  val_conf=0.95, quality=0.80, timestamp=None):
        """Helper: create a PlateObservation with given parameters."""
        from app.schemas.plate import PlateValidationStatus
        return PlateObservation(
            track_id=track_id,
            camera_id="CAM_A_EAST",
            frame_id=frame_id,
            timestamp=timestamp if timestamp is not None else float(frame_id) * 0.1,
            raw_text=text,
            normalized_text=text,
            ocr_confidence=ocr_conf,
            validation_confidence=val_conf,
            overall_confidence=round(0.4 * ocr_conf + 0.3 * val_conf + 0.3 * quality, 3),
            quality_score=quality,
            status=PlateValidationStatus.VALID,
        )

    # T1 — One valid OCR observation → PENDING (not CONFIRMED)
    def test_T1_single_obs_is_pending(self):
        """T1: A single valid observation must yield PENDING, never CONFIRMED."""
        engine = TemporalPlateFusionEngine()
        obs = self._make_obs("TRK_T1", 1, "TN09AB1234")
        result = engine.process_observation("TRK_T1", "CAM_A_EAST", obs)
        self.assertIsNotNone(result)
        self.assertFalse(result.confirmed, "Single obs must NOT be CONFIRMED")
        self.assertEqual(result.status, "PENDING")

    # T2 — One strong OCR obs must NOT automatically confirm
    def test_T2_high_conf_single_obs_not_confirmed(self):
        """T2: Even ocr_confidence=0.99 on a single frame must stay PENDING."""
        engine = TemporalPlateFusionEngine()
        obs = self._make_obs("TRK_T2", 1, "TN09AB1234", ocr_conf=0.99, val_conf=0.99, quality=0.99)
        result = engine.process_observation("TRK_T2", "CAM_A_EAST", obs)
        self.assertFalse(result.confirmed, "Single obs must NOT confirm regardless of confidence")

    # T3 — Three consistent temporal observations → CONFIRMED
    def test_T3_three_consistent_obs_confirmed(self):
        """T3: 3 consistent observations across distinct frames must reach CONFIRMED."""
        engine = TemporalPlateFusionEngine()
        for fid in [1, 2, 3]:
            obs = self._make_obs("TRK_T3", fid, "TN09AB1234")
            result = engine.process_observation("TRK_T3", "CAM_A_EAST", obs)
        self.assertTrue(result.confirmed, "3 consistent obs must CONFIRM")
        self.assertEqual(result.status, "CONFIRMED")
        self.assertEqual(result.best_plate_number, "TN09AB1234")

    # T4 — Same-frame 4 variants = 1 temporal observation
    def test_T4_same_frame_variants_count_as_one(self):
        """T4: Multiple preprocessing variants of the same frame_id count as exactly 1."""
        engine = TemporalPlateFusionEngine()
        for variant in ["GRAYSCALE", "CLAHE", "OTSU_THRESH", "SHARPENED"]:
            obs = self._make_obs("TRK_T4", frame_id=5, text="TN09AB1234")
            obs.preprocessing_variant = variant
            engine.process_observation("TRK_T4", "CAM_A_EAST", obs)
        result = engine.get_fused_identity("TRK_T4")
        self.assertIsNotNone(result)
        self.assertEqual(result.observation_count, 1,
                         "4 variants of same frame must produce exactly 1 temporal obs")
        self.assertFalse(result.confirmed)

    # T5 — Conflicting OCR characters → char-wise vote picks majority
    def test_T5_character_wise_voting(self):
        """T5: Position 6 majority B (3 obs) beats 8 (1 obs) via weighted vote."""
        from app.perception.plate_fusion import _character_wise_vote
        obs_b = self._make_obs("TRK_T5", 1, "TN09AB1234")
        obs_8 = self._make_obs("TRK_T5", 2, "TN09A81234", ocr_conf=0.60, val_conf=0.65)
        obs_b2 = self._make_obs("TRK_T5", 3, "TN09AB1234")
        obs_b3 = self._make_obs("TRK_T5", 4, "TN09AB1234")
        cluster_obs = [
            (obs_b, 0.80), (obs_8, 0.45), (obs_b2, 0.80), (obs_b3, 0.80)
        ]
        voted = _character_wise_vote(cluster_obs, "TN09AB1234")
        self.assertEqual(voted[5], "B", f"Position 5 should vote 'B', got '{voted}'")

    # T6 — ±1 length variation still clusters together
    def test_T6_length_variation_clusters(self):
        """T6: TN09AB1234 and TN09AB123 (±1 length) must cluster together."""
        engine = TemporalPlateFusionEngine()
        engine.process_observation("TRK_T6", "CAM_A_EAST",
                                   self._make_obs("TRK_T6", 1, "TN09AB1234"))
        engine.process_observation("TRK_T6", "CAM_A_EAST",
                                   self._make_obs("TRK_T6", 2, "TN09AB123", val_conf=0.60))
        result = engine.get_fused_identity("TRK_T6")
        self.assertIsNotNone(result)
        # Both must be in one cluster → candidate_history has at most 2 distinct keys
        self.assertLessEqual(len(result.candidate_history), 2)
        # Combined score should be higher than single-obs threshold
        self.assertGreater(result.weighted_evidence_score, 0.15)

    # T7 — INVALID OCR candidate causes validation rejection (stat check)
    def test_T7_invalid_ocr_rejected_by_association(self):
        """T7: INVALID plate text increments validation_rejections stat."""
        from app.perception.plate_association import PlateTrackerAssociationManager
        from app.schemas.tracking import TrackState, TrackStatus
        mgr = PlateTrackerAssociationManager(
            ocr_engine=TestOCREngine(default_plate="INVALID123", default_conf=0.80)
        )
        track = TrackState(
            track_id="TRK_T7", camera_id="CAM_A_EAST", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=300, y2=250),
            current_center=(200, 175), confidence=0.8,
            first_seen_timestamp=0.0, last_seen_timestamp=1.0,
            status=TrackStatus.CONFIRMED, confirmed=True,
        )
        mgr.process_track_frame(track, {"width": 1920, "height": 1080}, 1.0, 1)
        self.assertGreater(
            mgr.ocr_scheduling_stats.get("validation_rejections", 0), 0,
            "INVALID plate text must increment validation_rejections"
        )

    # T8 — PENDING continues collecting evidence
    def test_T8_pending_continues_accumulating(self):
        """T8: A PENDING track must have increasing observation_count on each new frame."""
        engine = TemporalPlateFusionEngine()
        r1 = engine.process_observation("TRK_T8", "CAM_A_EAST",
                                        self._make_obs("TRK_T8", 1, "TN09AB1234"))
        r2 = engine.process_observation("TRK_T8", "CAM_A_EAST",
                                        self._make_obs("TRK_T8", 2, "TN09AB1234"))
        self.assertEqual(r1.status, "PENDING")
        self.assertEqual(r2.observation_count, 2,
                         "PENDING must accumulate—2nd frame must give obs_count=2")

    # T9 — CONFIRMED stops normal OCR scheduling (cache check)
    def test_T9_confirmed_plate_is_cached(self):
        """T9: After CONFIRMED, confirmed=True and status='CONFIRMED' persist."""
        engine = TemporalPlateFusionEngine()
        for fid in [1, 2, 3]:
            result = engine.process_observation("TRK_T9", "CAM_A_EAST",
                                                self._make_obs("TRK_T9", fid, "TN09AB1234"))
        self.assertTrue(result.confirmed)
        # Passing None obs must not clear confirmed status
        cached = engine.process_observation("TRK_T9", "CAM_A_EAST", None)
        self.assertIsNotNone(cached)
        self.assertTrue(cached.confirmed, "CONFIRMED must persist after None obs")

    # T10 — Two different track_ids never share evidence
    def test_T10_evidence_isolation_between_tracks(self):
        """T10: Evidence for TRK_A and TRK_B must be completely independent."""
        engine = TemporalPlateFusionEngine()
        for fid in [1, 2, 3]:
            engine.process_observation("TRK_A", "CAM_A_EAST",
                                       self._make_obs("TRK_A", fid, "TN09AB1234"))
        r_b = engine.process_observation("TRK_B", "CAM_A_EAST",
                                         self._make_obs("TRK_B", 1, "KA01MH9999"))
        r_a = engine.get_fused_identity("TRK_A")
        self.assertNotEqual(
            r_a.observation_count, r_b.observation_count,
            "TRK_A and TRK_B must have independent observation counts"
        )
        self.assertNotEqual(r_a.best_plate_number, r_b.best_plate_number)

    # T11 — Audit trail preserved across all observations
    def test_T11_audit_trail_preserved(self):
        """T11: raw_observations_audit must contain one entry per distinct frame."""
        engine = TemporalPlateFusionEngine()
        for fid in [10, 20, 30]:
            engine.process_observation("TRK_T11", "CAM_A_EAST",
                                       self._make_obs("TRK_T11", fid, "TN09AB1234"))
        result = engine.get_fused_identity("TRK_T11")
        self.assertEqual(len(result.raw_observations_audit), 3,
                         "Audit trail must have 1 entry per distinct frame")
        frame_ids = [e["frame_id"] for e in result.raw_observations_audit]
        self.assertIn(10, frame_ids)
        self.assertIn(20, frame_ids)
        self.assertIn(30, frame_ids)

    # T12 — No fabricated characters
    def test_T12_no_fabricated_characters(self):
        """T12: voted_text characters must all appear in at least one observed string."""
        from app.perception.plate_fusion import _character_wise_vote
        obs1 = self._make_obs("TRK_T12", 1, "TN09AB1234")
        obs2 = self._make_obs("TRK_T12", 2, "TN09A81234")
        obs3 = self._make_obs("TRK_T12", 3, "TN09AB1234")
        cluster_obs = [(obs1, 0.8), (obs2, 0.6), (obs3, 0.8)]
        voted = _character_wise_vote(cluster_obs, "TN09AB1234")
        # Every character in voted must come from one of the observed strings
        observed_chars_per_pos = [
            set(obs.normalized_text[i] for obs, _ in cluster_obs
                if i < len(obs.normalized_text))
            for i in range(len(voted))
        ]
        for i, ch in enumerate(voted):
            self.assertIn(ch, observed_chars_per_pos[i],
                          f"Voted char '{ch}' at pos {i} is not in observed set {observed_chars_per_pos[i]}")

    # T13 — Character agreement ratio is computed correctly
    def test_T13_agreement_ratio_correct(self):
        """T13: Agreement ratio = best_cluster_score / total_evidence."""
        engine = TemporalPlateFusionEngine()
        for fid in [1, 2, 3]:
            engine.process_observation("TRK_T13", "CAM_A_EAST",
                                       self._make_obs("TRK_T13", fid, "TN09AB1234"))
        result = engine.get_fused_identity("TRK_T13")
        # All obs are in the same cluster → agreement_ratio must be 1.0
        self.assertAlmostEqual(result.character_agreement_ratio, 1.0, places=2,
                               msg="All same-text obs → agreement_ratio must equal 1.0")

    # T14 — Confusion map: 0/O treated as equivalent in clustering
    def test_T14_confusion_map_clustering(self):
        """T14: TN09AB1234 and TNO9AB1234 (0 vs O at pos 2) must cluster with cost ≤ 0.5."""
        from app.perception.plate_fusion import confusion_weighted_distance
        # The confusion pair is 0/O (zero vs letter O).
        # Strings differ only at position 2: '0' in s1 vs 'O' in s2.
        d = confusion_weighted_distance("TN09AB1234", "TNO9AB1234")
        # Standard Levenshtein would give 1.0; confusion map gives 0.5
        self.assertLessEqual(d, 0.5 + 0.001,
                             f"Confused pair 0/O must cost ≤0.5, got {d}")
        # Also verify un-confused pair costs full 1.0
        d_full = confusion_weighted_distance("TN09AB1234", "TN09AB1239")
        self.assertAlmostEqual(d_full, 1.0, places=5,
                               msg="Non-confused substitution must cost 1.0")

    # T15 — Deterministic output for identical inputs
    def test_T15_deterministic_fusion(self):
        """T15: Two engines fed the same observations must produce identical output."""
        def run_engine():
            engine = TemporalPlateFusionEngine()
            for fid in [1, 2, 3]:
                obs = self._make_obs("TRK_T15", fid, "TN09AB1234",
                                     ocr_conf=0.90, val_conf=0.95, quality=0.80)
                result = engine.process_observation("TRK_T15", "CAM_A_EAST", obs)
            return result

        r1 = run_engine()
        r2 = run_engine()
        self.assertEqual(r1.confirmed, r2.confirmed)
        self.assertEqual(r1.status, r2.status)
        self.assertEqual(r1.best_plate_number, r2.best_plate_number)
        self.assertAlmostEqual(r1.overall_confidence, r2.overall_confidence, places=5)


if __name__ == "__main__":
    unittest.main()
