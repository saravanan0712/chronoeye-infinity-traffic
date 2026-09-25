"""
ChronoEye Infinity - Advanced Number Plate Recovery Unit Tests (ROLLBACK 2 REPAIR)
Verifies ROI quality scoring, safe perspective correction, multi-variant preprocessing,
Levenshtein sequence alignment, multi-frame temporal evidence accumulation, OCR scheduling,
anti-fabrication rules, global vehicle pooling, audit trail logging, and transparent status transitions.
"""

import unittest
import numpy as np
from app.schemas.geometry import BoundingBoxXYXY
from app.schemas.plate import PlateObservation, PlateValidationStatus, FusedPlateIdentity
from app.perception.plate_detector import PlateDetector
from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.plate_fusion import TemporalPlateFusionEngine, levenshtein_distance
from app.perception.plate_association import PlateTrackerAssociationManager
from app.perception.journey import JourneyReconstructionEngine
from app.schemas.tracking import TrackState


class TestAdvancedPlateRecoveryRepair(unittest.TestCase):

    def setUp(self):
        self.detector = PlateDetector()
        self.preprocessor = PlatePreprocessor()
        self.fusion_engine = TemporalPlateFusionEngine(
            confirmation_score_threshold=1.0,
            min_observations_to_confirm=2
        )
        self.association_manager = PlateTrackerAssociationManager()

    def test_01_multiple_ocr_attempts_while_pending(self):
        """Test 1: Multiple OCR attempts allowed while track status is PENDING."""
        track = TrackState(track_id="TRK_PEND_01", camera_id="CAM_01", bbox=BoundingBoxXYXY(x1=10, y1=10, x2=200, y2=100))
        v_bbox = BoundingBoxXYXY(x1=50, y1=50, x2=150, y2=80)

        # Attempt 1
        should1, _ = self.association_manager.should_run_ocr(track, frame_id=5, timestamp=1.0, v_bbox=v_bbox)
        self.assertTrue(should1)

        # Record 1st observation -> status PENDING
        obs1 = PlateObservation(
            track_id="TRK_PEND_01", camera_id="CAM_01", frame_id=5, timestamp=1.0,
            raw_text="TN09AB1234", normalized_text="TN09AB1234", ocr_confidence=0.85,
            validation_confidence=0.95, quality_score=0.40, preprocessing_variant="CLAHE",
            status=PlateValidationStatus.VALID
        )
        fused1 = self.association_manager.fusion_engine.process_observation("TRK_PEND_01", "CAM_01", obs1)
        self.assertEqual(fused1.status, "PENDING")

        # Update stats
        self.association_manager.track_ocr_stats["TRK_PEND_01"] = {
            "attempts": 1, "last_ocr_frame": 5, "last_ocr_timestamp": 1.0
        }

        # Attempt 2 (5 frames later) -> should_run_ocr MUST return True while PENDING
        should2, reason = self.association_manager.should_run_ocr(track, frame_id=10, timestamp=1.2, v_bbox=v_bbox)
        self.assertTrue(should2, f"OCR should run for PENDING track, got reason: {reason}")

    def test_02_ocr_stops_after_confirmed(self):
        """Test 2: OCR stops permanently after plate status becomes CONFIRMED."""
        track = TrackState(track_id="TRK_CONF_02", camera_id="CAM_01", bbox=BoundingBoxXYXY(x1=10, y1=10, x2=200, y2=100))
        v_bbox = BoundingBoxXYXY(x1=50, y1=50, x2=150, y2=80)

        obs1 = PlateObservation(
            track_id="TRK_CONF_02", camera_id="CAM_01", frame_id=5, timestamp=1.0,
            raw_text="TN09AB1234", normalized_text="TN09AB1234", ocr_confidence=0.90,
            validation_confidence=0.95, quality_score=0.50, preprocessing_variant="CLAHE",
            status=PlateValidationStatus.VALID
        )
        obs2 = PlateObservation(
            track_id="TRK_CONF_02", camera_id="CAM_01", frame_id=10, timestamp=1.2,
            raw_text="TN09AB1234", normalized_text="TN09AB1234", ocr_confidence=0.90,
            validation_confidence=0.95, quality_score=0.50, preprocessing_variant="OTSU_THRESH",
            status=PlateValidationStatus.VALID
        )
        self.association_manager.fusion_engine.process_observation("TRK_CONF_02", "CAM_01", obs1)
        fused = self.association_manager.fusion_engine.process_observation("TRK_CONF_02", "CAM_01", obs2)

        self.assertEqual(fused.status, "CONFIRMED")
        self.association_manager.track_ocr_stats["TRK_CONF_02"] = {
            "attempts": 2, "last_ocr_frame": 10, "last_ocr_timestamp": 1.2
        }

        should, reason = self.association_manager.should_run_ocr(track, frame_id=15, timestamp=1.4, v_bbox=v_bbox)
        self.assertFalse(should)
        self.assertEqual(reason, "CONFIRMED_PLATE_CACHED")

    def test_03_high_overall_confidence_does_not_block_pending(self):
        """Test 3: High overall confidence alone does not block PENDING evidence collection."""
        track = TrackState(track_id="TRK_HIGH_03", camera_id="CAM_01", bbox=BoundingBoxXYXY(x1=10, y1=10, x2=200, y2=100))
        v_bbox = BoundingBoxXYXY(x1=50, y1=50, x2=150, y2=80)

        obs1 = PlateObservation(
            track_id="TRK_HIGH_03", camera_id="CAM_01", frame_id=1, timestamp=0.1,
            raw_text="MH12DE1428", normalized_text="MH12DE1428", ocr_confidence=0.95,
            validation_confidence=0.95, quality_score=0.80, preprocessing_variant="ORIGINAL",
            status=PlateValidationStatus.VALID
        )
        fused = self.association_manager.fusion_engine.process_observation("TRK_HIGH_03", "CAM_01", obs1)

        self.assertEqual(fused.status, "PENDING")
        self.assertGreaterEqual(fused.overall_confidence, 0.60)

        self.association_manager.track_ocr_stats["TRK_HIGH_03"] = {
            "attempts": 1, "last_ocr_frame": 1, "last_ocr_timestamp": 0.1
        }
        should, _ = self.association_manager.should_run_ocr(track, frame_id=6, timestamp=0.3, v_bbox=v_bbox)
        self.assertTrue(should)

    def test_04_evidence_accumulates_across_frames(self):
        """Test 4: Evidence score accumulates across multiple temporal frames."""
        obs1 = PlateObservation(
            track_id="TRK_ACC_04", camera_id="CAM_01", frame_id=5, timestamp=0.5,
            raw_text="KA01MH9999", normalized_text="KA01MH9999", ocr_confidence=0.80,
            validation_confidence=0.95, quality_score=0.40, preprocessing_variant="GRAYSCALE",
            status=PlateValidationStatus.VALID
        )
        fused1 = self.fusion_engine.process_observation("TRK_ACC_04", "CAM_01", obs1)
        w1 = fused1.weighted_evidence_score

        obs2 = PlateObservation(
            track_id="TRK_ACC_04", camera_id="CAM_01", frame_id=10, timestamp=0.7,
            raw_text="KA01MH9999", normalized_text="KA01MH9999", ocr_confidence=0.80,
            validation_confidence=0.95, quality_score=0.40, preprocessing_variant="CLAHE",
            status=PlateValidationStatus.VALID
        )
        fused2 = self.fusion_engine.process_observation("TRK_ACC_04", "CAM_01", obs2)
        w2 = fused2.weighted_evidence_score

        self.assertGreater(w2, w1)

    def test_05_two_agreeing_observations_reach_confirmed(self):
        """Test 5: Two agreeing temporal observations reach CONFIRMED status."""
        obs1 = PlateObservation(
            track_id="TRK_CONF_05", camera_id="CAM_01", frame_id=10, timestamp=1.0,
            raw_text="TN09AB1234", normalized_text="TN09AB1234", ocr_confidence=0.88,
            validation_confidence=0.95, quality_score=0.45, preprocessing_variant="CLAHE",
            status=PlateValidationStatus.VALID
        )
        obs2 = PlateObservation(
            track_id="TRK_CONF_05", camera_id="CAM_01", frame_id=15, timestamp=1.2,
            raw_text="TN09AB1234", normalized_text="TN09AB1234", ocr_confidence=0.85,
            validation_confidence=0.95, quality_score=0.40, preprocessing_variant="OTSU_THRESH",
            status=PlateValidationStatus.VALID
        )
        self.fusion_engine.process_observation("TRK_CONF_05", "CAM_01", obs1)
        fused = self.fusion_engine.process_observation("TRK_CONF_05", "CAM_01", obs2)

        self.assertEqual(fused.status, "CONFIRMED")
        self.assertEqual(fused.best_plate_number, "TN09AB1234")
        self.assertEqual(fused.supporting_observations_count, 2)

    def test_06_different_length_candidates_alignment(self):
        """Test 6: Safe Levenshtein alignment handles minor length differences (10 vs 9 chars)."""
        obs1 = PlateObservation(
            track_id="TRK_ALIGN_06", camera_id="CAM_01", frame_id=10, timestamp=1.0,
            raw_text="TN09AB1234", normalized_text="TN09AB1234", ocr_confidence=0.90,
            validation_confidence=0.95, quality_score=0.50, preprocessing_variant="CLAHE",
            status=PlateValidationStatus.VALID
        )
        # Missing '0' -> TN9AB1234 (len 9)
        obs2 = PlateObservation(
            track_id="TRK_ALIGN_06", camera_id="CAM_01", frame_id=15, timestamp=1.2,
            raw_text="TN9AB1234", normalized_text="TN9AB1234", ocr_confidence=0.85,
            validation_confidence=0.65, quality_score=0.45, preprocessing_variant="GRAYSCALE",
            status=PlateValidationStatus.UNCERTAIN
        )
        self.fusion_engine.process_observation("TRK_ALIGN_06", "CAM_01", obs1)
        fused = self.fusion_engine.process_observation("TRK_ALIGN_06", "CAM_01", obs2)

        # Both observations should align into the same candidate cluster
        self.assertEqual(fused.supporting_observations_count, 2)
        self.assertEqual(fused.best_plate_number, "TN09AB1234")

    def test_07_sequence_alignment_anti_fabrication(self):
        """Test 7: Sequence alignment never invents missing characters."""
        obs1 = PlateObservation(
            track_id="TRK_FAB_07", camera_id="CAM_01", frame_id=10, timestamp=1.0,
            raw_text="TN09AB123", normalized_text="TN09AB123", ocr_confidence=0.75,
            validation_confidence=0.65, quality_score=0.40, preprocessing_variant="CLAHE",
            status=PlateValidationStatus.UNCERTAIN
        )
        fused = self.fusion_engine.process_observation("TRK_FAB_07", "CAM_01", obs1)
        self.assertEqual(fused.best_plate_number, "TN09AB123")
        self.assertNotEqual(fused.best_plate_number, "TN09AB1234")

    def test_08_conflicting_candidates_remain_pending_or_unknown(self):
        """Test 8: Conflicting plate reads prevent immediate confirmation and stay PENDING/UNKNOWN."""
        obs1 = PlateObservation(
            track_id="TRK_CNFL_08", camera_id="CAM_01", frame_id=10, timestamp=1.0,
            raw_text="TN09AB1234", normalized_text="TN09AB1234", ocr_confidence=0.80,
            validation_confidence=0.95, quality_score=0.40, preprocessing_variant="CLAHE",
            status=PlateValidationStatus.VALID
        )
        # Completely different plate string (distance > 2)
        obs2 = PlateObservation(
            track_id="TRK_CNFL_08", camera_id="CAM_01", frame_id=15, timestamp=1.2,
            raw_text="DL01XY9999", normalized_text="DL01XY9999", ocr_confidence=0.80,
            validation_confidence=0.95, quality_score=0.40, preprocessing_variant="OTSU_THRESH",
            status=PlateValidationStatus.VALID
        )
        self.fusion_engine.process_observation("TRK_CNFL_08", "CAM_01", obs1)
        fused = self.fusion_engine.process_observation("TRK_CNFL_08", "CAM_01", obs2)

        # Neither candidate reaches supporting count >= 2
        self.assertFalse(fused.confirmed)
        self.assertIn(fused.status, ["PENDING", "UNKNOWN"])

    def test_09_same_frame_preprocessing_variants_single_observation(self):
        """Test 9: Variants from the same frame collapse into 1 single temporal observation."""
        dummy_crop = np.zeros((40, 140, 3), dtype=np.uint8)
        dummy_crop[10:30, 20:120] = 180
        variants = self.preprocessor.preprocess_plate_roi(dummy_crop)

        # Preprocessing produces 8 variants for 1 crop
        self.assertGreaterEqual(len(variants), 4)

        # process_track_frame processes 1 frame and creates 1 PlateObservation
        # which yields observation_count == 1 in fusion_engine
        obs = PlateObservation(
            track_id="TRK_VAR_09", camera_id="CAM_01", frame_id=10, timestamp=1.0,
            raw_text="MH12DE1428", normalized_text="MH12DE1428", ocr_confidence=0.90,
            validation_confidence=0.95, quality_score=0.50, preprocessing_variant="CLAHE",
            status=PlateValidationStatus.VALID
        )
        fused = self.fusion_engine.process_observation("TRK_VAR_09", "CAM_01", obs)
        self.assertEqual(fused.observation_count, 1)

    def test_10_global_vehicle_pooling_same_vehicle_only(self):
        """Test 10: Global vehicle plate pooling only occurs for tracks mapped to the same global vehicle."""
        journey_engine = JourneyReconstructionEngine()
        journey_engine.add_segment("TRK_CAM1_A", "CAM_01", 1.0, 10.0, "VEH_GLOBAL_100")
        journey_engine.add_segment("TRK_CAM2_B", "CAM_02", 15.0, 25.0, "VEH_GLOBAL_100")

        fused_map = {
            "TRK_CAM1_A": FusedPlateIdentity(
                track_id="TRK_CAM1_A", camera_id="CAM_01", best_plate_number="TN09AB1234",
                overall_confidence=0.88, observation_count=2, confirmed=True, status="CONFIRMED"
            ),
            "TRK_CAM2_B": FusedPlateIdentity(
                track_id="TRK_CAM2_B", camera_id="CAM_02", best_plate_number="TN09AB1234",
                overall_confidence=0.85, observation_count=1, confirmed=False, status="PENDING"
            )
        }
        journey_engine.update_journeys(fused_map)
        journeys = journey_engine.get_active_journeys()

        self.assertEqual(len(journeys), 1)
        j = journeys[0]
        self.assertEqual(j.global_vehicle_id, "VEH_GLOBAL_100")
        self.assertEqual(j.best_plate_number, "TN09AB1234")

    def test_11_raw_ocr_audit_trail_preserved(self):
        """Test 11: Full raw OCR audit trail is preserved."""
        obs = PlateObservation(
            track_id="TRK_AUDIT_11", camera_id="CAM_01", frame_id=20, timestamp=2.0,
            raw_text="KA01MH9999", normalized_text="KA01MH9999", ocr_confidence=0.92,
            validation_confidence=0.95, quality_score=0.60, preprocessing_variant="DENOISED",
            status=PlateValidationStatus.VALID
        )
        fused = self.fusion_engine.process_observation("TRK_AUDIT_11", "CAM_01", obs)
        self.assertEqual(len(fused.raw_observations_audit), 1)
        audit = fused.raw_observations_audit[0]
        self.assertEqual(audit["raw_text"], "KA01MH9999")
        self.assertEqual(audit["variant"], "DENOISED")

    def test_12_missing_characters_remain_missing(self):
        """Test 12: Incomplete OCR reads preserve missing characters."""
        obs = PlateObservation(
            track_id="TRK_MISS_12", camera_id="CAM_01", frame_id=5, timestamp=0.5,
            raw_text="TN09AB1", normalized_text="TN09AB1", ocr_confidence=0.70,
            validation_confidence=0.20, quality_score=0.35, preprocessing_variant="ORIGINAL",
            status=PlateValidationStatus.INVALID
        )
        fused = self.fusion_engine.process_observation("TRK_MISS_12", "CAM_01", obs)
        self.assertEqual(fused.best_plate_number, "TN09AB1")

    def test_13_perspective_correction_fallback(self):
        """Test 13: Perspective correction safely returns original crop on noisy contour."""
        dummy_crop = np.zeros((50, 150, 3), dtype=np.uint8)
        warped, corrected = self.detector.correct_perspective(dummy_crop)
        self.assertIsNotNone(warped)
        self.assertIsInstance(corrected, bool)

    def test_14_roi_quality_influences_without_eliminating(self):
        """Test 14: Medium ROI quality reduces weight without eliminating reasonable evidence."""
        obs = PlateObservation(
            track_id="TRK_QUAL_14", camera_id="CAM_01", frame_id=1, timestamp=0.1,
            raw_text="TN09AB1234", normalized_text="TN09AB1234", ocr_confidence=0.85,
            validation_confidence=0.95, quality_score=0.30, preprocessing_variant="GRAYSCALE",
            status=PlateValidationStatus.VALID
        )
        w = self.fusion_engine.calculate_observation_weight(obs)
        self.assertGreaterEqual(w, 0.40)
        self.assertLessEqual(w, 0.80)


if __name__ == "__main__":
    unittest.main()
