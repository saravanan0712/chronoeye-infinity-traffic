"""
ChronoEye Infinity - Stage 4 ANPR / OCR Verification Test Suite
Automated Python test suite verifying Stage 4 requirements across 15 explicit test cases.
"""

import os
import sys
import unittest
import json

# Add backend directory to Python path
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
from app.perception.tracker import VehicleTrackerManager


class TestStage4RealANPRPipeline(unittest.TestCase):
    """Stage 4: Real License Plate Recognition (ANPR / OCR) Verification Tests."""

    def setUp(self):
        self.detector = PlateDetector()
        self.preprocessor = PlatePreprocessor()
        self.ocr_engine = TestOCREngine(default_plate="TN09AB1234", default_conf=0.92)
        self.fusion_engine = TemporalPlateFusionEngine()
        self.association_manager = PlateTrackerAssociationManager(ocr_engine=self.ocr_engine)

    def test_1_ocr_configuration(self):
        """Test 4.1: Verify OCR engine configuration parameters."""
        engine = TestOCREngine(default_plate="KA01MH9999", default_conf=0.88)
        self.assertEqual(engine.default_plate, "KA01MH9999")
        self.assertEqual(engine.default_conf, 0.88)

    def test_2_ocr_initialization(self):
        """Test 4.2: Verify OCR engine initialization via factory."""
        engine = OCREngineFactory.create_engine(prefer_real=False)
        self.assertIsInstance(engine, TestOCREngine)

    def test_3_missing_invalid_input_handling(self):
        """Test 4.3: Verify OCR engine handles empty or None input lists gracefully."""
        raw, norm, conf, _ = self.ocr_engine.recognize_text([])
        self.assertEqual(norm, "TN09AB1234")  # Fallback test engine returns configured default or empty

    def test_4_plate_bounding_box_validation(self):
        """Test 4.4: Verify plate candidate ROI bounding box boundary validation."""
        valid_box = BoundingBoxXYXY(x1=100.0, y1=100.0, x2=200.0, y2=150.0)
        self.assertTrue(self.detector.validate_roi_bounds(valid_box, 1920, 1080))

        invalid_box = BoundingBoxXYXY(x1=-10.0, y1=100.0, x2=200.0, y2=150.0)
        self.assertFalse(self.detector.validate_roi_bounds(invalid_box, 1920, 1080))

    def test_5_text_normalization(self):
        """Test 4.5: Verify position-aware text normalizer character correction."""
        # 1. Whitespace & hyphen removal
        self.assertEqual(TextNormalizer.normalize("tn-09 ab 1234"), "TN09AB1234")
        # 2. Position-aware character corrections ('0' in state -> 'O', 'O' in RTO -> '0')
        self.assertEqual(TextNormalizer.normalize("0N O9 AB 1234"), "ON09AB1234")

    def test_6_confidence_filtering(self):
        """Test 4.6: Verify confidence filtering and overall confidence calculation."""
        raw, norm, ocr_conf, _ = self.ocr_engine.recognize_text([{"sim_plate": "TN09AB1234"}])
        val_status, val_conf = IndianPlateValidator.validate_format(norm)
        overall_conf = round(ocr_conf * 0.6 + val_conf * 0.4, 3)

        self.assertGreater(overall_conf, 0.85)

    def test_7_invalid_plate_rejection(self):
        """Test 4.7: Verify IndianPlateValidator rejects invalid registration text formats."""
        status, conf = IndianPlateValidator.validate_format("INVALID123")
        self.assertEqual(status, PlateValidationStatus.INVALID)
        self.assertLess(conf, 0.5)

    def test_8_track_to_plate_association(self):
        """Test 4.8: Verify PlateTrackerAssociationManager associates TRK_101 with recognized plate."""
        track = TrackState(
            track_id="TRK_101", camera_id="CAM_A_EAST", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True,
            raw_vehicle_reference="V_100_TN09AB1234"
        )
        frame_mock = {"width": 1920, "height": 1080}

        for f in range(1, 3):
            evidence = self.association_manager.process_track_frame(track, frame_mock, f * 0.1, f)

        self.assertEqual(evidence.track_id, "TRK_101")
        self.assertIsNotNone(evidence.associated_plate)
        self.assertEqual(evidence.associated_plate.best_plate_number, "TN09AB1234")

    def test_9_repeated_ocr_observation(self):
        """Test 4.9: Verify repeated multi-frame OCR observations increase observation_count."""
        obs = PlateObservation(
            track_id="TRK_101", camera_id="CAM_01", frame_id=1, timestamp=0.1,
            raw_text="TN09AB1234", normalized_text="TN09AB1234", ocr_confidence=0.90,
            validation_confidence=0.95, overall_confidence=0.92, status=PlateValidationStatus.VALID
        )
        for i in range(1, 5):
            fused = self.fusion_engine.process_observation("TRK_101", "CAM_01", obs)

        self.assertEqual(fused.observation_count, 4)
        self.assertTrue(fused.confirmed)

    def test_10_temporal_ocr_fusion(self):
        """Test 4.10: Verify TemporalPlateFusionEngine selects highest cumulative weighted plate candidate."""
        obs_good = PlateObservation(
            track_id="TRK_101", camera_id="CAM_01", frame_id=1, timestamp=0.1,
            raw_text="TN09AB1234", normalized_text="TN09AB1234", ocr_confidence=0.92,
            validation_confidence=0.95, overall_confidence=0.93, status=PlateValidationStatus.VALID
        )
        obs_noisy = PlateObservation(
            track_id="TRK_101", camera_id="CAM_01", frame_id=3, timestamp=0.3,
            raw_text="TN09A81234", normalized_text="TN09A81234", ocr_confidence=0.60,
            validation_confidence=0.65, overall_confidence=0.62, status=PlateValidationStatus.UNCERTAIN
        )

        for _ in range(3):
            self.fusion_engine.process_observation("TRK_101", "CAM_01", obs_good)
        fused = self.fusion_engine.process_observation("TRK_101", "CAM_01", obs_noisy)

        self.assertEqual(fused.best_plate_number, "TN09AB1234")

    def test_11_unknown_plate_handling(self):
        """Test 4.11: Verify unrecogized or uncertain plate returns clean evidence without fabrication."""
        mgr = PlateTrackerAssociationManager(ocr_engine=TestOCREngine(default_plate="", default_conf=0.0))
        track = TrackState(
            track_id="TRK_999", camera_id="CAM_01", vehicle_type="truck",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.8, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.TENTATIVE
        )
        frame_mock = {"width": 1920, "height": 1080}
        evidence = mgr.process_track_frame(track, frame_mock, 1.0, 1)

        self.assertEqual(evidence.track_id, "TRK_999")
        # Associated plate should be None or unconfirmed
        if evidence.associated_plate:
            self.assertFalse(evidence.associated_plate.confirmed)

    def test_12_schema_serialization(self):
        """Test 4.12: Verify Pydantic schema serialization for VehicleIdentityEvidence."""
        evidence = VehicleIdentityEvidence(
            track_id="TRK_101",
            camera_id="CAM_01",
            vehicle_type="car",
            last_updated_timestamp=1.0,
        )
        data = evidence.model_dump()
        self.assertEqual(data["track_id"], "TRK_101")
        self.assertEqual(data["vehicle_type"], "car")
        dump_json = json.dumps(data)
        self.assertIn("TRK_101", dump_json)

    def test_13_stage3_to_stage4_integration(self):
        """Test 4.13: Verify Stage 3 TrackState feeds directly into Stage 4 ANPR pipeline."""
        track = TrackState(
            track_id="TRK_202", camera_id="CAM_01", vehicle_type="bus",
            current_bbox=BoundingBoxXYXY(x1=200, y1=200, x2=500, y2=400),
            current_center=(350, 300), confidence=0.88, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True,
            raw_vehicle_reference="V_202_KA01MH9999"
        )
        frame_mock = {"width": 1920, "height": 1080}

        for f in range(1, 3):
            evidence = self.association_manager.process_track_frame(track, frame_mock, f * 0.1, f)

        self.assertEqual(evidence.track_id, "TRK_202")
        self.assertIsNotNone(evidence.associated_plate)

    def test_14_real_frame_processing(self):
        """Test 4.14: Verify processing with numpy array image frame."""
        try:
            import numpy as np
            real_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        except ImportError:
            real_frame = {"width": 1920, "height": 1080}

        track = TrackState(
            track_id="TRK_303", camera_id="CAM_01", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=300, y2=250),
            current_center=(200, 175), confidence=0.91, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True,
            raw_vehicle_reference="V_303_MH12DE1234"
        )
        evidence = self.association_manager.process_track_frame(track, real_frame, 1.0, 1)
        self.assertIsNotNone(evidence)

    def test_15_deterministic_reproducibility(self):
        """Test 4.15: Verify deterministic ALPR pipeline reproducibility."""
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

    def test_16_ocr_output_reaches_plate_observation(self):
        """Test 4.16: Verify valid OCR output reaches PlateObservation and fusion layer."""
        mgr = PlateTrackerAssociationManager(ocr_engine=TestOCREngine(default_plate="TN10AB1234", default_conf=0.92))
        track = TrackState(
            track_id="TRK_404", camera_id="CAM_01", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True
        )
        frame_mock = {"width": 1920, "height": 1080}
        evidence = mgr.process_track_frame(track, frame_mock, 1.0, 1)

        self.assertIsNotNone(evidence.associated_plate)
        self.assertEqual(evidence.associated_plate.best_plate_number, "TN10AB1234")
        self.assertGreater(mgr.ocr_scheduling_stats["valid_ocr_observations"], 0)
        self.assertGreater(mgr.ocr_scheduling_stats["plate_observations_created"], 0)

    def test_17_invalid_ocr_is_rejected(self):
        """Test 4.17: Verify invalid OCR text (e.g. INVALID123) is rejected by format validation."""
        status, conf = IndianPlateValidator.validate_format("INVALID123")
        self.assertEqual(status, PlateValidationStatus.INVALID)

        mgr = PlateTrackerAssociationManager(ocr_engine=TestOCREngine(default_plate="INVALID123", default_conf=0.90))
        track = TrackState(
            track_id="TRK_505", camera_id="CAM_01", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.TENTATIVE
        )
        frame_mock = {"width": 1920, "height": 1080}
        evidence = mgr.process_track_frame(track, frame_mock, 1.0, 1)

        self.assertEqual(mgr.ocr_scheduling_stats["validation_rejected"], 1)
        self.assertEqual(mgr.ocr_scheduling_stats["valid_ocr_observations"], 0)
        self.assertEqual(mgr.ocr_scheduling_stats["plate_observations_created"], 0)

    def test_18_realistic_ocr_strings_and_confusions(self):
        """Test 4.18: Verify normalization on realistic Indian OCR strings and position-aware confusions."""
        # 1. Standard spaced / hyphenated strings
        self.assertEqual(TextNormalizer.normalize("TN10AB1234"), "TN10AB1234")
        self.assertEqual(TextNormalizer.normalize("TN 10 AB 1234"), "TN10AB1234")
        self.assertEqual(TextNormalizer.normalize("TN-10-AB-1234"), "TN10AB1234")

        # 2. HSRP IND prefix stripping
        self.assertEqual(TextNormalizer.normalize("IND TN10AB1234"), "TN10AB1234")
        self.assertEqual(TextNormalizer.normalize("IND-TN10AB1234"), "TN10AB1234")

        # 3. Position-aware character confusions
        self.assertEqual(TextNormalizer.normalize("TN10A81234"), "TN10AB1234")  # 8 in series -> B
        self.assertEqual(TextNormalizer.normalize("TN10AB123A"), "TN10AB1234")  # A in number -> 4

    def test_19_diagnostic_counter_accounting_invariants(self):
        """Test 4.19: Verify diagnostic counter accounting equations hold strictly."""
        mgr = PlateTrackerAssociationManager(ocr_engine=TestOCREngine(default_plate="TN10AB1234", default_conf=0.92))
        track = TrackState(
            track_id="TRK_606", camera_id="CAM_01", vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150), confidence=0.9, first_seen_timestamp=0.0,
            last_seen_timestamp=1.0, status=TrackStatus.CONFIRMED, confirmed=True
        )
        frame_mock = {"width": 1920, "height": 1080}
        mgr.process_track_frame(track, frame_mock, 1.0, 1)

        stats = mgr.ocr_scheduling_stats
        self.assertEqual(stats["ocr_calls"], stats["ocr_returned_empty"] + stats["ocr_returned_text"])
        self.assertEqual(stats["ocr_returned_text"], stats["normalization_rejected"] + stats["validation_rejected"] + stats["validation_accepted"])
        self.assertEqual(stats["validation_accepted"], stats["valid_ocr_observations"])
        self.assertEqual(stats["valid_ocr_observations"], stats["plate_observations_created"])



class TestBoundingBoxSanitization(unittest.TestCase):
    """Regression test suite for extract_vehicle_crop bounding box sanitization."""

    def setUp(self):
        self.detector = PlateDetector()
        try:
            import numpy as np
            self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self.has_numpy = True
        except ImportError:
            self.frame = {"width": 1920, "height": 1080}
            self.has_numpy = False
        self.img_w = 1920
        self.img_h = 1080

    def _make_bbox(self, x1, y1, x2, y2):
        """Construct a BoundingBoxXYXY with raw coordinates, bypassing validation for test setup."""
        from pydantic import BaseModel
        # We pass using keyword args and force valid pydantic construction for setup only
        # For invalid-coordinate tests, use the raw float values directly on a mock
        class RawBBox:
            def __init__(self, x1, y1, x2, y2):
                self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2
        return RawBBox(x1, y1, x2, y2)

    def test_A_valid_bbox(self):
        """Test A: Valid bbox [100,100,500,400] produces a valid vehicle crop."""
        bbox = BoundingBoxXYXY(x1=100.0, y1=100.0, x2=500.0, y2=400.0)
        success, crop, validated_bbox = self.detector.extract_vehicle_crop(self.frame, bbox, self.img_w, self.img_h)
        self.assertTrue(success)
        self.assertIsNotNone(crop)
        self.assertEqual(float(validated_bbox.x1), 100.0)
        self.assertEqual(float(validated_bbox.y1), 100.0)

    def test_B_negative_coordinates_clamped(self):
        """Test B: Negative coordinates [-50,-30,500,400] are clamped to [0,0,500,400] and crop succeeds."""
        # Construct via raw object to bypass Pydantic validation on the input side
        raw = self._make_bbox(-50.0, -30.0, 500.0, 400.0)
        success, crop, validated_bbox = self.detector.extract_vehicle_crop(self.frame, raw, self.img_w, self.img_h)
        # After clamping: x1=0, y1=0, x2=500, y2=400 → valid crop
        self.assertTrue(success)
        self.assertIsNotNone(crop)
        self.assertGreaterEqual(float(validated_bbox.x1), 0.0)
        self.assertGreaterEqual(float(validated_bbox.y1), 0.0)

    def test_C_completely_outside_image(self):
        """Test C: BBox [-500,-500,-100,-100] is entirely outside image → safe rejection, no exception."""
        raw = self._make_bbox(-500.0, -500.0, -100.0, -100.0)
        success, crop, _ = self.detector.extract_vehicle_crop(self.frame, raw, self.img_w, self.img_h)
        self.assertFalse(success)
        self.assertIsNone(crop)

    def test_D_reversed_coordinates(self):
        """Test D: Reversed coordinates [500,400,100,100] → after clamping x2<=x1, safe rejection."""
        raw = self._make_bbox(500.0, 400.0, 100.0, 100.0)
        success, crop, _ = self.detector.extract_vehicle_crop(self.frame, raw, self.img_w, self.img_h)
        # x2=100 <= x1=500 after clamping. Safe rejection, no crash.
        self.assertFalse(success)
        self.assertIsNone(crop)

    def test_E_regression_y2_negative_2130(self):
        """Test E: Exact regression — y2=-2130 → no Pydantic crash, no fake crop, safe rejection."""
        raw = self._make_bbox(100.0, 50.0, 300.0, -2130.0)
        try:
            success, crop, _ = self.detector.extract_vehicle_crop(self.frame, raw, self.img_w, self.img_h)
        except Exception as exc:
            self.fail(f"extract_vehicle_crop raised an exception: {exc}")
        self.assertFalse(success)
        self.assertIsNone(crop)

    def test_F_nan_infinity_coordinates(self):
        """Test F: NaN and Infinity coordinates → safe rejection, no exception."""
        import math
        nan_raw = self._make_bbox(float("nan"), 100.0, 300.0, 400.0)
        inf_raw = self._make_bbox(100.0, float("inf"), 300.0, float("-inf"))
        for raw in [nan_raw, inf_raw]:
            try:
                success, crop, _ = self.detector.extract_vehicle_crop(self.frame, raw, self.img_w, self.img_h)
            except Exception as exc:
                self.fail(f"extract_vehicle_crop raised an exception for inf/nan: {exc}")
            self.assertFalse(success)
            self.assertIsNone(crop)


class TestFallbackFalsePositiveRejection(unittest.TestCase):
    """
    Regression test: a rejected fallback strip on a real NumPy vehicle crop
    must NOT produce a synthetic candidate — candidates must be empty.
    """

    def setUp(self):
        self.detector = PlateDetector()

    def test_G_all_black_numpy_crop_returns_no_candidates(self):
        """
        Test G: An all-black NumPy vehicle crop (mean=0, std=0) fails ALL
        fallback guards (brightness < 20, contrast < 10) and must return
        an empty candidate list — NOT a synthetic dict crop.
        """
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        # All-black 200×120 crop: fails brightness guard (mean=0 < 20)
        black_crop = np.zeros((120, 200, 3), dtype=np.uint8)
        v_bbox = BoundingBoxXYXY(x1=0.0, y1=0.0, x2=200.0, y2=120.0)

        candidates = self.detector.extract_plate_candidates(black_crop, v_bbox)

        self.assertEqual(
            len(candidates), 0,
            f"All-black NumPy crop must yield 0 candidates, got {len(candidates)}. "
            "Rejected fallback must not produce a synthetic dict candidate."
        )
        # Confirm the rejection counter incremented
        self.assertGreater(
            self.detector.fp_rejection_stats["fallback_dark_rejected"], 0,
            "fallback_dark_rejected counter must increment for all-black crop."
        )

    def test_H_uniform_bright_numpy_crop_returns_no_candidates(self):
        """
        Test H: An all-white (uniform, low-contrast) NumPy crop fails the
        contrast guard (std < 10) and must also return empty candidates.
        """
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        # All-white 200×120 crop: mean=255 (high brightness, passes guard 2),
        # but std=0 (fails contrast guard 3)
        white_crop = np.full((120, 200, 3), 255, dtype=np.uint8)
        v_bbox = BoundingBoxXYXY(x1=0.0, y1=0.0, x2=200.0, y2=120.0)

        candidates = self.detector.extract_plate_candidates(white_crop, v_bbox)

        self.assertEqual(
            len(candidates), 0,
            "Uniform-white NumPy crop must yield 0 candidates (fails contrast guard)."
        )
        self.assertGreater(
            self.detector.fp_rejection_stats["fallback_low_contrast_rejected"], 0,
            "fallback_low_contrast_rejected counter must increment for uniform-white crop."
        )

    def test_I_dict_mock_crop_still_returns_synthetic_candidate(self):
        """
        Test I: A dict-based mock input (non-NumPy) must still return a
        synthetic candidate, preserving simulation/test-path compatibility.
        """
        mock_crop = {"type": "vehicle_crop"}
        v_bbox = BoundingBoxXYXY(x1=100.0, y1=100.0, x2=300.0, y2=250.0)

        candidates = self.detector.extract_plate_candidates(mock_crop, v_bbox)

        self.assertGreater(
            len(candidates), 0,
            "Dict-based mock input must still produce a synthetic candidate for simulation paths."
        )
        plate_crop, plate_bbox = candidates[0]
        self.assertIsInstance(plate_crop, dict, "Synthetic candidate crop must be a dict for mock inputs.")


class TestYoloAndFallbackGeometricFiltering(unittest.TestCase):
    """
    Tests rigorous geometric constraints (aspect ratio, Area, width/height ratio)
    for both YOLO predictions and Fallback crops on real numpy frames.
    """
    def setUp(self):
        self.detector = PlateDetector()

    def _mock_yolo(self, bbox_xyxy):
        class MockBox:
            def __init__(self, box):
                class ConfMock:
                    def item(self): return 0.9
                self.conf = [ConfMock()]
                self.xyxy = [box]
                
        class MockResult:
            def __init__(self, boxes_list):
                self.boxes = [MockBox(b) for b in boxes_list]

        self.detector.plate_model = lambda img, conf, verbose: [MockResult([bbox_xyxy])]
        self.detector._is_yolo_available = True

    def test_A_large_bright_yolo_rejected(self):
        # A: Large bright vehicle-region YOLO candidate → rejected (cw > 60% v_w)
        import numpy as np
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        self._mock_yolo([0, 50, 150, 80])  # Width 150 > (200*0.6=120). Rejected.
        v_bbox = BoundingBoxXYXY(x1=0.0, y1=0.0, x2=200.0, y2=100.0)
        c = self.detector.extract_plate_candidates(img, v_bbox)
        self.assertEqual(len(c), 0, "Large bright YOLO candidate should be rejected.")
        self.assertGreater(self.detector.fp_rejection_stats["yolo_geometry_rejected"], 0)

    def test_B_large_wide_rear_yolo_rejected(self):
        # B: Large wide vehicle-rear candidate → rejected (area > 20%)
        import numpy as np
        img = np.full((200, 200, 3), 200, dtype=np.uint8)
        # cw=120 (<=60%), ch=80 (<=40%), area=9600 > 20% of 40,000 (8000). Rejected.
        self._mock_yolo([40, 60, 160, 140]) 
        v_bbox = BoundingBoxXYXY(x1=0.0, y1=0.0, x2=200.0, y2=200.0)
        c = self.detector.extract_plate_candidates(img, v_bbox)
        self.assertEqual(len(c), 0, "Excessive area candidate should be rejected.")
        self.assertGreater(self.detector.fp_rejection_stats["yolo_geometry_rejected"], 0)

    def test_C_tiny_yolo_candidate_rejected(self):
        # C: Tiny candidate → rejected (cw < 40 or ch < 12)
        import numpy as np
        img = np.full((100, 100, 3), 150, dtype=np.uint8)
        self._mock_yolo([50, 50, 70, 60])  # cw=20, ch=10. Rejected.
        v_bbox = BoundingBoxXYXY(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        c = self.detector.extract_plate_candidates(img, v_bbox)
        self.assertEqual(len(c), 0, "Tiny YOLO candidate should be rejected.")
        self.assertGreater(self.detector.fp_rejection_stats["yolo_geometry_rejected"], 0)

    def test_D_implausible_aspect_yolo_rejected(self):
        # D: Implausible aspect-ratio YOLO candidate → rejected (< 0.75 or > 6.0)
        import numpy as np
        img = np.full((200, 200, 3), 150, dtype=np.uint8)
        
        # 1. Square plate: cw=60, ch=60. Aspect = 1.0 (>= 0.75). Should be ACCEPTED now.
        self._mock_yolo([50, 100, 110, 160])
        v_bbox = BoundingBoxXYXY(x1=0.0, y1=0.0, x2=200.0, y2=200.0)
        c_square = self.detector.extract_plate_candidates(img, v_bbox)
        self.assertEqual(len(c_square), 1, "Square YOLO candidate should be accepted (aspect >= 0.75).")
        
        # 2. Too tall: cw=20, ch=40. Aspect = 0.5 (< 0.75). Rejected.
        self._mock_yolo([50, 100, 70, 140])
        c_tall = self.detector.extract_plate_candidates(img, v_bbox)
        self.assertEqual(len(c_tall), 0, "Too tall candidate should be rejected (aspect < 0.75).")
        
        # 3. Too wide: cw=100, ch=12. Aspect = 8.33 (> 6.0). Rejected.
        self._mock_yolo([50, 100, 150, 112])
        c_wide = self.detector.extract_plate_candidates(img, v_bbox)
        self.assertEqual(len(c_wide), 0, "Too wide YOLO candidate should be rejected (aspect > 6.0).")
        
        self.assertGreater(self.detector.fp_rejection_stats["yolo_geometry_rejected"], 0)

    def test_E_low_quality_yolo_rejected(self):
        # E: Low-quality/dark candidate → rejected via fallback path.
        # With v_h=100, v_w=200: strip cy1=65,cy2=88 → fc_h=23, fc_w=140.
        # aspect=140/23=6.08 > 6.0, so geometry guard fires before brightness check.
        import numpy as np
        img = np.zeros((100, 200, 3), dtype=np.uint8)  # Black image
        v_bbox = BoundingBoxXYXY(x1=0.0, y1=0.0, x2=200.0, y2=100.0)
        # Disable YOLO to test fallback
        self.detector._is_yolo_available = False
        c = self.detector.extract_plate_candidates(img, v_bbox)
        self.assertEqual(len(c), 0, "Dark fallback should be rejected.")
        self.assertGreater(self.detector.fp_rejection_stats["fallback_geometry_rejected"], 0)

    def test_F_valid_plate_yolo_accepted(self):
        # F: Valid plate-shaped YOLO candidate → accepted
        import numpy as np
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        # cw=100 (50%), ch=30 (15%), Aspect ~3.33. Area ~ 7.5%. Y center ~115 (lower). Passes all.
        # But wait, padding adds 8% / 12%. So 100 -> 108, 30 -> 36. Aspect = 3.0. Still passes.
        self._mock_yolo([50, 100, 150, 130])
        v_bbox = BoundingBoxXYXY(x1=0.0, y1=0.0, x2=200.0, y2=200.0)
        c = self.detector.extract_plate_candidates(img, v_bbox)
        self.assertEqual(len(c), 1, "Valid plate should be accepted by YOLO.")
        self.assertGreater(self.detector.fp_rejection_stats["yolo_accepted"], 0)
        
    def test_M_large_fallback_rejected(self):
        # M: Tall pillar → fallback rejected (uniform/low-contrast image).
        # v_w=100, v_h=300: strip cy1=195,cy2=264 → fc_h=69, fc_w=70, aspect≈1.01.
        # Geometry checks pass, but np.full(...,150) has std=0 < 10 → contrast guard fires.
        import numpy as np
        img = np.full((300, 100, 3), 150, dtype=np.uint8)
        v_bbox = BoundingBoxXYXY(x1=0.0, y1=0.0, x2=100.0, y2=300.0)
        self.detector._is_yolo_available = False
        c = self.detector.extract_plate_candidates(img, v_bbox)
        self.assertEqual(len(c), 0, "Fallback on uniform pillar image must be rejected.")
        self.assertGreater(self.detector.fp_rejection_stats["fallback_low_contrast_rejected"], 0)

    def test_N_valid_plate_fallback_accepted(self):
        # N: Valid plate-shaped fallback candidate → accepted
        # Vehicle box is well proportioned for fallback: v_w=200, v_h=150.
        # New strip: cy1=int(150*0.65)=97, cy2=int(150*0.88)=132, cx1=int(200*0.15)=30, cx2=int(200*0.85)=170
        # fc_h=35, fc_w=140, aspect=4.0. Good contrast within this region.
        import numpy as np
        img = np.zeros((150, 200, 3), dtype=np.uint8)
        # Paint noisy pixels that overlap new strip [97:132, 30:170] to pass contrast/brightness
        # cy1=97, cy2=132, cx1=30, cx2=170
        img[97:132, 30:170] = np.random.randint(50, 255, (35, 140, 3), dtype=np.uint8)
        v_bbox = BoundingBoxXYXY(x1=0.0, y1=0.0, x2=200.0, y2=150.0)
        self.detector._is_yolo_available = False
        c = self.detector.extract_plate_candidates(img, v_bbox)
        self.assertEqual(len(c), 1, "Valid high-contrast fallback strip must be accepted.")
        self.assertGreater(self.detector.fp_rejection_stats["fallback_accepted"], 0)
        
    def test_O_valid_aspect_excessive_area_rejected(self):
        # O: Candidate with valid aspect ratio but excessive vehicle-relative area → rejected
        # We test YOLO since fallback is hardcoded to 19.8% (< 35% check).
        import numpy as np
        img = np.full((100, 100, 3), 150, dtype=np.uint8)
        # cw=60, ch=40. Aspect = 1.5. Area = 2400. Vehicle Area = 10000. 2400 > 2000 (20%). Rejected.
        # Plus padding adds more.
        self._mock_yolo([20, 30, 80, 70]) 
        v_bbox = BoundingBoxXYXY(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        c = self.detector.extract_plate_candidates(img, v_bbox)
        self.assertEqual(len(c), 0, "Valid aspect but excessive area YOLO candidate should be rejected.")
        self.assertGreater(self.detector.fp_rejection_stats["yolo_geometry_rejected"], 0)


if __name__ == "__main__":
    unittest.main()
