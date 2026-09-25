"""
ChronoEye Infinity - Stacked / Two-Line License Plate Recognition Unit Tests
Verifies aspect-ratio-based stacked plate detection, upper/lower splitting,
combination logic (SX8525, TS5330), single-line pass-through, and incomplete split guardrails.
"""

import os
import sys
import unittest
import numpy as np

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.plate_association import PlateTrackerAssociationManager
from app.schemas.tracking import TrackState, TrackStatus
from app.schemas.detection import BoundingBoxXYXY


class MockSplitOCREngine:
    """Mock OCR engine that returns configurable responses for testing."""
    def __init__(self, top_res=("SX", "SX", 0.95, "GRAYSCALE"), bot_res=("8525", "8525", 0.98, "GRAYSCALE")):
        self.top_res = top_res
        self.bot_res = bot_res
        self.call_count = 0
        self.calls = []

    def recognize_text(self, variants):
        self.call_count += 1
        self.calls.append(variants)
        # Alternate or return based on call count
        if self.call_count % 2 == 1:
            return self.top_res
        else:
            return self.bot_res


class MockArtifactAwareOCREngine:
    """
    Mock OCR engine simulating artifact removal:
    - If the top crop is untrimmed (e.g. width >= 220 after target resize), peripheral rivets cause
      leading character hallucinations ('PSX' or 'CTS').
    - If the top crop is margin-trimmed (8% inset on left and right), peripheral rivets
      are excluded, correctly recovering central letters ('SX' or 'TS').
    """
    def __init__(self, artifact_text: str = "PSX", clean_text: str = "SX", bot_text: str = "8525"):
        self.artifact_text = artifact_text
        self.clean_text = clean_text
        self.bot_text = bot_text
        self.call_count = 0

    def recognize_text(self, variants):
        self.call_count += 1
        if self.call_count % 2 == 1:
            # Top crop variants
            is_trimmed = False
            if variants and isinstance(variants[0], dict) and "image" in variants[0]:
                img = variants[0]["image"]
                # Untrimmed 140/52 -> target_w + border = 235 px; trimmed 118/52 -> 202 px
                if img.shape[1] < 220:
                    is_trimmed = True
            elif variants and isinstance(variants[0], dict) and "raw" in variants[0]:
                raw = variants[0]["raw"]
                if hasattr(raw, "shape") and raw.shape[1] < 140:
                    is_trimmed = True

            if is_trimmed:
                return (self.clean_text, self.clean_text, 0.98, "GRAYSCALE")
            else:
                return (self.artifact_text, self.artifact_text, 0.85, "GRAYSCALE")
        else:
            return (self.bot_text, self.bot_text, 0.99, "GRAYSCALE")


class TestStackedPlateRecognition(unittest.TestCase):
    def setUp(self):
        self.preprocessor = PlatePreprocessor()

    def test_stacked_plate_geometry_detection(self):
        """Test aspect-ratio detection of stacked plates vs standard single-line plates."""
        # Stacked plate: width=140, height=100 -> aspect=1.40 (<= 2.2)
        stacked_crop = np.zeros((100, 140, 3), dtype=np.uint8)
        self.assertTrue(self.preprocessor.is_likely_stacked_plate(stacked_crop))

        # Standard single-line plate: width=300, height=80 -> aspect=3.75 (> 2.2)
        single_line_crop = np.zeros((80, 300, 3), dtype=np.uint8)
        self.assertFalse(self.preprocessor.is_likely_stacked_plate(single_line_crop))

        # Edge cases: None, empty array, zero dimensions
        self.assertFalse(self.preprocessor.is_likely_stacked_plate(None))
        self.assertFalse(self.preprocessor.is_likely_stacked_plate(np.zeros((0, 0, 3), dtype=np.uint8)))

    def test_stacked_plate_splitting(self):
        """Test upper and lower vertical splitting with midline overlap and upper margin trimming."""
        crop = np.zeros((100, 140, 3), dtype=np.uint8)
        top_crop, bot_crop = self.preprocessor.split_stacked_plate(crop, overlap_ratio=0.02)
        
        self.assertIsNotNone(top_crop)
        self.assertIsNotNone(bot_crop)
        # Top crop should cover 0 to 52% of height
        self.assertEqual(top_crop.shape[0], 52)
        # Top crop width is trimmed by 8% on left (11px) and 8% on right (11px): 140 - 22 = 118
        self.assertEqual(top_crop.shape[1], 118)
        # Bottom crop should cover 48% to 100% of height (52 rows) and remain full width (140)
        self.assertEqual(bot_crop.shape[0], 52)
        self.assertEqual(bot_crop.shape[1], 140)

        # When upper_margin_trim=0.0, top crop retains full width
        top_untrimmed, bot_untrimmed = self.preprocessor.split_stacked_plate(crop, upper_margin_trim=0.0)
        self.assertEqual(top_untrimmed.shape[1], 140)
        self.assertEqual(bot_untrimmed.shape[1], 140)

        # Invalid or tiny crops return (None, None)
        self.assertEqual(self.preprocessor.split_stacked_plate(None), (None, None))
        self.assertEqual(self.preprocessor.split_stacked_plate(np.zeros((5, 5, 3), dtype=np.uint8)), (None, None))

    def test_stacked_plate_sx8525_combination(self):
        """Test stacked plate recognition combines TOP='SX' + BOTTOM='8525' -> 'SX8525'."""
        crop = np.zeros((70, 100, 3), dtype=np.uint8)  # aspect = 1.43
        mock_engine = MockSplitOCREngine(
            top_res=("SX", "SX", 0.96, "GRAYSCALE"),
            bot_res=("8525", "8525", 0.99, "GRAYSCALE"),
        )
        result = self.preprocessor.process_stacked_plate(crop, mock_engine)
        self.assertIsNotNone(result)
        raw_text, norm_text, conf, variant = result
        self.assertEqual(raw_text, "SX8525")
        self.assertEqual(norm_text, "SX8525")
        self.assertGreaterEqual(conf, 0.95)
        self.assertEqual(variant, "STACKED_SPLIT")

    def test_stacked_plate_ts5330_combination(self):
        """Test stacked plate recognition combines TOP='TS' + BOTTOM='5330' -> 'TS5330'."""
        crop = np.zeros((80, 120, 3), dtype=np.uint8)  # aspect = 1.50
        mock_engine = MockSplitOCREngine(
            top_res=("TS", "TS", 0.92, "GRAYSCALE"),
            bot_res=("5330", "5330", 0.98, "GRAYSCALE"),
        )
        result = self.preprocessor.process_stacked_plate(crop, mock_engine)
        self.assertIsNotNone(result)
        raw_text, norm_text, conf, variant = result
        self.assertEqual(raw_text, "TS5330")
        self.assertEqual(norm_text, "TS5330")
        self.assertEqual(variant, "STACKED_SPLIT")

    def test_normal_single_line_plate_continues_through_existing_path(self):
        """Test standard single-line plates bypass stacked processing and use single-line path."""
        # Single-line plate crop: aspect = 3.0 (e.g. KW527)
        single_crop = np.zeros((50, 150, 3), dtype=np.uint8)
        mock_engine = MockSplitOCREngine(
            top_res=("KW527", "KW527", 0.95, "ORIGINAL"),
            bot_res=("KW527", "KW527", 0.95, "ORIGINAL"),
        )
        # process_stacked_plate must return None for aspect > 2.2
        res = self.preprocessor.process_stacked_plate(single_crop, mock_engine)
        self.assertIsNone(res)
        # Mock engine should not have been called because aspect ratio check failed fast
        self.assertEqual(mock_engine.call_count, 0)

    def test_incomplete_empty_split_results_no_fake_plates(self):
        """Test incomplete or empty split halves safely return None without creating fake plates."""
        crop = np.zeros((70, 100, 3), dtype=np.uint8)

        # Case 1: Empty top half
        engine_empty_top = MockSplitOCREngine(
            top_res=("", "", 0.0, "ORIGINAL"),
            bot_res=("8525", "8525", 0.90, "ORIGINAL"),
        )
        self.assertIsNone(self.preprocessor.process_stacked_plate(crop, engine_empty_top))

        # Case 2: Empty bottom half
        engine_empty_bot = MockSplitOCREngine(
            top_res=("SX", "SX", 0.90, "ORIGINAL"),
            bot_res=("", "", 0.0, "ORIGINAL"),
        )
        self.assertIsNone(self.preprocessor.process_stacked_plate(crop, engine_empty_bot))

        # Case 3: Digits only on both halves (no letters -> fake plate noise guard)
        engine_digits_only = MockSplitOCREngine(
            top_res=("12", "12", 0.85, "ORIGINAL"),
            bot_res=("3456", "3456", 0.85, "ORIGINAL"),
        )
        self.assertIsNone(self.preprocessor.process_stacked_plate(crop, engine_digits_only))

        # Case 4: Letters only on both halves (no digits -> fake plate noise guard)
        engine_letters_only = MockSplitOCREngine(
            top_res=("SX", "SX", 0.85, "ORIGINAL"),
            bot_res=("AB", "AB", 0.85, "ORIGINAL"),
        )
        self.assertIsNone(self.preprocessor.process_stacked_plate(crop, engine_letters_only))

    def test_upper_margin_trim_recovers_sx8525_from_psx8525(self):
        """
        Test that upper margin trimming shaves peripheral rivets/screws so OCR
        recovers clean 'SX8525' instead of hallucinating 'PSX8525'.
        """
        crop = np.zeros((100, 140, 3), dtype=np.uint8)  # aspect = 1.40 (stacked)
        mock_engine = MockArtifactAwareOCREngine(
            artifact_text="PSX", clean_text="SX", bot_text="8525"
        )
        # With default upper margin trim (8%), top crop is trimmed -> clean 'SX' + '8525' -> 'SX8525'
        result = self.preprocessor.process_stacked_plate(crop, mock_engine)
        self.assertIsNotNone(result)
        raw_text, norm_text, conf, variant = result
        self.assertEqual(norm_text, "SX8525")
        self.assertEqual(raw_text, "SX8525")
        self.assertGreaterEqual(conf, 0.95)

    def test_upper_margin_trim_recovers_ts5330_from_cts5330(self):
        """
        Test that upper margin trimming shaves peripheral rivets/screws so OCR
        recovers clean 'TS5330' instead of hallucinating 'CTS5330'.
        """
        crop = np.zeros((100, 140, 3), dtype=np.uint8)  # aspect = 1.40 (stacked)
        mock_engine = MockArtifactAwareOCREngine(
            artifact_text="CTS", clean_text="TS", bot_text="5330"
        )
        # With default upper margin trim (8%), top crop is trimmed -> clean 'TS' + '5330' -> 'TS5330'
        result = self.preprocessor.process_stacked_plate(crop, mock_engine)
        self.assertIsNotNone(result)
        raw_text, norm_text, conf, variant = result
        self.assertEqual(norm_text, "TS5330")
        self.assertEqual(raw_text, "TS5330")
        self.assertGreaterEqual(conf, 0.95)

    def test_fusion_resolution_psx8525_vs_sx8525(self):
        """Test temporal fusion resolves conflicting PSX8525 (0.854) vs SX8525 (0.994) to SX8525."""
        from app.perception.plate_fusion import TemporalPlateFusionEngine
        from app.schemas.plate import PlateObservation, PlateValidationStatus

        fusion = TemporalPlateFusionEngine()
        obs1 = PlateObservation(
            track_id="TRK_102", camera_id="CAM_A", frame_id=1, timestamp=0.04,
            raw_text="PSX8525", normalized_text="PSX8525", ocr_confidence=0.854,
            validation_status=PlateValidationStatus.FORMAT_MISMATCH, validation_confidence=0.40,
            quality_score=0.5, preprocessing_variant="STACKED_SPLIT"
        )
        fused1 = fusion.process_observation("TRK_102", "CAM_A", obs1)
        self.assertEqual(fused1.status, "PENDING")

        obs2 = PlateObservation(
            track_id="TRK_102", camera_id="CAM_A", frame_id=8, timestamp=0.32,
            raw_text="SX8525", normalized_text="SX8525", ocr_confidence=0.994,
            validation_status=PlateValidationStatus.FORMAT_MISMATCH, validation_confidence=0.40,
            quality_score=0.5, preprocessing_variant="STACKED_SPLIT"
        )
        fused2 = fusion.process_observation("TRK_102", "CAM_A", obs2)
        self.assertEqual(fused2.status, "CONFIRMED")
        self.assertEqual(fused2.best_plate_number, "SX8525")

    def test_fusion_resolution_ts5330_vs_pts5330(self):
        """Test temporal fusion resolves TS5330 (1.000) and PTS5330 (0.955) to TS5330."""
        from app.perception.plate_fusion import TemporalPlateFusionEngine
        from app.schemas.plate import PlateObservation, PlateValidationStatus

        fusion = TemporalPlateFusionEngine()
        obs1 = PlateObservation(
            track_id="TRK_106", camera_id="CAM_A", frame_id=80, timestamp=2.67,
            raw_text="TS5330", normalized_text="TS5330", ocr_confidence=1.000,
            validation_status=PlateValidationStatus.FORMAT_MISMATCH, validation_confidence=0.40,
            quality_score=0.5, preprocessing_variant="STACKED_SPLIT"
        )
        fused1 = fusion.process_observation("TRK_106", "CAM_A", obs1)
        self.assertEqual(fused1.status, "PENDING")

        obs2 = PlateObservation(
            track_id="TRK_106", camera_id="CAM_A", frame_id=85, timestamp=2.83,
            raw_text="PTS5330", normalized_text="PTS5330", ocr_confidence=0.955,
            validation_status=PlateValidationStatus.FORMAT_MISMATCH, validation_confidence=0.40,
            quality_score=0.5, preprocessing_variant="STACKED_SPLIT"
        )
        fused2 = fusion.process_observation("TRK_106", "CAM_A", obs2)
        self.assertEqual(fused2.status, "CONFIRMED")
        self.assertEqual(fused2.best_plate_number, "TS5330")

    def test_fusion_resolution_cts5330_vs_ts5330(self):
        """Test temporal fusion resolves conflicting CTS5330 (0.820) vs TS5330 (0.990) to TS5330."""
        from app.perception.plate_fusion import TemporalPlateFusionEngine
        from app.schemas.plate import PlateObservation, PlateValidationStatus

        fusion = TemporalPlateFusionEngine()
        obs1 = PlateObservation(
            track_id="TRK_106", camera_id="CAM_A", frame_id=75, timestamp=2.50,
            raw_text="CTS5330", normalized_text="CTS5330", ocr_confidence=0.820,
            validation_status=PlateValidationStatus.FORMAT_MISMATCH, validation_confidence=0.40,
            quality_score=0.5, preprocessing_variant="STACKED_SPLIT"
        )
        fused1 = fusion.process_observation("TRK_106", "CAM_A", obs1)
        self.assertEqual(fused1.status, "PENDING")

        obs2 = PlateObservation(
            track_id="TRK_106", camera_id="CAM_A", frame_id=80, timestamp=2.67,
            raw_text="TS5330", normalized_text="TS5330", ocr_confidence=0.990,
            validation_status=PlateValidationStatus.FORMAT_MISMATCH, validation_confidence=0.40,
            quality_score=0.5, preprocessing_variant="STACKED_SPLIT"
        )
        fused2 = fusion.process_observation("TRK_106", "CAM_A", obs2)
        self.assertEqual(fused2.status, "CONFIRMED")
        self.assertEqual(fused2.best_plate_number, "TS5330")

    def test_invalid_observation_preserves_confirmed_plate(self):
        """Test subsequent invalid/empty observation does not degrade a confirmed plate."""
        from app.perception.plate_fusion import TemporalPlateFusionEngine
        from app.schemas.plate import PlateObservation, PlateValidationStatus

        fusion = TemporalPlateFusionEngine()
        # Confirm a plate with 2 observations
        for fid, ts in [(1, 0.04), (6, 0.24)]:
            obs = PlateObservation(
                track_id="TRK_101", camera_id="CAM_A", frame_id=fid, timestamp=ts,
                raw_text="KW527", normalized_text="KW527", ocr_confidence=0.999,
                validation_status=PlateValidationStatus.FORMAT_MISMATCH, validation_confidence=0.40,
                quality_score=0.5, preprocessing_variant="ORIGINAL_RESIZED"
            )
            fusion.process_observation("TRK_101", "CAM_A", obs)

        confirmed = fusion.get_fused_identity("TRK_101")
        self.assertEqual(confirmed.status, "CONFIRMED")
        self.assertEqual(confirmed.best_plate_number, "KW527")

        # Now pass None / empty observation (representing rejected OCR)
        res = fusion.process_observation("TRK_101", "CAM_A", None)
        self.assertEqual(res.status, "CONFIRMED")
        self.assertEqual(res.best_plate_number, "KW527")

    def test_format_mismatch_validation_acceptance(self):
        """Test FORMAT_MISMATCH validation for readable non-standard plates (SX8525, TS5330, KW527)."""
        from app.perception.ocr_engine import IndianPlateValidator
        from app.schemas.plate import PlateValidationStatus

        for plate in ["SX8525", "TS5330", "KW527"]:
            status, conf = IndianPlateValidator.validate_format(plate)
            self.assertEqual(status, PlateValidationStatus.FORMAT_MISMATCH, f"Failed for {plate}")
            self.assertEqual(conf, 0.40, f"Failed for {plate}")

    def test_invalid_pure_digits_and_pure_letters_rejected(self):
        """Test pure numeric noise ('8525') and pure alphabetic noise ('LR') are rejected as INVALID."""
        from app.perception.ocr_engine import IndianPlateValidator
        from app.schemas.plate import PlateValidationStatus

        for noise in ["8525", "5330", "LR", "ABC", "12"]:
            status, conf = IndianPlateValidator.validate_format(noise)
            self.assertEqual(status, PlateValidationStatus.INVALID, f"Failed for {noise}")
            self.assertLessEqual(conf, 0.10, f"Failed for {noise}")

    def test_text_normalizer_preserves_stacked_plates(self):
        """Test TextNormalizer preserves literal characters of stacked/short plates without mutation."""
        from app.perception.ocr_engine import TextNormalizer

        # For strings with len < 8, TextNormalizer must not apply 10-character position corrections
        self.assertEqual(TextNormalizer.normalize("SX8525"), "SX8525")
        self.assertEqual(TextNormalizer.normalize("TS5330"), "TS5330")
        self.assertEqual(TextNormalizer.normalize("KW527"), "KW527")


if __name__ == "__main__":
    unittest.main()
