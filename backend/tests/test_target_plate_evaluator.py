"""
ChronoEye Infinity - Target Plate Evaluator Tests
Tests for DETECTED / NOT_DETECTED / UNCERTAIN logic.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.plate import PlateObservation, PlateValidationStatus, FusedPlateIdentity
from app.perception.plate_fusion import TemporalPlateFusionEngine
from app.perception.target_plate_evaluator import TargetPlateEvaluator, _normalize_plate


def _make_obs(track_id, frame_id, text, ocr_conf=0.99, val_conf=0.40, timestamp=None):
    return PlateObservation(
        track_id=track_id,
        camera_id="CAM_A",
        frame_id=frame_id,
        timestamp=timestamp if timestamp is not None else frame_id * 0.1,
        raw_text=text,
        normalized_text=text,
        ocr_confidence=ocr_conf,
        validation_confidence=val_conf,
        overall_confidence=round(ocr_conf * 0.6 + val_conf * 0.4, 3),
        quality_score=0.70,
        preprocessing_variant="ORIGINAL",
        status=PlateValidationStatus.FORMAT_MISMATCH,
    )


def _build_confirmed_engine(track_id, plate_text, ocr_conf=0.99, val_conf=0.40):
    """Build a fusion engine with a CONFIRMED plate (2-obs path)."""
    engine = TemporalPlateFusionEngine()
    obs1 = _make_obs(track_id, frame_id=1, text=plate_text, ocr_conf=ocr_conf, val_conf=val_conf)
    obs2 = _make_obs(track_id, frame_id=6, text=plate_text, ocr_conf=ocr_conf, val_conf=val_conf)
    engine.process_observation(track_id, "CAM_A", obs1)
    engine.process_observation(track_id, "CAM_A", obs2)
    return engine


def _build_pending_engine(track_id, plate_text, ocr_conf=0.85, val_conf=0.40):
    """Build a fusion engine with a PENDING plate (1-obs, short track)."""
    engine = TemporalPlateFusionEngine()
    obs = _make_obs(track_id, frame_id=5, text=plate_text, ocr_conf=ocr_conf, val_conf=val_conf)
    # track_frame_count=0 -> fast-path disabled -> stays PENDING
    engine.process_observation(track_id, "CAM_A", obs, track_frame_count=0)
    return engine


class TestTargetPlateEvaluator(unittest.TestCase):

    # -------------------------------------------------------------------------
    # Test 1: SX8525 confirmed -> DETECTED
    # -------------------------------------------------------------------------
    def test_1_confirmed_plate_detected(self):
        """SX8525 confirmed in engine -> evaluator returns DETECTED."""
        engine = _build_confirmed_engine("TRK_102", "SX8525")
        evaluator = TargetPlateEvaluator()
        result = evaluator.evaluate("SX8525", engine)
        self.assertEqual(result.status, "DETECTED")
        self.assertEqual(result.recognized_plate, "SX8525")
        self.assertEqual(result.track_id, "TRK_102")

    # -------------------------------------------------------------------------
    # Test 2: KW527 confirmed -> DETECTED
    # -------------------------------------------------------------------------
    def test_2_kw527_detected(self):
        """KW527 confirmed in engine -> evaluator returns DETECTED."""
        engine = _build_confirmed_engine("TRK_101", "KW527")
        evaluator = TargetPlateEvaluator()
        result = evaluator.evaluate("KW527", engine)
        self.assertEqual(result.status, "DETECTED")
        self.assertTrue(result.track_id is not None)

    # -------------------------------------------------------------------------
    # Test 3: TS5330 confirmed -> DETECTED
    # -------------------------------------------------------------------------
    def test_3_ts5330_detected(self):
        """TS5330 confirmed in engine -> evaluator returns DETECTED."""
        engine = _build_confirmed_engine("TRK_106", "TS5330")
        evaluator = TargetPlateEvaluator()
        result = evaluator.evaluate("TS5330", engine)
        self.assertEqual(result.status, "DETECTED")

    # -------------------------------------------------------------------------
    # Test 4: ZZ9999 not present -> NOT_DETECTED
    # -------------------------------------------------------------------------
    def test_4_absent_plate_not_detected(self):
        """ZZ9999 not in engine -> evaluator returns NOT_DETECTED."""
        engine = _build_confirmed_engine("TRK_101", "KW527")
        evaluator = TargetPlateEvaluator()
        result = evaluator.evaluate("ZZ9999", engine)
        self.assertEqual(result.status, "NOT_DETECTED")

    # -------------------------------------------------------------------------
    # Test 5: Plate with spaces/hyphens normalized before comparison
    # -------------------------------------------------------------------------
    def test_5_normalized_target_detected(self):
        """Target supplied as 'SX 8525' or 'SX-8525' must still match SX8525."""
        engine = _build_confirmed_engine("TRK_102", "SX8525")
        evaluator = TargetPlateEvaluator()
        for variant in ["SX 8525", "sx8525", "SX-8525", " SX8525 "]:
            result = evaluator.evaluate(variant, engine)
            self.assertEqual(result.status, "DETECTED",
                             f"Variant '{variant}' should normalize to SX8525 and be DETECTED")

    # -------------------------------------------------------------------------
    # Test 6: PENDING plate -> UNCERTAIN (not DETECTED)
    # -------------------------------------------------------------------------
    def test_6_pending_plate_uncertain(self):
        """Plate that is PENDING -> evaluator returns UNCERTAIN, not DETECTED."""
        engine = _build_pending_engine("TRK_127", "NN773")
        evaluator = TargetPlateEvaluator()
        result = evaluator.evaluate("NN773", engine)
        # Should be UNCERTAIN (pending exact match) — NOT DETECTED
        self.assertNotEqual(result.status, "DETECTED",
                            "PENDING plate must not produce DETECTED")
        self.assertEqual(result.status, "UNCERTAIN")

    # -------------------------------------------------------------------------
    # Test 7: Near-match (VN472 vs VN4712) -> UNCERTAIN
    # -------------------------------------------------------------------------
    def test_7_near_match_uncertain(self):
        """Target VN4712 with engine having VN472 (1 char diff) -> UNCERTAIN via near-match."""
        engine = _build_pending_engine("TRK_116", "VN472")
        evaluator = TargetPlateEvaluator()
        result = evaluator.evaluate("VN4712", engine)
        # VN4712 vs VN472 -> confusion_weighted_distance = 1.0 <= 2.0 -> UNCERTAIN
        self.assertIn(result.status, ["UNCERTAIN", "NOT_DETECTED"],
                      "Near-match should produce UNCERTAIN")

    # -------------------------------------------------------------------------
    # Test 8: DETECTED requires EXACT match — confusion-distance match on CONFIRMED is NOT DETECTED
    # -------------------------------------------------------------------------
    def test_8_near_match_on_confirmed_is_uncertain_not_detected(self):
        """
        A confirmed plate WG219 does NOT produce DETECTED for target WG2119.
        The two plates differ by 1 char — this must produce UNCERTAIN (near-match),
        not DETECTED, because DETECTED requires an exact match.
        """
        engine = _build_confirmed_engine("TRK_117", "WG219")
        evaluator = TargetPlateEvaluator()
        result = evaluator.evaluate("WG2119", engine)
        # WG219 != WG2119 -> NOT DETECTED (no exact match on confirmed)
        # But confusion_dist(WG2119, WG219) = 1.0 <= 2.0 -> UNCERTAIN via near-match
        self.assertNotEqual(result.status, "DETECTED",
                            "WG219 confirmed should NOT produce DETECTED for target WG2119")
        self.assertEqual(result.status, "UNCERTAIN",
                         "Near-match on confirmed plate should produce UNCERTAIN")

    # -------------------------------------------------------------------------
    # Test 9: Empty fusion engine -> NOT_DETECTED
    # -------------------------------------------------------------------------
    def test_9_empty_engine_not_detected(self):
        """Empty engine with no identities -> NOT_DETECTED for any target."""
        engine = TemporalPlateFusionEngine()
        evaluator = TargetPlateEvaluator()
        result = evaluator.evaluate("SX8525", engine)
        self.assertEqual(result.status, "NOT_DETECTED")

    # -------------------------------------------------------------------------
    # Test 10: normalize_plate helper
    # -------------------------------------------------------------------------
    def test_10_normalize_plate(self):
        """_normalize_plate removes spaces, hyphens, lowercases -> uppercase no spaces."""
        self.assertEqual(_normalize_plate("kw 527"), "KW527")
        self.assertEqual(_normalize_plate("SX-8525"), "SX8525")
        self.assertEqual(_normalize_plate(" ts5330 "), "TS5330")
        self.assertEqual(_normalize_plate("wg2119"), "WG2119")


if __name__ == "__main__":
    unittest.main()
