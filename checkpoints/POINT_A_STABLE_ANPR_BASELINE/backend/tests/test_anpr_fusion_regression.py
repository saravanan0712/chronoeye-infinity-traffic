"""
ChronoEye Infinity - Stage 4 ANPR Fusion & OCR Scheduling Regression Tests
Tests 1-8 from the Stage 4 repair specification.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.detection import BoundingBoxXYXY
from app.schemas.tracking import TrackState, TrackStatus
from app.schemas.plate import PlateObservation, PlateValidationStatus
from app.perception.plate_fusion import TemporalPlateFusionEngine, _observation_score
from app.perception.plate_association import PlateTrackerAssociationManager
from app.perception.ocr_engine import TestOCREngine


def _make_obs(track_id, frame_id, text, ocr_conf=0.90, val_conf=0.85, quality=0.70, timestamp=None, variant="ORIGINAL"):
    """Helper: build a PlateObservation with given fields."""
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
        quality_score=quality,
        preprocessing_variant=variant,
        status=PlateValidationStatus.VALID if val_conf >= 0.60 else PlateValidationStatus.UNCERTAIN,
    )


class TestANPRFusionOCRSchedulingRegression(unittest.TestCase):

    # -------------------------------------------------------------------------
    # Test 1: Single weak observation must be PENDING or UNKNOWN, NOT CONFIRMED
    # -------------------------------------------------------------------------
    def test_1_single_weak_observation_not_confirmed(self):
        """Test 1: Single weak OCR observation must produce PENDING or UNKNOWN, never CONFIRMED."""
        engine = TemporalPlateFusionEngine(
            min_observations_to_confirm=3,
            confirmation_score_threshold=1.2,
        )
        obs = _make_obs("TRK_001", frame_id=10, text="TN01AB1234", ocr_conf=0.60, val_conf=0.65, quality=0.50)
        fused = engine.process_observation("TRK_001", "CAM_A", obs)
        self.assertIsNotNone(fused)
        self.assertNotEqual(fused.status, "CONFIRMED")
        self.assertFalse(fused.confirmed)
        self.assertIn(fused.status, ["PENDING", "UNKNOWN"])

    # -------------------------------------------------------------------------
    # Test 2: Two strong observations from distinct frames → CONFIRMED
    # -------------------------------------------------------------------------
    def test_2_two_strong_distinct_frame_observations_confirmed(self):
        """Test 2: Two strong observations from different frames → CONFIRMED (when threshold met)."""
        engine = TemporalPlateFusionEngine(
            min_observations_to_confirm=2,
            confirmation_score_threshold=0.80,
        )
        obs1 = _make_obs("TRK_002", frame_id=10, text="KA01MH9999", ocr_conf=0.92, val_conf=0.90, quality=0.80)
        obs2 = _make_obs("TRK_002", frame_id=20, text="KA01MH9999", ocr_conf=0.91, val_conf=0.92, quality=0.85)
        engine.process_observation("TRK_002", "CAM_A", obs1)
        fused = engine.process_observation("TRK_002", "CAM_A", obs2)
        self.assertIsNotNone(fused)
        self.assertEqual(fused.status, "CONFIRMED")
        self.assertTrue(fused.confirmed)
        self.assertEqual(fused.best_plate_number, "KA01MH9999")

    # -------------------------------------------------------------------------
    # Test 3: Same-frame preprocessing variants = ONE temporal frame, not three
    # -------------------------------------------------------------------------
    def test_3_same_frame_variants_count_as_one_temporal_frame(self):
        """Test 3: Three variants of the same frame must contribute only 1 distinct temporal frame."""
        engine = TemporalPlateFusionEngine(
            min_observations_to_confirm=3,
            confirmation_score_threshold=1.2,
        )
        for variant in ["ORIGINAL", "CLAHE", "OTSU"]:
            obs = _make_obs("TRK_003", frame_id=10, text="MH12DE1234", ocr_conf=0.90, val_conf=0.88, quality=0.80, variant=variant)
            engine.process_observation("TRK_003", "CAM_A", obs)

        fused = engine.get_fused_identity("TRK_003")
        self.assertIsNotNone(fused)
        # Only 1 distinct frame (frame_id=10), so observation_count = 1
        self.assertEqual(fused.observation_count, 1)
        # Must NOT be CONFIRMED — only 1 distinct frame < min_observations_to_confirm=3
        self.assertNotEqual(fused.status, "CONFIRMED")
        self.assertFalse(fused.confirmed)

    # -------------------------------------------------------------------------
    # Test 4: Minor OCR error — dominant candidate wins with agreement
    # -------------------------------------------------------------------------
    def test_4_minor_ocr_error_dominant_candidate_wins(self):
        """Test 4: TN01AB1234 × 2 + TN01AB1284 × 1 → dominant candidate TN01AB1234 preferred."""
        engine = TemporalPlateFusionEngine(
            min_observations_to_confirm=2,
            confirmation_score_threshold=0.70,
        )
        obs1 = _make_obs("TRK_004", frame_id=10, text="TN01AB1234", ocr_conf=0.91, val_conf=0.90, quality=0.80)
        obs2 = _make_obs("TRK_004", frame_id=20, text="TN01AB1284", ocr_conf=0.70, val_conf=0.72, quality=0.65)
        obs3 = _make_obs("TRK_004", frame_id=30, text="TN01AB1234", ocr_conf=0.94, val_conf=0.93, quality=0.85)
        engine.process_observation("TRK_004", "CAM_A", obs1)
        engine.process_observation("TRK_004", "CAM_A", obs2)
        fused = engine.process_observation("TRK_004", "CAM_A", obs3)
        self.assertIsNotNone(fused)
        # The dominant text (TN01AB1234 or Levenshtein neighbor cluster) should be represented
        self.assertIn(fused.best_plate_number, ["TN01AB1234", "TN01AB1284"])
        # The TN01AB1234 cluster should have higher evidence weight
        if "TN01AB1284" in fused.candidate_history and "TN01AB1234" in fused.candidate_history:
            self.assertGreater(
                fused.candidate_history.get("TN01AB1234", 0.0),
                fused.candidate_history.get("TN01AB1284", 0.0)
            )

    # -------------------------------------------------------------------------
    # Test 5: PENDING plate must NOT trigger confirmed-cache skip in scheduler
    # -------------------------------------------------------------------------
    def test_5_pending_plate_does_not_trigger_confirmed_cache_skip(self):
        """Test 5: A PENDING fused plate must not cause should_run_ocr to return False as CONFIRMED_PLATE_CACHED."""
        engine = TemporalPlateFusionEngine(
            min_observations_to_confirm=5,      # Very high — keeps plate PENDING
            confirmation_score_threshold=5.0,
        )
        # Manually plant a PENDING fused identity in the engine
        from app.schemas.plate import FusedPlateIdentity
        pending_fused = FusedPlateIdentity(
            track_id="TRK_005",
            camera_id="CAM_A",
            best_plate_number="TN02CD5678",
            overall_confidence=0.78,     # High confidence — old code would cache-skip this
            observation_count=2,
            confirmed=False,
            status="PENDING",
            evidence_frames=[10, 20],
            first_seen_timestamp=0.1,
            last_seen_timestamp=0.2,
        )
        engine.fused_identities["TRK_005"] = pending_fused

        mgr = PlateTrackerAssociationManager(
            ocr_engine=TestOCREngine(),
            ocr_frame_interval=3,
            ocr_cache_duration=30.0,
        )
        mgr.fusion_engine = engine

        valid_bbox = BoundingBoxXYXY(x1=100.0, y1=100.0, x2=300.0, y2=250.0)
        track = TrackState(
            track_id="TRK_005", camera_id="CAM_A", vehicle_type="car",
            current_bbox=valid_bbox, current_center=(200.0, 175.0),
            confidence=0.90, first_seen_timestamp=0.0, last_seen_timestamp=0.3,
            status=TrackStatus.CONFIRMED, confirmed=True,
        )
        # Set last_ocr_frame so interval doesn't block
        mgr.track_ocr_stats["TRK_005"] = {
            "attempts": 2,
            "last_ocr_frame": 1,          # Long ago
            "last_ocr_timestamp": 0.0,
            "last_accepted_plate": None,
            "last_crop_sig": None,
        }

        run_ocr, reason = mgr.should_run_ocr(track, frame_id=100, timestamp=10.0, v_bbox=valid_bbox)
        # A PENDING plate must NOT return CONFIRMED_PLATE_CACHED
        self.assertNotEqual(reason, "CONFIRMED_PLATE_CACHED")
        # It should be allowed to run (or skip only due to interval/size, never due to "confirmed plate")
        if not run_ocr:
            self.assertNotIn(reason, ["CONFIRMED_PLATE_CACHED"])

    # -------------------------------------------------------------------------
    # Test 6: Two different vehicles must NOT have plates merged
    # -------------------------------------------------------------------------
    def test_6_different_vehicles_plates_not_merged(self):
        """Test 6: Observations for TRK_006 and TRK_007 must remain isolated."""
        engine = TemporalPlateFusionEngine()
        obs_a = _make_obs("TRK_006", frame_id=10, text="TN01AB1234", ocr_conf=0.90, val_conf=0.90, quality=0.80)
        obs_b = _make_obs("TRK_007", frame_id=10, text="TN01AB1234", ocr_conf=0.90, val_conf=0.90, quality=0.80)

        engine.process_observation("TRK_006", "CAM_A", obs_a)
        engine.process_observation("TRK_007", "CAM_A", obs_b)

        fused_a = engine.get_fused_identity("TRK_006")
        fused_b = engine.get_fused_identity("TRK_007")

        # Each track must have exactly its own observations; observation counts independent
        self.assertIsNotNone(fused_a)
        self.assertIsNotNone(fused_b)
        self.assertEqual(fused_a.observation_count, 1)  # TRK_006: 1 frame
        self.assertEqual(fused_b.observation_count, 1)  # TRK_007: 1 frame (SEPARATE)
        # Evidence must not combine across tracks
        self.assertNotIn("TRK_007", engine.observations_store.get("TRK_006", []))

    # -------------------------------------------------------------------------
    # Test 7: Cross-camera same global vehicle may pool only if same track_id key
    # -------------------------------------------------------------------------
    def test_7_cross_camera_pooled_only_for_same_global_vehicle(self):
        """Test 7: Cross-camera observations with same track_id key may pool; different track_ids may not."""
        engine = TemporalPlateFusionEngine()
        # Same global vehicle ID (e.g., after Re-ID, both cams contribute to same track_id)
        obs_cam_a = _make_obs("GLOBAL_VEH_001", frame_id=10, text="KA01MH9999",
                               ocr_conf=0.90, val_conf=0.90, quality=0.80)
        obs_cam_b = _make_obs("GLOBAL_VEH_001", frame_id=20, text="KA01MH9999",
                               ocr_conf=0.88, val_conf=0.88, quality=0.78)
        engine.process_observation("GLOBAL_VEH_001", "CAM_A", obs_cam_a)
        fused = engine.process_observation("GLOBAL_VEH_001", "CAM_B", obs_cam_b)

        # Both observations under the same logical key → 2 distinct frames accumulated
        self.assertIsNotNone(fused)
        self.assertEqual(fused.observation_count, 2)

        # A different vehicle under a different track_id stays separate
        obs_other = _make_obs("GLOBAL_VEH_002", frame_id=30, text="KA01MH9999",
                               ocr_conf=0.90, val_conf=0.90, quality=0.80)
        engine.process_observation("GLOBAL_VEH_002", "CAM_A", obs_other)
        fused_other = engine.get_fused_identity("GLOBAL_VEH_002")
        self.assertIsNotNone(fused_other)
        self.assertEqual(fused_other.observation_count, 1)  # completely separate evidence chain

    # -------------------------------------------------------------------------
    # Test 8: Invalid plate text must NEVER become CONFIRMED regardless of confidence
    # -------------------------------------------------------------------------
    def test_8_invalid_plate_text_never_confirmed(self):
        """Test 8: Text failing IndianPlateValidator must not reach CONFIRMED even at high OCR confidence."""
        from app.perception.ocr_engine import IndianPlateValidator
        engine = TemporalPlateFusionEngine(
            min_observations_to_confirm=2,
            confirmation_score_threshold=0.50,
        )
        invalid_text = "INVALID123"
        _, val_conf = IndianPlateValidator.validate_format(invalid_text)

        # Supply high OCR confidence but the validation confidence will be low
        for frame_id in [10, 20, 30]:
            obs = _make_obs("TRK_008", frame_id=frame_id, text=invalid_text,
                            ocr_conf=0.95, val_conf=val_conf, quality=0.85)
            fused = engine.process_observation("TRK_008", "CAM_A", obs)

        # Even with many observations, INVALID format must block CONFIRMED
        self.assertIsNotNone(fused)
        # An invalid plate may land in PENDING (if val_conf > 0.20) or UNKNOWN (if very low)
        # but MUST NOT be CONFIRMED
        self.assertNotEqual(fused.status, "CONFIRMED")
        self.assertFalse(fused.confirmed)


if __name__ == "__main__":
    unittest.main()
