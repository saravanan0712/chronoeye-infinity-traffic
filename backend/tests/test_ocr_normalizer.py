"""
Unit tests for post-OCR text normalization (2-row plate prefix corruption).
"""

import pytest
from app.perception.ocr_normalizer import generate_normalized_plate_candidates
from app.schemas.plate import PlateObservation, PlateValidationStatus
from app.perception.plate_fusion import TemporalPlateFusionEngine
from app.perception.ocr_engine import IndianPlateValidator


def test_pts5330_to_ts5330():
    """Verify PTS5330 generates secondary candidate TS5330."""
    candidates = generate_normalized_plate_candidates("PTS5330", "PTS5330")
    assert "PTS5330" in candidates
    assert "TS5330" in candidates
    assert candidates[0] == "PTS5330"  # Original OCR candidate preserved first
    assert candidates[1] == "TS5330"


def test_fjz8479_to_jz8479():
    """Verify FJZ8479 generates secondary candidate JZ8479."""
    candidates = generate_normalized_plate_candidates("FJZ8479", "FJZ8479")
    assert "FJZ8479" in candidates
    assert "JZ8479" in candidates
    assert candidates[0] == "FJZ8479"
    assert candidates[1] == "JZ8479"


def test_pind7823_prefix_normalization():
    """Verify PIND7823 generates secondary candidate ND7823 (stripped 2 chars)."""
    candidates = generate_normalized_plate_candidates("PIND7823", "PIND7823")
    assert "PIND7823" in candidates
    assert "ND7823" in candidates
    assert candidates[0] == "PIND7823"


def test_shorter_than_seven_unchanged():
    """Verify strings shorter than 7 characters remain unchanged."""
    for short_text in ["X0147", "KW527", "VN472", "TN782", "NN773", "TS533", "YA82"]:
        candidates = generate_normalized_plate_candidates(short_text, short_text)
        assert candidates == [short_text]


def test_invalid_stripped_candidates_rejected():
    """Verify invalid stripped candidates (digits only, letters only, mixed) are rejected."""
    # Digits only: stripping yields digits only (no letters)
    assert generate_normalized_plate_candidates("1234567", "1234567") == ["1234567"]
    # Letters only: stripping yields letters only (no digits)
    assert generate_normalized_plate_candidates("ABCDEFG", "ABCDEFG") == ["ABCDEFG"]
    # Mixed pattern: letters then digits then letters then digits
    assert generate_normalized_plate_candidates("AB12C34", "AB12C34") == ["AB12C34"]


def test_original_ocr_candidate_available():
    """Verify original raw/normalized OCR candidate is always preserved as first candidate."""
    raw = "PTS5330"
    norm = "PTS5330"
    candidates = generate_normalized_plate_candidates(raw, norm)
    assert len(candidates) >= 1
    assert candidates[0] == norm


def test_no_duplicate_candidate_explosion():
    """Verify deduplication works and candidate list remains small and clean."""
    candidates = generate_normalized_plate_candidates("PTS5330", "PTS5330")
    assert len(candidates) == len(set(candidates))
    assert len(candidates) <= 3


def test_fusion_with_normalized_secondary_candidate():
    """Verify temporal fusion correctly fuses TS5330 when normalized candidate is used."""
    engine = TemporalPlateFusionEngine()

    obs1 = PlateObservation(
        track_id="TRK_TEST",
        camera_id="CAM_A",
        frame_id=1,
        timestamp=0.1,
        raw_text="PTS5330",
        normalized_text="TS5330",
        ocr_confidence=0.95,
        validation_status=PlateValidationStatus.FORMAT_MISMATCH,
        validation_confidence=0.40,
        quality_score=0.5,
        preprocessing_variant="STACKED_SPLIT+PREFIX_NORM",
        source="ALPR_PIPELINE"
    )
    obs2 = PlateObservation(
        track_id="TRK_TEST",
        camera_id="CAM_A",
        frame_id=2,
        timestamp=0.2,
        raw_text="PTS5330",
        normalized_text="TS5330",
        ocr_confidence=0.95,
        validation_status=PlateValidationStatus.FORMAT_MISMATCH,
        validation_confidence=0.40,
        quality_score=0.5,
        preprocessing_variant="STACKED_SPLIT+PREFIX_NORM",
        source="ALPR_PIPELINE"
    )

    engine.process_observation("TRK_TEST", "CAM_A", obs1, track_frame_count=15)
    fused = engine.process_observation("TRK_TEST", "CAM_A", obs2, track_frame_count=15)

    assert fused is not None
    assert fused.best_plate_number == "TS5330"
    assert fused.confirmed is True


def test_text_normalizer_pts5330_positive():
    """Positive test: PTS5330 is normalized to TS5330 by TextNormalizer."""
    from app.perception.ocr_engine import TextNormalizer
    assert TextNormalizer.normalize("PTS5330") == "TS5330"
    assert TextNormalizer.normalize("FYA8262") == "YA8262"
    assert TextNormalizer.normalize("EYA8262") == "YA8262"


def test_text_normalizer_preserves_existing_correct_plates():
    """Negative test: verified correct real-video plates must remain strictly unchanged."""
    from app.perception.ocr_engine import TextNormalizer
    assert TextNormalizer.normalize("KW527") == "KW527"
    assert TextNormalizer.normalize("SX8525") == "SX8525"
    assert TextNormalizer.normalize("KT4540") == "KT4540"
    assert TextNormalizer.normalize("VN4712") == "VN4712"
    assert TextNormalizer.normalize("CC435") == "CC435"
    assert TextNormalizer.normalize("YB6433") == "YB6433"
    assert TextNormalizer.normalize("YA8262") == "YA8262"
    assert TextNormalizer.normalize("TN09AB1234") == "TN09AB1234"


def test_text_normalizer_preserves_legitimate_pfe_plates():
    """Negative test: legitimate short plates starting with P, F, or E must NOT be stripped."""
    from app.perception.ocr_engine import TextNormalizer
    # PB is valid Punjab state code
    assert TextNormalizer.normalize("PB0123") == "PB0123"
    assert TextNormalizer.normalize("PB1001") == "PB1001"
    # PY is valid Puducherry state code
    assert TextNormalizer.normalize("PY1234") == "PY1234"
    # Digits directly following single letter -> not stripped
    assert TextNormalizer.normalize("P1234") == "P1234"
    # Letters only -> not stripped
    assert TextNormalizer.normalize("PABCD") == "PABCD"

