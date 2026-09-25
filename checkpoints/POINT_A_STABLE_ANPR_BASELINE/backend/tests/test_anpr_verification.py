"""
ChronoEye Infinity - Unit Test Suite for ANPR Verification Workflow
Tests crop sampling, CSV recording schema, and evaluation metric calculation.
"""

import os
import csv
import pytest
import numpy as np
from app.perception.anpr_verification import ANPRVerifier, CSV_COLUMNS


@pytest.fixture
def temp_verifier(tmp_path):
    output_dir = str(tmp_path / "diagnostic_crops")
    verifier = ANPRVerifier(output_dir=output_dir, max_samples_per_track=2, max_total_samples=5)
    return verifier, output_dir


def test_verifier_initialization_and_csv_creation(temp_verifier):
    verifier, output_dir = temp_verifier
    assert os.path.exists(output_dir)
    assert os.path.exists(verifier.csv_path)

    with open(verifier.csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == CSV_COLUMNS


def test_representative_sampling_and_crop_logging(temp_verifier):
    verifier, output_dir = temp_verifier
    dummy_crop = np.zeros((40, 100, 3), dtype=np.uint8)

    # 1. First sample for TRK_01
    crop_id1 = verifier.log_verification_sample(
        crop_image=dummy_crop,
        frame_number=10,
        track_id="TRK_01",
        timestamp=0.33,
        raw_ocr="TN09AB1234",
        normalized_ocr="TN09AB1234",
        ocr_confidence=0.92,
        indian_format="VALID",
        indian_validation_reason="conf_0.95",
    )
    assert crop_id1 == "plate_crop_001.png"
    assert os.path.exists(os.path.join(output_dir, "plate_crop_001.png"))

    # 2. Duplicate same OCR for TRK_01 should be skipped
    crop_id_dup = verifier.log_verification_sample(
        crop_image=dummy_crop,
        frame_number=15,
        track_id="TRK_01",
        timestamp=0.50,
        raw_ocr="TN09AB1234",
        normalized_ocr="TN09AB1234",
        ocr_confidence=0.93,
    )
    assert crop_id_dup is None

    # 3. New OCR for TRK_01 within max per track limit (2)
    crop_id2 = verifier.log_verification_sample(
        crop_image=dummy_crop,
        frame_number=20,
        track_id="TRK_01",
        timestamp=0.66,
        raw_ocr="TN09AB1235",
        normalized_ocr="TN09AB1235",
        ocr_confidence=0.88,
    )
    assert crop_id2 == "plate_crop_002.png"

    # Verify CSV content
    with open(verifier.csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["crop_id"] == "plate_crop_001.png"
        assert rows[0]["raw_ocr"] == "TN09AB1234"
        assert rows[0]["human_ground_truth"] == ""
        assert rows[0]["human_result"] == ""


def test_evaluate_verification_metrics(temp_verifier):
    verifier, output_dir = temp_verifier

    # Write synthetic annotated CSV
    annotated_rows = [
        CSV_COLUMNS,
        ["plate_crop_001.png", "10", "TRK_01", "0.33", "TN09AB1234", "TN09AB1234", "0.92", "1.0", "VALID", "", "0.0", "TN09AB1234", "CORRECT"],
        ["plate_crop_002.png", "20", "TRK_02", "0.66", "KA01MH9999", "KA01MH9999", "0.85", "1.0", "VALID", "", "0.0", "KA01MH9999", "CORRECT"],
        ["plate_crop_003.png", "30", "TRK_03", "1.00", "8335", "8335", "0.75", "1.0", "INVALID", "", "0.0", "KA01AB8335", "WRONG"],
        ["plate_crop_004.png", "40", "TRK_04", "1.33", "BLURRED", "BLURRED", "0.20", "1.0", "INVALID", "", "0.0", "", "UNCLEAR"],
        ["plate_crop_005.png", "50", "TRK_05", "1.66", "MH12AB1234", "MH12AB1234", "0.50", "1.0", "INVALID", "", "0.0", "MH12AB1234", "CORRECT"],
    ]

    with open(verifier.csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(annotated_rows)

    metrics = ANPRVerifier.evaluate_verification_csv(verifier.csv_path)

    assert metrics["total_samples"] == 5
    assert metrics["correct_recognitions"] == 3
    assert metrics["wrong_recognitions"] == 1
    assert metrics["unclear_samples"] == 1
    assert metrics["clear_samples"] == 4
    assert metrics["accuracy_among_clear"] == 0.75  # 3/4
    assert metrics["rejected_but_correct_count"] == 1  # plate_crop_005 (INVALID format, but CORRECT human_result)
    assert metrics["high_conf_wrong_count"] == 1  # plate_crop_003 (conf=0.75 >= 0.70, WRONG)
