"""
ChronoEye Infinity — Vehicle-Rear Set01 Evaluation Harness Unit Tests
Validates ground-truth parsing, normalization, cross-camera vehicle pairing,
evaluation matcher logic, metric calculations, and seven-signal reporting.

STRICT PROTOCOL: Evaluation harness tests only. Does not modify production code.
"""

import os
import sys
import json
import tempfile
import unittest
from typing import Dict, List, Any

# Ensure backend and scripts are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.prepare_vehicle_rear_set01_gt import (
    normalize_plate_string,
    parse_vehicle_rear_xml,
    parse_vehicle_rear_json,
    extract_cross_camera_ground_truth,
    prepare_set01_ground_truth,
)
from scripts.evaluate_vehicle_rear_set01 import (
    compute_iou,
    match_track_to_gt,
    evaluate_predictions_against_gt,
)


class TestVehicleRearGroundTruthPreparation(unittest.TestCase):
    """Tests for Vehicle-Rear GT parsing, normalization, and pairing."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_gt_plate_normalization(self):
        """Test 1: GT normalization correctly formats and cleans plate strings."""
        self.assertEqual(normalize_plate_string("abc-1234"), "ABC1234")
        self.assertEqual(normalize_plate_string("  TN 09 AB 1234  "), "TN09AB1234")
        self.assertEqual(normalize_plate_string("KA01-MH-9999"), "KA01MH9999")
        self.assertEqual(normalize_plate_string(""), "")
        self.assertEqual(normalize_plate_string(None), "")

    def test_02_xml_ground_truth_parsing(self):
        """Test 2: XML parser extracts vehicle attributes, bounding boxes, and frames."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<vehicles>
    <vehicle id="101">
        <plate>ABC1234</plate>
        <type>car</type>
        <make>Toyota</make>
        <model>Corolla</model>
        <color>Silver</color>
        <year>2018</year>
        <motorcycle>0</motorcycle>
        <quality>good</quality>
        <discard>0</discard>
        <frame number="10">
            <plate x="100" y="200" width="80" height="25" />
            <vehicle x="50" y="150" width="300" height="200" />
        </frame>
        <frame number="15">
            <plate x="110" y="205" width="82" height="26" />
            <vehicle x="55" y="155" width="305" height="205" />
        </frame>
    </vehicle>
</vehicles>
"""
        xml_path = os.path.join(self.temp_dir.name, "vehicles_c1.xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        vehicles = parse_vehicle_rear_xml(xml_path, "Camera1")
        self.assertIn("ABC1234", vehicles)
        v = vehicles["ABC1234"]
        self.assertEqual(v["vehicle_gt_id"], "ABC1234")
        self.assertEqual(v["normalized_plate"], "ABC1234")
        self.assertEqual(v["vehicle_type"], "car")
        self.assertEqual(v["color"], "Silver")
        self.assertEqual(v["observation_count"], 2)
        self.assertEqual(v["min_frame"], 10)
        self.assertEqual(v["max_frame"], 15)

    def test_03_common_vehicle_cross_camera_extraction(self):
        """Test 3: Cross-camera pairing identifies overlapping and single-camera vehicles."""
        cam1_data = {
            "ABC1234": {
                "vehicle_gt_id": "ABC1234",
                "normalized_plate": "ABC1234",
                "raw_plate": "ABC1234",
                "vehicle_type": "car",
                "observations": [{"frame": 10}],
                "observation_count": 1,
                "min_frame": 10,
                "max_frame": 10,
            },
            "XYZ9999": {
                "vehicle_gt_id": "XYZ9999",
                "normalized_plate": "XYZ9999",
                "raw_plate": "XYZ9999",
                "vehicle_type": "truck",
                "observations": [{"frame": 5}],
                "observation_count": 1,
                "min_frame": 5,
                "max_frame": 5,
            },
        }
        cam2_data = {
            "ABC1234": {
                "vehicle_gt_id": "ABC1234",
                "normalized_plate": "ABC1234",
                "raw_plate": "ABC1234",
                "vehicle_type": "car",
                "observations": [{"frame": 50}],
                "observation_count": 1,
                "min_frame": 50,
                "max_frame": 50,
            },
            "KL07AA1111": {
                "vehicle_gt_id": "KL07AA1111",
                "normalized_plate": "KL07AA1111",
                "raw_plate": "KL07AA1111",
                "vehicle_type": "bus",
                "observations": [{"frame": 60}],
                "observation_count": 1,
                "min_frame": 60,
                "max_frame": 60,
            },
        }

        cross_cam, single_cam = extract_cross_camera_ground_truth(cam1_data, cam2_data)

        self.assertEqual(len(cross_cam), 1)
        self.assertEqual(cross_cam[0]["gt_identity"], "ABC1234")
        self.assertEqual(cross_cam[0]["camera1_summary"]["min_frame"], 10)
        self.assertEqual(cross_cam[0]["camera2_summary"]["min_frame"], 50)

        self.assertEqual(len(single_cam), 2)
        single_plates = {v["normalized_plate"] for v in single_cam}
        self.assertIn("XYZ9999", single_plates)
        self.assertIn("KL07AA1111", single_plates)


class TestVehicleRearEvaluationMetrics(unittest.TestCase):
    """Tests for post-inference evaluation matcher, precision/recall, purity, and signals."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_04_positive_same_vehicle_match(self):
        """Test 4: Correctly associated cross-camera vehicles are scored as True Positives."""
        gt_cross = {
            "cross_camera_vehicles": [
                {"gt_identity": "ABC1234", "normalized_plate": "ABC1234"},
                {"gt_identity": "XYZ5678", "normalized_plate": "XYZ5678"},
            ],
            "single_camera_vehicles": [],
        }

        # Predictions where ABC1234 is associated under VEH_1 in both cams, XYZ5678 under VEH_2
        predictions = {
            "journeys": [
                {
                    "global_vehicle_id": "VEH_1",
                    "plate": "ABC1234",
                    "status": "CONFIRMED",
                    "cameras": ["Camera1", "Camera2"],
                    "segments": [
                        {"camera_id": "Camera1", "plate_number": "ABC1234"},
                        {"camera_id": "Camera2", "plate_number": "ABC1234"},
                    ],
                },
                {
                    "global_vehicle_id": "VEH_2",
                    "plate": "XYZ5678",
                    "status": "CONFIRMED",
                    "cameras": ["Camera1", "Camera2"],
                    "segments": [
                        {"camera_id": "Camera1", "plate_number": "XYZ5678"},
                        {"camera_id": "Camera2", "plate_number": "XYZ5678"},
                    ],
                },
            ],
            "transitions": [
                {
                    "from_camera": "Camera1",
                    "to_camera": "Camera2",
                    "transition_score": 0.88,
                    "transition_decision": "CONFIRMED",
                    "transition_breakdown": {
                        "plate_similarity": 1.0,
                        "appearance_similarity": 0.85,
                        "visual_features_similarity": 0.80,
                        "vehicle_type_similarity": 1.0,
                        "direction_similarity": 0.90,
                        "temporal_compatibility": 0.95,
                        "spatial_compatibility": 0.90,
                    },
                }
            ],
        }

        report = evaluate_predictions_against_gt(
            predictions_data=predictions,
            gt_cross_data=gt_cross,
            output_dir=self.temp_dir.name,
        )

        self.assertEqual(report["counts"]["true_positives"], 2)
        self.assertEqual(report["counts"]["false_negatives"], 0)
        self.assertEqual(report["counts"]["false_positives"], 0)
        self.assertEqual(report["metrics"]["cross_camera_precision"], 1.0)
        self.assertEqual(report["metrics"]["cross_camera_recall"], 1.0)
        self.assertEqual(report["metrics"]["cross_camera_f1"], 1.0)
        self.assertEqual(report["metrics"]["global_id_purity"], 1.0)
        self.assertEqual(report["metrics"]["id_switches_count"], 0)
        self.assertEqual(report["metrics"]["journey_reconstruction_accuracy"], 1.0)

    def test_05_negative_different_vehicle_separation(self):
        """Test 5: Distinct vehicles receiving different Global IDs do not produce False Positives."""
        gt_cross = {
            "cross_camera_vehicles": [
                {"gt_identity": "ABC1234", "normalized_plate": "ABC1234"},
            ],
            "single_camera_vehicles": [
                {"vehicle_gt_id": "DIFF999", "normalized_plate": "DIFF999"},
            ],
        }

        predictions = {
            "journeys": [
                {
                    "global_vehicle_id": "VEH_101",
                    "plate": "ABC1234",
                    "status": "CONFIRMED",
                    "cameras": ["Camera1", "Camera2"],
                    "segments": [
                        {"camera_id": "Camera1", "plate_number": "ABC1234"},
                        {"camera_id": "Camera2", "plate_number": "ABC1234"},
                    ],
                },
                {
                    "global_vehicle_id": "VEH_102",
                    "plate": "DIFF999",
                    "status": "ACTIVE",
                    "cameras": ["Camera1"],
                    "segments": [
                        {"camera_id": "Camera1", "plate_number": "DIFF999"},
                    ],
                },
            ],
            "transitions": [],
        }

        report = evaluate_predictions_against_gt(
            predictions_data=predictions,
            gt_cross_data=gt_cross,
            output_dir=self.temp_dir.name,
        )

        self.assertEqual(report["counts"]["true_positives"], 1)
        self.assertEqual(report["counts"]["false_positives"], 0)
        self.assertEqual(report["counts"]["true_negatives"], 1)
        self.assertEqual(report["metrics"]["global_id_purity"], 1.0)

    def test_06_global_id_purity_calculation(self):
        """Test 6: Global ID purity is correctly reduced when multiple GT vehicles merge."""
        gt_cross = {
            "cross_camera_vehicles": [
                {"gt_identity": "CAR_A", "normalized_plate": "CAR_A"},
                {"gt_identity": "CAR_B", "normalized_plate": "CAR_B"},
            ],
            "single_camera_vehicles": [],
        }

        # Faulty prediction: CAR_A and CAR_B both mapped to VEH_MERGED
        predictions = {
            "journeys": [
                {
                    "global_vehicle_id": "VEH_MERGED",
                    "plate": "CAR_A",
                    "status": "CONFIRMED",
                    "cameras": ["Camera1", "Camera2"],
                    "segments": [
                        {"camera_id": "Camera1", "plate_number": "CAR_A"},
                        {"camera_id": "Camera2", "plate_number": "CAR_A"},
                    ],
                }
            ],
            "raw_records": [
                {"global_vehicle_id": "VEH_MERGED", "camera_id": "Camera1", "matched_gt_id": "CAR_A"},
                {"global_vehicle_id": "VEH_MERGED", "camera_id": "Camera2", "matched_gt_id": "CAR_A"},
                {"global_vehicle_id": "VEH_MERGED", "camera_id": "Camera2", "matched_gt_id": "CAR_B"},
            ],
            "transitions": [],
        }

        report = evaluate_predictions_against_gt(
            predictions_data=predictions,
            gt_cross_data=gt_cross,
            output_dir=self.temp_dir.name,
        )

        # 2 CAR_A out of 3 total records in VEH_MERGED -> purity = 2/3 ≈ 0.6667
        self.assertAlmostEqual(report["metrics"]["global_id_purity"], 0.6667, places=3)

    def test_07_id_switch_calculation(self):
        """Test 7: ID switches are incremented when the same GT vehicle changes Global IDs."""
        gt_cross = {
            "cross_camera_vehicles": [
                {"gt_identity": "CAR_SPLIT", "normalized_plate": "CAR_SPLIT"},
            ],
            "single_camera_vehicles": [],
        }

        # Predictions: CAR_SPLIT got VEH_1 in Camera 1 and VEH_2 in Camera 2
        predictions = {
            "journeys": [
                {
                    "global_vehicle_id": "VEH_1",
                    "plate": "CAR_SPLIT",
                    "status": "ACTIVE",
                    "cameras": ["Camera1"],
                    "segments": [{"camera_id": "Camera1", "plate_number": "CAR_SPLIT"}],
                },
                {
                    "global_vehicle_id": "VEH_2",
                    "plate": "CAR_SPLIT",
                    "status": "ACTIVE",
                    "cameras": ["Camera2"],
                    "segments": [{"camera_id": "Camera2", "plate_number": "CAR_SPLIT"}],
                },
            ],
            "transitions": [],
        }

        report = evaluate_predictions_against_gt(
            predictions_data=predictions,
            gt_cross_data=gt_cross,
            output_dir=self.temp_dir.name,
        )

        self.assertEqual(report["counts"]["false_negatives"], 1)
        self.assertEqual(report["counts"]["true_positives"], 0)
        self.assertEqual(report["metrics"]["id_switches_count"], 1)

    def test_08_unknown_unobserved_handling(self):
        """Test 8: Unobserved/unknown vehicles are tracked as False Negatives without crashing."""
        gt_cross = {
            "cross_camera_vehicles": [
                {"gt_identity": "GHOST_VEHICLE", "normalized_plate": "GHOST_VEHICLE"},
            ],
            "single_camera_vehicles": [],
        }

        # No predictions produced
        predictions = {
            "journeys": [],
            "transitions": [],
            "raw_records": [],
        }

        report = evaluate_predictions_against_gt(
            predictions_data=predictions,
            gt_cross_data=gt_cross,
            output_dir=self.temp_dir.name,
        )

        self.assertEqual(report["counts"]["false_negatives"], 1)
        self.assertEqual(report["counts"]["true_positives"], 0)
        self.assertEqual(report["metrics"]["cross_camera_recall"], 0.0)

    def test_09_seven_signal_report_and_csv_preservation(self):
        """Test 9: Seven-signal values and availability are faithfully preserved in CSV and JSON."""
        gt_cross = {"cross_camera_vehicles": [], "single_camera_vehicles": []}
        predictions = {
            "journeys": [],
            "transitions": [
                {
                    "from_camera": "Camera1",
                    "to_camera": "Camera2",
                    "from_track_id": "TRK_1",
                    "to_track_id": "TRK_2",
                    "transition_score": 0.85,
                    "transition_decision": "CONFIRMED",
                    "rejection_reason": "NONE",
                    "transition_breakdown": {
                        "plate_similarity": 1.0,
                        "appearance_similarity": 0.70,
                        "visual_features_similarity": 0.65,
                        "vehicle_type_similarity": 1.0,
                        "direction_similarity": 0.90,
                        "temporal_compatibility": 0.85,
                        "spatial_compatibility": 0.80,
                    },
                }
            ],
        }

        report = evaluate_predictions_against_gt(
            predictions_data=predictions,
            gt_cross_data=gt_cross,
            output_dir=self.temp_dir.name,
        )

        self.assertTrue(report["signal_availability"]["plate"])
        self.assertTrue(report["signal_availability"]["appearance"])
        self.assertTrue(report["signal_availability"]["visual_features"])
        self.assertTrue(report["signal_availability"]["vehicle_type"])
        self.assertTrue(report["signal_availability"]["direction"])
        self.assertTrue(report["signal_availability"]["temporal"])
        self.assertTrue(report["signal_availability"]["spatial"])

        csv_path = os.path.join(self.temp_dir.name, "set01_signal_analysis.csv")
        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # Header + 1 transition row
            self.assertIn("CONFIRMED", lines[1])

    def test_10_journey_reconstruction_accuracy_bounds_and_grounding(self):
        """Test 10: Journey Reconstruction Accuracy is bounded [0.0, 1.0] and GT-grounded."""
        # 1. Zero correct journeys -> produces 0.0
        gt_two = {
            "cross_camera_vehicles": [
                {"gt_identity": "GT_A", "normalized_plate": "GT_A"},
                {"gt_identity": "GT_B", "normalized_plate": "GT_B"},
            ],
            "single_camera_vehicles": [],
        }
        no_preds = {"journeys": [], "transitions": [], "raw_records": []}
        rep_zero = evaluate_predictions_against_gt(no_preds, gt_two, output_dir=self.temp_dir.name)
        self.assertEqual(rep_zero["metrics"]["journey_reconstruction_accuracy"], 0.0)

        # 2. All GT journeys correctly reconstructed -> produces 1.0
        perfect_preds = {
            "journeys": [
                {
                    "global_vehicle_id": "V1",
                    "plate": "GT_A",
                    "status": "CONFIRMED",
                    "cameras": ["Camera1", "Camera2"],
                    "segments": [
                        {"camera_id": "Camera1", "plate_number": "GT_A"},
                        {"camera_id": "Camera2", "plate_number": "GT_A"},
                    ],
                },
                {
                    "global_vehicle_id": "V2",
                    "plate": "GT_B",
                    "status": "ACTIVE",
                    "cameras": ["Camera1", "Camera2"],
                    "segments": [
                        {"camera_id": "Camera1", "plate_number": "GT_B"},
                        {"camera_id": "Camera2", "plate_number": "GT_B"},
                    ],
                },
            ],
            "transitions": [],
        }
        rep_full = evaluate_predictions_against_gt(perfect_preds, gt_two, output_dir=self.temp_dir.name)
        self.assertEqual(rep_full["metrics"]["journey_reconstruction_accuracy"], 1.0)
        self.assertEqual(rep_full["counts"]["correct_reconstructed_journeys"], 2)

        # 3. Extra predicted journeys cannot cause metric to exceed 1.0 (bounding test)
        # 1 GT vehicle, but 5 multi-camera predicted journeys
        gt_single_cross = {
            "cross_camera_vehicles": [
                {"gt_identity": "GT_A", "normalized_plate": "GT_A"},
            ],
            "single_camera_vehicles": [],
        }
        extra_preds = {
            "journeys": [
                {
                    "global_vehicle_id": "V1",
                    "plate": "GT_A",
                    "status": "CONFIRMED",
                    "cameras": ["Camera1", "Camera2"],
                    "segments": [
                        {"camera_id": "Camera1", "plate_number": "GT_A"},
                        {"camera_id": "Camera2", "plate_number": "GT_A"},
                    ],
                },
                {
                    "global_vehicle_id": "V_EXTRA1",
                    "plate": "OTHER1",
                    "status": "CONFIRMED",
                    "cameras": ["Camera1", "Camera2"],
                    "segments": [
                        {"camera_id": "Camera1", "plate_number": "OTHER1"},
                        {"camera_id": "Camera2", "plate_number": "OTHER1"},
                    ],
                },
                {
                    "global_vehicle_id": "V_EXTRA2",
                    "plate": "OTHER2",
                    "status": "ACTIVE",
                    "cameras": ["Camera1", "Camera2"],
                    "segments": [
                        {"camera_id": "Camera1", "plate_number": "OTHER2"},
                        {"camera_id": "Camera2", "plate_number": "OTHER2"},
                    ],
                },
                {
                    "global_vehicle_id": "V_EXTRA3",
                    "plate": "OTHER3",
                    "status": "CONFIRMED",
                    "cameras": ["Camera1", "Camera2"],
                    "segments": [
                        {"camera_id": "Camera1", "plate_number": "OTHER3"},
                        {"camera_id": "Camera2", "plate_number": "OTHER3"},
                    ],
                },
                {
                    "global_vehicle_id": "V_EXTRA4",
                    "plate": "OTHER4",
                    "status": "PROBABLE",
                    "cameras": ["Camera1", "Camera2"],
                    "segments": [
                        {"camera_id": "Camera1", "plate_number": "OTHER4"},
                        {"camera_id": "Camera2", "plate_number": "OTHER4"},
                    ],
                },
            ],
            "transitions": [],
        }
        rep_bounded = evaluate_predictions_against_gt(extra_preds, gt_single_cross, output_dir=self.temp_dir.name)
        self.assertEqual(rep_bounded["metrics"]["journey_reconstruction_accuracy"], 1.0)
        self.assertLessEqual(rep_bounded["metrics"]["journey_reconstruction_accuracy"], 1.0)
        self.assertGreaterEqual(rep_bounded["metrics"]["journey_reconstruction_accuracy"], 0.0)
        self.assertEqual(rep_bounded["counts"]["correct_reconstructed_journeys"], 1)

        # 4. Partial reconstruction: 1 of 2 GT vehicles reconstructed -> 0.5
        partial_preds = {
            "journeys": [
                {
                    "global_vehicle_id": "V1",
                    "plate": "GT_A",
                    "status": "CONFIRMED",
                    "cameras": ["Camera1", "Camera2"],
                    "segments": [
                        {"camera_id": "Camera1", "plate_number": "GT_A"},
                        {"camera_id": "Camera2", "plate_number": "GT_A"},
                    ],
                },
            ],
            "transitions": [],
        }
        rep_partial = evaluate_predictions_against_gt(partial_preds, gt_two, output_dir=self.temp_dir.name)
        self.assertEqual(rep_partial["metrics"]["journey_reconstruction_accuracy"], 0.5)
        self.assertEqual(rep_partial["counts"]["correct_reconstructed_journeys"], 1)

        # 5. Empty GT list -> 0.0 without division by zero or errors
        gt_empty = {"cross_camera_vehicles": [], "single_camera_vehicles": []}
        rep_empty = evaluate_predictions_against_gt(extra_preds, gt_empty, output_dir=self.temp_dir.name)
        self.assertEqual(rep_empty["metrics"]["journey_reconstruction_accuracy"], 0.0)

    def test_11_timestamp_conversion_and_formatting(self):
        """Test 11: Format timestamp converts seconds to HH:MM:SS.mmm format."""
        from scripts.evaluate_vehicle_rear_set01 import format_timestamp_hms
        self.assertEqual(format_timestamp_hms(0.0), "00:00:00.000")
        self.assertEqual(format_timestamp_hms(0.038), "00:00:00.038")
        self.assertEqual(format_timestamp_hms(65.456), "00:01:05.456")
        self.assertEqual(format_timestamp_hms(3665.123), "01:01:05.123")
        self.assertEqual(format_timestamp_hms(None), "")
        self.assertEqual(format_timestamp_hms(-1.0), "")

    def test_12_anpr_events_csv_generation_and_schema(self):
        """Test 12: Generates set01_anpr_events.csv with all required columns and event types."""
        import csv
        gt_cross = {
            "cross_camera_vehicles": [
                {"gt_identity": "ABC1234", "normalized_plate": "ABC1234"},
            ],
            "single_camera_vehicles": [],
        }
        predictions = {
            "journeys": [
                {
                    "global_vehicle_id": "VEH_101",
                    "journey_id": "JRN_101",
                    "plate": "ABC1234",
                    "status": "CONFIRMED",
                    "cameras": ["Camera1"],
                    "segments": [
                        {
                            "camera_id": "Camera1",
                            "local_track_id": "TRK_101",
                            "frame_index": 12,
                            "timestamp": 0.48,
                            "plate_number": "ABC1234",
                        }
                    ],
                }
            ],
            "transitions": [],
            "raw_records": [
                {
                    "event_type": "ACCEPTED_OBSERVATION",
                    "camera_id": "Camera1",
                    "source_video": "Camera1/Set01.mp4",
                    "frame_index": 12,
                    "timestamp_seconds": 0.48,
                    "timestamp_hms": "00:00:00.480",
                    "local_track_id": "TRK_101",
                    "plate_text_raw": "ABC-1234",
                    "plate_text_norm": "ABC1234",
                    "ocr_confidence": 0.92,
                    "recognition_status": "VALID",
                    "global_vehicle_id": "VEH_101",
                    "journey_id": "JRN_101",
                }
            ],
        }

        report = evaluate_predictions_against_gt(
            predictions_data=predictions,
            gt_cross_data=gt_cross,
            output_dir=self.temp_dir.name,
        )

        anpr_csv = os.path.join(self.temp_dir.name, "set01_anpr_events.csv")
        self.assertTrue(os.path.exists(anpr_csv))
        with open(anpr_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertEqual(r["event_type"], "ACCEPTED_OBSERVATION")
            self.assertEqual(r["camera_id"], "Camera1")
            self.assertEqual(r["frame_index"], "12")
            self.assertEqual(r["local_track_id"], "TRK_101")
            self.assertEqual(r["plate_text_norm"], "ABC1234")
            self.assertEqual(r["recognition_status"], "VALID")
            self.assertEqual(r["global_vehicle_id"], "VEH_101")

    def test_13_journey_lookup_csv_and_missing_cameras(self):
        """Test 13: set01_journey_lookup.csv properly distinguishes cross-camera vs single-camera."""
        import csv
        gt_cross = {
            "cross_camera_vehicles": [
                {"gt_identity": "ABC1234", "normalized_plate": "ABC1234"},
            ],
            "single_camera_vehicles": [
                {"vehicle_gt_id": "SINGLE99", "normalized_plate": "SINGLE99"},
            ],
        }
        predictions = {
            "journeys": [
                # Cross-camera journey
                {
                    "global_vehicle_id": "VEH_101",
                    "journey_id": "JRN_101",
                    "plate": "ABC1234",
                    "vehicle_type": "car",
                    "status": "CONFIRMED",
                    "cameras": ["Camera1", "Camera2"],
                    "segments": [
                        {
                            "camera_id": "Camera1",
                            "local_track_id": "TRK_101",
                            "frame_index": 10,
                            "timestamp": 0.40,
                            "plate_number": "ABC1234",
                        },
                        {
                            "camera_id": "Camera2",
                            "local_track_id": "TRK_205",
                            "frame_index": 300,
                            "timestamp": 12.00,
                            "plate_number": "ABC1234",
                        },
                    ],
                },
                # Single-camera journey (Camera 1 only)
                {
                    "global_vehicle_id": "VEH_102",
                    "journey_id": "JRN_102",
                    "plate": "SINGLE99",
                    "vehicle_type": "truck",
                    "status": "ACTIVE",
                    "cameras": ["Camera1"],
                    "segments": [
                        {
                            "camera_id": "Camera1",
                            "local_track_id": "TRK_102",
                            "frame_index": 15,
                            "timestamp": 0.60,
                            "plate_number": "SINGLE99",
                        }
                    ],
                },
            ],
            "transitions": [],
            "raw_records": [],
        }

        evaluate_predictions_against_gt(
            predictions_data=predictions,
            gt_cross_data=gt_cross,
            output_dir=self.temp_dir.name,
        )

        lookup_csv = os.path.join(self.temp_dir.name, "set01_journey_lookup.csv")
        self.assertTrue(os.path.exists(lookup_csv))
        with open(lookup_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 2)

            # Row 1: Cross-camera vehicle
            r1 = rows[0]
            self.assertEqual(r1["global_vehicle_id"], "VEH_101")
            self.assertEqual(r1["cam1_observed"], "True")
            self.assertEqual(r1["cam1_track_id"], "TRK_101")
            self.assertEqual(r1["cam1_first_ts_sec"], "0.4")
            self.assertEqual(r1["cam2_observed"], "True")
            self.assertEqual(r1["cam2_track_id"], "TRK_205")
            self.assertEqual(r1["cam2_first_ts_sec"], "12.0")
            self.assertEqual(float(r1["transit_time_seconds"]), 11.6)

            # Row 2: Single-camera vehicle (Camera 1 only)
            r2 = rows[1]
            self.assertEqual(r2["global_vehicle_id"], "VEH_102")
            self.assertEqual(r2["cam1_observed"], "True")
            self.assertEqual(r2["cam1_track_id"], "TRK_102")
            self.assertEqual(r2["cam2_observed"], "False")
            self.assertEqual(r2["cam2_track_id"], "")
            self.assertEqual(r2["cam2_first_ts_sec"], "")
            self.assertEqual(r2["transit_time_seconds"], "")  # NOT invented!

    def test_14_audited_plate_metrics_separation(self):
        """Test 14: Plate exact match accuracy, precision, recall, and F1 are computed independently."""
        gt_cross = {
            "cross_camera_vehicles": [
                {"gt_identity": "ABC1234", "normalized_plate": "ABC1234"},
                {"gt_identity": "XYZ5678", "normalized_plate": "XYZ5678"},
            ],
            "single_camera_vehicles": [],
        }

        # 1 of 2 GT plates matched (ABC1234 matched, XYZ5678 not predicted)
        predictions = {
            "journeys": [
                {
                    "global_vehicle_id": "VEH_1",
                    "plate": "ABC1234",
                    "status": "CONFIRMED",
                    "cameras": ["Camera1"],
                    "segments": [{"camera_id": "Camera1", "plate_number": "ABC1234"}],
                },
            ],
            "transitions": [],
            "raw_records": [],
        }

        report = evaluate_predictions_against_gt(
            predictions_data=predictions,
            gt_cross_data=gt_cross,
            output_dir=self.temp_dir.name,
        )

        metrics = report["metrics"]
        # 1 match out of 2 GT plates: Recall = 0.5, Precision = 1.0 (1 correct out of 1 predicted target), F1 = 0.6667
        self.assertEqual(metrics["plate_exact_match_accuracy"], 0.5)
        self.assertEqual(metrics["plate_recall"], 0.5)
        self.assertEqual(metrics["plate_precision"], 1.0)
        self.assertAlmostEqual(metrics["plate_f1"], 2 * 1.0 * 0.5 / (1.0 + 0.5), places=4)

    def test_15_regression_camera_local_track_id_collision(self):
        """
        Test 15 (Phase 4 Regression):
        Demonstrates that JourneyReconstructionEngine.track_to_journey_map is keyed by track_id alone.
        When Camera 1 and Camera 2 both emit TRK_101 for different vehicles, TRK_101 hits branch 1
        and merges into Camera 1's journey without candidate matching.
        """
        from app.schemas.tracking import TrackState, BoundingBoxXYXY
        from app.schemas.plate import VehicleIdentityEvidence
        from app.perception.journey import JourneyReconstructionEngine

        engine = JourneyReconstructionEngine()

        # Cam 1 emits TRK_101
        track_c1 = TrackState(
            track_id="TRK_101",
            camera_id="Camera1",
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=10, y1=10, x2=100, y2=100),
            current_center=(55.0, 55.0),
            confidence=0.9,
            first_seen_timestamp=1.0,
            last_seen_timestamp=1.0,
        )
        ev_c1 = VehicleIdentityEvidence(
            track_id="TRK_101",
            camera_id="Camera1",
            vehicle_type="car",
            last_updated_timestamp=1.0,
        )
        j1 = engine.process_track_evidence(evidence=ev_c1, track=track_c1)
        self.assertEqual(j1.global_vehicle_id, "VEH_101")
        self.assertIn("TRK_101", engine.track_to_journey_map)

        # Cam 2 independently emits TRK_101 (a different vehicle in Cam 2)
        track_c2 = TrackState(
            track_id="TRK_101",
            camera_id="Camera2",
            vehicle_type="truck",
            current_bbox=BoundingBoxXYXY(x1=500, y1=500, x2=800, y2=800),
            current_center=(650.0, 650.0),
            confidence=0.9,
            first_seen_timestamp=10.0,
            last_seen_timestamp=10.0,
        )
        ev_c2 = VehicleIdentityEvidence(
            track_id="TRK_101",
            camera_id="Camera2",
            vehicle_type="truck",
            last_updated_timestamp=10.0,
        )
        j2 = engine.process_track_evidence(evidence=ev_c2, track=track_c2)

        # Demonstrates collision: j2 is the SAME journey JRN_101 because track_to_journey_map is unnamespaced
        self.assertEqual(j2.journey_id, j1.journey_id)
        self.assertEqual(engine.track_to_journey_map["TRK_101"], j1.journey_id)


if __name__ == "__main__":
    unittest.main()
