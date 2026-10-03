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


if __name__ == "__main__":
    unittest.main()
