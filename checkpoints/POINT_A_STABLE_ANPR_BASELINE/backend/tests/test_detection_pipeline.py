"""
ChronoEye Infinity - Phase 2 Verification Test Suite
Automated Python test suite verifying Vehicle Detection Pipeline requirements.
"""

import os
import sys
import unittest

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.detection import (
    DetectionEvent,
    BoundingBoxXYXY,
    DetectorConfig,
)
from app.perception.detector import YOLOVehicleDetector
from app.perception.frame_source import SyntheticFrameSource
from app.perception.adapter import DetectionAdapter
from app.schemas.simulation import (
    CameraObservation,
    VehicleType,
    BoundingBox,
)


class TestPhase2VehicleDetectionPipeline(unittest.TestCase):

    def setUp(self):
        self.config = DetectorConfig(
            confidence_threshold=0.40,
            iou_threshold=0.50,
            allowed_classes=["car", "bus", "truck", "motorcycle", "ambulance"],
        )
        self.detector = YOLOVehicleDetector(config=self.config)

    def test_detector_initialization_and_config(self):
        """Test 1: Verify detector initialization and configuration."""
        self.assertIsNotNone(self.detector.config)
        self.assertEqual(self.detector.config.confidence_threshold, 0.40)
        self.assertEqual(self.detector.config.iou_threshold, 0.50)
        self.assertIn("car", self.detector.config.allowed_classes)

    def test_detection_event_schema_validation(self):
        """Test 2: Verify DetectionEvent schema instantiation and property calculations."""
        bbox = BoundingBoxXYXY(x1=100.0, y1=50.0, x2=200.0, y2=150.0)
        event = DetectionEvent(
            camera_id="CAM_A_EAST",
            frame_id=1,
            timestamp=12.5,
            class_name="car",
            class_id=2,
            confidence=0.91,
            bbox=bbox,
            bbox_center=bbox.center,
            bbox_area=bbox.area,
        )

        self.assertTrue(event.detection_id.startswith("DET_"))
        self.assertEqual(event.camera_id, "CAM_A_EAST")
        self.assertEqual(event.bbox_center, (150.0, 100.0))
        self.assertEqual(event.bbox_area, 10000.0)
        self.assertEqual(event.bbox.width, 100.0)
        self.assertEqual(event.bbox.height, 100.0)

    def test_bounding_box_validation(self):
        """Test 3: Verify bounding box validator rejects invalid coordinates."""
        # 1. Valid box
        valid_box = BoundingBoxXYXY(x1=10.0, y1=10.0, x2=50.0, y2=50.0)
        self.assertTrue(self.detector.validate_bounding_box(valid_box, 1920, 1080))

        # 2. Negative coordinate box
        neg_box = BoundingBoxXYXY(x1=-10.0, y1=10.0, x2=50.0, y2=50.0)
        self.assertFalse(self.detector.validate_bounding_box(neg_box, 1920, 1080))

        # 3. Out-of-bounds box
        oob_box = BoundingBoxXYXY(x1=10.0, y1=10.0, x2=2000.0, y2=50.0)
        self.assertFalse(self.detector.validate_bounding_box(oob_box, 1920, 1080))

        # 4. Zero area / inverted box
        inv_box = BoundingBoxXYXY(x1=50.0, y1=50.0, x2=50.0, y2=50.0)
        self.assertFalse(self.detector.validate_bounding_box(inv_box, 1920, 1080))

    def test_class_filtering(self):
        """Test 4: Verify class filtering removes unallowed classes."""
        det_car = DetectionEvent(
            camera_id="CAM_01", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.85,
            bbox=BoundingBoxXYXY(x1=10, y1=10, x2=50, y2=50),
            bbox_center=(30, 30), bbox_area=1600,
        )
        det_person = DetectionEvent(
            camera_id="CAM_01", frame_id=1, timestamp=0.0,
            class_name="person", class_id=0, confidence=0.85,
            bbox=BoundingBoxXYXY(x1=10, y1=10, x2=50, y2=50),
            bbox_center=(30, 30), bbox_area=1600,
        )

        filtered = self.detector.filter_detections([det_car, det_person])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].class_name, "car")

    def test_confidence_filtering(self):
        """Test 5: Verify confidence filtering removes low-confidence detections."""
        det_high = DetectionEvent(
            camera_id="CAM_01", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.75,
            bbox=BoundingBoxXYXY(x1=10, y1=10, x2=50, y2=50),
            bbox_center=(30, 30), bbox_area=1600,
        )
        det_low = DetectionEvent(
            camera_id="CAM_01", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.25,  # Below 0.40 threshold
            bbox=BoundingBoxXYXY(x1=10, y1=10, x2=50, y2=50),
            bbox_center=(30, 30), bbox_area=1600,
        )

        filtered = self.detector.filter_detections([det_high, det_low])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].confidence, 0.75)

    def test_frame_processing_and_inference(self):
        """Test 6: Verify frame source ingestion and detector execution."""
        frame_source = SyntheticFrameSource(camera_id="CAM_TEST", total_frames=5)
        success, frame, meta = frame_source.read_frame()
        self.assertTrue(success)

        events = self.detector.detect_frame(frame, camera_id="CAM_TEST", frame_id=meta.frame_id, timestamp=meta.timestamp)
        self.assertGreater(len(events), 0)
        self.assertEqual(events[0].camera_id, "CAM_TEST")

        stats = self.detector.get_performance_stats()
        self.assertEqual(stats["frames_processed"], 1)
        self.assertGreater(stats["total_detections"], 0)

    def test_phase_1_simulation_adapter(self):
        """Test 7: Verify Phase 1 CameraObservation adapt to DetectionEvent."""
        obs = CameraObservation(
            observation_id="OBS_99",
            camera_id="CAM_A_EAST",
            junction_id="J_A",
            road_id="R_AB",
            timestamp=14.2,
            frame_id=120,
            vehicle_id="V_100_TN09AB1234",
            tracking_id=100,
            vehicle_type=VehicleType.CAR,
            plate_number="TN09AB1234",
            speed_kmh=45.0,
            lane_id=0,
            position_x=150.0,
            position_y=100.0,
            bbox=BoundingBox(x=100.0, y=50.0, width=64.0, height=48.0),
            detection_confidence=0.96,
            ocr_confidence=0.92,
        )

        det_event = DetectionAdapter.from_camera_observation(obs)

        self.assertEqual(det_event.camera_id, "CAM_A_EAST")
        self.assertEqual(det_event.frame_id, 120)
        self.assertEqual(det_event.timestamp, 14.2)
        self.assertEqual(det_event.class_name, "car")
        self.assertEqual(det_event.confidence, 0.96)
        self.assertEqual(det_event.bbox.x1, 100.0)
        self.assertEqual(det_event.bbox.y1, 50.0)
        self.assertEqual(det_event.bbox.x2, 164.0)
        self.assertEqual(det_event.bbox.y2, 98.0)
        self.assertEqual(det_event.raw_vehicle_reference, "V_100_TN09AB1234")


if __name__ == "__main__":
    unittest.main()
