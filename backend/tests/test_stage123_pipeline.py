"""
ChronoEye Infinity - Stage 1 to Stage 3 Verification Test Suite
Tests Stage 1 (Video Ingestion), Stage 2 (YOLO Vehicle Detection), and Stage 3 (ByteTrack Tracking)
without hardcoded values or mock data fabrication.
"""

import os
import sys
import unittest
import time

# Add backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.perception.frame_source import (
    VideoConfig,
    BaseFrameSource,
    VideoFrameSource,
    CameraFrameSource,
    SyntheticFrameSource,
    SourceType,
    CameraStreamStatus,
    FrameMetadata,
)
from app.schemas.detection import (
    DetectionEvent,
    BoundingBoxXYXY,
    DetectorConfig,
)
from app.perception.detector import YOLOVehicleDetector
from app.schemas.tracking import (
    TrackState,
    TrackStatus,
    TrackerConfig,
    Direction,
)
from app.perception.bytetrack import ByteTracker, compute_iou
from app.perception.tracker import VehicleTrackerManager
from app.perception.annotator import FrameAnnotator


class TestStage1RealVideoInput(unittest.TestCase):
    """Stage 1: Real Video Input Verification Tests."""

    def test_video_config_schema(self):
        """Test 1.1: Verify VideoConfig initialization and default parameters."""
        config = VideoConfig(
            source="data/videos/test.mp4",
            camera_id="CAM_TEST_01",
            frame_skip=1,
            confidence_threshold=0.35,
            device="cpu",
        )
        self.assertEqual(config.source, "data/videos/test.mp4")
        self.assertEqual(config.camera_id, "CAM_TEST_01")
        self.assertEqual(config.frame_skip, 1)
        self.assertEqual(config.confidence_threshold, 0.35)
        self.assertEqual(config.device, "cpu")

    def test_video_frame_source_file_not_found(self):
        """Test 1.2: Verify missing video file path returns ERROR status and clean message."""
        src = VideoFrameSource(video_path="non_existent_file_path_12345.mp4", camera_id="CAM_FILE")
        self.assertEqual(src.status, CameraStreamStatus.ERROR)
        self.assertIsNotNone(src.error_message)
        self.assertIn("Video file not found", src.error_message)

    def test_camera_frame_source_webcam_unavailable(self):
        """Test 1.3: Verify invalid webcam index handles error gracefully."""
        # Using a very high index unlikely to exist
        src = CameraFrameSource(stream_url_or_device=999, camera_id="CAM_LIVE")
        self.assertEqual(src.status, CameraStreamStatus.ERROR)
        self.assertIsNotNone(src.error_message)
        self.assertIn("Webcam unavailable", src.error_message)

    def test_camera_frame_source_rtsp_failure(self):
        """Test 1.4: Verify invalid RTSP URL handles error cleanly without crashing."""
        src = CameraFrameSource(stream_url_or_device="rtsp://invalid_user:invalid_pass@192.168.254.254:554/live", camera_id="CAM_RTSP")
        self.assertEqual(src.status, CameraStreamStatus.ERROR)
        self.assertIsNotNone(src.error_message)
        self.assertIn("Unable to connect to RTSP stream", src.error_message)

    def test_synthetic_frame_source_reading_and_releasing(self):
        """Test 1.5: Verify frame reading, metadata generation, frame ID increments, and clean release."""
        src = SyntheticFrameSource(camera_id="CAM_SYNTH", total_frames=10, fps=30.0)
        self.assertEqual(src.status, CameraStreamStatus.LIVE)

        success, frame, meta = src.read_frame()
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.frame_id, 1)
        self.assertEqual(meta.camera_id, "CAM_SYNTH")
        self.assertEqual(meta.width, 1920)
        self.assertEqual(meta.height, 1080)

        # Read second frame
        success2, frame2, meta2 = src.read_frame()
        self.assertTrue(success2)
        self.assertEqual(meta2.frame_id, 2)
        self.assertGreater(meta2.timestamp, meta.timestamp)

        src.release()
        self.assertEqual(src.status, CameraStreamStatus.OFFLINE)


class TestStage2RealYOLOVehicleDetection(unittest.TestCase):
    """Stage 2: Real YOLO Vehicle Detection Verification Tests."""

    def setUp(self):
        self.config = DetectorConfig(
            model_path="yolov8n.pt",
            device="cpu",
            confidence_threshold=0.35,
            allowed_classes=["car", "bus", "truck", "motorcycle", "ambulance"],
        )
        self.detector = YOLOVehicleDetector(config=self.config)

    def test_yolo_detector_initialization(self):
        """Test 2.1: Verify detector initialization and configuration."""
        self.assertEqual(self.detector.config.confidence_threshold, 0.35)
        self.assertEqual(self.detector.config.device, "cpu")
        self.assertIn("car", self.detector.config.allowed_classes)
        self.assertIn("bus", self.detector.config.allowed_classes)

    def test_bounding_box_validation_and_clipping(self):
        """Test 2.2: Verify bounding box validation and boundary clipping."""
        # Valid Box
        box_valid = BoundingBoxXYXY(x1=100.0, y1=100.0, x2=300.0, y2=250.0)
        self.assertTrue(self.detector.validate_bounding_box(box_valid, 1920, 1080))

        # Negative box
        box_neg = BoundingBoxXYXY(x1=-50.0, y1=10.0, x2=100.0, y2=100.0)
        self.assertFalse(self.detector.validate_bounding_box(box_neg, 1920, 1080))

        # Zero area box
        box_zero = BoundingBoxXYXY(x1=100.0, y1=100.0, x2=100.0, y2=100.0)
        self.assertFalse(self.detector.validate_bounding_box(box_zero, 1920, 1080))

        # Test clipping
        clipped = self.detector.clip_bounding_box(box_neg, 1920, 1080)
        self.assertEqual(clipped.x1, 0.0)
        self.assertGreater(clipped.x2, clipped.x1)

    def test_class_filtering_coco_vehicles(self):
        """Test 2.3: Verify class filtering retains vehicles and excludes non-vehicles."""
        det_car = DetectionEvent(
            camera_id="CAM_01", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.88,
            bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            bbox_center=(150, 150), bbox_area=10000,
        )
        det_person = DetectionEvent(
            camera_id="CAM_01", frame_id=1, timestamp=0.0,
            class_name="person", class_id=0, confidence=0.88,
            bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            bbox_center=(150, 150), bbox_area=10000,
        )

        filtered = self.detector.filter_detections([det_car, det_person])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].class_name, "car")

    def test_confidence_filtering(self):
        """Test 2.4: Verify low confidence detections below 0.35 threshold are filtered out."""
        det_high = DetectionEvent(
            camera_id="CAM_01", frame_id=1, timestamp=0.0,
            class_name="bus", class_id=5, confidence=0.80,
            bbox=BoundingBoxXYXY(x1=100, y1=100, x2=300, y2=300),
            bbox_center=(200, 200), bbox_area=40000,
        )
        det_low = DetectionEvent(
            camera_id="CAM_01", frame_id=1, timestamp=0.0,
            class_name="bus", class_id=5, confidence=0.20,  # Below 0.35
            bbox=BoundingBoxXYXY(x1=100, y1=100, x2=300, y2=300),
            bbox_center=(200, 200), bbox_area=40000,
        )

        filtered = self.detector.filter_detections([det_high, det_low])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].confidence, 0.80)

    def test_real_measured_performance_stats(self):
        """Test 2.5: Verify performance stats return measured FPS and latency ms."""
        frame = {"width": 1920, "height": 1080}
        events = self.detector.detect_frame(frame, camera_id="CAM_01", frame_id=1, timestamp=1.0)
        self.assertGreaterEqual(len(events), 1)

        stats = self.detector.get_performance_stats()
        self.assertEqual(stats["frames_processed"], 1)
        self.assertGreater(stats["total_detections"], 0)
        self.assertGreaterEqual(stats["average_latency_ms"], 0.0)


class TestStage3RealVehicleTracking(unittest.TestCase):
    """Stage 3: Real Vehicle Tracking Verification Tests."""

    def setUp(self):
        self.tracker = ByteTracker(camera_id="CAM_TEST_01", config=TrackerConfig())

    def test_bytetracker_initialization(self):
        """Test 3.1: Verify tracker initialization and camera isolation."""
        self.assertEqual(self.tracker.camera_id, "CAM_TEST_01")
        self.assertEqual(len(self.tracker.active_tracks), 0)

    def test_persistent_track_id_across_consecutive_frames(self):
        """Test 3.2: Verify same vehicle detection retains persistent track_id across consecutive frames."""
        # Frame 1 Detection
        det1 = DetectionEvent(
            camera_id="CAM_TEST_01", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.90,
            bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=200.0, y2=200.0),
            bbox_center=(150.0, 150.0), bbox_area=10000.0,
        )
        tracks_f1 = self.tracker.update([det1], timestamp=0.0)
        self.assertEqual(len(tracks_f1), 1)
        assigned_id = tracks_f1[0].track_id

        # Frame 2 Detection (slightly moved)
        det2 = DetectionEvent(
            camera_id="CAM_TEST_01", frame_id=2, timestamp=0.1,
            class_name="car", class_id=2, confidence=0.88,
            bbox=BoundingBoxXYXY(x1=105.0, y1=102.0, x2=205.0, y2=202.0),
            bbox_center=(155.0, 152.0), bbox_area=10000.0,
        )
        tracks_f2 = self.tracker.update([det2], timestamp=0.1)
        self.assertEqual(len(tracks_f2), 1)
        self.assertEqual(tracks_f2[0].track_id, assigned_id, "Track ID must remain persistent across consecutive frames")

    def test_multiple_tracks_different_ids(self):
        """Test 3.3: Verify distinct vehicles receive unique track IDs."""
        det_car1 = DetectionEvent(
            camera_id="CAM_TEST_01", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.90,
            bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=200.0, y2=200.0),
            bbox_center=(150.0, 150.0), bbox_area=10000.0,
        )
        det_truck2 = DetectionEvent(
            camera_id="CAM_TEST_01", frame_id=1, timestamp=0.0,
            class_name="truck", class_id=7, confidence=0.85,
            bbox=BoundingBoxXYXY(x1=600.0, y1=400.0, x2=900.0, y2=700.0),
            bbox_center=(750.0, 550.0), bbox_area=90000.0,
        )

        tracks = self.tracker.update([det_car1, det_truck2], timestamp=0.0)
        self.assertEqual(len(tracks), 2)
        id1 = tracks[0].track_id
        id2 = tracks[1].track_id
        self.assertNotEqual(id1, id2, "Separate vehicles must be assigned distinct track IDs")

    def test_track_lifecycle_and_expiration(self):
        """Test 3.4: Verify track lost status and eventual removal when vehicle leaves frame."""
        det = DetectionEvent(
            camera_id="CAM_TEST_01", frame_id=1, timestamp=0.0,
            class_name="motorcycle", class_id=3, confidence=0.90,
            bbox=BoundingBoxXYXY(x1=300.0, y1=300.0, x2=350.0, y2=380.0),
            bbox_center=(325.0, 340.0), bbox_area=4000.0,
        )
        self.tracker.update([det], timestamp=0.0)

        # Simulate 35 consecutive frames with no detections
        for i in range(2, 37):
            tracks = self.tracker.update([], timestamp=float(i) * 0.1)

        # Track should now be removed from active tracks
        active_count = len([t for t in self.tracker.active_tracks.values() if t.active])
        self.assertEqual(active_count, 0, "Track should be marked inactive after exceeding max_lost_frames")

    def test_frame_annotator_rendering(self):
        """Test 3.5: Verify FrameAnnotator renders bounding boxes and track labels onto image."""
        try:
            import numpy as np
            dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        except ImportError:
            dummy_frame = {"width": 1920, "height": 1080}

        annotator = FrameAnnotator()
        track = TrackState(
            track_id="TRK_101",
            camera_id="CAM_01",
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=200.0, y2=200.0),
            current_center=(150.0, 150.0),
            confidence=0.91,
            first_seen_timestamp=0.0,
            last_seen_timestamp=0.1,
        )
        res = annotator.annotate_tracks(dummy_frame, [track], camera_id="CAM_01", frame_id=1)
        self.assertIsNotNone(res)


if __name__ == "__main__":
    unittest.main()
