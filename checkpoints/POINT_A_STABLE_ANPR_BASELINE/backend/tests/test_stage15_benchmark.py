"""
Unit tests for PipelineBenchmarkTracker (Component 8)
Verifies:
1. benchmark initialization
2. latency recording
3. frame counting
4. OCR attempt counting
5. plate confirmation counting
6. UNKNOWN counting
7. global identity counting
8. journey counting
9. FPS calculation
10. benchmark reset/finalization
"""

import unittest
import time
from app.core.benchmark import PipelineBenchmarkTracker


class TestPipelineBenchmarkTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = PipelineBenchmarkTracker(
            source_fps=26.3,
            frame_skip=0,
            device="cpu",
            source="test_video.mp4",
            resolution="1920x1080",
        )

    def test_01_initialization(self):
        """Verify default configuration and initial counts."""
        rep = self.tracker.get_summary_report()
        self.assertEqual(rep["source_fps"], 26.3)
        self.assertEqual(rep["frame_skip"], 0)
        self.assertEqual(rep["device"], "cpu")
        self.assertEqual(rep["frames_read"], 0)
        self.assertEqual(rep["frames_processed"], 0)

    def test_02_latency_recording(self):
        """Verify latency recording across stages."""
        self.tracker.record_latency("ingestion", 1.5)
        self.tracker.record_latency("yolo", 15.0)
        self.tracker.record_latency("bytetrack", 2.0)
        self.tracker.record_latency("plate_detection", 3.0)
        self.tracker.record_latency("ocr", 85.0)
        self.tracker.record_latency("reid", 5.0)
        self.tracker.record_latency("total", 111.5)

        rep = self.tracker.get_summary_report()
        self.assertEqual(rep["ingestion_latency_ms"], 1.5)
        self.assertEqual(rep["yolo_latency_ms"], 15.0)
        self.assertEqual(rep["bytetrack_latency_ms"], 2.0)
        self.assertEqual(rep["plate_detection_latency_ms"], 3.0)
        self.assertEqual(rep["ocr_latency_ms"], 85.0)
        self.assertEqual(rep["reid_latency_ms"], 5.0)
        self.assertEqual(rep["total_pipeline_latency_ms"], 111.5)

    def test_03_frame_counting(self):
        """Verify frame read and processed increments."""
        for _ in range(10):
            self.tracker.record_frame(read=True, processed=True)
        rep = self.tracker.get_summary_report()
        self.assertEqual(rep["frames_read"], 10)
        self.assertEqual(rep["frames_processed"], 10)

    def test_04_ocr_attempt_counting(self):
        """Verify OCR attempts, confirmed, and unknown counts."""
        self.tracker.record_ocr_attempt(confirmed=True, is_unknown=False)
        self.tracker.record_ocr_attempt(confirmed=False, is_unknown=True)
        self.tracker.record_ocr_attempt(confirmed=True, is_unknown=False)

        rep = self.tracker.get_summary_report()
        self.assertEqual(rep["ocr_attempts"], 3)
        self.assertEqual(rep["confirmed_plates"], 2)
        self.assertEqual(rep["unknown_plate_observations"], 1)

    def test_05_global_identity_counting(self):
        """Verify global vehicle identity counting."""
        self.tracker.record_global_identity("VEH_001")
        self.tracker.record_global_identity("VEH_002")
        self.tracker.record_global_identity("VEH_001")  # duplicate

        rep = self.tracker.get_summary_report()
        self.assertEqual(rep["global_vehicle_identities"], 2)

    def test_06_journey_counting(self):
        """Verify journey counting."""
        self.tracker.record_journey("JRN_001")
        self.tracker.record_journey("JRN_002")

        rep = self.tracker.get_summary_report()
        self.assertEqual(rep["reconstructed_journeys"], 2)

    def test_07_timer_start_stop(self):
        """Verify monotonic stage timers."""
        self.tracker.start_timer("yolo")
        time.sleep(0.01)
        latency = self.tracker.stop_timer("yolo")
        self.assertGreater(latency, 5.0)  # at least ~10ms
        rep = self.tracker.get_summary_report()
        self.assertGreater(rep["yolo_latency_ms"], 5.0)

    def test_08_fps_calculation(self):
        """Verify effective FPS calculation based on pipeline runtime."""
        self.tracker.start_pipeline()
        time.sleep(0.05)
        for _ in range(5):
            self.tracker.record_frame(read=True, processed=True)
        self.tracker.stop_pipeline()

        rep = self.tracker.get_summary_report()
        self.assertGreater(rep["effective_fps"], 0.0)
        self.assertGreater(rep["total_runtime_seconds"], 0.0)

    def test_09_reset(self):
        """Verify tracker reset."""
        self.tracker.record_frame()
        self.tracker.record_ocr_attempt(confirmed=True)
        self.tracker.reset()
        rep = self.tracker.get_summary_report()
        self.assertEqual(rep["frames_read"], 0)
        self.assertEqual(rep["ocr_attempts"], 0)


if __name__ == "__main__":
    unittest.main()
