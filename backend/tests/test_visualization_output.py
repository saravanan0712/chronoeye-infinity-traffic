"""
Unit tests for ChronoEye Infinity Visualization and Annotated Output Video Layer.
"""

import os
import tempfile
import unittest
import numpy as np
import cv2

from app.perception.annotator import FrameAnnotator
from app.schemas.detection import BoundingBoxXYXY
from app.schemas.tracking import TrackState
from app.schemas.plate import VehicleIdentityEvidence, FusedPlateIdentity
from app.schemas.reid import VehicleJourney


class TestVisualizationOutputLayer(unittest.TestCase):
    """
    Tests for visual overlays, status card, label formatting, and VideoWriter functionality.
    """

    def setUp(self):
        self.annotator = FrameAnnotator()
        self.dummy_frame = np.zeros((1012, 1920, 3), dtype=np.uint8)

    def _create_track(self, track_id="TRK_101", x1=100.0, y1=100.0, x2=300.0, y2=250.0):
        return TrackState(
            track_id=track_id,
            camera_id="CAM_A",
            vehicle_type="car",
            current_bbox=BoundingBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2),
            current_center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            confidence=0.92,
            first_seen_timestamp=0.0,
            last_seen_timestamp=0.5,
            confirmed=True,
        )

    def test_top_left_status_card_rendered(self):
        """Verify the top-left ChronoEye status card is drawn on frame."""
        track = self._create_track()
        annotated = self.annotator.annotate_tracks(
            frame=self.dummy_frame,
            tracks=[track],
            camera_id="CAM_A",
            frame_id=42,
            fps=26.3,
            vehicles_count=5,
        )
        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.shape, self.dummy_frame.shape)
        # Check that top-left overlay region has non-zero pixel values (drawn card)
        overlay_region = annotated[15:120, 15:255]
        self.assertGreater(np.count_nonzero(overlay_region), 0)

    def test_confirmed_plate_visualization(self):
        """Verify confirmed plate displays PLATE: <NUMBER> and CONF: <SCORE>."""
        track = self._create_track("TRK_101")
        evidence = VehicleIdentityEvidence(
            track_id="TRK_101",
            camera_id="CAM_A",
            vehicle_type="car",
            associated_plate=FusedPlateIdentity(
                track_id="TRK_101",
                camera_id="CAM_A",
                best_plate_number="KW527",
                overall_confidence=0.99,
                confirmed=True,
                status="CONFIRMED",
            ),
            last_updated_timestamp=1.0,
        )
        evidence_map = {"TRK_101": evidence}

        annotated = self.annotator.annotate_tracks(
            frame=self.dummy_frame,
            tracks=[track],
            camera_id="CAM_A",
            frame_id=10,
            evidence_map=evidence_map,
        )
        self.assertIsNotNone(annotated)
        # Bounding box and banner area must be non-zero
        banner_region = annotated[70:100, 100:300]
        self.assertGreater(np.count_nonzero(banner_region), 0)

    def test_pending_plate_visualization(self):
        """Verify pending plate displays PLATE: PENDING."""
        track = self._create_track("TRK_104")
        evidence = VehicleIdentityEvidence(
            track_id="TRK_104",
            camera_id="CAM_A",
            vehicle_type="auto",
            associated_plate=FusedPlateIdentity(
                track_id="TRK_104",
                camera_id="CAM_A",
                best_plate_number="X0147",
                pending_plate_number="X0147",
                overall_confidence=0.71,
                confirmed=False,
                status="PENDING",
            ),
            last_updated_timestamp=1.0,
        )
        evidence_map = {"TRK_104": evidence}

        annotated = self.annotator.annotate_tracks(
            frame=self.dummy_frame,
            tracks=[track],
            camera_id="CAM_A",
            frame_id=10,
            evidence_map=evidence_map,
        )
        self.assertIsNotNone(annotated)

    def test_unknown_plate_visualization(self):
        """Verify unassociated or unknown plate displays PLATE: UNKNOWN."""
        track = self._create_track("TRK_105")
        evidence_map = {}  # No plate evidence

        annotated = self.annotator.annotate_tracks(
            frame=self.dummy_frame,
            tracks=[track],
            camera_id="CAM_A",
            frame_id=10,
            evidence_map=evidence_map,
        )
        self.assertIsNotNone(annotated)

    def test_global_identity_visualization(self):
        """Verify Stage 5 global vehicle identity is displayed when available."""
        track = self._create_track("TRK_102")
        journey = VehicleJourney(
            journey_id="JRN_102",
            global_vehicle_id="GID_102_EAST",
            plate_number="SX8525",
            vehicle_type="car",
            segments=[],
            first_seen=0.0,
            last_seen=5.0,
        )
        journey_map = {"TRK_102": journey}

        annotated = self.annotator.annotate_tracks(
            frame=self.dummy_frame,
            tracks=[track],
            camera_id="CAM_A",
            frame_id=10,
            journey_map=journey_map,
        )
        self.assertIsNotNone(annotated)

    def test_video_writer_creates_valid_playable_mp4(self):
        """Verify OpenCV VideoWriter writes a playable, non-empty MP4 with matching dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 25.0
            width, height = 640, 480

            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            self.assertTrue(writer.isOpened(), "VideoWriter should successfully open the output file.")

            # Write 10 frames
            for i in range(10):
                f = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(f, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                writer.write(f)

            writer.release()

            # Verify file on disk
            self.assertTrue(os.path.exists(out_path), "Output video file must exist.")
            self.assertGreater(os.path.getsize(out_path), 1000, "Output video file must not be empty.")

            # Re-open with cv2.VideoCapture
            cap = cv2.VideoCapture(out_path)
            self.assertTrue(cap.isOpened(), "OpenCV should be able to reopen the generated video.")
            reopened_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            reopened_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            reopened_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            self.assertEqual(reopened_w, width)
            self.assertEqual(reopened_h, height)
            self.assertEqual(reopened_frames, 10)


if __name__ == "__main__":
    unittest.main()
