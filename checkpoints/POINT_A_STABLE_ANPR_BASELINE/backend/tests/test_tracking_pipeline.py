"""
ChronoEye Infinity - Phase 3 Verification Test Suite
Automated Python test suite verifying Vehicle Tracking Pipeline requirements across 14 test cases.
"""

import os
import sys
import unittest

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.detection import DetectionEvent, BoundingBoxXYXY
from app.schemas.tracking import (
    TrackState,
    TrackStatus,
    TrackerConfig,
    TrackerBackend,
    Direction,
)
from app.perception.bytetrack import ByteTracker
from app.perception.tracker import VehicleTrackerManager
from app.perception.adapter import DetectionAdapter
from app.perception.motion import MotionEstimator
from app.schemas.simulation import CameraObservation, VehicleType, BoundingBox
from app.simulation.engine import TrafficSimulationEngine


class TestPhase3VehicleTrackingPipeline(unittest.TestCase):

    def setUp(self):
        self.config = TrackerConfig(
            high_conf_thresh=0.60,
            low_conf_thresh=0.20,
            iou_threshold=0.30,
            max_lost_frames=5,
            min_confirmation_hits=3,
        )
        self.tracker = ByteTracker(camera_id="CAM_A_EAST", config=self.config)
        self.manager = VehicleTrackerManager(config=self.config)

    def test_1_tracker_initialization_and_config(self):
        """Test 1: Tracker initialization and configuration."""
        self.assertEqual(self.tracker.camera_id, "CAM_A_EAST")
        self.assertEqual(self.tracker.config.high_conf_thresh, 0.60)
        self.assertEqual(self.tracker.config.min_confirmation_hits, 3)
        self.assertEqual(len(self.tracker.active_tracks), 0)

    def test_2_single_detection_creates_tentative_track(self):
        """Test 2: Single detection creates a tentative track."""
        det = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.85,
            bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=160.0, y2=140.0),
            bbox_center=(130.0, 120.0), bbox_area=2400.0,
        )
        tracks = self.tracker.update([det], timestamp=0.0)

        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        self.assertTrue(track.track_id.startswith("TRK_"))
        self.assertEqual(track.status, TrackStatus.TENTATIVE)
        self.assertFalse(track.confirmed)
        self.assertEqual(track.hits, 1)

    def test_3_repeated_matching_detections_produce_persistent_track_id(self):
        """Test 3: Repeated matching detections produce one persistent track ID."""
        for frame in range(1, 5):
            det = DetectionEvent(
                camera_id="CAM_A_EAST", frame_id=frame, timestamp=frame * 0.1,
                class_name="car", class_id=2, confidence=0.85,
                bbox=BoundingBoxXYXY(x1=100.0 + frame * 2, y1=100.0, x2=160.0 + frame * 2, y2=140.0),
                bbox_center=(130.0 + frame * 2, 120.0), bbox_area=2400.0,
            )
            tracks = self.tracker.update([det], timestamp=frame * 0.1)

        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        self.assertEqual(track.hits, 4)
        self.assertEqual(track.status, TrackStatus.CONFIRMED)
        self.assertTrue(track.confirmed)

    def test_4_multiple_vehicles_receive_different_track_ids(self):
        """Test 4: Multiple non-overlapping vehicles receive different track IDs."""
        det1 = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.85,
            bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=160.0, y2=140.0),
            bbox_center=(130.0, 120.0), bbox_area=2400.0,
        )
        det2 = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=1, timestamp=0.0,
            class_name="bus", class_id=5, confidence=0.90,
            bbox=BoundingBoxXYXY(x1=500.0, y1=500.0, x2=600.0, y2=580.0),
            bbox_center=(550.0, 540.0), bbox_area=8000.0,
        )

        tracks = self.tracker.update([det1, det2], timestamp=0.0)
        self.assertEqual(len(tracks), 2)
        self.assertNotEqual(tracks[0].track_id, tracks[1].track_id)

    def test_5_vehicle_movement_updates_trajectory(self):
        """Test 5: Vehicle movement updates trajectory point history."""
        for step in range(5):
            det = DetectionEvent(
                camera_id="CAM_A_EAST", frame_id=step + 1, timestamp=step * 1.0,
                class_name="car", class_id=2, confidence=0.90,
                bbox=BoundingBoxXYXY(x1=100.0 + step * 10, y1=100.0, x2=160.0 + step * 10, y2=140.0),
                bbox_center=(130.0 + step * 10, 120.0), bbox_area=2400.0,
            )
            tracks = self.tracker.update([det], timestamp=step * 1.0)

        track = tracks[0]
        self.assertEqual(len(track.trajectory), 5)
        self.assertEqual(track.trajectory[0].center, (130.0, 120.0))
        self.assertEqual(track.trajectory[-1].center, (170.0, 120.0))

    def test_6_velocity_estimation(self):
        """Test 6: Velocity estimation works."""
        prev_center = (100.0, 100.0)
        curr_center = (120.0, 100.0)
        dt = 2.0

        vx, vy, speed, direction = MotionEstimator.calculate_velocity_and_direction(prev_center, curr_center, dt)
        self.assertEqual(vx, 10.0)  # (120-100)/2 = 10 px/s
        self.assertEqual(vy, 0.0)
        self.assertEqual(speed, 10.0)
        self.assertEqual(direction, Direction.EAST)

    def test_7_direction_estimation(self):
        """Test 7: Direction estimation categories work."""
        # 1. Moving East
        _, _, _, dir_east = MotionEstimator.calculate_velocity_and_direction((100, 100), (150, 100), 1.0)
        self.assertEqual(dir_east, Direction.EAST)

        # 2. Moving North (dy negative in image coordinates)
        _, _, _, dir_north = MotionEstimator.calculate_velocity_and_direction((100, 100), (100, 50), 1.0)
        self.assertEqual(dir_north, Direction.NORTH)

        # 3. Stationary
        _, _, _, dir_stat = MotionEstimator.calculate_velocity_and_direction((100, 100), (100.5, 100.5), 1.0)
        self.assertEqual(dir_stat, Direction.STATIONARY)

    def test_8_temporary_detection_loss_retains_id(self):
        """Test 8: Temporary detection loss does not immediately create a new ID."""
        det = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.85,
            bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=160.0, y2=140.0),
            bbox_center=(130.0, 120.0), bbox_area=2400.0,
        )
        self.tracker.update([det], timestamp=0.0)

        # Frame 2: Missed detection (empty)
        tracks_lost = self.tracker.update([], timestamp=1.0)
        self.assertEqual(len(tracks_lost), 1)
        self.assertEqual(tracks_lost[0].status, TrackStatus.LOST)
        self.assertEqual(tracks_lost[0].missed_frames, 1)

        # Frame 3: Detection reappears
        tracks_recovered = self.tracker.update([det], timestamp=2.0)
        self.assertEqual(len(tracks_recovered), 1)
        self.assertEqual(tracks_recovered[0].track_id, tracks_lost[0].track_id)
        self.assertEqual(tracks_recovered[0].missed_frames, 0)

    def test_9_track_expiration_after_max_missed_frames(self):
        """Test 9: Track expiration works after configured max missed frames."""
        det = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.85,
            bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=160.0, y2=140.0),
            bbox_center=(130.0, 120.0), bbox_area=2400.0,
        )
        self.tracker.update([det], timestamp=0.0)

        # Miss 6 consecutive frames (max_lost_frames=5)
        for i in range(1, 7):
            tracks = self.tracker.update([], timestamp=i * 1.0)

        self.assertEqual(len(tracks), 0)
        self.assertEqual(len(self.tracker.removed_tracks), 1)

    def test_10_low_and_high_conf_association(self):
        """Test 10: Low-confidence detection recovers active track in Stage 2."""
        # 1. High conf detection establishes track
        det_high = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.85,
            bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=160.0, y2=140.0),
            bbox_center=(130.0, 120.0), bbox_area=2400.0,
        )
        tracks1 = self.tracker.update([det_high], timestamp=0.0)
        orig_id = tracks1[0].track_id

        # 2. Next frame has low confidence (0.35, between low=0.20 and high=0.60)
        det_low = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=2, timestamp=1.0,
            class_name="car", class_id=2, confidence=0.35,
            bbox=BoundingBoxXYXY(x1=102.0, y1=100.0, x2=162.0, y2=140.0),
            bbox_center=(132.0, 120.0), bbox_area=2400.0,
        )
        tracks2 = self.tracker.update([det_low], timestamp=1.0)

        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].track_id, orig_id)
        self.assertEqual(tracks2[0].hits, 2)

    def test_11_detection_event_to_track_adapter(self):
        """Test 11: Phase 2 DetectionEvent -> Phase 3 TrackState adapter works."""
        obs = CameraObservation(
            observation_id="OBS_101",
            camera_id="CAM_A_EAST",
            junction_id="J_A",
            road_id="R_AB",
            timestamp=5.0,
            frame_id=50,
            vehicle_id="V_100_TN09AB1234",
            tracking_id=100,
            vehicle_type=VehicleType.CAR,
            plate_number="TN09AB1234",
            speed_kmh=40.0,
            lane_id=0,
            position_x=120.0,
            position_y=80.0,
            bbox=BoundingBox(x=100.0, y=50.0, width=64.0, height=48.0),
            detection_confidence=0.95,
            ocr_confidence=0.90,
        )

        det_event = DetectionAdapter.from_camera_observation(obs)
        tracks = self.manager.update("CAM_A_EAST", [det_event], timestamp=5.0)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].camera_id, "CAM_A_EAST")
        self.assertEqual(tracks[0].raw_vehicle_reference, "V_100_TN09AB1234")

    def test_12_phase_1_to_2_to_3_end_to_end_integration(self):
        """Test 12: End-to-end integration: Phase 1 simulation -> Phase 2 detection -> Phase 3 tracking."""
        sim = TrafficSimulationEngine(seed=42)

        # Run 10 simulation steps
        for _ in range(10):
            snapshot = sim.step(dt_seconds=1.0)

        obs_list = sim.recent_observations
        self.assertGreater(len(obs_list), 0)

        det_events = DetectionAdapter.batch_convert(obs_list)
        self.assertEqual(len(det_events), len(obs_list))

        # Feed detections into tracking manager
        active_tracks = self.manager.update(
            camera_id=det_events[0].camera_id,
            detections=det_events,
            timestamp=sim.sim_time,
        )
        self.assertGreater(len(active_tracks), 0)

    def test_13_camera_local_tracking_separation(self):
        """Test 13: Camera-local tracking isolation produces distinct track namespaces."""
        det_cam_a = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.85,
            bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=160.0, y2=140.0),
            bbox_center=(130.0, 120.0), bbox_area=2400.0,
        )
        det_cam_b = DetectionEvent(
            camera_id="CAM_B_WEST", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.85,
            bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=160.0, y2=140.0),
            bbox_center=(130.0, 120.0), bbox_area=2400.0,
        )

        tracks_a = self.manager.update("CAM_A_EAST", [det_cam_a], timestamp=0.0)
        tracks_b = self.manager.update("CAM_B_WEST", [det_cam_b], timestamp=0.0)

        self.assertEqual(tracks_a[0].camera_id, "CAM_A_EAST")
        self.assertEqual(tracks_b[0].camera_id, "CAM_B_WEST")
        # Ensure separate camera tracker instances
        self.assertIn("CAM_A_EAST", self.manager.trackers)
        self.assertIn("CAM_B_WEST", self.manager.trackers)

    def test_14_deterministic_reproducibility(self):
        """Test 14: Deterministic tracking reproducibility across identical runs."""
        manager1 = VehicleTrackerManager(config=self.config)
        manager2 = VehicleTrackerManager(config=self.config)

        det = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.85,
            bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=160.0, y2=140.0),
            bbox_center=(130.0, 120.0), bbox_area=2400.0,
        )

        t1 = manager1.update("CAM_A_EAST", [det], timestamp=0.0)
        t2 = manager2.update("CAM_A_EAST", [det], timestamp=0.0)

        self.assertEqual(t1[0].track_id, t2[0].track_id)
        self.assertEqual(t1[0].current_center, t2[0].current_center)

    def test_15_detection_quality_gating(self):
        """Test 15: Detection quality gating rejects invalid area, extreme aspect ratios, and zero coordinates."""
        invalid_det_area = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.85,
            bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=105.0, y2=105.0), # 25 px^2 area < 100 min
            bbox_center=(102.5, 102.5), bbox_area=25.0,
        )
        invalid_det_aspect = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.85,
            bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=600.0, y2=105.0), # 500x5 aspect ratio 100 > 5.0
            bbox_center=(350.0, 102.5), bbox_area=2500.0,
        )
        valid_det = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.85,
            bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=200.0, y2=180.0),
            bbox_center=(150.0, 140.0), bbox_area=8000.0,
        )

        tracks = self.tracker.update([invalid_det_area, invalid_det_aspect, valid_det], timestamp=0.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].current_bbox.area, 8000.0)

    def test_16_track_reactivation_after_occlusion(self):
        """Test 16: Lost track reactivates seamlessly with motion projection when vehicle emerges."""
        # 1. Initialize track over 3 frames moving East
        for f in range(1, 4):
            det = DetectionEvent(
                camera_id="CAM_A_EAST", frame_id=f, timestamp=f * 0.1,
                class_name="car", class_id=2, confidence=0.90,
                bbox=BoundingBoxXYXY(x1=100.0 + f * 50, y1=100.0, x2=160.0 + f * 50, y2=140.0),
                bbox_center=(130.0 + f * 50, 120.0), bbox_area=2400.0,
            )
            tracks = self.tracker.update([det], timestamp=f * 0.1)

        orig_id = tracks[0].track_id
        self.assertEqual(tracks[0].status, TrackStatus.CONFIRMED)

        # 2. Miss 2 frames (Occlusion / LOST state)
        tracks_lost = self.tracker.update([], timestamp=0.4)
        self.assertEqual(tracks_lost[0].status, TrackStatus.LOST)

        # 3. Vehicle emerges near predicted location -> REACTIVATED state
        det_emerge = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=6, timestamp=0.6,
            class_name="car", class_id=2, confidence=0.90,
            bbox=BoundingBoxXYXY(x1=100.0 + 6 * 50, y1=100.0, x2=160.0 + 6 * 50, y2=140.0),
            bbox_center=(130.0 + 6 * 50, 120.0), bbox_area=2400.0,
        )
        tracks_reactivated = self.tracker.update([det_emerge], timestamp=0.6)
        self.assertEqual(len(tracks_reactivated), 1)
        self.assertEqual(tracks_reactivated[0].track_id, orig_id)
        self.assertEqual(tracks_reactivated[0].status, TrackStatus.REACTIVATED)
        self.assertGreater(tracks_reactivated[0].reactivation_count, 0)

    def test_17_class_consistency_voting(self):
        """Test 17: Dominant vehicle class is maintained despite single noisy misclassifications."""
        classes = ["car", "car", "truck", "car"]  # 3 cars vs 1 noisy truck
        for i, cls in enumerate(classes):
            det = DetectionEvent(
                camera_id="CAM_A_EAST", frame_id=i + 1, timestamp=i * 0.1,
                class_name=cls, class_id=2, confidence=0.85,
                bbox=BoundingBoxXYXY(x1=100.0 + i * 2, y1=100.0, x2=160.0 + i * 2, y2=140.0),
                bbox_center=(130.0 + i * 2, 120.0), bbox_area=2400.0,
            )
            tracks = self.tracker.update([det], timestamp=i * 0.1)

        self.assertEqual(tracks[0].vehicle_type, "car")
        self.assertEqual(tracks[0].class_votes["car"], 3)
        self.assertEqual(tracks[0].class_votes["truck"], 1)

    def test_18_duplicate_track_suppression(self):
        """Test 18: Heavily overlapping duplicate active tracks are suppressed."""
        det1 = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.90,
            bbox=BoundingBoxXYXY(x1=100.0, y1=100.0, x2=160.0, y2=140.0),
            bbox_center=(130.0, 120.0), bbox_area=2400.0,
        )
        det2 = DetectionEvent(
            camera_id="CAM_A_EAST", frame_id=1, timestamp=0.0,
            class_name="car", class_id=2, confidence=0.88,
            bbox=BoundingBoxXYXY(x1=101.0, y1=101.0, x2=161.0, y2=141.0), # 95%+ IoU duplicate
            bbox_center=(131.0, 121.0), bbox_area=2400.0,
        )

        tracks = self.tracker.update([det1, det2], timestamp=0.0)
        # One of the duplicates should be suppressed by intra-frame NMS or duplicate track handler
        self.assertEqual(len(tracks), 1)

    def test_19_track_quality_score_and_bounded_trajectory(self):
        """Test 19: Track quality score is calculated and trajectory memory is bounded."""
        self.tracker.config.max_trajectory_length = 5
        for f in range(1, 10):
            det = DetectionEvent(
                camera_id="CAM_A_EAST", frame_id=f, timestamp=f * 0.1,
                class_name="car", class_id=2, confidence=0.90,
                bbox=BoundingBoxXYXY(x1=100.0 + f, y1=100.0, x2=160.0 + f, y2=140.0),
                bbox_center=(130.0 + f, 120.0), bbox_area=2400.0,
            )
            tracks = self.tracker.update([det], timestamp=f * 0.1)

        track = tracks[0]
        self.assertGreater(track.track_quality_score, 0.0)
        self.assertLessEqual(track.track_quality_score, 1.0)
        self.assertEqual(len(track.trajectory), 5)  # Bounded to max 5


if __name__ == "__main__":
    unittest.main()

