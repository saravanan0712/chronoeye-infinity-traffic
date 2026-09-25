"""
ChronoEye Infinity - Multi-Camera Runner Unit & Integration Test Suite
Verifies multi-camera orchestration, cross-camera identity merging, spatial-temporal constraints,
thread safety, and reporting in multi_camera_runner.py.
"""

import os
import sys
import unittest
import threading
from typing import Tuple

# Add backend and root directories to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.schemas.detection import BoundingBoxXYXY
from app.schemas.tracking import TrackState, TrackStatus, Direction
from app.schemas.plate import FusedPlateIdentity, VehicleIdentityEvidence
from app.schemas.reid import ReIDConfig
from app.perception.camera_topology import CityCameraTopology
from app.perception.reid_engine import ReIDMatchingEngine
from app.perception.journey import JourneyReconstructionEngine

from multi_camera_runner import (
    MultiCameraRunner,
    MultiCameraRunnerConfig,
    CameraSourceConfig,
    parse_camera_arg,
)


class TestMultiCameraRunner(unittest.TestCase):
    """Deterministic unit tests for MultiCameraRunner and cross-camera journey reconstruction."""

    def setUp(self):
        self.topology = CityCameraTopology()
        self.config = ReIDConfig()
        self.matching_engine = ReIDMatchingEngine(topology=self.topology, config=self.config)
        self.journey_engine = JourneyReconstructionEngine(
            matching_engine=self.matching_engine, config=self.config
        )
        self.runner_cfg = MultiCameraRunnerConfig(
            cameras=[
                CameraSourceConfig(camera_id="CAM_A_EAST", source="dummy_a.mp4"),
                CameraSourceConfig(camera_id="CAM_B_WEST", source="dummy_b.mp4"),
            ],
            verbose=False,
        )
        self.runner = MultiCameraRunner(
            config=self.runner_cfg,
            journey_engine=self.journey_engine,
            matching_engine=self.matching_engine,
            topology=self.topology,
        )

    def _create_mock_track(
        self,
        track_id: str,
        camera_id: str,
        timestamp: float,
        plate: str = "TN09AB1234",
        vehicle_type: str = "car",
        direction: Direction = Direction.EAST,
    ) -> Tuple[VehicleIdentityEvidence, TrackState]:
        track = TrackState(
            track_id=track_id,
            camera_id=camera_id,
            vehicle_type=vehicle_type,
            current_bbox=BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=200),
            current_center=(150, 150),
            confidence=0.9,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=direction,
        )
        fused_plate = FusedPlateIdentity(
            track_id=track_id,
            camera_id=camera_id,
            best_plate_number=plate,
            overall_confidence=0.95,
            observation_count=5,
            confirmed=True,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
        )
        evidence = VehicleIdentityEvidence(
            track_id=track_id,
            camera_id=camera_id,
            vehicle_type=vehicle_type,
            associated_plate=fused_plate,
            last_updated_timestamp=timestamp,
        )
        return evidence, track

    def test_single_camera_creates_journey(self):
        """Test 1: Single camera observation creates a new global vehicle journey."""
        ev, trk = self._create_mock_track("TRK_101", "CAM_A_EAST", 10.0, "KA01MJ5000")
        journey = self.runner.feed_track_evidence(ev, trk)

        self.assertIsNotNone(journey)
        self.assertTrue(journey.global_vehicle_id.startswith("VEH_"))
        self.assertTrue(journey.journey_id.startswith("JRN_"))
        self.assertEqual(journey.plate_number, "KA01MJ5000")
        self.assertEqual(len(journey.segments), 1)
        self.assertEqual(journey.cameras, ["CAM_A_EAST"])

    def test_two_cameras_same_plate_merge(self):
        """Test 2: Two cameras observing the same plate with feasible travel time merge into ONE journey."""
        # Distance between CAM_A_EAST and CAM_B_WEST is 300m in topology
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 10.0, "TS08EP9999")
        ev_b, trk_b = self._create_mock_track("TRK_205", "CAM_B_WEST", 35.0, "TS08EP9999")

        j_a = self.runner.feed_track_evidence(ev_a, trk_a)
        j_b = self.runner.feed_track_evidence(ev_b, trk_b)

        # Must merge into identical global vehicle and journey ID
        self.assertEqual(j_a.global_vehicle_id, j_b.global_vehicle_id)
        self.assertEqual(j_a.journey_id, j_b.journey_id)
        self.assertEqual(len(j_b.segments), 2)
        self.assertEqual(j_b.cameras, ["CAM_A_EAST", "CAM_B_WEST"])
        self.assertEqual(len(self.journey_engine.journeys), 1)

    def test_two_cameras_different_plates_no_merge(self):
        """Test 3: Distinct vehicles (different plate, class, and appearance) do not merge; create two distinct global identities."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 10.0, "MH02CB1234", vehicle_type="car", direction=Direction.EAST)
        ev_b, trk_b = self._create_mock_track("TRK_205", "CAM_B_WEST", 35.0, "DL01AA8888", vehicle_type="bus", direction=Direction.WEST)

        # Orthogonal visual appearance embeddings
        self.journey_engine.track_embeddings["TRK_101"] = (10.0, [1.0] + [0.0] * 127)
        self.journey_engine.track_embeddings["TRK_205"] = (35.0, [0.0, 1.0] + [0.0] * 126)

        j_a = self.runner.feed_track_evidence(ev_a, trk_a)
        j_b = self.runner.feed_track_evidence(ev_b, trk_b)

        self.assertNotEqual(j_a.global_vehicle_id, j_b.global_vehicle_id)
        self.assertNotEqual(j_a.journey_id, j_b.journey_id)
        self.assertEqual(len(self.journey_engine.journeys), 2)

    def test_impossible_travel_rejected(self):
        """Test 4: Physically impossible travel speed rejects merge and flags separate journey."""
        # Distance is 300m, time delta = 0.5s -> speed = 600 m/s (impossible)
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 10.0, "TN01AB0001")
        ev_b, trk_b = self._create_mock_track("TRK_205", "CAM_B_WEST", 10.5, "TN01AB0001")

        j_a = self.runner.feed_track_evidence(ev_a, trk_a)
        j_b = self.runner.feed_track_evidence(ev_b, trk_b)

        # Must NOT merge due to impossible travel constraint
        self.assertNotEqual(j_a.global_vehicle_id, j_b.global_vehicle_id)
        self.assertEqual(len(self.journey_engine.journeys), 2)

    def test_three_camera_journey_reconstruction(self):
        """Test 5: Vehicle traveling through 3 cameras (A -> B -> C) reconstructs 3-segment journey."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 10.0, "KA05XY1111")
        ev_b, trk_b = self._create_mock_track("TRK_202", "CAM_B_WEST", 35.0, "KA05XY1111")
        ev_c, trk_c = self._create_mock_track("TRK_303", "CAM_C_NORTH", 65.0, "KA05XY1111")

        j_a = self.runner.feed_track_evidence(ev_a, trk_a)
        j_b = self.runner.feed_track_evidence(ev_b, trk_b)
        j_c = self.runner.feed_track_evidence(ev_c, trk_c)

        self.assertEqual(j_a.global_vehicle_id, j_c.global_vehicle_id)
        self.assertEqual(len(j_c.segments), 3)
        self.assertEqual(j_c.cameras, ["CAM_A_EAST", "CAM_B_WEST", "CAM_C_NORTH"])
        self.assertEqual(j_c.first_seen, 10.0)
        self.assertEqual(j_c.last_seen, 65.0)

    def test_thread_safety_no_duplicate_ids(self):
        """Test 6: Thread-safe locking under concurrent multi-camera evidence injection."""
        threads = []
        errors = []

        def worker_feed(cam_id: str, start_idx: int):
            try:
                for i in range(15):
                    # Co-present simultaneous tracks on same camera at t=0.0 cannot merge into same journey
                    plate = f"TN{start_idx:02d}XX{i:04d}"
                    ev, trk = self._create_mock_track(f"TRK_{cam_id}_{i}", cam_id, 0.0, plate)
                    self.runner.feed_track_evidence(ev, trk)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=worker_feed, args=("CAM_A_EAST", 10))
        t2 = threading.Thread(target=worker_feed, args=("CAM_B_WEST", 20))
        t3 = threading.Thread(target=worker_feed, args=("CAM_C_NORTH", 30))

        for t in [t1, t2, t3]:
            t.start()
        for t in [t1, t2, t3]:
            t.join()

        self.assertEqual(len(errors), 0)
        all_global_ids = [j.global_vehicle_id for j in self.journey_engine.journeys.values()]
        # Check no duplicates in assigned global IDs and all 45 journeys uniquely created
        self.assertEqual(len(all_global_ids), len(set(all_global_ids)))
        self.assertEqual(len(all_global_ids), 45)

    def test_journey_summary_report_fields(self):
        """Test 7: Validate all required fields in the summary report."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 10.0, "AP09CC7777")
        ev_b, trk_b = self._create_mock_track("TRK_205", "CAM_B_WEST", 35.0, "AP09CC7777")
        self.runner.feed_track_evidence(ev_a, trk_a)
        self.runner.feed_track_evidence(ev_b, trk_b)

        res = self.runner._build_result(total_runtime=1.23)
        self.assertGreaterEqual(len(res.journey_summary), 1)
        item = res.journey_summary[0]

        expected_fields = [
            "global_vehicle_id",
            "journey_id",
            "status",
            "plate_number",
            "vehicle_type",
            "number_of_segments",
            "camera_sequence",
            "local_track_ids",
            "first_timestamp",
            "last_timestamp",
        ]
        for field_name in expected_fields:
            self.assertIn(field_name, item)

        self.assertEqual(item["plate_number"], "AP09CC7777")
        self.assertEqual(item["number_of_segments"], 2)
        self.assertEqual(item["camera_sequence"], ["CAM_A_EAST", "CAM_B_WEST"])
        self.assertIn("TRK_101", item["local_track_ids"])
        self.assertIn("TRK_205", item["local_track_ids"])

    def test_search_plate(self):
        """Test 8: Search plate finds matching journey records."""
        ev_a, trk_a = self._create_mock_track("TRK_101", "CAM_A_EAST", 10.0, "TS07FA1234")
        self.runner.feed_track_evidence(ev_a, trk_a)

        results = self.runner.search_plate("TS07FA1234")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["plate_number"], "TS07FA1234")
        self.assertEqual(results[0]["cameras"], ["CAM_A_EAST"])

        not_found = self.runner.search_plate("NONEXISTENT")
        self.assertEqual(len(not_found), 0)

    def test_parse_camera_arg(self):
        """Test 9: Verify CLI camera string parser."""
        c1 = parse_camera_arg("CAM_A:data/videos/cam_a.mp4")
        self.assertEqual(c1.camera_id, "CAM_A")
        self.assertEqual(c1.source, "data/videos/cam_a.mp4")
        self.assertEqual(c1.timestamp_offset_seconds, 0.0)

        c2 = parse_camera_arg("CAM_B:data/videos/cam_b.mp4:12.5")
        self.assertEqual(c2.camera_id, "CAM_B")
        self.assertEqual(c2.source, "data/videos/cam_b.mp4")
        self.assertEqual(c2.timestamp_offset_seconds, 12.5)


if __name__ == "__main__":
    unittest.main()
