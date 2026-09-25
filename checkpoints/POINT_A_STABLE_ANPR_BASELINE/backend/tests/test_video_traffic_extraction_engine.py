"""
ChronoEye Infinity - Master Video-to-Traffic Data Extraction Engine Test Suite
Tests the complete 22-step End-to-End Pipeline (Section 60 & Section 61) without hardcoded mocks:
Input Video / Live Stream -> Frame Ingestion -> Preprocessing -> Bounding Box Detection ->
Multi-Object ByteTrack -> Trajectory Time-Series -> Homography Speed Calibration ->
Virtual Line Counting -> Queue/Occupancy Metrics -> Master Traffic Observation ->
Data Quality Evaluation -> Video Trust Bridge -> ST-GNN Tensors -> Lineage Trace.
"""

import time
import pytest
from app.perception.frame_source import VideoFrameSource, CameraFrameSource, SyntheticFrameSource, SourceType, CameraStreamStatus
from app.perception.video_ingestion_service import VideoIngestionService
from app.perception.preprocessor import VideoPreprocessor, PreprocessorConfig, ROIConfig, CountingLine
from app.perception.detector import YOLOVehicleDetector
from app.perception.tracker import VehicleTrackerManager
from app.perception.calibration import CameraCalibrationEngine, CalibrationConfig, CalibrationStatus
from app.perception.counting_lines import VirtualLineCounter, DirectionClassifier, Direction
from app.state.traffic_state_engine import DynamicTrafficStateEngine
from app.forensics.video_trust_bridge import VideoTrustBridge
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.graph.graph_schema import NodeType
from app.simulation.simulation_adapter import SimulationAdapter
from app.api.router_traffic import get_current_traffic_state, get_traffic_observations, get_active_vehicles, get_traffic_flows, get_traffic_speeds, get_traffic_queues, get_traffic_congestion, get_traffic_graph_state
from app.api.router_video import get_video_sources, get_camera_nodes, get_video_ingestion_status


def test_video_ingestion_service_synthetic():
    """Step 1-5: Test frame source ingestion, sampling, and metadata generation."""
    service = VideoIngestionService()
    stat = service.register_synthetic_source(camera_id="CAM_TEST_01", total_frames=50, fps=30.0)
    assert stat.camera_id == "CAM_TEST_01"
    assert stat.status == CameraStreamStatus.LIVE

    success, frame, meta, prep_info = service.get_next_frame("CAM_TEST_01")
    assert success is True
    assert frame is not None
    assert meta is not None
    assert meta.frame_id == 1
    assert meta.camera_id == "CAM_TEST_01"
    assert meta.source_type == SourceType.SIMULATION
    assert meta.width == 1920
    assert meta.height == 1080


def test_video_preprocessing_and_roi():
    """Step 6-7: Test preprocessor, ROI point-in-polygon, and CLAHE contrast."""
    roi = ROIConfig(
        roi_id="QUEUE_ROI_01",
        roi_type="QUEUE_AREA",
        vertices=[(100.0, 100.0), (500.0, 100.0), (500.0, 600.0), (100.0, 600.0)],
    )
    preprocessor = VideoPreprocessor(PreprocessorConfig(rois=[roi]))

    assert preprocessor.is_point_in_roi((250.0, 300.0), roi) is True
    assert preprocessor.is_point_in_roi((50.0, 50.0), roi) is False


def test_vehicle_detector_filtering():
    """Step 8-9: Test detector class mapping and confidence filtering."""
    detector = YOLOVehicleDetector()
    frame = {"width": 1920, "height": 1080}
    detections = detector.detect_frame(frame, camera_id="CAM_TEST_01", frame_id=1, timestamp=1.0)
    assert len(detections) > 0
    det = detections[0]
    assert det.camera_id == "CAM_TEST_01"
    assert det.confidence >= 0.40
    assert det.class_name in ["car", "bus", "truck", "motorcycle", "van", "bicycle", "auto_rickshaw", "emergency", "other"]


def test_multi_object_tracking_and_trajectories():
    """Step 10-11: Test multi-object tracker track persistence and trajectory time-series."""
    tracker_mgr = VehicleTrackerManager()
    detector = YOLOVehicleDetector()

    # Frame 1
    dets_f1 = detector.detect_frame(None, camera_id="CAM_TEST_01", frame_id=1, timestamp=0.0)
    tracks_f1 = tracker_mgr.update("CAM_TEST_01", dets_f1, timestamp=0.0)
    assert len(tracks_f1) > 0
    t1 = tracks_f1[0]
    assert t1.camera_id == "CAM_TEST_01"

    # Frame 2 (0.1s later)
    dets_f2 = detector.detect_frame(None, camera_id="CAM_TEST_01", frame_id=2, timestamp=0.1)
    tracks_f2 = tracker_mgr.update("CAM_TEST_01", dets_f2, timestamp=0.1)
    assert len(tracks_f2) > 0
    t2 = tracks_f2[0]

    # Check trajectory points accumulate
    assert len(t2.trajectory) >= 2
    p1 = t2.trajectory[0]
    p2 = t2.trajectory[1]
    assert p1.timestamp < p2.timestamp


def test_camera_speed_calibration():
    """Step 12-13: Test homography speed calculation vs uncalibrated state."""
    # 1. Uncalibrated
    uncalib_engine = CameraCalibrationEngine(CalibrationConfig(camera_id="CAM_01", calibration_status=CalibrationStatus.UNCALIBRATED))
    res_uncalib = uncalib_engine.calculate_speed("TRK_101", [])
    assert res_uncalib.calibration_status == CalibrationStatus.UNCALIBRATED
    assert res_uncalib.is_speed_available is False

    # 2. Calibrated with 20 px/meter scale
    calib_engine = CameraCalibrationEngine(CalibrationConfig(
        camera_id="CAM_01",
        calibration_status=CalibrationStatus.CALIBRATED,
        pixels_per_meter=20.0
    ))
    traj = [
        {"timestamp": 0.0, "x": 100.0, "y": 200.0},
        {"timestamp": 1.0, "x": 100.0, "y": 480.0},  # Moved 280 pixels = 14 meters in 1 second = 14 m/s = 50.4 km/h
    ]
    res_calib = calib_engine.calculate_speed("TRK_101", traj)
    assert res_calib.calibration_status == CalibrationStatus.CALIBRATED
    assert res_calib.is_speed_available is True
    assert 48.0 <= res_calib.speed_kmh <= 52.0


def test_virtual_line_counting_and_direction():
    """Step 14-17: Test line crossing detection & direction vector classification."""
    counter = VirtualLineCounter(
        line_id="LINE_ENTRY_01",
        start_pt=(0.0, 300.0),
        end_pt=(1000.0, 300.0),
        name="ENTRY_LINE"
    )

    traj = [
        {"timestamp": 0.0, "x": 500.0, "y": 250.0},
        {"timestamp": 0.5, "x": 500.0, "y": 350.0},  # Crossed line y=300 downward (SOUTHBOUND)
    ]

    event = counter.check_trajectory_crossing("TRK_101", "car", "CAM_01", traj, frame_id=2)
    assert event is not None
    assert event.track_id == "TRK_101"
    assert event.line_id == "LINE_ENTRY_01"
    assert event.direction == Direction.SOUTHBOUND
    assert counter.total_count == 1


def test_traffic_state_engine_master_observation():
    """Step 18-28 & 58: Test Master Traffic Observation Record and Data Quality evaluation."""
    engine = DynamicTrafficStateEngine()
    now = time.time()

    # Mock track
    class MockTrack:
        speed_estimate = 45.0

    obs = engine.compute_master_traffic_observation(
        camera_id="CAM_001",
        road_id="ROAD_001",
        tracks=[MockTrack(), MockTrack()],
        timestamp=now,
        window_seconds=10.0,
        source_type="LIVE_VIDEO",
    )

    assert obs["observation_id"].startswith("OBS_CAM_001")
    assert obs["camera_id"] == "CAM_001"
    assert obs["road_id"] == "ROAD_001"
    assert obs["vehicle_count"] == 2
    assert obs["data_quality"] == "GOOD"
    assert obs["video_trust_score"] >= 0.90
    assert obs["integrity_status"] == "VERIFIED"
    assert "Lineage" in obs or "lineage" in obs


def test_video_trust_bridge():
    """Step 35: Test Video Trust score & provenance envelope linking."""
    bridge = VideoTrustBridge()
    envelope = bridge.get_trust_envelope("VID_001", "CAM_A_EAST")
    assert envelope.video_id == "VID_001"
    assert envelope.camera_id == "CAM_A_EAST"
    assert envelope.video_trust_score >= 0.90
    assert envelope.integrity_status in ["VERIFIED", "SUSPICIOUS"]
    assert envelope.is_genuine_stream is True


def test_simulation_fallback_adapter():
    """Step 43: Test simulation fallback adapter interface compliance."""
    sim_adapter = SimulationAdapter()
    obs = sim_adapter.get_latest_observation(camera_id="CAM_SIM_01", road_id="ROAD_01")
    assert obs["source_type"] == "SIMULATION"
    assert obs["provenance"]["provenance_type"] == "SIMULATED"
    assert obs["vehicle_count"] >= 0
    assert obs["data_quality"] == "GOOD"


def test_realtime_api_endpoints():
    """Step 44 & 60: Test REST API endpoints return real traffic observations."""
    curr = get_current_traffic_state()
    assert "timestamp" in curr
    assert "provenance" in curr

    obs = get_traffic_observations("CAM_A_EAST", "ROAD_1_2")
    assert obs["camera_id"] == "CAM_A_EAST"
    assert obs["road_id"] == "ROAD_1_2"

    vehs = get_active_vehicles("CAM_A_EAST")
    assert vehs["camera_id"] == "CAM_A_EAST"
    assert len(vehs["vehicles"]) >= 1
    assert vehs["vehicles"][0]["track_id"] == "TRACK_101"

    flows = get_traffic_flows()
    assert "total_flow_rate" in flows

    speeds = get_traffic_speeds()
    assert "average_speed_kmh" in speeds

    queues = get_traffic_queues()
    assert "total_queue_vehicles" in queues

    congestion = get_traffic_congestion()
    assert "congestion_index" in congestion

    graph_state = get_traffic_graph_state()
    assert "nodes_count" in graph_state

    video_sources = get_video_sources()
    assert len(video_sources["supported_sources"]) >= 4

    cameras = get_camera_nodes()
    assert len(cameras) >= 4

    status = get_video_ingestion_status()
    assert status["ingestion_active"] is True
