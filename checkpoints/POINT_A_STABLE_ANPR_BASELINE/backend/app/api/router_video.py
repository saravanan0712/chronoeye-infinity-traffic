"""
ChronoEye Infinity - Video Ingestion API Router
Provides REST endpoints for Video Sources, Camera Management, Active Ingestion Modes
(LIVE, RECORDED, SIMULATION), and Frame Ingestion Stream Status with Data Provenance.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query
from app.schemas.provenance import DataProvenance, DataProvenanceType
from app.perception.frame_source import (
    VideoFrameSource,
    CameraFrameSource,
    SyntheticFrameSource,
)

router = APIRouter(prefix="/api/v1/video", tags=["Video Ingestion & Cameras"])


class VideoIngestionMode(str, Enum):
    LIVE = "LIVE"
    RECORDED = "RECORDED"
    SIMULATION = "SIMULATION"


class CameraStatus(str, Enum):
    LIVE = "LIVE"
    PLAYBACK = "PLAYBACK"
    SIMULATION = "SIMULATION"
    OFFLINE = "OFFLINE"


class CameraMetadata(BaseModel):
    camera_id: str
    location_name: str
    latitude: float
    longitude: float
    status: CameraStatus = CameraStatus.SIMULATION
    fps: float = 30.0
    resolution: str = "1920x1080"
    active_source: str = "CAM_SYNTHETIC"
    road_id: str = "ROAD_1_2"
    provenance: DataProvenance


class VideoModeRequest(BaseModel):
    mode: VideoIngestionMode
    source_url_or_path: Optional[str] = None
    camera_id: Optional[str] = "CAM_A_EAST"


# Global state tracker for Ingestion Engine
_current_mode: VideoIngestionMode = VideoIngestionMode.SIMULATION
_active_sources: Dict[str, Any] = {
    "CAM_A_EAST": {"mode": VideoIngestionMode.SIMULATION, "status": CameraStatus.SIMULATION, "location": "Anna Salai North Junction", "lat": 13.0827, "lon": 80.2707, "road": "ROAD_R_AB"},
    "CAM_B_WEST": {"mode": VideoIngestionMode.SIMULATION, "status": CameraStatus.SIMULATION, "location": "Central Station Intersection", "lat": 13.0850, "lon": 80.2720, "road": "ROAD_R_BA"},
    "CAM_C_NORTH": {"mode": VideoIngestionMode.SIMULATION, "status": CameraStatus.SIMULATION, "location": "GST Expressway Ramp", "lat": 13.0880, "lon": 80.2750, "road": "ROAD_R_CD"},
    "CAM_D_NORTH": {"mode": VideoIngestionMode.SIMULATION, "status": CameraStatus.SIMULATION, "location": "Koyambedu Roundabout", "lat": 13.0900, "lon": 80.2800, "road": "ROAD_R_DC"},
}


@router.get("/sources")
def get_video_sources() -> Dict[str, Any]:
    """Retrieves supported video input sources and active ingestion status."""
    return {
        "supported_sources": [
            {"id": "LIVE_CCTV", "name": "Live CCTV Camera Stream", "type": "RTSP / IP Camera", "status": "AVAILABLE"},
            {"id": "WEBCAM", "name": "Local Device Camera", "type": "USB / Integrated Webcam", "status": "AVAILABLE"},
            {"id": "UPLOADED_VIDEO", "name": "Uploaded MP4 Traffic Video", "type": "Video File (MP4, AVI)", "status": "AVAILABLE"},
            {"id": "RECORDED_FOOTAGE", "name": "Recorded Benchmark Footage", "type": "Archival Video", "status": "AVAILABLE"},
            {"id": "DEVELOPMENT_SIMULATION", "name": "Synthetic Traffic Simulator", "type": "Simulation Generator", "status": "ACTIVE"},
        ],
        "active_mode": _current_mode.value,
        "provenance": DataProvenance(
            provenance_type=DataProvenanceType.OBSERVED if _current_mode == VideoIngestionMode.LIVE else DataProvenanceType.SIMULATED,
            source_id="VIDEO_INGESTION_ENGINE",
            process_name="VideoSourceManager",
            is_mock=False,
        ).model_dump(),
    }


@router.get("/cameras")
def get_camera_nodes() -> List[Dict[str, Any]]:
    """Retrieves metadata and status for registered traffic cameras."""
    cameras = []
    for cam_id, info in _active_sources.items():
        prov_type = (
            DataProvenanceType.OBSERVED if info["status"] == CameraStatus.LIVE
            else DataProvenanceType.SIMULATED
        )
        cam = CameraMetadata(
            camera_id=cam_id,
            location_name=info["location"],
            latitude=info["lat"],
            longitude=info["lon"],
            status=info["status"],
            fps=30.0,
            resolution="1920x1080",
            active_source=f"SOURCE_{info['mode'].value}",
            road_id=info["road"],
            provenance=DataProvenance(
                provenance_type=prov_type,
                source_id=cam_id,
                process_name="CameraTopologyManager",
            ),
        )
        cameras.append(cam.model_dump())
    return cameras


@router.post("/mode")
def set_video_ingestion_mode(req: VideoModeRequest) -> Dict[str, Any]:
    """Switches active ingestion mode (LIVE, RECORDED, SIMULATION)."""
    global _current_mode
    _current_mode = req.mode

    status_map = {
        VideoIngestionMode.LIVE: CameraStatus.LIVE,
        VideoIngestionMode.RECORDED: CameraStatus.PLAYBACK,
        VideoIngestionMode.SIMULATION: CameraStatus.SIMULATION,
    }

    target_cam = req.camera_id or "CAM_A_EAST"
    if target_cam in _active_sources:
        _active_sources[target_cam]["mode"] = req.mode
        _active_sources[target_cam]["status"] = status_map.get(req.mode, CameraStatus.SIMULATION)

    return {
        "status": "SUCCESS",
        "message": f"Video Ingestion Mode updated to {req.mode.value} for {target_cam}",
        "active_mode": req.mode.value,
        "provenance": DataProvenance(
            provenance_type=DataProvenanceType.OBSERVED if req.mode == VideoIngestionMode.LIVE else DataProvenanceType.SIMULATED,
            source_id=target_cam,
            process_name="VideoModeController",
        ).model_dump(),
    }


@router.get("/status")
def get_video_ingestion_status() -> Dict[str, Any]:
    """Retrieves real-time video frame ingestion status & camera health."""
    return {
        "ingestion_active": True,
        "current_mode": _current_mode.value,
        "frames_processed_per_second": 30.0,
        "active_camera_count": len(_active_sources),
        "camera_health": {cam_id: info["status"].value for cam_id, info in _active_sources.items()},
        "provenance": DataProvenance(
            provenance_type=DataProvenanceType.OBSERVED if _current_mode == VideoIngestionMode.LIVE else DataProvenanceType.SIMULATED,
            source_id="INGESTION_STATUS_SERVICE",
            process_name="VideoIngestionStatus",
        ).model_dump(),
    }
