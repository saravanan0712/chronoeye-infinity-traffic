"""
ChronoEye Infinity - Phase 3: Vehicle Tracking Schemas
Defines TrackState models, lifecycle status enums, direction vectors,
trajectory points, and tracker configuration schemas.
"""

import uuid
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field, model_validator
from app.schemas.detection import BoundingBoxXYXY, DetectionEvent


class TrackStatus(str, Enum):
    NEW = "NEW"
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    REACTIVATED = "REACTIVATED"
    REMOVED = "REMOVED"


class Direction(str, Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"
    NORTHEAST = "NORTHEAST"
    NORTHWEST = "NORTHWEST"
    SOUTHEAST = "SOUTHEAST"
    SOUTHWEST = "SOUTHWEST"
    STATIONARY = "STATIONARY"
    UNKNOWN = "UNKNOWN"


class TrackerBackend(str, Enum):
    BYTETRACK = "bytetrack"
    DEEPSORT = "deepsort"


class TrajectoryPoint(BaseModel):
    """
    Single point in a vehicle track trajectory sequence.
    """
    timestamp: float
    frame_id: int
    bbox: BoundingBoxXYXY
    center: Tuple[float, float]
    confidence: float
    speed_estimate: float = 0.0  # Image-space displacement pixels/sec or normalized speed
    direction: Direction = Direction.UNKNOWN
    camera_id: str


class TrackState(BaseModel):
    """
    Persistent vehicle track state produced by Phase 3 tracking pipeline.
    Maintains track lifecycle, motion vectors, trajectory history, and detection links.
    """
    track_id: str = Field(default_factory=lambda: f"TRK_{uuid.uuid4().hex[:8]}")
    camera_id: str
    vehicle_type: str  # "car", "bus", "truck", "motorcycle", "ambulance"
    current_bbox: BoundingBoxXYXY
    current_center: Tuple[float, float]
    confidence: float
    first_seen_timestamp: float
    last_seen_timestamp: float
    age: int = 1
    hits: int = 1
    missed_frames: int = 0
    status: TrackStatus = TrackStatus.TENTATIVE
    confirmed: bool = False
    active: bool = True
    velocity_x: float = 0.0  # pixels/sec in X
    velocity_y: float = 0.0  # pixels/sec in Y
    speed_estimate: float = 0.0
    direction: Direction = Direction.UNKNOWN
    trajectory: List[TrajectoryPoint] = Field(default_factory=list)
    detection_history: List[str] = Field(default_factory=list)  # List of detection_ids
    raw_vehicle_reference: Optional[str] = None  # Links to Phase 1 simulation vehicle ID if adapted
    predicted_bbox: Optional[BoundingBoxXYXY] = None
    raw_bbox: Optional[BoundingBoxXYXY] = None
    class_votes: Dict[str, int] = Field(default_factory=dict)
    track_quality_score: float = 1.0
    reactivation_count: int = 0
    history: List[Any] = Field(default_factory=list)

    @property
    def bbox(self) -> BoundingBoxXYXY:
        """Backward-compatible property alias for current_bbox."""
        return self.current_bbox

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_fields(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "camera_id" not in values:
                values["camera_id"] = "CAM_UNKNOWN"

            if "current_bbox" not in values and "bbox" in values:
                b = values["bbox"]
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    values["current_bbox"] = BoundingBoxXYXY(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]))
                else:
                    values["current_bbox"] = b

            if "vehicle_type" not in values:
                if "class_name" in values:
                    values["vehicle_type"] = values["class_name"]
                else:
                    values["vehicle_type"] = "car"

            if "speed_estimate" not in values and "velocity" in values:
                v = values["velocity"]
                if isinstance(v, (int, float)):
                    values["speed_estimate"] = float(v)

            if "current_center" not in values:
                if "current_bbox" in values:
                    bbox = values["current_bbox"]
                    if hasattr(bbox, "center"):
                        values["current_center"] = bbox.center
                    elif isinstance(bbox, dict):
                        x1 = bbox.get("x1", 0)
                        y1 = bbox.get("y1", 0)
                        x2 = bbox.get("x2", 0)
                        y2 = bbox.get("y2", 0)
                        values["current_center"] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    else:
                        values["current_center"] = (0.0, 0.0)
                else:
                    values["current_center"] = (0.0, 0.0)

            if "confidence" not in values:
                values["confidence"] = 1.0
            
            if "first_seen_timestamp" not in values:
                values["first_seen_timestamp"] = 0.0

            if "last_seen_timestamp" not in values:
                values["last_seen_timestamp"] = values["first_seen_timestamp"]
        return values


class TrackerConfig(BaseModel):
    """
    Configuration parameters for multi-object tracking pipeline.
    """
    tracker_backend: TrackerBackend = TrackerBackend.BYTETRACK
    high_conf_thresh: float = 0.60
    low_conf_thresh: float = 0.20
    iou_threshold: float = 0.30
    max_lost_frames: int = 30
    min_confirmation_hits: int = 3
    max_track_age: int = 300
    track_id_prefix: str = "TRK_"

    # Advanced Quality Gating & Multi-Factor Association Parameters
    min_bbox_area: float = 100.0
    max_aspect_ratio: float = 5.0
    duplicate_iou_thresh: float = 0.80
    weight_iou: float = 0.40
    weight_motion: float = 0.25
    weight_size: float = 0.15
    weight_class: float = 0.10
    weight_confidence: float = 0.10
    bbox_smoothing_alpha: float = 0.70
    max_trajectory_length: int = 500
    max_velocity_pixels_per_sec: float = 1500.0

