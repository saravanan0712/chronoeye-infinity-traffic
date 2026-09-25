"""
ChronoEye Infinity - Phase 2: Vehicle Detection Schemas
Defines structured detection event schemas, bounding box conventions (x1, y1, x2, y2),
detector configuration, and class mappings for perception pipelines.
"""

import uuid
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, model_validator


class BoundingBoxXYXY(BaseModel):
    """
    Standard bounding box format (x1, y1, x2, y2).
    Coordinates are in pixels.
    """
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def check_coordinates(self) -> "BoundingBoxXYXY":
        if self.x2 < self.x1:
            raise ValueError("x2 must be greater than or equal to x1")
        if self.y2 < self.y1:
            raise ValueError("y2 must be greater than or equal to y1")
        return self

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height


class DetectionEvent(BaseModel):
    """
    Normalized Detection Event produced by Phase 2 vehicle detector.
    This schema is consumed by Phase 3 (Tracking), Phase 4 (OCR), and Phase 5 (Re-ID).
    """
    detection_id: str = Field(default_factory=lambda: f"DET_{uuid.uuid4().hex[:8]}")
    camera_id: str
    frame_id: int
    timestamp: float
    class_name: str  # "car", "bus", "truck", "motorcycle", "emergency"
    class_id: int    # COCO class ID or custom class ID
    confidence: float
    bbox: BoundingBoxXYXY
    bbox_center: Tuple[float, float]
    bbox_area: float
    image_width: int = 1920
    image_height: int = 1080
    source: str = "YOLO_V8"
    raw_vehicle_reference: Optional[str] = None  # Preserves vehicle ID link when adapted from simulation


class DetectorConfig(BaseModel):
    """
    Configuration parameters for YOLO Vehicle Detector.
    """
    model_path: str = "yolov8n.pt"
    device: str = "cpu"  # "cpu" or "cuda"
    confidence_threshold: float = 0.40
    iou_threshold: float = 0.50
    allowed_classes: List[str] = Field(
        default_factory=lambda: [
            "car", "bus", "truck", "motorcycle", "van", "bicycle",
            "auto_rickshaw", "emergency", "ambulance", "other"
        ]
    )
    coco_class_mapping: Dict[int, str] = Field(
        default_factory=lambda: {
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }
    )
