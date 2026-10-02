"""
ChronoEye Infinity - Phase 5: Cross-Camera Re-ID & Journey Reconstruction Schemas
Defines schemas for Re-ID matching scores, decision states, camera topology graph elements,
journey segments, and global spatio-temporal vehicle journeys.
"""

import uuid
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field, model_validator
from app.schemas.detection import BoundingBoxXYXY


class MatchDecision(str, Enum):
    MATCH_CONFIRMED = "MATCH_CONFIRMED"
    MATCH_PROBABLE = "MATCH_PROBABLE"
    MATCH_REJECTED = "MATCH_REJECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ReIDScoreBreakdown(BaseModel):
    """
    Detailed breakdown of multi-evidence cross-camera similarity metrics.
    """
    plate_similarity: float = 0.0
    appearance_similarity: float = 0.0
    visual_features_similarity: float = 0.0
    vehicle_type_similarity: float = 0.0
    direction_similarity: float = 0.0
    temporal_compatibility: float = 0.0
    spatial_compatibility: float = 0.0
    overall_score: float = 0.0
    decision: MatchDecision = MatchDecision.INSUFFICIENT_EVIDENCE
    rejection_reason: Optional[str] = None


class ReIDConfig(BaseModel):
    """
    Configurable weights and physical thresholds for Re-ID matching engine.
    """
    w_plate: float = 0.35
    w_appearance: float = 0.15
    w_visual_features: float = 0.10
    w_type: float = 0.10
    w_direction: float = 0.10
    w_time: float = 0.05
    w_space: float = 0.15

    confirm_threshold: float = 0.75
    probable_threshold: float = 0.50

    max_speed_kmh: float = 120.0  # Maximum physically possible travel speed
    min_speed_kmh: float = 5.0    # Minimum travel speed
    max_time_gap_seconds: float = 3600.0  # 1 hour max gap between camera observations
    journey_timeout_seconds: float = 60.0  # Max inactivity time before journey is marked COMPLETED


class JourneySegment(BaseModel):
    """
    Single camera observation segment in a reconstructed vehicle journey.
    """
    segment_id: str = Field(default_factory=lambda: f"SEG_{uuid.uuid4().hex[:8]}")
    camera_id: str
    track_id: str
    timestamp: float = 0.0
    timestamp_uncertainty_seconds: Optional[float] = None
    bbox: Optional[BoundingBoxXYXY] = None
    speed_estimate: float = 0.0
    direction: str = "UNKNOWN"
    plate_number: Optional[str] = None
    plate_confidence: Optional[float] = None   # Step 3A: FusedPlateIdentity.overall_confidence
    plate_status: Optional[str] = None         # Step 3A: FusedPlateIdentity.status ("CONFIRMED"/"PENDING"/"UNKNOWN")
    appearance_embedding: Optional[List[float]] = None
    visual_features: Optional[Dict[str, Any]] = None
    transition_decision: Optional[MatchDecision] = None
    transition_score: Optional[float] = None
    transition_breakdown: Optional[ReIDScoreBreakdown] = None
    has_unobserved_gap: bool = False

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_fields(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "timestamp" not in values:
                if "enter_timestamp" in values:
                    values["timestamp"] = values["enter_timestamp"]
                elif "exit_timestamp" in values:
                    values["timestamp"] = values["exit_timestamp"]
            if "speed_estimate" not in values and "average_speed" in values:
                values["speed_estimate"] = values["average_speed"]
        return values

    @property
    def enter_timestamp(self) -> float:
        return self.timestamp

    @property
    def exit_timestamp(self) -> float:
        return self.timestamp

    @property
    def average_speed(self) -> float:
        return self.speed_estimate


class VehicleJourney(BaseModel):
    """
    Reconstructed cross-camera spatio-temporal vehicle journey.
    """
    journey_id: str = Field(default_factory=lambda: f"JRN_{uuid.uuid4().hex[:8]}")
    global_vehicle_id: str = Field(default_factory=lambda: f"VEH_{uuid.uuid4().hex[:8]}")
    plate_number: Optional[str] = None
    vehicle_type: str = "car"
    segments: List[JourneySegment] = Field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    cameras: List[str] = Field(default_factory=list)
    overall_confidence: float = 1.0
    status: str = "ACTIVE"  # "ACTIVE", "COMPLETE"

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_fields(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "plate_number" not in values and "license_plate" in values:
                values["plate_number"] = values["license_plate"]
            if "overall_confidence" not in values and "reconstruction_confidence" in values:
                values["overall_confidence"] = values["reconstruction_confidence"]
            
            # To handle missing required first_seen/last_seen for tests constructing VehicleJourneys manually
            if "first_seen" not in values and "segments" in values and values["segments"]:
                first_seg = values["segments"][0]
                if isinstance(first_seg, dict):
                    values["first_seen"] = first_seg.get("enter_timestamp", first_seg.get("timestamp", 0.0))
                    last_seg = values["segments"][-1]
                    values["last_seen"] = last_seg.get("exit_timestamp", last_seg.get("timestamp", values["first_seen"]))
                elif hasattr(first_seg, "timestamp"):
                    values["first_seen"] = first_seg.timestamp
                    values["last_seen"] = values["segments"][-1].timestamp
        return values

    @property
    def license_plate(self) -> Optional[str]:
        return self.plate_number

    @property
    def total_distance_km(self) -> float:
        return 0.0

    @property
    def total_travel_time_seconds(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def reconstruction_confidence(self) -> float:
        return self.overall_confidence

    @property
    def best_plate_number(self) -> Optional[str]:
        return self.plate_number


class CheckpointObservationStatus(str, Enum):
    OBSERVED = "OBSERVED"
    NOT_OBSERVED = "NOT_OBSERVED"
    UNKNOWN = "UNKNOWN"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class CheckpointQueryResult(BaseModel):
    """
    Structured response for vehicle-checkpoint crossing query.
    Supports queries like:
      - 'Did global vehicle VEH_101 cross checkpoint CAM_B?'
      - 'Did plate TN09AB1111 cross checkpoint CAM_C?'
    """
    query_type: str = "CHECKPOINT_CROSSING"
    target_vehicle_id: Optional[str] = None
    target_plate_number: Optional[str] = None
    checkpoint_id: str
    status: CheckpointObservationStatus
    timestamp: Optional[float] = None
    timestamp_uncertainty_seconds: Optional[float] = None
    confidence: float = 0.0
    plate_number: Optional[str] = None
    plate_status: Optional[str] = None
    journey_id: Optional[str] = None
    supporting_segments_count: int = 0
    has_unobserved_gap: bool = False
    evidence: Dict[str, Any] = Field(default_factory=dict)
    uncertainty: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)

