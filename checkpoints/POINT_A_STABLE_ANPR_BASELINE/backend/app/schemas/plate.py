"""
ChronoEye Infinity - Phase 4: License Plate Recognition (ALPR/OCR) Schemas
Defines schemas for single-frame plate observations, plate validation statuses,
temporally fused plate identities, and vehicle track-plate identity evidence links.
"""

import uuid
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field
from app.schemas.detection import BoundingBoxXYXY


class PlateValidationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    UNCERTAIN = "UNCERTAIN"
    FORMAT_MISMATCH = "FORMAT_MISMATCH"


class PlateObservation(BaseModel):
    """
    Single-frame license plate observation extracted from a vehicle ROI.
    """
    plate_id: str = Field(default_factory=lambda: f"PLT_{uuid.uuid4().hex[:8]}")
    track_id: str
    camera_id: str
    frame_id: int
    timestamp: float
    bbox: Optional[BoundingBoxXYXY] = None
    raw_text: str
    normalized_text: str
    ocr_confidence: float
    validation_confidence: float = 0.0
    overall_confidence: float = 0.0
    quality_score: float = 0.0
    preprocessing_variant: str = "ORIGINAL"
    status: PlateValidationStatus = PlateValidationStatus.UNCERTAIN
    source: str = "EASY_OCR"  # "EASY_OCR", "TEST_OCR", "SIMULATION"


class FusedPlateIdentity(BaseModel):
    """
    Temporally fused plate identity accumulated over consecutive track observations.
    """
    track_id: str
    camera_id: str
    best_plate_number: str
    overall_confidence: float
    observation_count: int = 1
    confirmed: bool = False
    status: str = "UNKNOWN"  # "CONFIRMED", "PENDING", "UNKNOWN"
    pending_plate_number: Optional[str] = None
    evidence_frames: List[int] = Field(default_factory=list)
    supporting_observations_count: int = 0
    weighted_evidence_score: float = 0.0
    character_agreement_ratio: float = 0.0
    raw_observations_audit: List[Dict[str, Any]] = Field(default_factory=list)
    first_seen_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0
    candidate_history: Dict[str, float] = Field(default_factory=dict)  # plate -> cumulative weighted score


class VehicleIdentityEvidence(BaseModel):
    """
    Unified vehicle identity evidence associating a Phase 3 persistent TrackState
    with physical license plate evidence and simulation ground-truth references.
    """
    track_id: str
    camera_id: str
    vehicle_type: str
    associated_plate: Optional[FusedPlateIdentity] = None
    raw_vehicle_reference: Optional[str] = None  # Link to Phase 1 simulation vehicle ID
    last_updated_timestamp: float
