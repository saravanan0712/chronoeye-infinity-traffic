"""
ChronoEye Infinity - Phase 12: Anomaly & Incident Detection Schemas
Defines data models for traffic incident types, severity levels, status transitions, and incident events.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class IncidentType(str, Enum):
    SUDDEN_QUEUE_GROWTH = "SUDDEN_QUEUE_GROWTH"
    ABNORMAL_SPEED_DROP = "ABNORMAL_SPEED_DROP"
    STOPPED_VEHICLE = "STOPPED_VEHICLE"
    FLOW_COLLAPSE = "FLOW_COLLAPSE"
    ROAD_BLOCKAGE = "ROAD_BLOCKAGE"
    UNEXPECTED_CONGESTION = "UNEXPECTED_CONGESTION"


class IncidentSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentStatus(str, Enum):
    ACTIVE = "ACTIVE"
    VERIFIED = "VERIFIED"
    RESOLVED = "RESOLVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class IncidentEvent(BaseModel):
    """
    Structured Traffic Incident Event detected by divergence between predicted and observed states.
    """
    incident_id: str
    segment_id: str
    incident_type: IncidentType
    severity: IncidentSeverity = IncidentSeverity.MEDIUM
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    detected_at: float
    resolved_at: Optional[float] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)
    status: IncidentStatus = IncidentStatus.ACTIVE
