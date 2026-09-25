"""
ChronoEye Infinity - Phase 14: API & WebSocket Schemas
Defines request and response models for REST endpoints and structured WebSocket real-time broadcast messages.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from app.schemas.provenance import DataProvenance, DataProvenanceType


class HealthResponse(BaseModel):
    """System health check response schema."""
    status: str = "HEALTHY"
    version: str = "1.0.0"
    active_phases: List[str] = Field(
        default_factory=lambda: [f"Phase {i}" for i in range(1, 19)]
    )
    provenance: DataProvenance = Field(
        default_factory=lambda: DataProvenance(
            provenance_type=DataProvenanceType.OBSERVED,
            source_id="API_GATEWAY",
            process_name="ChronoEye_HealthService",
            is_mock=False,
        )
    )


class RouteOptimizationRequest(BaseModel):
    """Route optimization request payload."""
    origin: str
    destination: str
    departure_timestamp: float = 0.0
    algorithm: str = "ChronoEyePredictiveAStar"


class EmergencyCorridorRequest(BaseModel):
    """Emergency green corridor activation request payload."""
    vehicle_id: str = "AMB_911"
    vehicle_type: str = "AMBULANCE"
    origin_junction: str
    destination_junction: str


class WSMessage(BaseModel):
    """Real-time WebSocket streaming message envelope with Data Provenance tracking."""
    event_type: str
    timestamp: float
    provenance: DataProvenance = Field(
        default_factory=lambda: DataProvenance(
            provenance_type=DataProvenanceType.OBSERVED,
            source_id="CHRONOEYE_STATE_ENGINE",
            process_name="DynamicTrafficStateEngine",
        )
    )
    payload: Dict[str, Any]

