"""
ChronoEye Infinity - Phase 13: Emergency Green Corridor Schemas
Defines data models for emergency vehicle priority corridors, signal preemption schedules,
and corridor lifecycle states.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from app.optimization.signal_schema import SignalPhase


class CorridorStatus(str, Enum):
    PLANNING = "PLANNING"
    ACTIVE_GREEN_WAVE = "ACTIVE_GREEN_WAVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class EmergencyVehicle(BaseModel):
    """
    Emergency vehicle profile.
    """
    vehicle_id: str = "AMB_911"
    vehicle_type: str = "AMBULANCE"
    origin_junction: str
    destination_junction: str
    current_location_junction: str
    speed_kmh: float = 60.0


class SignalPrioritySchedule(BaseModel):
    """
    Signal priority preemption window scheduled for a target junction.
    """
    signal_id: str
    junction_id: str
    expected_arrival_timestamp: float
    green_start_timestamp: float
    green_end_timestamp: float
    priority_phase: SignalPhase = SignalPhase.PHASE_NORTH_SOUTH
    preemption_active: bool = True


class CorridorPlan(BaseModel):
    """
    Complete Emergency Green Corridor plan recommendation.
    """
    corridor_id: str
    emergency_vehicle: EmergencyVehicle
    ordered_junctions: List[str] = Field(default_factory=list)
    path_roads: List[str] = Field(default_factory=list)
    total_distance_km: float = 0.0
    estimated_total_travel_time_seconds: float = 0.0
    predicted_arrival_times: Dict[str, float] = Field(default_factory=dict)
    signal_priority_schedule: Dict[str, SignalPrioritySchedule] = Field(default_factory=dict)
    status: CorridorStatus = CorridorStatus.PLANNING
