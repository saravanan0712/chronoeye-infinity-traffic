"""
ChronoEye Infinity - Phase 1: Traffic Simulation Schemas
Defines core data models, enums, and state representations for urban topology,
vehicle agents, traffic signals, camera FOVs, and traffic metrics.
"""

from enum import Enum
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field


class VehicleType(str, Enum):
    CAR = "CAR"
    BUS = "BUS"
    TRUCK = "TRUCK"
    MOTORCYCLE = "MOTORCYCLE"
    EMERGENCY = "EMERGENCY"


class SignalMode(str, Enum):
    FIXED = "FIXED"
    REACTIVE = "REACTIVE"
    PREDICTIVE = "PREDICTIVE"
    EMERGENCY = "EMERGENCY"


class SignalPhase(str, Enum):
    NORTH_SOUTH_GREEN = "NORTH_SOUTH_GREEN"
    NORTH_SOUTH_YELLOW = "NORTH_SOUTH_YELLOW"
    EAST_WEST_GREEN = "EAST_WEST_GREEN"
    EAST_WEST_YELLOW = "EAST_WEST_YELLOW"
    EMERGENCY_OVERRIDE = "EMERGENCY_OVERRIDE"


class VehicleStatus(str, Enum):
    MOVING = "MOVING"
    QUEUED = "QUEUED"
    ARRIVED = "ARRIVED"


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class VehicleState(BaseModel):
    vehicle_id: str
    tracking_id: int
    vehicle_type: VehicleType
    plate_number: str
    color: str
    speed_kmh: float = 0.0
    desired_speed_kmh: float = 50.0
    position_x: float = 0.0
    position_y: float = 0.0
    current_road_id: str
    lane_id: int = 0
    distance_on_road: float = 0.0
    route: List[str] = Field(default_factory=list)  # List of road IDs
    route_index: int = 0
    destination_junction_id: str
    status: VehicleStatus = VehicleStatus.MOVING
    confidence: float = 1.0
    first_seen_timestamp: float = 0.0
    last_updated_timestamp: float = 0.0


class CameraDefinition(BaseModel):
    camera_id: str
    name: str
    junction_id: str
    road_id: str
    position_x: float
    position_y: float
    fov_range_meters: float = 50.0


class CameraObservation(BaseModel):
    observation_id: str
    camera_id: str
    junction_id: str
    road_id: str
    timestamp: float
    frame_id: int
    vehicle_id: str
    tracking_id: int
    vehicle_type: VehicleType
    plate_number: str
    speed_kmh: float
    lane_id: int
    position_x: float
    position_y: float
    bbox: BoundingBox
    detection_confidence: float = 0.95
    ocr_confidence: float = 0.92
    source: str = "SIMULATED_CAMERA"


class TrafficSignalState(BaseModel):
    signal_id: str
    junction_id: str
    mode: SignalMode = SignalMode.FIXED
    current_phase: SignalPhase = SignalPhase.NORTH_SOUTH_GREEN
    phase_timer: float = 0.0
    ns_green_duration: float = 45.0
    ew_green_duration: float = 45.0
    yellow_duration: float = 5.0
    emergency_corridor_active: bool = False
    emergency_route: List[str] = Field(default_factory=list)


class RoadSegmentDefinition(BaseModel):
    road_id: str
    name: str
    source_junction_id: str
    target_junction_id: str
    length_meters: float
    speed_limit_kmh: float = 60.0
    num_lanes: int = 2
    capacity: int = 50
    active_vehicle_ids: List[str] = Field(default_factory=list)
    is_blocked: bool = False
    blockage_severity: float = 0.0  # 0.0 to 1.0


class JunctionDefinition(BaseModel):
    junction_id: str
    name: str
    position_x: float
    position_y: float
    connected_road_ids: List[str] = Field(default_factory=list)
    signal_id: Optional[str] = None


class TrafficStateMetrics(BaseModel):
    timestamp: float
    junction_id: str
    road_id: str
    vehicle_count: int
    average_speed_kmh: float
    queue_length_meters: float
    occupancy_pct: float
    congestion_score: float  # 0.0 (free) to 1.0 (gridlock)
