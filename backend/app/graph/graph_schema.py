"""
ChronoEye Infinity - Phase 6: Spatio-Temporal Traffic Graph Schemas
Defines node types, edge types, node attributes, edge attributes, and temporal graph data contracts.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    VEHICLE = "VEHICLE"
    CAMERA = "CAMERA"
    ROAD = "ROAD"
    LANE = "LANE"
    JUNCTION = "JUNCTION"
    SIGNAL = "SIGNAL"


class EdgeType(str, Enum):
    VEHICLE_OBSERVED_BY_CAMERA = "VEHICLE_OBSERVED_BY_CAMERA"
    VEHICLE_TRAVELS_ON_LANE = "VEHICLE_TRAVELS_ON_LANE"
    LANE_BELONGS_TO_ROAD = "LANE_BELONGS_TO_ROAD"
    ROAD_CONNECTS_TO_JUNCTION = "ROAD_CONNECTS_TO_JUNCTION"
    ROAD_CONNECTS_JUNCTION = "ROAD_CONNECTS_TO_JUNCTION"   # Backward-compat alias
    JUNCTION_CONTROLLED_BY_SIGNAL = "JUNCTION_CONTROLLED_BY_SIGNAL"
    VEHICLE_TRANSITIONS_TO_CAMERA = "VEHICLE_TRANSITIONS_TO_CAMERA"
    VEHICLE_TRANSITIONS_TO_ROAD = "VEHICLE_TRANSITIONS_TO_ROAD"
    VEHICLE_TRANSITIONS_TO_LANE = "VEHICLE_TRANSITIONS_TO_LANE"
    JUNCTION_CONNECTS_TO_JUNCTION = "JUNCTION_CONNECTS_TO_JUNCTION"



class VehicleNode(BaseModel):
    vehicle_id: str
    plate_number: Optional[str] = None
    vehicle_type: str = "car"
    first_seen: float = 0.0
    last_seen: float = 0.0
    observation_count: int = 1
    current_camera: Optional[str] = None
    current_road: Optional[str] = None
    current_lane: Optional[str] = None
    current_speed: float = 0.0
    current_direction: str = "UNKNOWN"


class CameraNodeGraph(BaseModel):
    camera_id: str
    latitude: float = 0.0
    longitude: float = 0.0
    road_name: str = ""
    direction: str = "EAST"
    active_vehicle_count: int = 0
    observation_count: int = 0


class RoadNode(BaseModel):
    road_id: str
    road_name: str
    length_meters: float = 500.0
    speed_limit_kmh: float = 60.0
    capacity: int = 100
    current_vehicle_count: int = 0
    average_speed: float = 60.0
    density: float = 0.0  # vehicles per km
    congestion_score: float = 0.0  # 0.0 (free flow) to 1.0 (jammed)


class LaneNode(BaseModel):
    lane_id: str
    road_id: str
    direction: str = "FORWARD"
    capacity: int = 50


class JunctionNode(BaseModel):
    junction_id: str
    latitude: float = 0.0
    longitude: float = 0.0
    incoming_roads: List[str] = Field(default_factory=list)
    outgoing_roads: List[str] = Field(default_factory=list)


class SignalNode(BaseModel):
    signal_id: str
    junction_id: str
    active_phase: str = "GREEN_NS"
    mode: str = "FIXED"  # FIXED, REACTIVE, PREDICTIVE, EMERGENCY
    green_time: float = 30.0
    cycle_length: float = 90.0


class EdgeAttributes(BaseModel):
    edge_type: EdgeType
    timestamp: float = 0.0
    confidence: float = 1.0
    distance: float = 0.0
    travel_time: float = 0.0
    speed: float = 0.0
    direction: str = "UNKNOWN"
    camera_id: Optional[str] = None
    source: str
    target: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
