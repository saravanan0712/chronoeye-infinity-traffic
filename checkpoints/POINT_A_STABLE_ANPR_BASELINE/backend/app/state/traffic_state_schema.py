"""
ChronoEye Infinity - Phase 7: Dynamic Traffic State Schemas
Defines data models for road segment traffic states, congestion levels, ML-ready feature vectors,
and network-wide traffic snapshots.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class CongestionLevel(str, Enum):
    FREE_FLOW = "FREE_FLOW"
    MODERATE = "MODERATE"
    HEAVY = "HEAVY"
    SEVERELY_CONGESTED = "SEVERELY_CONGESTED"
    STATIONARY_GRIDLOCK = "STATIONARY_GRIDLOCK"


class RoadSegmentState(BaseModel):
    """
    Dynamic state vector for a single road segment at timestamp t.
    """
    segment_id: str
    road_name: str
    timestamp: float
    vehicle_count: int = 0
    flow_rate: float = 0.0          # vehicles / hour
    density: float = 0.0            # vehicles / km
    average_speed: float = 60.0     # km/h
    occupancy: float = 0.0          # 0.0 to 1.0
    queue_length: int = 0           # queued vehicle count
    queue_length_meters: float = 0.0
    travel_time: float = 30.0       # estimated travel time seconds
    congestion_score: float = 0.0   # 0.0 (free) to 1.0 (jammed)
    congestion_level: CongestionLevel = CongestionLevel.FREE_FLOW
    normalized_features: List[float] = Field(default_factory=list)  # [norm_flow, norm_density, norm_speed, occupancy, norm_queue, congestion_score]


class NetworkTrafficSnapshot(BaseModel):
    """
    Network-wide traffic snapshot aggregating all road segment states at timestamp t.
    """
    timestamp: float
    segment_states: Dict[str, RoadSegmentState] = Field(default_factory=dict)
    network_average_speed: float = 60.0
    network_congestion_score: float = 0.0
    total_active_vehicles: int = 0

    # Backward-compatible property aliases (older API contract)
    @property
    def total_network_vehicles(self) -> int:
        """Alias for total_active_vehicles (backward compatibility)."""
        return self.total_active_vehicles

    @property
    def network_congestion_index(self) -> float:
        """Alias for network_congestion_score (backward compatibility)."""
        return self.network_congestion_score

