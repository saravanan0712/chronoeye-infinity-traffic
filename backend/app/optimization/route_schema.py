"""
ChronoEye Infinity - Phase 11: Route Optimization Schemas
Defines data models for route nodes, segments, cost components, and optimal route recommendations.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class RouteNode(BaseModel):
    """
    Junction node spatial coordinates.
    """
    node_id: str
    lat: float = 13.0827
    lon: float = 80.2707


class RouteSegment(BaseModel):
    """
    Road segment edge attributes for route graph traversal.
    """
    road_id: str
    from_junction: str
    to_junction: str
    length_meters: float = 500.0
    travel_time_seconds: float = 30.0
    congestion_score: float = 0.0
    uncertainty_std: float = 0.0
    has_incident: bool = False


class OptimizationRoute(BaseModel):
    """
    Optimized route recommendation output.
    """
    origin_junction: str
    destination_junction: str
    path_junctions: List[str] = Field(default_factory=list)
    path_roads: List[str] = Field(default_factory=list)
    total_distance_km: float = 0.0
    total_travel_time_seconds: float = 0.0
    total_predicted_delay_seconds: float = 0.0
    computation_latency_ms: float = 0.0
    algorithm_name: str = "ChronoEyePredictiveAStar"
    confidence_lower_seconds: Optional[float] = None
    confidence_upper_seconds: Optional[float] = None
