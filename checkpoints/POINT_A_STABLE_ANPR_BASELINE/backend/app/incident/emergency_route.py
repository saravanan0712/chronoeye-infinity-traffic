"""
ChronoEye Infinity - Phase 13: Emergency Route Planner
Calculates optimal fast-path routing corridors for emergency vehicles.
"""

from typing import List, Dict, Optional, Any
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.optimization.route_schema import OptimizationRoute
from app.optimization.predictive_router import ChronoEyePredictiveAStarRouter
from app.incident.emergency_schema import EmergencyVehicle


class EmergencyRoutePlanner:
    """
    Emergency Route Planner utilizing predictive graph traversal to select optimal corridors.
    """

    def __init__(self, router: Optional[ChronoEyePredictiveAStarRouter] = None):
        self.router = router or ChronoEyePredictiveAStarRouter()

    def plan_emergency_route(
        self,
        builder: SpatioTemporalGraphBuilder,
        vehicle: EmergencyVehicle,
    ) -> OptimizationRoute:
        """
        Plans optimal priority route for emergency vehicle.
        """
        route = self.router.find_route(
            builder,
            origin=vehicle.current_location_junction,
            destination=vehicle.destination_junction,
        )

        # Emergency vehicle speed boost factor (prioritized green wave reduces travel time by 35%)
        route.total_travel_time_seconds = round(route.total_travel_time_seconds * 0.65, 2)
        route.algorithm_name = "EmergencyPredictiveCorridor"

        return route
