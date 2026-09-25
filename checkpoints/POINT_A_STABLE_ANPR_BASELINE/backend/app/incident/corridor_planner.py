"""
ChronoEye Infinity - Phase 13: Corridor Planner
Calculates sequential arrival timestamps at each junction along the emergency vehicle corridor.
"""

from typing import List, Dict, Optional, Any
from app.optimization.route_schema import OptimizationRoute


class CorridorPlanner:
    """
    Corridor Arrival Time Planner.
    """

    def calculate_junction_arrivals(
        self,
        route: OptimizationRoute,
        departure_timestamp: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculates expected arrival timestamps at each ordered junction node along route.
        """
        arrivals: Dict[str, float] = {}

        if not route.path_junctions:
            return arrivals

        num_juncs = len(route.path_junctions)
        time_per_segment = (
            route.total_travel_time_seconds / float(max(1, num_juncs - 1))
            if num_juncs > 1
            else 0.0
        )

        current_t = departure_timestamp
        for i, junc in enumerate(route.path_junctions):
            arrivals[junc] = round(current_t, 2)
            current_t += time_per_segment

        return arrivals
