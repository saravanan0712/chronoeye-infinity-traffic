"""
ChronoEye Infinity - Phase 11: Spatio-Temporal Edge Cost Calculator
Calculates edge cost C(e, t) combining predicted travel time, congestion score, uncertainty penalty,
length distance, and incident penalties.
"""

from typing import Dict, Any, Optional
from app.optimization.route_schema import RouteSegment


class SpatioTemporalEdgeCostCalculator:
    """
    Predictive Spatio-Temporal Edge Cost Calculator.
    """

    def __init__(
        self,
        w_congestion: float = 10.0,
        w_uncertainty: float = 2.0,
        w_distance: float = 1.0,
        w_incident: float = 300.0,
    ):
        self.w_congestion = w_congestion
        self.w_uncertainty = w_uncertainty
        self.w_distance = w_distance
        self.w_incident = w_incident

    def calculate_cost(
        self,
        segment: RouteSegment,
        predicted_travel_time: Optional[float] = None,
        predicted_congestion: Optional[float] = None,
        uncertainty_std: Optional[float] = None,
        has_incident: Optional[bool] = None,
    ) -> float:
        """
        Calculates scalar predictive traversal cost for road segment e.
        """
        t_travel = predicted_travel_time if predicted_travel_time is not None else segment.travel_time_seconds
        cong = predicted_congestion if predicted_congestion is not None else segment.congestion_score
        unc = uncertainty_std if uncertainty_std is not None else segment.uncertainty_std
        inc = has_incident if has_incident is not None else segment.has_incident

        dist_km = segment.length_meters / 1000.0

        t_term = max(1.0, float(t_travel))
        c_term = self.w_congestion * max(0.0, float(cong)) * 30.0
        u_term = self.w_uncertainty * max(0.0, float(unc))
        d_term = self.w_distance * dist_km
        i_term = self.w_incident if inc else 0.0

        total_cost = t_term + c_term + u_term + d_term + i_term
        return round(float(total_cost), 2)
