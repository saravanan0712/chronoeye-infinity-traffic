"""
ChronoEye Infinity - Phase 7: Travel Time Estimator
Estimates actual travel time T (seconds) for traversing a road segment based on current average speed.
"""

class TravelTimeEstimator:
    """
    Road Segment Travel Time Estimator.
    Calculates expected traversal duration in seconds.
    """

    @staticmethod
    def estimate_travel_time(
        length_meters: float = 500.0,
        avg_speed_kmh: float = 60.0,
        speed_limit_kmh: float = 60.0,
    ) -> float:
        """
        Estimates travel time T (seconds) for road segment.
        T = length_meters / (avg_speed_m_s).
        Enforces T >= T_free_flow and caps stationary gridlock delay.
        """
        if length_meters <= 0:
            return 0.0

        v_limit_m_s = max(1.0, (speed_limit_kmh * 1000.0) / 3600.0)
        t_free_flow = length_meters / v_limit_m_s

        if avg_speed_kmh <= 0.5:
            # Stationary gridlock: return 10x free-flow travel time cap
            return round(float(t_free_flow * 10.0), 2)

        v_avg_m_s = (avg_speed_kmh * 1000.0) / 3600.0
        t_est = length_meters / v_avg_m_s

        # Bound output: minimum travel time is free flow travel time
        t_final = max(t_free_flow, t_est)
        return round(float(t_final), 2)
