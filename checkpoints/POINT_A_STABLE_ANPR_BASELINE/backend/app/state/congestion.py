"""
ChronoEye Infinity - Phase 7: Congestion Scoring & Level Analyzer
Computes normalized 0.0 to 1.0 congestion score and assigns discrete CongestionLevel enum.
"""

from app.state.traffic_state_schema import CongestionLevel


class CongestionAnalyzer:
    """
    Congestion Scoring & Classification Engine.
    """

    @staticmethod
    def calculate_congestion_score(
        density: float,
        capacity_density: float,
        avg_speed: float,
        speed_limit: float,
        queue_count: int,
        capacity: int,
    ) -> float:
        """
        Calculates normalized 0.0 to 1.0 congestion score.
        Formula: 0.4 * density_ratio + 0.4 * speed_deficit + 0.2 * queue_ratio
        """
        cap_dens = max(1.0, capacity_density)
        spd_lim = max(1.0, speed_limit)
        cap = max(1, capacity)

        density_ratio = min(1.0, max(0.0, density / cap_dens))
        speed_deficit = max(0.0, 1.0 - (avg_speed / spd_lim))
        queue_ratio = min(1.0, max(0.0, queue_count / float(cap)))

        congestion = 0.4 * density_ratio + 0.4 * speed_deficit + 0.2 * queue_ratio
        return round(min(1.0, max(0.0, float(congestion))), 3)

    @staticmethod
    def classify_congestion_level(score: float) -> CongestionLevel:
        """
        Classifies congestion score into discrete CongestionLevel enum.
        """
        if score < 0.25:
            return CongestionLevel.FREE_FLOW
        elif score < 0.50:
            return CongestionLevel.MODERATE
        elif score < 0.75:
            return CongestionLevel.HEAVY
        elif score < 0.90:
            return CongestionLevel.SEVERELY_CONGESTED
        else:
            return CongestionLevel.STATIONARY_GRIDLOCK
