"""
ChronoEye Infinity - Phase 7: Traffic Flow Calculator
Calculates traffic flow rate q (vehicles per hour) passing across road segment observation windows.
"""

from typing import List, Optional


class FlowCalculator:
    """
    Traffic Flow Calculator.
    Converts vehicle counts over observation time windows into hourly flow rate q (veh/hr).
    """

    @staticmethod
    def calculate_flow_rate(vehicle_count: int, window_seconds: float = 60.0) -> float:
        """
        Calculates hourly traffic flow rate: q = (N / delta_t) * 3600.
        """
        if vehicle_count <= 0 or window_seconds <= 0:
            return 0.0

        flow = (vehicle_count / window_seconds) * 3600.0
        return round(float(flow), 2)
