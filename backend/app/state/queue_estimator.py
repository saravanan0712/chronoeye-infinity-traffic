"""
ChronoEye Infinity - Phase 7: Traffic Queue Estimator
Estimates queued/stopped vehicle counts and spatial queue length in meters.
"""

from typing import List, Tuple


class QueueEstimator:
    """
    Queue Length Estimator.
    Identifies queued vehicles based on speed thresholds (v < 5.0 km/h) and estimates spatial queue length.
    """

    @staticmethod
    def estimate_queue(
        vehicle_speeds: List[float],
        speed_thresh_kmh: float = 5.0,
        avg_vehicle_spacing_m: float = 7.0,
    ) -> Tuple[int, float]:
        """
        Estimates queued vehicle count and spatial queue length in meters.
        Returns: (queue_count, queue_length_meters)
        """
        if not vehicle_speeds:
            return 0, 0.0

        queue_count = sum(1 for v in vehicle_speeds if v < speed_thresh_kmh)
        queue_length_m = queue_count * avg_vehicle_spacing_m
        return queue_count, round(float(queue_length_m), 2)
