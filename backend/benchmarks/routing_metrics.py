"""
ChronoEye Infinity - Routing Metrics Engine
Computes route travel time, predicted travel time, ETA error, delay avoided, and reroute count.
"""

from typing import Dict, Any


def calculate_routing_metrics(actual_travel_time: float, predicted_travel_time: float, baseline_travel_time: float, distance_meters: float) -> Dict[str, float]:
    """Calculates navigation route evaluation metrics."""
    eta_error = abs(actual_travel_time - predicted_travel_time)
    delay_avoided = max(0.0, baseline_travel_time - actual_travel_time)

    return {
        "actual_travel_time_s": round(actual_travel_time, 2),
        "predicted_travel_time_s": round(predicted_travel_time, 2),
        "eta_error_s": round(eta_error, 2),
        "delay_avoided_s": round(delay_avoided, 2),
        "distance_meters": round(distance_meters, 1),
    }
