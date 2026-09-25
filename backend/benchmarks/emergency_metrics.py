"""
ChronoEye Infinity - Emergency Corridor Metrics Engine
Computes emergency travel time, delay, speedup percentage, and corridor clearance time.
"""

from typing import Dict, Any


def calculate_emergency_metrics(emergency_travel_time: float, baseline_travel_time: float, clearance_time: float) -> Dict[str, float]:
    """Calculates green corridor preemption metrics."""
    delay = max(0.0, emergency_travel_time - 35.0)
    speedup = ((baseline_travel_time - emergency_travel_time) / baseline_travel_time) * 100.0 if baseline_travel_time > 0 else 0.0

    return {
        "emergency_travel_time_s": round(emergency_travel_time, 2),
        "baseline_travel_time_s": round(baseline_travel_time, 2),
        "emergency_delay_s": round(delay, 2),
        "speedup_percentage": round(speedup, 2),
        "clearance_time_s": round(clearance_time, 2),
    }
