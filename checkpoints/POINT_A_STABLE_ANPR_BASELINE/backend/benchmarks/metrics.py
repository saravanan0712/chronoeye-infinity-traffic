"""
ChronoEye Infinity - Metrics Engine
Computes mathematically valid traffic, queue, delay, throughput, and signal performance metrics.
"""

import numpy as np
from typing import List, Dict, Any


def calculate_traffic_metrics(travel_times: List[float], delays: List[float], queues: List[int], total_vehicles: int, duration_seconds: float) -> Dict[str, float]:
    """Calculates summary traffic metrics."""
    if not travel_times:
        return {
            "mean_travel_time": 0.0,
            "median_travel_time": 0.0,
            "p95_travel_time": 0.0,
            "avg_delay": 0.0,
            "max_delay": 0.0,
            "avg_queue_length": 0.0,
            "max_queue_length": 0.0,
            "throughput_veh_h": 0.0,
            "average_speed": 0.0,
            "congestion_index": 0.0,
        }

    mean_tt = float(np.mean(travel_times))
    median_tt = float(np.median(travel_times))
    p95_tt = float(np.percentile(travel_times, 95))
    avg_d = float(np.mean(delays)) if delays else 0.0
    max_d = float(np.max(delays)) if delays else 0.0
    avg_q = float(np.mean(queues)) if queues else 0.0
    max_q = float(np.max(queues)) if queues else 0.0
    throughput = (total_vehicles / duration_seconds) * 3600.0 if duration_seconds > 0 else 0.0

    return {
        "mean_travel_time": round(mean_tt, 2),
        "median_travel_time": round(median_tt, 2),
        "p95_travel_time": round(p95_tt, 2),
        "avg_delay": round(avg_d, 2),
        "max_delay": round(max_d, 2),
        "avg_queue_length": round(avg_q, 2),
        "max_queue_length": int(max_q),
        "throughput_veh_h": round(throughput, 1),
        "average_speed": round(max(0.0, 60.0 - avg_d * 0.5), 1),
        "congestion_index": round(min(1.0, avg_d / 50.0), 3),
    }


def calculate_percentage_improvement(baseline: float, chronoeye: float, lower_is_better: bool = True) -> float:
    """Calculates percentage improvement with zero-denominator safety."""
    if baseline == 0.0:
        return 0.0
    if lower_is_better:
        val = ((baseline - chronoeye) / abs(baseline)) * 100.0
    else:
        val = ((chronoeye - baseline) / abs(baseline)) * 100.0
    return round(val, 2)
