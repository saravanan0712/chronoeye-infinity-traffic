"""
ChronoEye Infinity - Research Metrics Calculation Engine
Computes travel, traffic, control, routing, and incident metrics, plus lower/higher-is-better improvement % formula.
"""

import numpy as np
from typing import Dict, List, Any


def compute_metrics(delays: List[float], travel_times: List[float], queues: List[float], throughputs: List[float]) -> Dict[str, float]:
    """Computes aggregate trial run metrics."""
    return {
        "mean_travel_time": float(np.mean(travel_times)) if travel_times else 0.0,
        "median_travel_time": float(np.median(travel_times)) if travel_times else 0.0,
        "p95_travel_time": float(np.percentile(travel_times, 95)) if travel_times else 0.0,
        "avg_delay": float(np.mean(delays)) if delays else 0.0,
        "max_delay": float(np.max(delays)) if delays else 0.0,
        "avg_queue_length": float(np.mean(queues)) if queues else 0.0,
        "max_queue_length": float(np.max(queues)) if queues else 0.0,
        "throughput": float(np.mean(throughputs)) if throughputs else 0.0,
        "average_speed": max(0.0, 60.0 - float(np.mean(delays)) * 0.5) if delays else 0.0,
        "congestion_index": min(1.0, float(np.mean(delays)) / 50.0) if delays else 0.0,
    }


def compute_improvement_percentage(baseline: float, chronoeye: float, lower_is_better: bool = True) -> float:
    """Computes improvement percentage with zero denominator protection."""
    if baseline == 0.0:
        return 0.0
    if lower_is_better:
        val = ((baseline - chronoeye) / abs(baseline)) * 100.0
    else:
        val = ((chronoeye - baseline) / abs(baseline)) * 100.0
    return round(val, 2)
