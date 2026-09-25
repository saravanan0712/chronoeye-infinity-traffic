"""
ChronoEye Infinity - Research Statistical Aggregation Engine
Computes mean, median, std, min, max, and 95% confidence interval bounds.
"""

import numpy as np
from typing import List, Dict, Any


def compute_statistical_aggregates(values: List[float]) -> Dict[str, float]:
    """Computes mean, median, std, min, max, and 95% CI."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0}

    arr = np.array(values, dtype=float)
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    margin = 1.96 * (s / np.sqrt(len(arr))) if len(arr) > 1 else 0.0

    return {
        "mean": round(m, 2),
        "median": round(float(np.median(arr)), 2),
        "std": round(s, 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "ci95_lower": round(m - margin, 2),
        "ci95_upper": round(m + margin, 2),
    }
