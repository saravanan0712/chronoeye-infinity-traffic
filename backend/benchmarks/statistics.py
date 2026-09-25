"""
ChronoEye Infinity - Statistical Aggregation Engine
Computes mean, median, std, min, max, and 95% confidence intervals across multi-seed benchmark runs.
"""

import numpy as np
from typing import List, Dict, Any


def compute_summary_statistics(values: List[float]) -> Dict[str, float]:
    """Computes statistical distribution parameters for repeated runs."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0}

    arr = np.array(values, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    margin = 1.96 * (std_val / np.sqrt(len(arr))) if len(arr) > 1 else 0.0

    return {
        "mean": round(mean_val, 2),
        "median": round(float(np.median(arr)), 2),
        "std": round(std_val, 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "ci95_lower": round(mean_val - margin, 2),
        "ci95_upper": round(mean_val + margin, 2),
    }
