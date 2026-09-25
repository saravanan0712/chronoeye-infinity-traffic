"""
ChronoEye Infinity - Phase 9: Prediction Interval Calculator
Computes 80%, 90%, and 95% confidence prediction intervals [lower_bound, upper_bound]
enforcing physical domain lower bounds (e.g. non-negative speed, flow, density, queue).
"""

import math
from typing import Tuple, Dict


class PredictionIntervalCalculator:
    """
    Prediction Interval Calculator.
    """

    CONFIDENCE_Z_SCORES: Dict[float, float] = {
        0.80: 1.282,
        0.90: 1.645,
        0.95: 1.960,
    }

    @classmethod
    def calculate_interval(
        cls,
        mean: float,
        std: float,
        confidence_level: float = 0.95,
        min_bound: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Calculates prediction interval [lower_bound, upper_bound] for given mean, std, and confidence level.
        Enforces physical non-negativity bound.
        """
        z = cls.CONFIDENCE_Z_SCORES.get(confidence_level, 1.960)
        margin = z * max(0.0, std)

        lower = max(min_bound, mean - margin)
        upper = max(lower, mean + margin)

        return round(float(lower), 2), round(float(upper), 2)
