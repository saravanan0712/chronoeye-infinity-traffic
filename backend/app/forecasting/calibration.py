"""
ChronoEye Infinity - Phase 9: Uncertainty Calibration & Empirical Coverage Analyzer
Calculates Empirical Interval Coverage Ratio (PICP), Mean Interval Width (MPIW),
and Expected Calibration Error (ECE).
"""

import math
from typing import List, Dict, Tuple


class UncertaintyCalibrator:
    """
    Uncertainty Calibration & Coverage Evaluator.
    """

    @staticmethod
    def calculate_picp(y_true: List[float], lower_bounds: List[float], upper_bounds: List[float]) -> float:
        """
        Calculates Prediction Interval Coverage Probability (PICP).
        Ratio of true target values falling within predicted [lower, upper] intervals.
        """
        if not y_true or not lower_bounds or not upper_bounds or len(y_true) != len(lower_bounds):
            return 0.0

        covered = sum(1 for y, low, high in zip(y_true, lower_bounds, upper_bounds) if low <= y <= high)
        coverage = covered / float(len(y_true))
        return round(float(coverage), 4)

    @staticmethod
    def calculate_mpiw(lower_bounds: List[float], upper_bounds: List[float]) -> float:
        """
        Calculates Mean Prediction Interval Width (MPIW).
        Average width (upper - lower) of prediction intervals.
        """
        if not lower_bounds or not upper_bounds or len(lower_bounds) != len(upper_bounds):
            return 0.0

        widths = [high - low for low, high in zip(lower_bounds, upper_bounds)]
        mpiw = sum(widths) / float(len(widths))
        return round(float(mpiw), 3)

    @staticmethod
    def calculate_ece(y_true: List[float], y_pred: List[float], stds: List[float]) -> float:
        """
        Calculates Expected Calibration Error (ECE) for standard deviation estimates.
        """
        if not y_true or not y_pred or not stds or len(y_true) != len(y_pred):
            return 0.0

        errors = [abs(y - p) for y, p in zip(y_true, y_pred)]
        # Bin standard deviations and compute gap between expected error and observed error
        mean_expected_err = sum(stds) / float(len(stds))
        mean_observed_err = sum(errors) / float(len(errors))

        ece = abs(mean_expected_err - mean_observed_err)
        return round(float(ece), 4)
