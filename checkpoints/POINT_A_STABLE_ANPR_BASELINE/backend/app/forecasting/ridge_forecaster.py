"""
ChronoEye Infinity - Phase 8: Level 2 Ridge Regression Traffic Forecaster
Uses L2-regularized linear regression on lagged temporal traffic features to predict multi-horizon traffic state.
Fully CPU compatible with zero hard external ML library dependencies (uses NumPy or Pure Python fallback).
"""

import math
from typing import List, Dict, Optional, Any
from app.state.traffic_state_schema import NetworkTrafficSnapshot
from app.forecasting.forecasting_schema import (
    ForecastHorizon,
    SegmentForecast,
    ForecastingModelType,
)
from app.forecasting.feature_window import FeatureWindowExtractor


class RidgeTrafficForecaster:
    """
    Level 2 Ridge Regression Traffic Forecaster.
    """

    HORIZON_SECONDS = {
        ForecastHorizon.PLUS_5MIN: 300.0,
        ForecastHorizon.PLUS_10MIN: 600.0,
        ForecastHorizon.PLUS_15MIN: 900.0,
        ForecastHorizon.PLUS_30MIN: 1800.0,
    }

    def __init__(self, alpha: float = 1.0, window_size: int = 12):
        self.alpha = alpha
        self.window_size = window_size
        self.weights: Optional[List[List[float]]] = None  # Feature_dim x Target_dim
        self.bias: Optional[List[float]] = None

    def fit(self, X: List[List[float]], Y: List[List[float]]):
        """
        Fits Ridge Regression model: W = (X^T X + alpha * I)^(-1) X^T Y
        """
        if not X or not Y or len(X) != len(Y):
            return

        try:
            import numpy as np

            X_mat = np.array(X, dtype=np.float64)
            Y_mat = np.array(Y, dtype=np.float64)

            # Add bias column
            num_samples = X_mat.shape[0]
            ones = np.ones((num_samples, 1), dtype=np.float64)
            X_b = np.hstack([ones, X_mat])

            num_feats = X_b.shape[1]
            I = np.eye(num_feats, dtype=np.float64)
            I[0, 0] = 0.0  # Do not regularize bias

            # Ridge normal equation
            W_all = np.linalg.solve(X_b.T @ X_b + self.alpha * I, X_b.T @ Y_mat)

            self.bias = W_all[0, :].tolist()
            self.weights = W_all[1:, :].tolist()

        except Exception:
            # Fallback mean predictor if matrix inversion fails / small dataset
            num_targets = len(Y[0]) if Y else 5
            num_feats = len(X[0]) if X else 72
            self.bias = [float(sum(row[t] for row in Y)) / float(len(Y)) for t in range(num_targets)]
            self.weights = [[0.0 for _ in range(num_targets)] for _ in range(num_feats)]

    def predict(
        self,
        history: List[NetworkTrafficSnapshot],
        horizon: ForecastHorizon,
        segment_id: str = "ROAD_R_AB",
    ) -> SegmentForecast:
        """
        Predicts future traffic conditions using trained Ridge Regression weights.
        """
        if not history:
            return SegmentForecast(segment_id=segment_id, target_timestamp=300.0, horizon=horizon)

        current_t = history[-1].timestamp
        target_t = current_t + self.HORIZON_SECONDS[horizon]

        # Extract lag window
        window = FeatureWindowExtractor.extract_lag_windows(history, self.window_size, segment_id)
        flat_x = FeatureWindowExtractor.flatten_window(window)

        if self.weights is None or self.bias is None:
            # Fallback persistence prediction if un-trained
            seg = history[-1].segment_states.get(segment_id)
            if seg:
                return SegmentForecast(
                    segment_id=segment_id,
                    target_timestamp=target_t,
                    horizon=horizon,
                    predicted_flow=seg.flow_rate,
                    predicted_queue=seg.queue_length,
                    predicted_density=seg.density,
                    predicted_speed=seg.average_speed,
                    predicted_travel_time=seg.travel_time,
                )
            return SegmentForecast(segment_id=segment_id, target_timestamp=target_t, horizon=horizon)

        # Dot product prediction: Y_pred = X * W + bias
        num_targets = len(self.bias)
        y_pred = list(self.bias)

        min_len = min(len(flat_x), len(self.weights))
        for i in range(min_len):
            for t in range(num_targets):
                y_pred[t] += flat_x[i] * self.weights[i][t]

        p_flow = max(0.0, round(y_pred[0], 2))
        p_queue = max(0, int(round(y_pred[1])))
        p_density = max(0.0, round(y_pred[2], 2))
        p_speed = max(0.0, round(y_pred[3], 2))
        p_tt = max(1.0, round(y_pred[4], 2))

        return SegmentForecast(
            segment_id=segment_id,
            target_timestamp=target_t,
            horizon=horizon,
            predicted_flow=p_flow,
            predicted_queue=p_queue,
            predicted_density=p_density,
            predicted_speed=p_speed,
            predicted_travel_time=p_tt,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes model parameters for checkpoint saving."""
        return {
            "alpha": self.alpha,
            "window_size": self.window_size,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RidgeTrafficForecaster":
        """Deserializes model parameters from checkpoint."""
        model = cls(alpha=data.get("alpha", 1.0), window_size=data.get("window_size", 12))
        model.weights = data.get("weights")
        model.bias = data.get("bias")
        return model
