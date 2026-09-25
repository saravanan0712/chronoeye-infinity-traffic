"""
ChronoEye Infinity - Phase 8: Forecasting Evaluation Metrics Engine
Computes Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
Compares baseline vs Ridge vs ST-GNN models across all 4 forecasting horizons (+5m, +10m, +15m, +30m).
"""

import math
from typing import List, Dict, Any
from app.forecasting.forecasting_schema import (
    ForecastHorizon,
    ModelEvaluationMetrics,
)


class ForecastingEvaluator:
    """
    Forecasting Performance Evaluation Engine.
    """

    @staticmethod
    def calculate_mae(y_true: List[float], y_pred: List[float]) -> float:
        """Calculates Mean Absolute Error (MAE)."""
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return 0.0
        mae = sum(abs(a - p) for a, p in zip(y_true, y_pred)) / float(len(y_true))
        return round(float(mae), 3)

    @staticmethod
    def calculate_rmse(y_true: List[float], y_pred: List[float]) -> float:
        """Calculates Root Mean Squared Error (RMSE)."""
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return 0.0
        mse = sum((a - p) ** 2 for a, p in zip(y_true, y_pred)) / float(len(y_true))
        return round(float(math.sqrt(mse)), 3)

    @staticmethod
    def calculate_mape(y_true: List[float], y_pred: List[float]) -> float:
        """Calculates Mean Absolute Percentage Error (MAPE) in %."""
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return 0.0
        valid_pairs = [(a, p) for a, p in zip(y_true, y_pred) if abs(a) > 0.01]
        if not valid_pairs:
            return 0.0
        mape = (sum(abs((a - p) / float(a)) for a, p in valid_pairs) / float(len(valid_pairs))) * 100.0
        return round(float(mape), 2)

    @classmethod
    def evaluate_predictions(
        self,
        model_name: str,
        horizon: ForecastHorizon,
        y_true: List[float],
        y_pred: List[float],
    ) -> ModelEvaluationMetrics:
        """
        Computes MAE, RMSE, and MAPE for prediction outputs.
        """
        mae = self.calculate_mae(y_true, y_pred)
        rmse = self.calculate_rmse(y_true, y_pred)
        mape = self.calculate_mape(y_true, y_pred)

        return ModelEvaluationMetrics(
            model_name=model_name,
            horizon=horizon,
            mae=mae,
            rmse=rmse,
            mape=mape,
        )

    @staticmethod
    def compare_models(metrics_list: List[ModelEvaluationMetrics]) -> Dict[str, Any]:
        """
        Compares multiple model evaluation metrics and returns best performing model per horizon.
        """
        comparison = {}
        for m in metrics_list:
            h_key = m.horizon.value
            if h_key not in comparison or m.mae < comparison[h_key]["mae"]:
                comparison[h_key] = {
                    "best_model": m.model_name,
                    "mae": m.mae,
                    "rmse": m.rmse,
                    "mape": m.mape,
                }
        return comparison
