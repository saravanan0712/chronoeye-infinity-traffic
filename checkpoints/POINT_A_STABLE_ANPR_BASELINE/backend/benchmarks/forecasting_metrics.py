"""
ChronoEye Infinity - Forecasting Metrics Engine
Computes MAE, RMSE, MAPE (with zero-denominator safety), and forecast bias across horizons.
"""

import numpy as np
from typing import List, Dict, Any


def calculate_forecasting_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Calculates forecast accuracy metrics."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "bias": 0.0}

    arr_true = np.array(y_true, dtype=float)
    arr_pred = np.array(y_pred, dtype=float)

    mae = float(np.mean(np.abs(arr_true - arr_pred)))
    rmse = float(np.sqrt(np.mean((arr_true - arr_pred) ** 2)))
    
    # Safe MAPE calculation ignoring zero ground truth values
    non_zero = arr_true != 0.0
    if np.any(non_zero):
        mape = float(np.mean(np.abs((arr_true[non_zero] - arr_pred[non_zero]) / arr_true[non_zero]))) * 100.0
    else:
        mape = 0.0

    bias = float(np.mean(arr_pred - arr_true))

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "bias": round(bias, 2),
    }
