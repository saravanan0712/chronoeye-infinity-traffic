"""
ChronoEye Infinity - Traffic Forecasting Baseline Models.
Implements Persistence (Last Observed Value), Moving Average, and Linear Regression Baselines.
"""

import math
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from app.models.stgnn.loss import compute_all_metrics


class BaselineEvaluationResult(BaseModel):
    model_name: str
    horizon: str
    target: str
    mae: float
    rmse: float
    mape: float
    evaluated_samples_count: int


class PersistenceBaseline:
    """
    Persistence / Last Value Baseline.
    Predicts future state at all horizons using the most recent historical observation:
    Y_pred(t + H) = X(t)
    """

    def __init__(self, target_indices_in_features: Optional[Dict[str, int]] = None):
        self.target_indices = target_indices_in_features or {
            "flow_rate": 1,
            "density": 2,
            "congestion": 5,
            "travel_time": 1,
        }

    def predict(
        self,
        x: torch.Tensor,
        num_horizons: int,
        num_targets: int,
        feature_to_target_map: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        x: [B, T_in, N, F]
        returns: [B, H, N, T_target]
        """
        B, T_in, N, F = x.shape
        last_step = x[:, -1, :, :]  # [B, N, F]

        if feature_to_target_map is not None:
            selected_features = last_step[:, :, feature_to_target_map]  # [B, N, T_target]
        else:
            selected_features = last_step[:, :, :num_targets]

        # Repeat across all horizons
        preds = selected_features.unsqueeze(1).repeat(1, num_horizons, 1, 1)  # [B, H, N, T_target]
        return preds


class MovingAverageBaseline:
    """
    Moving Average / Historical Mean Baseline.
    Predicts future state as the mean of the input history window:
    Y_pred(t + H) = (1 / T_in) * sum_{k=0}^{T_in-1} X(t - k)
    """

    def __init__(self, target_indices_in_features: Optional[Dict[str, int]] = None):
        self.target_indices = target_indices_in_features

    def predict(
        self,
        x: torch.Tensor,
        num_horizons: int,
        num_targets: int,
        feature_to_target_map: Optional[List[int]] = None,
        x_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, T_in, N, F]
        returns: [B, H, N, T_target]
        """
        B, T_in, N, F = x.shape
        
        if x_mask is not None:
            valid_x = torch.where(x_mask.bool(), x, torch.nan)
            mean_step = torch.nanmean(valid_x, dim=1)  # [B, N, F]
            mean_step = torch.nan_to_num(mean_step, nan=0.0)
        else:
            mean_step = torch.mean(x, dim=1)  # [B, N, F]

        if feature_to_target_map is not None:
            selected_features = mean_step[:, :, feature_to_target_map]  # [B, N, T_target]
        else:
            selected_features = mean_step[:, :, :num_targets]

        preds = selected_features.unsqueeze(1).repeat(1, num_horizons, 1, 1)  # [B, H, N, T_target]
        return preds


class LinearRegressionBaseline:
    """
    Linear Ridge Regression Baseline.
    Fits a linear ridge regressor on flattened historical input features per node.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights: Optional[torch.Tensor] = None  # [T_in * F, H * T_target]

    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor):
        """
        x_train: [B, T_in, N, F]
        y_train: [B, H, N, T_target]
        """
        B, T_in, N, F = x_train.shape
        _, H, _, T_target = y_train.shape

        # Reshape to 2D regression: [B * N, T_in * F] -> [B * N, H * T_target]
        X_mat = x_train.permute(0, 2, 1, 3).contiguous().view(B * N, T_in * F)
        Y_mat = y_train.permute(0, 2, 1, 3).contiguous().view(B * N, H * T_target)

        # Replace NaNs with 0
        X_mat = torch.nan_to_num(X_mat, nan=0.0)
        Y_mat = torch.nan_to_num(Y_mat, nan=0.0)

        # Ridge solution: W = (X^T X + alpha * I)^(-1) X^T Y
        XtX = torch.matmul(X_mat.t(), X_mat)
        reg = self.alpha * torch.eye(XtX.size(0), device=x_train.device)
        inv_term = torch.inverse(XtX + reg)
        self.weights = torch.matmul(torch.matmul(inv_term, X_mat.t()), Y_mat)

    def predict(self, x: torch.Tensor, num_horizons: int, num_targets: int) -> torch.Tensor:
        """
        x: [B, T_in, N, F]
        returns: [B, H, N, T_target]
        """
        B, T_in, N, F = x.shape
        if self.weights is None:
            # Fallback to persistence if not fitted
            return PersistenceBaseline().predict(x, num_horizons, num_targets)

        X_mat = x.permute(0, 2, 1, 3).contiguous().view(B * N, T_in * F)
        X_mat = torch.nan_to_num(X_mat, nan=0.0)
        Y_pred = torch.matmul(X_mat, self.weights)  # [B * N, H * T_target]

        preds = Y_pred.view(B, N, num_horizons, num_targets).permute(0, 2, 1, 3).contiguous()
        return preds
