"""
ChronoEye Infinity - ST-GNN Loss Functions & Masked Evaluation Metrics.
Computes MAE, RMSE, and MAPE exclusively over valid ground-truth target entries,
safely ignoring unobserved/masked values and avoiding division by zero.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class MaskedLoss(nn.Module):
    """
    Masked Loss for spatio-temporal regression targets.
    Calculates loss only where target_mask is True and target is not NaN.
    """

    def __init__(self, loss_type: str = "mae", epsilon: float = 1.0):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.epsilon = epsilon

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        pred: [..., Target_Dim]
        target: [..., Target_Dim]
        mask: [..., Target_Dim] boolean mask (True = valid, False = missing)
        """
        valid_mask = ~torch.isnan(target)
        if mask is not None:
            valid_mask = valid_mask & mask.bool()

        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=pred.requires_grad)

        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]

        if self.loss_type == "mae":
            return torch.mean(torch.abs(pred_valid - target_valid))
        elif self.loss_type in ("mse", "rmse"):
            mse = torch.mean((pred_valid - target_valid) ** 2)
            return torch.sqrt(mse) if self.loss_type == "rmse" else mse
        elif self.loss_type == "mape":
            denom = torch.clamp(torch.abs(target_valid), min=self.epsilon)
            return torch.mean(torch.abs(pred_valid - target_valid) / denom) * 100.0
        elif self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_valid, target_valid)
        else:
            return torch.mean(torch.abs(pred_valid - target_valid))


def MaskedMAE(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> float:
    """Computes Masked Mean Absolute Error."""
    loss_fn = MaskedLoss(loss_type="mae")
    val = loss_fn(pred, target, mask).item()
    return float(val) if not math.isnan(val) else 0.0


def MaskedRMSE(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> float:
    """Computes Masked Root Mean Squared Error."""
    loss_fn = MaskedLoss(loss_type="rmse")
    val = loss_fn(pred, target, mask).item()
    return float(val) if not math.isnan(val) else 0.0


def MaskedMAPE(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    epsilon: float = 5.0,
) -> float:
    """
    Computes Masked Mean Absolute Percentage Error (%).
    epsilon: prevents division-by-zero for zero/near-zero traffic flow/speed.
    """
    loss_fn = MaskedLoss(loss_type="mape", epsilon=epsilon)
    val = loss_fn(pred, target, mask).item()
    return float(val) if not math.isnan(val) else 0.0


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    epsilon: float = 5.0,
) -> Dict[str, float]:
    """
    Computes MAE, RMSE, and MAPE metrics dictionary over valid entries.
    """
    return {
        "mae": round(MaskedMAE(pred, target, mask), 4),
        "rmse": round(MaskedRMSE(pred, target, mask), 4),
        "mape": round(MaskedMAPE(pred, target, mask, epsilon=epsilon), 4),
    }
