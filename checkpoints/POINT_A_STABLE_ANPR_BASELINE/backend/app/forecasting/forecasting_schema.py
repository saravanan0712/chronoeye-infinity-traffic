"""
ChronoEye Infinity - Phase 8: Traffic Forecasting Schemas
Defines data models for multi-horizon traffic predictions (+5, +10, +15, +30 min),
network forecast snapshots, model types, and evaluation metrics (MAE, RMSE, MAPE).
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, model_validator


class ForecastHorizon(str, Enum):
    PLUS_5MIN = "PLUS_5MIN"     # +5 minutes (1 step)
    PLUS_10MIN = "PLUS_10MIN"   # +10 minutes (2 steps)
    PLUS_15MIN = "PLUS_15MIN"   # +15 minutes (3 steps)
    PLUS_30MIN = "PLUS_30MIN"   # +30 minutes (6 steps)


class ForecastingModelType(str, Enum):
    PERSISTENCE = "PERSISTENCE"
    RIDGE = "RIDGE"
    ST_GNN = "ST_GNN"


class SegmentForecast(BaseModel):
    """
    Traffic prediction vector for a single road segment at a specific forecast horizon.
    """
    segment_id: str
    target_timestamp: float
    horizon: ForecastHorizon
    predicted_flow: float = 0.0          # vehicles / hour
    predicted_queue: int = 0             # queued vehicle count
    predicted_density: float = 0.0        # vehicles / km
    predicted_speed: float = 60.0        # km/h
    predicted_travel_time: float = 30.0   # seconds
    predicted_congestion: float = 0.0     # congestion score (0.0 - 1.0)
    provenance: Optional[Dict[str, Any]] = Field(default_factory=dict)
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None

    # Backward-compatible property alias
    @property
    def predicted_queue_length(self) -> int:
        """Alias for predicted_queue (backward compatibility)."""
        return self.predicted_queue



class NetworkForecastSnapshot(BaseModel):
    """
    Network-wide traffic forecast snapshot across all road segments at a specific forecast horizon.
    """
    snapshot_id: str = "FC_SNAPSHOT"
    timestamp: float
    horizon: ForecastHorizon
    model_type: ForecastingModelType
    segment_forecasts: Dict[str, SegmentForecast] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_model_type(cls, values: Any) -> Any:
        if isinstance(values, dict) and "model_type" in values:
            val = values["model_type"]
            if isinstance(val, str):
                values["model_type"] = val.upper()
        return values

    def __len__(self) -> int:
        return len(self.segment_forecasts)

    def __contains__(self, item: Any) -> bool:
        return item in self.segment_forecasts

    def __getitem__(self, item: str) -> SegmentForecast:
        return self.segment_forecasts[item]

    def __iter__(self):
        return iter(self.segment_forecasts)

    def items(self):
        return self.segment_forecasts.items()

    def values(self):
        return self.segment_forecasts.values()

    def get(self, key: str, default: Any = None) -> Any:
        return self.segment_forecasts.get(key, default)


class ModelEvaluationMetrics(BaseModel):
    """
    Evaluation metrics for a forecasting model at a specific horizon.
    """
    model_name: str
    horizon: ForecastHorizon
    mae: float = 0.0    # Mean Absolute Error
    rmse: float = 0.0   # Root Mean Squared Error
    mape: float = 0.0   # Mean Absolute Percentage Error (%)
