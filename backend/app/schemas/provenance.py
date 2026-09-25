"""
ChronoEye Infinity - Master Data Provenance System
Defines traceable data provenance models classifying system outputs into
OBSERVED, PREDICTED, SIMULATED, and CONTROL values with full lineage metadata.
"""

from enum import Enum
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field


class DataProvenanceType(str, Enum):
    OBSERVED = "OBSERVED"     # Real video / camera computer vision observations
    PREDICTED = "PREDICTED"   # ST-GNN forecasting or predictive model outputs
    SIMULATED = "SIMULATED"   # Synthetic traffic generator or fallback simulation data
    CONTROL = "CONTROL"       # Signal optimization or emergency green corridor decisions


class DataProvenance(BaseModel):
    """
    Data Provenance tracking envelope for every operational state & dashboard metric.
    Ensures complete lineage tracing (Source -> Model/Engine -> API -> UI).
    """
    provenance_type: DataProvenanceType = DataProvenanceType.OBSERVED
    source_id: str = "CAM_A_EAST"
    process_name: str = "ComputerVision_Tracker"
    timestamp: float = 0.0
    confidence: float = 1.0
    is_mock: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
