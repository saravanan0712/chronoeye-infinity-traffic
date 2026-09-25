"""
ChronoEye Infinity - ST-GNN Model Configuration.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class STGNNConfig(BaseModel):
    """
    Configuration parameters for the Spatio-Temporal Graph Neural Network.
    """
    model_name: str = "ChronoEye-STGNN"
    input_dim: int = 8                # Number of input node features
    output_dim: int = 4               # Number of output target features (flow, density, congestion, travel_time)
    hidden_dim: int = 64              # Hidden representation dimension
    num_spatial_layers: int = 2       # Number of spatial graph convolution layers
    num_temporal_layers: int = 2      # Number of temporal GRU / sequence layers
    input_sequence_length: int = 3    # Input history length T_in (e.g. 3 snapshots = 15 min)
    forecast_horizons: List[str] = Field(
        default_factory=lambda: ["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"]
    )
    dropout: float = 0.10
    use_residual: bool = True
    use_layer_norm: bool = True
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    random_seed: int = 42
    device: str = "cpu"

    @property
    def num_horizons(self) -> int:
        return len(self.forecast_horizons)
