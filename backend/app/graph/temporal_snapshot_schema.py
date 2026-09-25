"""
ChronoEye Infinity - Module 3 Step 2: Temporal Graph Data Contract
Defines typed schemas for temporal node features, temporal edge features,
windowed temporal graph snapshots G(t) = (V, E, X(t), A(t)), and the temporal graph dataset contract.
"""

import uuid
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, model_validator

from app.graph.graph_schema import NodeType, EdgeType
from app.forecasting.forecasting_schema import ForecastHorizon


class TemporalNodeFeatures(BaseModel):
    """
    Node-level traffic features for a single node within a temporal snapshot window.
    Features that cannot be measured/derived remain None (never fabricated).
    """
    node_id: str
    node_type: NodeType = NodeType.CAMERA
    timestamp: Optional[float] = None
    vehicle_count: Optional[int] = None
    flow_rate: Optional[float] = None          # vehicles / hour
    density: Optional[float] = None            # vehicles / km
    average_speed: Optional[float] = None      # km/h
    queue_length: Optional[int] = None         # queued vehicles count
    congestion: Optional[float] = None         # 0.0 (free flow) to 1.0 (gridlock)
    incoming_flow: Optional[float] = None      # incoming vehicles / hour
    outgoing_flow: Optional[float] = None      # outgoing vehicles / hour
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def handle_field_aliases(cls, values: Any) -> Any:
        if isinstance(values, dict):
            # Support 'flow' as alias for 'flow_rate'
            if "flow_rate" not in values and "flow" in values:
                values["flow_rate"] = values["flow"]
            elif "flow" not in values and "flow_rate" in values:
                values["flow"] = values["flow_rate"]
        return values

    @property
    def flow(self) -> Optional[float]:
        """Alias for flow_rate."""
        return self.flow_rate


class TemporalEdgeFeatures(BaseModel):
    """
    Edge-level transition features and relationships between graph nodes within a temporal snapshot.
    Preserves observed transitions, unobserved gaps, travel time distributions, and ReID evidence.
    """
    source: str
    target: str
    edge_type: EdgeType = EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA
    timestamp: Optional[float] = None
    vehicle_count: Optional[int] = None
    flow_rate: Optional[float] = None          # transitions / hour
    travel_time: Optional[float] = None        # seconds
    mean_travel_time: Optional[float] = None   # seconds
    median_travel_time: Optional[float] = None # seconds
    transition_count: Optional[int] = None     # total transitions
    observed_transition_count: Optional[int] = None
    unobserved_gap_count: Optional[int] = None
    has_unobserved_gap: bool = False
    confidence: Optional[float] = None         # 0.0 to 1.0 mean confidence
    mean_transition_confidence: Optional[float] = None
    direction: Optional[str] = None
    uncertainty: Optional[float] = None        # seconds or normalized uncertainty
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def handle_field_aliases(cls, values: Any) -> Any:
        if isinstance(values, dict):
            # Support 'source_node' / 'target_node' aliases
            if "source" not in values and "source_node" in values:
                values["source"] = values["source_node"]
            if "target" not in values and "target_node" in values:
                values["target"] = values["target_node"]
            # Support 'flow' as alias for 'flow_rate'
            if "flow_rate" not in values and "flow" in values:
                values["flow_rate"] = values["flow"]
            elif "flow" not in values and "flow_rate" in values:
                values["flow"] = values["flow_rate"]
            # Support 'mean_travel_time' alias for 'travel_time'
            if "mean_travel_time" not in values and "travel_time" in values:
                values["mean_travel_time"] = values["travel_time"]
            elif "travel_time" not in values and "mean_travel_time" in values:
                values["travel_time"] = values["mean_travel_time"]
            # Support 'mean_transition_confidence' alias for 'confidence'
            if "mean_transition_confidence" not in values and "confidence" in values:
                values["mean_transition_confidence"] = values["confidence"]
            elif "confidence" not in values and "mean_transition_confidence" in values:
                values["confidence"] = values["mean_transition_confidence"]
        return values

    @property
    def source_node(self) -> str:
        """Alias for source."""
        return self.source

    @property
    def target_node(self) -> str:
        """Alias for target."""
        return self.target

    @property
    def flow(self) -> Optional[float]:
        """Alias for flow_rate."""
        return self.flow_rate


class TemporalGraphSnapshot(BaseModel):
    """
    Single discrete temporal graph snapshot G(t) = (V, E, X(t), A(t))
    representing network topology and dynamic traffic features across a time window [start_time, end_time].
    """
    snapshot_id: str = Field(default_factory=lambda: f"SNAP_{uuid.uuid4().hex[:8]}")
    start_time: float
    end_time: float
    nodes: List[TemporalNodeFeatures] = Field(default_factory=list)
    edges: List[TemporalEdgeFeatures] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def get_node(self, node_id: str) -> Optional[TemporalNodeFeatures]:
        """Finds a node by its ID."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def get_edge(self, source: str, target: str) -> Optional[TemporalEdgeFeatures]:
        """Finds an edge between source and target."""
        for e in self.edges:
            if e.source == source and e.target == target:
                return e
        return None


class TemporalGraphDatasetContract(BaseModel):
    """
    Contract for a sequence of temporal graph snapshots G(t_0), G(t_1), ..., G(t_k)
    configured for downstream ST-GNN forecasting dataset construction and model ingestion.
    """
    dataset_id: str = Field(default_factory=lambda: f"DS_{uuid.uuid4().hex[:8]}")
    snapshots: List[TemporalGraphSnapshot] = Field(default_factory=list)
    feature_names: List[str] = Field(
        default_factory=lambda: [
            "vehicle_count",
            "flow_rate",
            "density",
            "average_speed",
            "queue_length",
            "congestion",
            "incoming_flow",
            "outgoing_flow",
        ]
    )
    target_names: List[str] = Field(
        default_factory=lambda: [
            "flow_rate",
            "density",
            "congestion",
            "travel_time",
        ]
    )
    window_seconds: float = 300.0   # Default 5-minute window
    stride_seconds: float = 300.0   # Default 5-minute stride (non-overlapping)
    forecast_horizons: List[Union[ForecastHorizon, str]] = Field(
        default_factory=lambda: [
            ForecastHorizon.PLUS_5MIN,
            ForecastHorizon.PLUS_10MIN,
            ForecastHorizon.PLUS_15MIN,
        ]
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> TemporalGraphSnapshot:
        return self.snapshots[idx]

    def __iter__(self):
        return iter(self.snapshots)

    def add_snapshot(self, snapshot: TemporalGraphSnapshot):
        """Appends a new chronological snapshot to the dataset."""
        self.snapshots.append(snapshot)
