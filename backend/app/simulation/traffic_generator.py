"""
ChronoEye Infinity - Phase 1 & 14: Synthetic Traffic Generator
Provides SimulationConfig and SyntheticTrafficGenerator adapter around TrafficSimulationEngine
for FastAPI API endpoints, REST routers, and empirical benchmark suites.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from app.simulation.engine import TrafficSimulationEngine


class SimulationConfig(BaseModel):
    num_intersections: int = Field(default=4, description="Number of junctions in simulation")
    seed: int = Field(default=42, description="Random seed for traffic generation")
    spawn_interval: float = Field(default=3.0, description="Vehicle spawn interval in seconds")


class SimulationSnapshotWrapper(dict):
    """
    Dual-interface simulation snapshot object:
    Functions as a canonical dictionary (for JSON serialization & dict APIs)
    and supports attribute access (.detections, .overall_flow, etc.) for object-oriented callers.
    """
    def __init__(self, data: Dict[str, Any]):
        super().__init__(data)

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        if name == "detections":
            return self.get("detections", [])
        if name == "overall_flow":
            metrics = self.get("traffic_metrics", [])
            return sum(m.get("flow_rate", 0.0) for m in metrics) if metrics else 0.0
        if name == "overall_speed":
            metrics = self.get("traffic_metrics", [])
            speeds = [m.get("average_speed_kmh", 0.0) for m in metrics]
            return sum(speeds) / len(speeds) if speeds else 60.0
        if name == "overall_density":
            metrics = self.get("traffic_metrics", [])
            densities = [m.get("density", 0.0) for m in metrics]
            return sum(densities) / len(densities) if densities else 0.0
        if name == "total_queue":
            metrics = self.get("traffic_metrics", [])
            return sum(m.get("queue_length_meters", 0.0) for m in metrics) if metrics else 0.0
        if name == "congestion_score":
            metrics = self.get("traffic_metrics", [])
            congs = [m.get("congestion_score", 0.0) for m in metrics]
            return sum(congs) / len(congs) if congs else 0.0
        if name == "active_vehicles":
            return self.get("active_vehicle_count", 0)
        if name == "traffic_state":
            metrics = self.get("traffic_metrics", [])
            congs = [m.get("congestion_score", 0.20) for m in metrics] if isinstance(metrics, list) else []
            avg_cong = sum(congs) / len(congs) if congs else 0.20
            if avg_cong < 0.25:
                return "FREE_FLOW"
            elif avg_cong < 0.50:
                return "MODERATE"
            elif avg_cong < 0.75:
                return "HEAVY"
            else:
                return "CONGESTED"
        raise AttributeError(f"'SimulationSnapshotWrapper' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class SyntheticTrafficGenerator:
    """
    Synthetic Traffic Generator adapting TrafficSimulationEngine for Phase 14 FastAPI API & benchmarks.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        if config is None:
            config = SimulationConfig()
        self.config = config
        self.engine = TrafficSimulationEngine(seed=self.config.seed)

    def reset(self, seed: Optional[int] = None):
        """Resets the simulation engine state."""
        if seed is not None:
            self.config.seed = seed
        self.engine.reset(seed=self.config.seed)

    def step(self, dt_seconds: float = 1.0) -> SimulationSnapshotWrapper:
        """Advances simulation state by dt_seconds and returns snapshot dictionary."""
        snap = self.engine.step(dt_seconds=dt_seconds)
        return SimulationSnapshotWrapper(snap) if isinstance(snap, dict) else snap
