"""
ChronoEye Infinity - Benchmark Scenario Definitions
Configures 12 standard urban traffic scenarios for empirical evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class ScenarioConfig:
    name: str
    injection_rate: float = 1.0
    num_intersections: int = 4
    has_road_blockage: bool = False
    has_emergency_vehicle: bool = False
    affected_segment: str = "ROAD_1_2"
    duration_steps: int = 20
    seed: int = 42


SCENARIOS_REGISTRY: Dict[str, ScenarioConfig] = {
    "free_flow": ScenarioConfig(name="free_flow", injection_rate=0.4),
    "normal_traffic": ScenarioConfig(name="normal_traffic", injection_rate=1.0),
    "rush_hour": ScenarioConfig(name="rush_hour", injection_rate=2.5),
    "demand_surge": ScenarioConfig(name="demand_surge", injection_rate=3.5),
    "persistent_congestion": ScenarioConfig(name="persistent_congestion", injection_rate=2.8),
    "road_blockage": ScenarioConfig(name="road_blockage", injection_rate=1.2, has_road_blockage=True, affected_segment="ROAD_1_2"),
    "multi_junction_propagation": ScenarioConfig(name="multi_junction_propagation", injection_rate=2.2),
    "forecasted_congestion": ScenarioConfig(name="forecasted_congestion", injection_rate=2.0),
    "emergency_corridor": ScenarioConfig(name="emergency_corridor", injection_rate=1.5, has_emergency_vehicle=True),
    "incident_recovery": ScenarioConfig(name="incident_recovery", injection_rate=1.5, has_road_blockage=True),
    "mixed_traffic_demand": ScenarioConfig(name="mixed_traffic_demand", injection_rate=1.8),
    "overloaded_stress": ScenarioConfig(name="overloaded_stress", injection_rate=4.0),
}


def get_scenario_config(name: str, seed: int = 42, steps: int = 20) -> ScenarioConfig:
    """Retrieves scenario configuration with custom seed and duration."""
    base = SCENARIOS_REGISTRY.get(name, ScenarioConfig(name=name))
    base.seed = seed
    base.duration_steps = steps
    return base
