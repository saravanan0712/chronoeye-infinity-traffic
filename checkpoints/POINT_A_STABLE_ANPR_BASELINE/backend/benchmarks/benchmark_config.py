"""
ChronoEye Infinity - Benchmark Configuration
Defines benchmark scenarios, seeds, parameters, and baseline execution modes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class BenchmarkConfig:
    scenarios: List[str] = field(default_factory=lambda: [
        "free_flow",
        "normal_traffic",
        "rush_hour",
        "demand_surge",
        "persistent_congestion",
        "road_blockage",
        "multi_junction_propagation",
        "forecasted_congestion",
        "emergency_corridor",
        "incident_recovery",
        "mixed_traffic_demand",
        "overloaded_stress"
    ])
    seeds: List[int] = field(default_factory=lambda: [42, 101, 202, 303, 404])
    simulation_duration_steps: int = 20
    num_intersections: int = 4
    output_dir: str = "e:/chronoeye/data/benchmarks"
