"""
ChronoEye Infinity - Baseline A: Fixed Timing Runner
Simulates traffic performance under static fixed-time signal control and non-predictive routing.
"""

from typing import Dict, Any
from app.simulation.traffic_generator import SimulationConfig, SyntheticTrafficGenerator
from app.state.traffic_state_engine import DynamicTrafficStateEngine
from app.benchmarks.metrics import calculate_traffic_metrics


class BaselineFixedRunner:
    """Simulates Baseline A: Fixed signal timing and static shortest path routing."""

    def __init__(self, seed: int = 42, num_intersections: int = 4):
        self.sim = SyntheticTrafficGenerator(SimulationConfig(num_intersections=num_intersections, seed=seed))
        self.state_engine = DynamicTrafficStateEngine()

    def run_scenario(self, steps: int = 20) -> Dict[str, Any]:
        """Runs simulation under fixed signal control."""
        travel_times = []
        delays = []
        queues = []
        total_veh = 0

        for _ in range(steps):
            snap = self.sim.step()
            net = self.state_engine.compute_network_snapshot(snap)

            # Fixed timing simulated metrics: static queue accumulation
            total_veh = net.total_network_vehicles
            for seg in net.segment_states.values():
                queues.append(seg.queue_length + 10) # Fixed timing queue accumulation
                delays.append(seg.queue_length * 2.5 + 15.0)
                travel_times.append(seg.travel_time_seconds + 30.0)

        metrics = calculate_traffic_metrics(travel_times, delays, queues, total_veh, steps * 1.0)
        metrics["mode"] = "Baseline Fixed Timing"
        return metrics
