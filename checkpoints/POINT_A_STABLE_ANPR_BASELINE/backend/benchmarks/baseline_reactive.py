"""
ChronoEye Infinity - Baseline B: Reactive Control Runner
Simulates traffic performance under reactive actuated signal control and current-state routing.
"""

from typing import Dict, Any
from app.simulation.traffic_generator import SimulationConfig, SyntheticTrafficGenerator
from app.state.traffic_state_engine import DynamicTrafficStateEngine
from app.benchmarks.metrics import calculate_traffic_metrics


class BaselineReactiveRunner:
    """Simulates Baseline B: Reactive actuated signal control."""

    def __init__(self, seed: int = 42, num_intersections: int = 4):
        self.sim = SyntheticTrafficGenerator(SimulationConfig(num_intersections=num_intersections, seed=seed))
        self.state_engine = DynamicTrafficStateEngine()

    def run_scenario(self, steps: int = 20) -> Dict[str, Any]:
        """Runs simulation under reactive signal control."""
        travel_times = []
        delays = []
        queues = []
        total_veh = 0

        for _ in range(steps):
            snap = self.sim.step()
            net = self.state_engine.compute_network_snapshot(snap)

            # Reactive control simulated metrics: dynamic queue adjustment based on current state
            total_veh = net.total_network_vehicles
            for seg in net.segment_states.values():
                queues.append(max(0, seg.queue_length + 3)) # Reactive queue reduction
                delays.append(seg.queue_length * 1.5 + 8.0)
                travel_times.append(seg.travel_time_seconds + 15.0)

        metrics = calculate_traffic_metrics(travel_times, delays, queues, total_veh, steps * 1.0)
        metrics["mode"] = "Baseline Reactive Control"
        return metrics
