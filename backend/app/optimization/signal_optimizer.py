"""
ChronoEye Infinity - Phase 10: Traffic Signal Optimization Engine
Master engine running rolling-horizon signal controllers and conducting reproducible comparative experiments
across Scenarios A to F (Low, Heavy, Demand Spike, Directional Imbalance, Incident, Emergency Vehicle).
"""

from typing import Dict, List, Optional, Any
from app.optimization.signal_schema import (
    SignalPhase,
    SignalState,
    IntersectionState,
    SignalOptimizationPlan,
)
from app.optimization.fixed_controller import FixedController
from app.optimization.reactive_controller import ReactiveController
from app.optimization.predictive_controller import ChronoEyePredictiveController


class TrafficSignalOptimizationEngine:
    """
    Traffic Signal Optimization & Experimentation Engine.
    """

    def __init__(self):
        self.fixed_controller = FixedController(fixed_green_duration=30.0)
        self.reactive_controller = ReactiveController(queue_diff_threshold=3)
        self.predictive_controller = ChronoEyePredictiveController()

    def run_scenario_experiment(
        self,
        scenario_name: str = "default",
        num_steps: int = 60,
        initial_inter_state: Optional[IntersectionState] = None,
    ) -> Dict[str, Any]:

        """
        Executes reproducible simulation experiment comparing Fixed vs Reactive vs ChronoEye Predictive controllers.
        Returns metrics: average_delay, maximum_queue, average_queue, throughput, travel_time.
        """
        inter = initial_inter_state or IntersectionState(
            intersection_id="JUNC_A", queue_ns=10, queue_ew=10, flow_ns=300.0, flow_ew=300.0
        )

        results = {}

        for ctrl_name, controller in [
            ("Fixed_Controller", self.fixed_controller),
            ("Reactive_Controller", self.reactive_controller),
            ("ChronoEye_Predictive", self.predictive_controller),
        ]:
            sig_state = SignalState(signal_id="SIG_01", current_phase=SignalPhase.PHASE_NORTH_SOUTH)
            sim_queue_ns = float(inter.queue_ns)
            sim_queue_ew = float(inter.queue_ew)

            total_delay = 0.0
            max_queue = 0
            queue_history = []
            vehicles_cleared = 0

            for t in range(num_steps):
                cur_inter = IntersectionState(
                    intersection_id=inter.intersection_id,
                    queue_ns=int(sim_queue_ns),
                    queue_ew=int(sim_queue_ew),
                    flow_ns=inter.flow_ns,
                    flow_ew=inter.flow_ew,
                    has_emergency_vehicle_ns=inter.has_emergency_vehicle_ns,
                    has_emergency_vehicle_ew=inter.has_emergency_vehicle_ew,
                )

                if ctrl_name == "ChronoEye_Predictive":
                    plan = self.predictive_controller.decide_signal_plan(
                        sig_state, cur_inter, forecast_queue_ns=sim_queue_ns*1.1, forecast_queue_ew=sim_queue_ew*1.1
                    )
                elif ctrl_name == "Reactive_Controller":
                    plan = self.reactive_controller.decide_signal_plan(sig_state, cur_inter)
                else:
                    plan = self.fixed_controller.decide_signal_plan(sig_state, cur_inter)

                # Execute signal phase step
                if plan.is_phase_switch:
                    sig_state.current_phase = plan.recommended_phase
                    sig_state.elapsed_green = 0.0
                else:
                    sig_state.elapsed_green += 5.0

                # Traffic dynamics step
                if sig_state.current_phase == SignalPhase.PHASE_NORTH_SOUTH:
                    cleared_ns = min(sim_queue_ns, 3)
                    sim_queue_ns -= cleared_ns
                    sim_queue_ew += (inter.flow_ew / 3600.0) * 5.0
                    vehicles_cleared += cleared_ns
                else:
                    cleared_ew = min(sim_queue_ew, 3)
                    sim_queue_ew -= cleared_ew
                    sim_queue_ns += (inter.flow_ns / 3600.0) * 5.0
                    vehicles_cleared += cleared_ew

                total_q = sim_queue_ns + sim_queue_ew
                queue_history.append(total_q)
                total_delay += total_q * 5.0
                if total_q > max_queue:
                    max_queue = int(total_q)

            avg_q = sum(queue_history) / float(len(queue_history)) if queue_history else 0.0
            avg_delay = total_delay / float(max(1, vehicles_cleared))
            throughput = (vehicles_cleared / float(num_steps * 5)) * 3600.0
            avg_travel_time = 30.0 + avg_delay

            results[ctrl_name] = {
                "average_delay": round(avg_delay, 2),
                "maximum_queue": max_queue,
                "average_queue": round(avg_q, 2),
                "throughput": round(throughput, 2),
                "travel_time": round(avg_travel_time, 2),
            }

        return {
            "scenario_name": scenario_name,
            "controllers": results,
        }
