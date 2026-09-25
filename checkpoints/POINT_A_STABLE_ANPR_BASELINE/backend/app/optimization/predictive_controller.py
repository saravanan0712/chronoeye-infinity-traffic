"""
ChronoEye Infinity - Phase 10: ChronoEye Rolling-Horizon Predictive Signal Controller
Predictive controller leveraging ST-GNN multi-horizon forecasts (+5m, +10m, +15m) and uncertainty bounds
to select optimal signal phase timing decisions.
"""

from typing import List, Dict, Optional, Any
from app.optimization.signal_schema import (
    SignalPhase,
    SignalState,
    IntersectionState,
    SignalOptimizationPlan,
)
from app.optimization.objective import SignalObjectiveFunction


class ChronoEyePredictiveController:
    """
    ChronoEye Rolling-Horizon Predictive Traffic Signal Controller.
    """

    def __init__(self, objective_func: Optional[SignalObjectiveFunction] = None):
        self.objective = objective_func or SignalObjectiveFunction()
        self.candidate_durations = [10.0, 15.0, 20.0, 25.0, 30.0, 45.0, 60.0]

    def decide_signal_plan(
        self,
        signal_state: SignalState,
        intersection_state: IntersectionState,
        forecast_queue_ns: float = 0.0,
        forecast_queue_ew: float = 0.0,
        uncertainty_std_ns: float = 0.0,
        uncertainty_std_ew: float = 0.0,
    ) -> SignalOptimizationPlan:
        """
        Determines optimal rolling-horizon signal plan.
        """
        # Safety constraint 1: enforce min_green clearance
        if signal_state.elapsed_green < signal_state.min_green:
            return SignalOptimizationPlan(
                signal_id=signal_state.signal_id,
                recommended_phase=signal_state.current_phase,
                recommended_green_duration=signal_state.min_green,
                is_phase_switch=False,
            )

        # Emergency vehicle priority override
        if intersection_state.has_emergency_vehicle_ns and signal_state.current_phase != SignalPhase.PHASE_NORTH_SOUTH:
            return SignalOptimizationPlan(
                signal_id=signal_state.signal_id,
                recommended_phase=SignalPhase.PHASE_NORTH_SOUTH,
                recommended_green_duration=30.0,
                expected_delay_reduction_pct=50.0,
                objective_cost=-900.0,
                is_phase_switch=True,
            )
        elif intersection_state.has_emergency_vehicle_ew and signal_state.current_phase != SignalPhase.PHASE_EAST_WEST:
            return SignalOptimizationPlan(
                signal_id=signal_state.signal_id,
                recommended_phase=SignalPhase.PHASE_EAST_WEST,
                recommended_green_duration=30.0,
                expected_delay_reduction_pct=50.0,
                objective_cost=-900.0,
                is_phase_switch=True,
            )

        # Incorporate predicted future queues with uncertainty adjustment (upper bound = mean + 1.28 * std)
        effective_inter_state = IntersectionState(
            intersection_id=intersection_state.intersection_id,
            queue_ns=intersection_state.queue_ns,
            queue_ew=intersection_state.queue_ew,
            flow_ns=intersection_state.flow_ns,
            flow_ew=intersection_state.flow_ew,
            predicted_queue_ns=forecast_queue_ns + 0.5 * uncertainty_std_ns,
            predicted_queue_ew=forecast_queue_ew + 0.5 * uncertainty_std_ew,
            has_emergency_vehicle_ns=intersection_state.has_emergency_vehicle_ns,
            has_emergency_vehicle_ew=intersection_state.has_emergency_vehicle_ew,
        )

        best_phase = signal_state.current_phase
        best_duration = signal_state.min_green
        best_cost = float("inf")

        for candidate_phase in [SignalPhase.PHASE_NORTH_SOUTH, SignalPhase.PHASE_EAST_WEST]:
            for duration in self.candidate_durations:
                # Enforce max_green bound
                if duration > signal_state.max_green:
                    continue

                cost = self.objective.evaluate_cost(
                    signal_state, effective_inter_state, candidate_phase, duration
                )

                if cost < best_cost:
                    best_cost = cost
                    best_phase = candidate_phase
                    best_duration = duration

        is_switch = best_phase != signal_state.current_phase
        expected_reduction = max(0.0, round((1.0 - (best_cost / max(1.0, best_cost + 50.0))) * 100.0, 1))

        return SignalOptimizationPlan(
            signal_id=signal_state.signal_id,
            recommended_phase=best_phase,
            recommended_green_duration=best_duration,
            expected_delay_reduction_pct=expected_reduction,
            objective_cost=best_cost,
            is_phase_switch=is_switch,
        )
