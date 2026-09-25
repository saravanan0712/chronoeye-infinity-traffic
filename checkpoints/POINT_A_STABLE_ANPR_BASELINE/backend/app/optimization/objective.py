"""
ChronoEye Infinity - Phase 10: Traffic Signal Objective Function
Calculates composite objective cost J for candidate signal timing decisions.
Cost = w_q * Queue + w_w * WaitingTime + w_c * PredictedCongestion + w_s * SwitchPenalty - w_e * EmergencyBonus
"""

from typing import Dict, Any
from app.optimization.signal_schema import SignalPhase, SignalState, IntersectionState


class SignalObjectiveFunction:
    """
    Traffic Signal Optimization Objective Function.
    """

    def __init__(
        self,
        w_queue: float = 1.0,
        w_wait: float = 0.5,
        w_congestion: float = 10.0,
        w_switch: float = 5.0,
        w_emergency: float = 100.0,
    ):
        self.w_queue = w_queue
        self.w_wait = w_wait
        self.w_congestion = w_congestion
        self.w_switch = w_switch
        self.w_emergency = w_emergency

    def evaluate_cost(
        self,
        signal_state: SignalState,
        intersection_state: IntersectionState,
        candidate_phase: SignalPhase,
        green_duration: float,
    ) -> float:
        """
        Evaluates objective cost J for proposed phase and green duration.
        Lower cost J indicates better signal control decision.
        """
        is_switch = candidate_phase != signal_state.current_phase

        # Unserved queue penalty
        if candidate_phase == SignalPhase.PHASE_NORTH_SOUTH:
            unserved_queue = intersection_state.queue_ew + max(0.0, intersection_state.predicted_queue_ew)
            served_queue = intersection_state.queue_ns
            emergency_served = intersection_state.has_emergency_vehicle_ns
            emergency_blocked = intersection_state.has_emergency_vehicle_ew
        else:
            unserved_queue = intersection_state.queue_ns + max(0.0, intersection_state.predicted_queue_ns)
            served_queue = intersection_state.queue_ew
            emergency_served = intersection_state.has_emergency_vehicle_ew
            emergency_blocked = intersection_state.has_emergency_vehicle_ns

        # Queue term
        queue_cost = self.w_queue * unserved_queue

        # Waiting time term (accumulates on unserved direction)
        wait_cost = self.w_wait * (unserved_queue * green_duration)

        # Predicted congestion term
        cong_cost = self.w_congestion * (unserved_queue / max(1.0, served_queue + unserved_queue))

        # Switch penalty (yellow & all-red lost clearance time)
        switch_cost = self.w_switch * (signal_state.yellow_time + signal_state.all_red_time) if is_switch else 0.0

        # Emergency vehicle bonus / penalty
        emergency_cost = 0.0
        if emergency_served:
            emergency_cost -= self.w_emergency * 10.0
        if emergency_blocked:
            emergency_cost += self.w_emergency * 10.0

        total_cost = queue_cost + wait_cost + cong_cost + switch_cost + emergency_cost
        return round(float(total_cost), 2)
