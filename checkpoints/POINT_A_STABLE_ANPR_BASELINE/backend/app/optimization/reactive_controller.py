"""
ChronoEye Infinity - Phase 10: Actuated / Reactive Signal Controller
Actuated controller that dynamically extends or switches green phases based on real-time queue sensing.
"""

from app.optimization.signal_schema import (
    SignalPhase,
    SignalState,
    IntersectionState,
    SignalOptimizationPlan,
)


class ReactiveController:
    """
    Actuated / Reactive Signal Controller.
    Responds dynamically to real-time queue imbalances and emergency vehicle presence.
    """

    def __init__(self, queue_diff_threshold: int = 3, green_extension_step: float = 5.0):
        self.queue_diff_threshold = queue_diff_threshold
        self.green_extension_step = green_extension_step

    def decide_signal_plan(
        self,
        signal_state: SignalState,
        intersection_state: IntersectionState,
    ) -> SignalOptimizationPlan:
        """
        Determines reactive signal control action based on current queue state.
        """
        # Safety constraint 1: enforce min_green clearance
        if signal_state.elapsed_green < signal_state.min_green:
            return SignalOptimizationPlan(
                signal_id=signal_state.signal_id,
                recommended_phase=signal_state.current_phase,
                recommended_green_duration=signal_state.min_green,
                is_phase_switch=False,
            )

        # Safety constraint 2: max_green force switch
        if signal_state.elapsed_green >= signal_state.max_green:
            next_phase = (
                SignalPhase.PHASE_EAST_WEST
                if signal_state.current_phase == SignalPhase.PHASE_NORTH_SOUTH
                else SignalPhase.PHASE_NORTH_SOUTH
            )
            return SignalOptimizationPlan(
                signal_id=signal_state.signal_id,
                recommended_phase=next_phase,
                recommended_green_duration=signal_state.min_green,
                is_phase_switch=True,
            )

        # 1. Emergency vehicle priority override
        if intersection_state.has_emergency_vehicle_ns and signal_state.current_phase != SignalPhase.PHASE_NORTH_SOUTH:
            return SignalOptimizationPlan(
                signal_id=signal_state.signal_id,
                recommended_phase=SignalPhase.PHASE_NORTH_SOUTH,
                recommended_green_duration=20.0,
                is_phase_switch=True,
            )
        elif intersection_state.has_emergency_vehicle_ew and signal_state.current_phase != SignalPhase.PHASE_EAST_WEST:
            return SignalOptimizationPlan(
                signal_id=signal_state.signal_id,
                recommended_phase=SignalPhase.PHASE_EAST_WEST,
                recommended_green_duration=20.0,
                is_phase_switch=True,
            )

        # 2. Queue imbalance evaluation
        q_ns = intersection_state.queue_ns
        q_ew = intersection_state.queue_ew

        if signal_state.current_phase == SignalPhase.PHASE_NORTH_SOUTH:
            if q_ew - q_ns >= self.queue_diff_threshold:
                # Switch to East-West
                return SignalOptimizationPlan(
                    signal_id=signal_state.signal_id,
                    recommended_phase=SignalPhase.PHASE_EAST_WEST,
                    recommended_green_duration=self.green_extension_step,
                    is_phase_switch=True,
                )
            else:
                # Extend North-South
                return SignalOptimizationPlan(
                    signal_id=signal_state.signal_id,
                    recommended_phase=SignalPhase.PHASE_NORTH_SOUTH,
                    recommended_green_duration=self.green_extension_step,
                    is_phase_switch=False,
                )
        else:
            if q_ns - q_ew >= self.queue_diff_threshold:
                # Switch to North-South
                return SignalOptimizationPlan(
                    signal_id=signal_state.signal_id,
                    recommended_phase=SignalPhase.PHASE_NORTH_SOUTH,
                    recommended_green_duration=self.green_extension_step,
                    is_phase_switch=True,
                )
            else:
                # Extend East-West
                return SignalOptimizationPlan(
                    signal_id=signal_state.signal_id,
                    recommended_phase=SignalPhase.PHASE_EAST_WEST,
                    recommended_green_duration=self.green_extension_step,
                    is_phase_switch=False,
                )
