"""
ChronoEye Infinity - Phase 10: Fixed-Time Pre-Timed Signal Controller
Baseline pre-timed controller operating on static fixed-duration green phase cycles.
"""

from app.optimization.signal_schema import (
    SignalPhase,
    SignalState,
    IntersectionState,
    SignalOptimizationPlan,
)


class FixedController:
    """
    Fixed-Time Pre-Timed Signal Controller.
    Switches phases at fixed static green duration thresholds (default 30 seconds).
    """

    def __init__(self, fixed_green_duration: float = 30.0):
        self.fixed_green_duration = fixed_green_duration

    def decide_signal_plan(
        self,
        signal_state: SignalState,
        intersection_state: IntersectionState,
    ) -> SignalOptimizationPlan:
        """
        Determines fixed signal control action.
        """
        # Safety constraint: enforce min_green clearance
        if signal_state.elapsed_green < signal_state.min_green:
            return SignalOptimizationPlan(
                signal_id=signal_state.signal_id,
                recommended_phase=signal_state.current_phase,
                recommended_green_duration=self.fixed_green_duration,
                is_phase_switch=False,
            )

        # Switch phase if elapsed_green >= fixed_green_duration
        if signal_state.elapsed_green >= self.fixed_green_duration:
            next_phase = (
                SignalPhase.PHASE_EAST_WEST
                if signal_state.current_phase == SignalPhase.PHASE_NORTH_SOUTH
                else SignalPhase.PHASE_NORTH_SOUTH
            )
            return SignalOptimizationPlan(
                signal_id=signal_state.signal_id,
                recommended_phase=next_phase,
                recommended_green_duration=self.fixed_green_duration,
                is_phase_switch=True,
            )

        return SignalOptimizationPlan(
            signal_id=signal_state.signal_id,
            recommended_phase=signal_state.current_phase,
            recommended_green_duration=self.fixed_green_duration,
            is_phase_switch=False,
        )
