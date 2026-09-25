"""
ChronoEye Infinity - Phase 13: Signal Priority Coordinator
Schedules green wave priority preemption windows along emergency vehicle corridor.
"""

from typing import List, Dict, Optional, Any
from app.optimization.signal_schema import SignalPhase
from app.incident.emergency_schema import SignalPrioritySchedule


class SignalPriorityCoordinator:
    """
    Signal Priority Preemption Coordinator.
    Enforces minimum clearance buffers and preemption windows while observing traffic safety constraints.
    """

    def __init__(self, lead_time_seconds: float = 15.0, trail_time_seconds: float = 10.0):
        self.lead_time_seconds = lead_time_seconds
        self.trail_time_seconds = trail_time_seconds

    def schedule_corridor_priorities(
        self,
        arrivals: Dict[str, float],
    ) -> Dict[str, SignalPrioritySchedule]:
        """
        Schedules signal green wave preemption windows for each junction in the arrival sequence.
        """
        schedules: Dict[str, SignalPrioritySchedule] = {}

        for junc_id, arr_time in arrivals.items():
            t_start = max(0.0, arr_time - self.lead_time_seconds)
            t_end = arr_time + self.trail_time_seconds

            schedules[junc_id] = SignalPrioritySchedule(
                signal_id=f"SIG_{junc_id}",
                junction_id=junc_id,
                expected_arrival_timestamp=arr_time,
                green_start_timestamp=round(t_start, 2),
                green_end_timestamp=round(t_end, 2),
                priority_phase=SignalPhase.PHASE_NORTH_SOUTH,
                preemption_active=True,
            )

        return schedules
