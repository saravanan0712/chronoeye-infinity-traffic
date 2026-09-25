"""
ChronoEye Infinity - Phase 13: Emergency Green Corridor Engine
Master orchestrator managing emergency corridor activation, signal priority preemption,
emergency cancellation, and safe return to normal control.
"""

from typing import List, Dict, Optional, Any
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.incident.emergency_schema import (
    EmergencyVehicle,
    SignalPrioritySchedule,
    CorridorStatus,
    CorridorPlan,
)
from app.incident.emergency_route import EmergencyRoutePlanner
from app.incident.corridor_planner import CorridorPlanner
from app.incident.signal_priority import SignalPriorityCoordinator


class EmergencyCorridorEngine:
    """
    Master Emergency Green Corridor Engine.
    """

    def __init__(self):
        self.route_planner = EmergencyRoutePlanner()
        self.corridor_planner = CorridorPlanner()
        self.priority_coordinator = SignalPriorityCoordinator()
        self.active_corridors: Dict[str, CorridorPlan] = {}
        self._corridor_counter = 100

    def create_corridor_plan(
        self,
        builder: SpatioTemporalGraphBuilder,
        vehicle: EmergencyVehicle,
        departure_timestamp: float = 0.0,
    ) -> CorridorPlan:
        """
        Creates and activates green wave priority corridor plan for emergency vehicle.
        """
        corridor_id = f"CORRIDOR_{self._corridor_counter}"
        self._corridor_counter += 1

        route = self.route_planner.plan_emergency_route(builder, vehicle)
        arrivals = self.corridor_planner.calculate_junction_arrivals(route, departure_timestamp)
        schedules = self.priority_coordinator.schedule_corridor_priorities(arrivals)

        plan = CorridorPlan(
            corridor_id=corridor_id,
            emergency_vehicle=vehicle,
            ordered_junctions=route.path_junctions,
            path_roads=route.path_roads,
            total_distance_km=route.total_distance_km,
            estimated_total_travel_time_seconds=route.total_travel_time_seconds,
            predicted_arrival_times=arrivals,
            signal_priority_schedule=schedules,
            status=CorridorStatus.ACTIVE_GREEN_WAVE,
        )

        self.active_corridors[corridor_id] = plan
        return plan

    def cancel_corridor(self, corridor_id: str) -> bool:
        """
        Cancels active emergency corridor and releases signal preemption overrides.
        """
        if corridor_id in self.active_corridors:
            self.active_corridors[corridor_id].status = CorridorStatus.CANCELLED
            for sch in self.active_corridors[corridor_id].signal_priority_schedule.values():
                sch.preemption_active = False
            return True
        return False

    def return_to_normal_control(self, corridor_id: str) -> bool:
        """
        Completes corridor operation and smoothly restores normal traffic signal control.
        """
        if corridor_id in self.active_corridors:
            self.active_corridors[corridor_id].status = CorridorStatus.COMPLETED
            for sch in self.active_corridors[corridor_id].signal_priority_schedule.values():
                sch.preemption_active = False
            return True
        return False
