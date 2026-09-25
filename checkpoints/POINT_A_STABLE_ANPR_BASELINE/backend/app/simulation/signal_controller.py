"""
ChronoEye Infinity - Phase 1: Traffic Signal Controller Module
Manages signal timing, phase transitions, reactive queue extensions,
predictive schedule adjustments, and emergency green wave overrides.
"""

from typing import Dict, List, Optional
from app.schemas.simulation import (
    TrafficSignalState,
    SignalMode,
    SignalPhase,
    RoadSegmentDefinition,
)


class SignalControllerManager:
    """
    Controls dynamic traffic signals across junctions in the simulation.
    Supports FIXED, REACTIVE, PREDICTIVE, and EMERGENCY signal management modes.
    """

    def __init__(self, signals: Dict[str, TrafficSignalState]):
        self.signals = signals

    def update_signals(
        self,
        dt_seconds: float,
        roads: Dict[str, RoadSegmentDefinition],
        predicted_queues: Optional[Dict[str, float]] = None,
    ):
        """
        Updates timers and triggers phase changes according to signal mode.
        """
        for signal_id, signal in self.signals.items():
            if signal.emergency_corridor_active:
                signal.current_phase = SignalPhase.EMERGENCY_OVERRIDE
                continue

            signal.phase_timer += dt_seconds

            if signal.mode == SignalMode.FIXED:
                self._update_fixed_signal(signal)
            elif signal.mode == SignalMode.REACTIVE:
                self._update_reactive_signal(signal, roads)
            elif signal.mode == SignalMode.PREDICTIVE:
                self._update_predictive_signal(signal, roads, predicted_queues)

    def _update_fixed_signal(self, signal: TrafficSignalState):
        """Standard fixed-cycle signal controller (45s Green / 5s Yellow)."""
        if signal.current_phase == SignalPhase.NORTH_SOUTH_GREEN:
            if signal.phase_timer >= signal.ns_green_duration:
                signal.current_phase = SignalPhase.NORTH_SOUTH_YELLOW
                signal.phase_timer = 0.0
        elif signal.current_phase == SignalPhase.NORTH_SOUTH_YELLOW:
            if signal.phase_timer >= signal.yellow_duration:
                signal.current_phase = SignalPhase.EAST_WEST_GREEN
                signal.phase_timer = 0.0
        elif signal.current_phase == SignalPhase.EAST_WEST_GREEN:
            if signal.phase_timer >= signal.ew_green_duration:
                signal.current_phase = SignalPhase.EAST_WEST_YELLOW
                signal.phase_timer = 0.0
        elif signal.current_phase == SignalPhase.EAST_WEST_YELLOW:
            if signal.phase_timer >= signal.yellow_duration:
                signal.current_phase = SignalPhase.NORTH_SOUTH_GREEN
                signal.phase_timer = 0.0

    def _update_reactive_signal(
        self, signal: TrafficSignalState, roads: Dict[str, RoadSegmentDefinition]
    ):
        """
        Reactive signal controller: dynamically adjusts green time based on
        current active queue/vehicle counts on incoming roads.
        """
        # Collect incoming North-South vs East-West vehicle counts for this junction
        j_id = signal.junction_id
        ns_vehicles = 0
        ew_vehicles = 0

        for road in roads.values():
            if road.target_junction_id == j_id:
                if "AC" in road.road_id or "CA" in road.road_id or "BD" in road.road_id or "DB" in road.road_id:
                    ns_vehicles += len(road.active_vehicle_ids)
                else:
                    ew_vehicles += len(road.active_vehicle_ids)

        total_vehicles = ns_vehicles + ew_vehicles
        if total_vehicles > 0:
            signal.ns_green_duration = max(20.0, min(80.0, 90.0 * (ns_vehicles / total_vehicles)))
            signal.ew_green_duration = max(20.0, min(80.0, 90.0 * (ew_vehicles / total_vehicles)))

        self._update_fixed_signal(signal)

    def _update_predictive_signal(
        self,
        signal: TrafficSignalState,
        roads: Dict[str, RoadSegmentDefinition],
        predicted_queues: Optional[Dict[str, float]] = None,
    ):
        """
        Predictive signal controller: uses forecasted future arrival queues
        to proactively adjust upcoming green time splits.
        """
        if predicted_queues:
            j_id = signal.junction_id
            ns_pred = sum(
                q for r_id, q in predicted_queues.items()
                if roads.get(r_id) and roads[r_id].target_junction_id == j_id
                and ("AC" in r_id or "CA" in r_id or "BD" in r_id or "DB" in r_id)
            )
            ew_pred = sum(
                q for r_id, q in predicted_queues.items()
                if roads.get(r_id) and roads[r_id].target_junction_id == j_id
                and ("AB" in r_id or "BA" in r_id or "CD" in r_id or "DC" in r_id)
            )

            total_pred = ns_pred + ew_pred
            if total_pred > 0:
                signal.ns_green_duration = max(15.0, min(85.0, 90.0 * (ns_pred / total_pred)))
                signal.ew_green_duration = max(15.0, min(85.0, 90.0 * (ew_pred / total_pred)))

        self._update_fixed_signal(signal)

    def set_emergency_corridor(self, route_junction_ids: List[str], active: bool):
        """
        Activates/deactivates emergency green wave corridor along a junction sequence.
        """
        for j_id in route_junction_ids:
            sig_id = f"SIG_{j_id}"
            if sig_id in self.signals:
                self.signals[sig_id].emergency_corridor_active = active
                if active:
                    self.signals[sig_id].current_phase = SignalPhase.EMERGENCY_OVERRIDE
                else:
                    self.signals[sig_id].current_phase = SignalPhase.NORTH_SOUTH_GREEN
                    self.signals[sig_id].phase_timer = 0.0

    def can_vehicle_pass(self, road_id: str, junction_id: str) -> bool:
        """
        Checks if a signal allows traffic flow from road_id into junction_id.
        """
        sig_id = f"SIG_{junction_id}"
        if sig_id not in self.signals:
            return True  # Unsignaled

        sig = self.signals[sig_id]

        if sig.current_phase == SignalPhase.EMERGENCY_OVERRIDE:
            return True  # Emergency vehicle priority override

        is_ns_road = "AC" in road_id or "CA" in road_id or "BD" in road_id or "DB" in road_id

        if is_ns_road:
            return sig.current_phase == SignalPhase.NORTH_SOUTH_GREEN
        else:
            return sig.current_phase == SignalPhase.EAST_WEST_GREEN
