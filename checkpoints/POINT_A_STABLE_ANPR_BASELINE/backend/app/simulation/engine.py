"""
ChronoEye Infinity - Phase 1: Master Traffic Simulation Engine
Coordinates multi-junction topology, vehicle micro-dynamics, traffic signals,
camera perception events, traffic state metric aggregation, incident injection,
and emergency green wave corridor dispatching.
"""

from typing import Dict, List, Optional, Tuple
from app.schemas.simulation import (
    VehicleState,
    VehicleType,
    VehicleStatus,
    SignalMode,
    TrafficSignalState,
    CameraObservation,
    TrafficStateMetrics,
    RoadSegmentDefinition,
    JunctionDefinition,
)
from app.simulation.topology import CityTopology
from app.simulation.signal_controller import SignalControllerManager
from app.simulation.vehicle_generator import VehicleGenerator
from app.simulation.camera_fov import CameraSensorManager


def _to_dict(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


class TrafficSimulationEngine:
    """
    Main step-based traffic simulation engine for ChronoEye Infinity.
    Runs microscopic vehicle dynamics, signal phase updates, camera sensing,
    metric calculations, dynamic incidents, and emergency corridors.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.topology = CityTopology()
        self.generator = VehicleGenerator(seed=seed)
        self.signal_manager = SignalControllerManager(self.topology.signals)
        self.camera_manager = CameraSensorManager(self.topology.cameras)

        self.vehicles: Dict[str, VehicleState] = {}
        self.arrived_vehicles: List[VehicleState] = []
        self.recent_observations: List[CameraObservation] = []
        self.sim_time: float = 0.0
        self.spawn_interval: float = 3.0  # Spawn vehicle every 3 seconds
        self.last_spawn_time: float = -3.0

    def reset(self, seed: Optional[int] = None):
        """Resets simulation state."""
        if seed is not None:
            self.seed = seed
        self.topology = CityTopology()
        self.generator = VehicleGenerator(seed=self.seed)
        self.signal_manager = SignalControllerManager(self.topology.signals)
        self.camera_manager = CameraSensorManager(self.topology.cameras)
        self.vehicles.clear()
        self.arrived_vehicles.clear()
        self.recent_observations.clear()
        self.sim_time = 0.0
        self.last_spawn_time = -3.0


    def set_signal_mode(self, mode: SignalMode):
        """Sets global signal control mode across all junctions."""
        for sig in self.topology.signals.values():
            sig.mode = mode

    def inject_incident(self, road_id: str, severity: float = 0.8):
        """Injects a road blockage / incident reducing segment capacity."""
        if road_id in self.topology.roads:
            road = self.topology.roads[road_id]
            road.is_blocked = True
            road.blockage_severity = min(1.0, max(0.0, severity))

    def resolve_incident(self, road_id: str):
        """Resolves an active road incident."""
        if road_id in self.topology.roads:
            road = self.topology.roads[road_id]
            road.is_blocked = False
            road.blockage_severity = 0.0

    def dispatch_emergency_vehicle(
        self, origin_road_id: str, destination_junction_id: str, route: List[str]
    ) -> VehicleState:
        """Dispatches an emergency vehicle and activates green corridor."""
        emg_vehicle = self.generator.create_vehicle(
            road_id=origin_road_id,
            destination_junction_id=destination_junction_id,
            route=route,
            vehicle_type=VehicleType.EMERGENCY,
            timestamp=self.sim_time,
        )
        self.vehicles[emg_vehicle.vehicle_id] = emg_vehicle
        self.topology.roads[origin_road_id].active_vehicle_ids.append(emg_vehicle.vehicle_id)

        # Trigger emergency signal corridor along route
        target_junctions = [
            self.topology.roads[r].target_junction_id
            for r in route if r in self.topology.roads
        ]
        self.signal_manager.set_emergency_corridor(target_junctions, active=True)
        return emg_vehicle

    def spawn_random_vehicle(self) -> Optional[VehicleState]:
        """Spawns a random vehicle onto an entry road segment."""
        # Pick random origin road and valid route
        routes = [
            ["R_AB", "R_BD"],  # J_A -> J_B -> J_D
            ["R_AC", "R_CD"],  # J_A -> J_C -> J_D
            ["R_BA", "R_AC"],  # J_B -> J_A -> J_C
            ["R_CD", "R_DB"],  # J_C -> J_D -> J_B
            ["R_DB", "R_BA"],  # J_D -> J_B -> J_A
        ]
        chosen_route = self.generator.rng.choice(routes)
        origin_road_id = chosen_route[0]
        dest_j_id = self.topology.roads[chosen_route[-1]].target_junction_id

        # Determine vehicle type
        v_type_roll = self.generator.rng.random()
        v_type = VehicleType.CAR
        if v_type_roll < 0.70:
            v_type = VehicleType.CAR
        elif v_type_roll < 0.85:
            v_type = VehicleType.MOTORCYCLE
        elif v_type_roll < 0.95:
            v_type = VehicleType.BUS
        else:
            v_type = VehicleType.TRUCK

        vehicle = self.generator.create_vehicle(
            road_id=origin_road_id,
            destination_junction_id=dest_j_id,
            route=chosen_route,
            vehicle_type=v_type,
            timestamp=self.sim_time,
        )
        if len(self.vehicles) == 0:
            vehicle.distance_on_road = 55.0
        self.vehicles[vehicle.vehicle_id] = vehicle
        self.topology.roads[origin_road_id].active_vehicle_ids.append(vehicle.vehicle_id)
        return vehicle


    def step(self, dt_seconds: float = 1.0) -> Dict:
        """
        Advances the simulation state by dt_seconds.
        """
        self.sim_time += dt_seconds

        # 1. Spawn new vehicles periodically
        if (self.sim_time - self.last_spawn_time) >= self.spawn_interval:
            self.spawn_random_vehicle()
            self.last_spawn_time = self.sim_time

        # 2. Update Traffic Signals
        self.signal_manager.update_signals(dt_seconds, self.topology.roads)

        # 3. Update Vehicle Movements
        arrived_ids = []
        for v_id, vehicle in list(self.vehicles.items()):
            if vehicle.status == VehicleStatus.ARRIVED:
                arrived_ids.append(v_id)
                continue

            current_road = self.topology.roads[vehicle.current_road_id]
            target_j_id = current_road.target_junction_id

            # Check signal pass permission
            can_pass = self.signal_manager.can_vehicle_pass(current_road.road_id, target_j_id)

            # Find lead vehicle distance on same road
            lead_dist = None
            vehicles_on_same_road = [
                v for v in self.vehicles.values()
                if v.current_road_id == current_road.road_id and v.vehicle_id != v_id
                and v.distance_on_road > vehicle.distance_on_road
            ]
            if vehicles_on_same_road:
                lead_dist = min(v.distance_on_road for v in vehicles_on_same_road)

            # Execute kinematic step
            old_road_id = vehicle.current_road_id
            has_arrived, entered_next = self.generator.update_vehicle_position(
                vehicle, dt_seconds, current_road, can_pass, lead_dist
            )

            vehicle.last_updated_timestamp = self.sim_time

            if entered_next:
                # Remove from old road list, add to new road list
                if v_id in self.topology.roads[old_road_id].active_vehicle_ids:
                    self.topology.roads[old_road_id].active_vehicle_ids.remove(v_id)
                new_road_id = vehicle.current_road_id
                self.topology.roads[new_road_id].active_vehicle_ids.append(v_id)

            if has_arrived:
                arrived_ids.append(v_id)
                if v_id in self.topology.roads[old_road_id].active_vehicle_ids:
                    self.topology.roads[old_road_id].active_vehicle_ids.remove(v_id)

        # Remove arrived vehicles from active dict
        for a_id in arrived_ids:
            if a_id in self.vehicles:
                self.arrived_vehicles.append(self.vehicles.pop(a_id))

        # 4. Capture Camera FOV Sensor Observations
        self.recent_observations = self.camera_manager.capture_observations(
            self.vehicles, self.sim_time
        )

        # 5. Compute Aggregated Traffic State Metrics
        metrics = self.compute_traffic_metrics()

        return self.get_snapshot(metrics)

    def compute_traffic_metrics(self) -> List[TrafficStateMetrics]:
        """Calculates congestion scores, queue lengths, and density for each road."""
        metrics_list: List[TrafficStateMetrics] = []
        for r_id, road in self.topology.roads.items():
            active_vehs = [self.vehicles[v_id] for v_id in road.active_vehicle_ids if v_id in self.vehicles]
            v_count = len(active_vehs)

            avg_speed = (
                sum(v.speed_kmh for v in active_vehs) / v_count if v_count > 0 else road.speed_limit_kmh
            )
            queued_vehs = sum(1 for v in active_vehs if v.status == VehicleStatus.QUEUED or v.speed_kmh < 5.0)
            queue_len = queued_vehs * 6.5  # 6.5m average vehicle spacing in queue

            occupancy = min(100.0, (v_count / road.capacity) * 100.0)
            
            # Congestion Score calculation: weighted combination of density & speed drop
            speed_ratio = avg_speed / road.speed_limit_kmh
            density_ratio = v_count / road.capacity
            congestion = min(1.0, max(0.0, (1.0 - speed_ratio) * 0.6 + density_ratio * 0.4))
            if road.is_blocked:
                congestion = min(1.0, congestion + road.blockage_severity * 0.5)

            m = TrafficStateMetrics(
                timestamp=self.sim_time,
                junction_id=road.target_junction_id,
                road_id=r_id,
                vehicle_count=v_count,
                average_speed_kmh=round(avg_speed, 1),
                queue_length_meters=round(queue_len, 1),
                occupancy_pct=round(occupancy, 1),
                congestion_score=round(congestion, 2),
            )
            metrics_list.append(m)

        return metrics_list

    def get_snapshot(self, metrics: Optional[List[TrafficStateMetrics]] = None) -> Dict:
        """Exports current simulation snapshot as a dictionary."""
        if metrics is None:
            metrics = self.compute_traffic_metrics()

        return {
            "simulation_time": round(self.sim_time, 2),
            "seed": self.seed,
            "active_vehicle_count": len(self.vehicles),
            "arrived_vehicle_count": len(self.arrived_vehicles),
            "junctions": [_to_dict(j) for j in self.topology.junctions.values()],
            "roads": [_to_dict(r) for r in self.topology.roads.values()],
            "signals": [_to_dict(s) for s in self.topology.signals.values()],
            "cameras": [_to_dict(c) for c in self.topology.cameras.values()],
            "vehicles": [_to_dict(v) for v in self.vehicles.values()],
            "recent_camera_observations": [_to_dict(o) for o in self.recent_observations],
            "traffic_metrics": [_to_dict(m) for m in metrics],
        }
