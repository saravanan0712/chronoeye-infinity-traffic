"""
ChronoEye Infinity - Phase 1: Vehicle Agent & Kinematic Movement Generator
Generates vehicle agents with realistic attributes, routes, and microscopic
kinematic updates (car-following model, acceleration, signal queuing).
"""

import random
from typing import Dict, List, Optional, Tuple
from app.schemas.simulation import (
    VehicleState,
    VehicleType,
    VehicleStatus,
    RoadSegmentDefinition,
    JunctionDefinition,
)


class VehicleGenerator:
    """
    Manages generation, movement physics, and route updates of vehicle agents.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.next_tracking_id = 100
        self.colors = ["Red", "Blue", "Black", "White", "Silver", "Grey", "Yellow"]
        self.plate_state_codes = ["TN09", "KA01", "MH12", "DL03", "AP09", "KL07"]

    def set_seed(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_license_plate(self) -> str:
        prefix = self.rng.choice(self.plate_state_codes)
        letters = "".join(self.rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2))
        digits = "".join(self.rng.choices("0123456789", k=4))
        return f"{prefix}{letters}{digits}"

    def create_vehicle(
        self,
        road_id: str,
        destination_junction_id: str,
        route: List[str],
        vehicle_type: VehicleType = VehicleType.CAR,
        timestamp: float = 0.0,
    ) -> VehicleState:
        t_id = self.next_tracking_id
        self.next_tracking_id += 1

        plate = self.generate_license_plate() if vehicle_type != VehicleType.EMERGENCY else "EMG9999"
        color = self.rng.choice(self.colors) if vehicle_type != VehicleType.EMERGENCY else "Red-White"
        desired_speed = 60.0 if vehicle_type == VehicleType.CAR else (
            70.0 if vehicle_type == VehicleType.EMERGENCY else 45.0
        )

        return VehicleState(
            vehicle_id=f"V_{t_id}_{plate}",
            tracking_id=t_id,
            vehicle_type=vehicle_type,
            plate_number=plate,
            color=color,
            speed_kmh=0.0,
            desired_speed_kmh=desired_speed,
            position_x=0.0,
            position_y=0.0,
            current_road_id=road_id,
            lane_id=self.rng.randint(0, 1),
            distance_on_road=0.0,
            route=route,
            route_index=0,
            destination_junction_id=destination_junction_id,
            status=VehicleStatus.MOVING,
            first_seen_timestamp=timestamp,
            last_updated_timestamp=timestamp,
        )

    def update_vehicle_position(
        self,
        vehicle: VehicleState,
        dt_seconds: float,
        road: RoadSegmentDefinition,
        can_pass_signal: bool,
        lead_vehicle_distance: Optional[float] = None,
    ) -> Tuple[bool, bool]:
        """
        Updates kinematic position using a car-following acceleration model.
        Returns: (has_arrived_at_destination, has_entered_next_road)
        """
        if vehicle.status == VehicleStatus.ARRIVED:
            return True, False

        # Convert speed to m/s
        v_m_s = vehicle.speed_kmh / 3.6
        v_desired_m_s = vehicle.desired_speed_kmh / 3.6
        max_speed_road = min(v_desired_m_s, (road.speed_limit_kmh / 3.6) * (1.0 - road.blockage_severity))

        target_speed = max_speed_road

        # 1. Car-following constraint (safe distance to lead vehicle)
        if lead_vehicle_distance is not None:
            gap = lead_vehicle_distance - vehicle.distance_on_road - 5.0  # 5m vehicle length
            if gap < 15.0:
                target_speed = min(target_speed, max(0.0, gap / 2.0))

        # 2. Signal constraint at end of road
        dist_to_signal = road.length_meters - vehicle.distance_on_road
        if dist_to_signal <= 35.0 and not can_pass_signal:
            target_speed = min(target_speed, max(0.0, dist_to_signal / 3.0))
            if dist_to_signal <= 5.0 and v_m_s < 0.5:
                vehicle.status = VehicleStatus.QUEUED
                vehicle.speed_kmh = 0.0
                return False, False

        # 3. Acceleration / Deceleration
        if v_m_s < target_speed:
            v_m_s = min(target_speed, v_m_s + 2.5 * dt_seconds)  # 2.5 m/s^2 acceleration
        elif v_m_s > target_speed:
            v_m_s = max(target_speed, v_m_s - 4.0 * dt_seconds)  # 4.0 m/s^2 braking

        vehicle.speed_kmh = v_m_s * 3.6
        if vehicle.speed_kmh > 1.0:
            vehicle.status = VehicleStatus.MOVING

        # Advance distance
        vehicle.distance_on_road += v_m_s * dt_seconds

        # Check if reached end of segment
        if vehicle.distance_on_road >= road.length_meters:
            if can_pass_signal:
                vehicle.route_index += 1
                if vehicle.route_index < len(vehicle.route):
                    # Transition to next road in route
                    vehicle.current_road_id = vehicle.route[vehicle.route_index]
                    vehicle.distance_on_road = 0.0
                    return False, True  # Entered next road
                else:
                    vehicle.status = VehicleStatus.ARRIVED
                    return True, False  # Arrived at destination

        return False, False
