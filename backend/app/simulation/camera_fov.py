"""
ChronoEye Infinity - Phase 1: Camera FOV Sensor Module
Simulates CCTV camera field-of-view observations of active vehicles passing through
camera coverage zones, generating synthetic perception metadata.
"""

import uuid
from typing import Dict, List
from app.schemas.simulation import (
    CameraDefinition,
    CameraObservation,
    VehicleState,
    BoundingBox,
)


class CameraSensorManager:
    """
    Monitors active vehicle spatial positions against registered camera FOV ranges
    and emits synthetic vision/ANPR observations.
    """

    def __init__(self, cameras: Dict[str, CameraDefinition]):
        self.cameras = cameras
        self.frame_counter = 0

    def capture_observations(
        self,
        vehicles: Dict[str, VehicleState],
        current_time: float,
    ) -> List[CameraObservation]:
        self.frame_counter += 1
        observations: List[CameraObservation] = []

        for cam_id, cam in self.cameras.items():
            for v_id, vehicle in vehicles.items():
                if vehicle.current_road_id == cam.road_id:
                    # Vehicle is on the camera's road segment. Check FOV proximity.
                    # Simulating FOV trigger zone near camera location
                    if 50.0 <= vehicle.distance_on_road <= (50.0 + cam.fov_range_meters):
                        # Generate synthetic bounding box
                        bbox = BoundingBox(
                            x=round(100.0 + vehicle.distance_on_road * 1.2, 1),
                            y=round(80.0 + vehicle.lane_id * 30.0, 1),
                            width=64.0,
                            height=48.0,
                        )

                        obs = CameraObservation(
                            observation_id=f"OBS_{uuid.uuid4().hex[:8]}",
                            camera_id=cam.camera_id,
                            junction_id=cam.junction_id,
                            road_id=cam.road_id,
                            timestamp=round(current_time, 2),
                            frame_id=self.frame_counter,
                            vehicle_id=vehicle.vehicle_id,
                            tracking_id=vehicle.tracking_id,
                            vehicle_type=vehicle.vehicle_type,
                            plate_number=vehicle.plate_number,
                            speed_kmh=round(vehicle.speed_kmh, 1),
                            lane_id=vehicle.lane_id,
                            position_x=round(cam.position_x + vehicle.distance_on_road, 1),
                            position_y=round(cam.position_y + vehicle.lane_id * 5.0, 1),
                            bbox=bbox,
                            detection_confidence=0.96,
                            ocr_confidence=0.94,
                            source="SIMULATED_CCTV",
                        )
                        observations.append(obs)

        return observations
