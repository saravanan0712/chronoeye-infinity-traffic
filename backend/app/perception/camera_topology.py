"""
ChronoEye Infinity - Phase 5: Camera Topology & Spatio-Temporal Constraint Model
Maintains urban CCTV camera spatial graph, road distances, directional connectivity,
and physical speed/travel-time feasibility bounds.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field


class CameraNode(BaseModel):
    """
    Spatiotemporal metadata for a single CCTV camera sensor.
    """
    camera_id: str
    latitude: float
    longitude: float
    road_name: str
    direction: str  # "EAST", "WEST", "NORTH", "SOUTH"
    connected_cameras: Dict[str, float] = Field(default_factory=dict)  # target_cam_id -> distance_meters


class CityCameraTopology:
    """
    Spatio-temporal urban camera graph topology manager.
    Enforces physical travel time bounds: Δt_min = distance / v_max.
    """

    def __init__(self):
        self.cameras: Dict[str, CameraNode] = {}
        self._initialize_default_topology()

    def _initialize_default_topology(self):
        """Initializes 4-camera arterial network matching Phase 1 simulation graph."""
        cam_a = CameraNode(
            camera_id="CAM_A_EAST",
            latitude=12.9716,
            longitude=77.5946,
            road_name="R_AB",
            direction="EAST",
            connected_cameras={"CAM_B_WEST": 300.0, "CAM_C_NORTH": 250.0},
        )
        cam_b = CameraNode(
            camera_id="CAM_B_WEST",
            latitude=12.9716,
            longitude=77.5973,
            road_name="R_BA",
            direction="WEST",
            connected_cameras={"CAM_A_EAST": 300.0, "CAM_D_NORTH": 250.0},
        )
        cam_c = CameraNode(
            camera_id="CAM_C_NORTH",
            latitude=12.9738,
            longitude=77.5946,
            road_name="R_CD",
            direction="NORTH",
            connected_cameras={"CAM_A_EAST": 250.0, "CAM_D_NORTH": 300.0},
        )
        cam_d = CameraNode(
            camera_id="CAM_D_NORTH",
            latitude=12.9738,
            longitude=77.5973,
            road_name="R_DC",
            direction="NORTH",
            connected_cameras={"CAM_B_WEST": 250.0, "CAM_C_NORTH": 300.0},
        )

        for cam in [cam_a, cam_b, cam_c, cam_d]:
            self.cameras[cam.camera_id] = cam

    def add_camera(self, camera_node: CameraNode):
        self.cameras[camera_node.camera_id] = camera_node

    def is_directly_connected(self, cam_a_id: str, cam_b_id: str) -> bool:
        """
        Checks whether two cameras are directly connected adjacent neighbors in the topology.
        Returns True if same camera or if cam_b is in cam_a's connected_cameras list.
        """
        if cam_a_id == cam_b_id:
            return True
        if cam_a_id in self.cameras:
            return cam_b_id in self.cameras[cam_a_id].connected_cameras
        return False

    def get_distance_meters(self, cam_a_id: str, cam_b_id: str) -> float:
        """
        Retrieves road network distance between two cameras in meters.
        Falls back to Haversine geographic distance if direct edge is absent.
        """
        if cam_a_id == cam_b_id:
            return 0.0

        if cam_a_id in self.cameras and cam_b_id in self.cameras[cam_a_id].connected_cameras:
            return self.cameras[cam_a_id].connected_cameras[cam_b_id]

        if cam_a_id in self.cameras and cam_b_id in self.cameras:
            # Haversine distance estimate
            node_a = self.cameras[cam_a_id]
            node_b = self.cameras[cam_b_id]
            return self._haversine_distance(
                node_a.latitude, node_a.longitude, node_b.latitude, node_b.longitude
            )

        return 500.0  # Default fallback distance meters

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371000.0  # Earth radius meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def compute_min_travel_time(self, cam_a_id: str, cam_b_id: str, max_speed_kmh: float = 120.0) -> float:
        """
        Computes minimum physically possible travel time (seconds): distance / v_max.
        """
        dist_m = self.get_distance_meters(cam_a_id, cam_b_id)
        max_v_m_s = (max_speed_kmh * 1000.0) / 3600.0
        return dist_m / max_v_m_s

    def is_temporally_feasible(
        self,
        cam_a_id: str,
        cam_b_id: str,
        delta_t_seconds: float,
        max_speed_kmh: float = 120.0,
        min_speed_kmh: float = 5.0,
        max_time_gap_seconds: float = 3600.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Enforces physical travel bounds.
        Rejects matches where travel time Δt is shorter than minimum possible travel time,
        or longer than maximum allowable time gap.
        """
        if delta_t_seconds < 0:
            return False, "NEGATIVE_TIME_DELTA"

        if cam_a_id == cam_b_id:
            return True, None

        min_travel_t = self.compute_min_travel_time(cam_a_id, cam_b_id, max_speed_kmh)
        if delta_t_seconds < min_travel_t:
            return False, f"PHYSICALLY_IMPOSSIBLE_SPEED (Δt={delta_t_seconds:.1f}s < min_t={min_travel_t:.1f}s)"

        if delta_t_seconds > max_time_gap_seconds:
            return False, f"TIME_GAP_EXCEEDED (Δt={delta_t_seconds:.1f}s > max={max_time_gap_seconds:.1f}s)"

        return True, None
