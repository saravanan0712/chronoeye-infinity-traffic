"""
ChronoEye Infinity - Phase 1: Urban Topology Builder
Constructs the multi-junction spatio-temporal road network graph.
"""

from typing import Dict, List, Tuple
from app.schemas.simulation import (
    JunctionDefinition,
    RoadSegmentDefinition,
    TrafficSignalState,
    CameraDefinition,
    SignalMode,
    SignalPhase,
)


class CityTopology:
    """
    Defines a multi-junction city road network with connected roads,
    traffic signals, and CCTV camera sensor locations.
    """

    def __init__(self):
        self.junctions: Dict[str, JunctionDefinition] = {}
        self.roads: Dict[str, RoadSegmentDefinition] = {}
        self.signals: Dict[str, TrafficSignalState] = {}
        self.cameras: Dict[str, CameraDefinition] = {}
        self._build_default_network()

    def _build_default_network(self):
        """
        Builds a 4-junction urban arterial network:
        
            (Junction_A) <== Road_A_B ==> (Junction_B)
                 ||                            ||
             Road_A_C                      Road_B_D
                 ||                            ||
            (Junction_C) <== Road_C_D ==> (Junction_D)
        """
        # 1. Define Junctions
        j_a = JunctionDefinition(
            junction_id="J_A",
            name="Northwest Intersection",
            position_x=100.0,
            position_y=100.0,
            signal_id="SIG_J_A",
        )
        j_b = JunctionDefinition(
            junction_id="J_B",
            name="Northeast Intersection",
            position_x=600.0,
            position_y=100.0,
            signal_id="SIG_J_B",
        )
        j_c = JunctionDefinition(
            junction_id="J_C",
            name="Southwest Intersection",
            position_x=100.0,
            position_y=600.0,
            signal_id="SIG_J_C",
        )
        j_d = JunctionDefinition(
            junction_id="J_D",
            name="Southeast Intersection",
            position_x=600.0,
            position_y=600.0,
            signal_id="SIG_J_D",
        )

        for j in [j_a, j_b, j_c, j_d]:
            self.junctions[j.junction_id] = j

        # 2. Define Bi-directional Road Segments
        road_definitions = [
            ("R_AB", "Road AB (Eastbound)", "J_A", "J_B", 500.0, 60.0),
            ("R_BA", "Road BA (Westbound)", "J_B", "J_A", 500.0, 60.0),
            ("R_AC", "Road AC (Southbound)", "J_A", "J_C", 500.0, 50.0),
            ("R_CA", "Road CA (Northbound)", "J_C", "J_A", 500.0, 50.0),
            ("R_BD", "Road BD (Southbound)", "J_B", "J_D", 500.0, 50.0),
            ("R_DB", "Road DB (Northbound)", "J_D", "J_B", 500.0, 50.0),
            ("R_CD", "Road CD (Eastbound)", "J_C", "J_D", 500.0, 60.0),
            ("R_DC", "Road DC (Westbound)", "J_D", "J_C", 500.0, 60.0),
        ]

        for r_id, r_name, src, tgt, length, speed in road_definitions:
            road = RoadSegmentDefinition(
                road_id=r_id,
                name=r_name,
                source_junction_id=src,
                target_junction_id=tgt,
                length_meters=length,
                speed_limit_kmh=speed,
                num_lanes=2,
                capacity=60,
            )
            self.roads[r_id] = road
            self.junctions[src].connected_road_ids.append(r_id)

        # 3. Define Traffic Signals for Each Junction
        for j_id in self.junctions:
            sig = TrafficSignalState(
                signal_id=f"SIG_{j_id}",
                junction_id=j_id,
                mode=SignalMode.FIXED,
                current_phase=SignalPhase.NORTH_SOUTH_GREEN,
                phase_timer=0.0,
                ns_green_duration=45.0,
                ew_green_duration=45.0,
                yellow_duration=5.0,
            )
            self.signals[sig.signal_id] = sig

        # 4. Define CCTV Camera Sensor Placements
        cam_definitions = [
            ("CAM_A_EAST", "Camera A-East (R_AB)", "J_A", "R_AB", 150.0, 100.0),
            ("CAM_B_WEST", "Camera B-West (R_BA)", "J_B", "R_BA", 550.0, 100.0),
            ("CAM_C_NORTH", "Camera C-North (R_CA)", "J_C", "R_CA", 100.0, 550.0),
            ("CAM_D_NORTH", "Camera D-North (R_DB)", "J_D", "R_DB", 600.0, 550.0),
        ]

        for c_id, c_name, j_id, r_id, px, py in cam_definitions:
            cam = CameraDefinition(
                camera_id=c_id,
                name=c_name,
                junction_id=j_id,
                road_id=r_id,
                position_x=px,
                position_y=py,
                fov_range_meters=60.0,
            )
            self.cameras[c_id] = cam

    def get_outgoing_roads(self, junction_id: str) -> List[RoadSegmentDefinition]:
        return [r for r in self.roads.values() if r.source_junction_id == junction_id]

    def get_incoming_roads(self, junction_id: str) -> List[RoadSegmentDefinition]:
        return [r for r in self.roads.values() if r.target_junction_id == junction_id]
