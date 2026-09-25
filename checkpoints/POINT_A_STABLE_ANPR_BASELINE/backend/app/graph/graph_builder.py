"""
ChronoEye Infinity - Phase 6: Spatio-Temporal Traffic Graph Builder
Constructs and maintains heterogeneous NetworkX graph containing Vehicles, Cameras, Roads, Lanes, Junctions, and Signals.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import networkx as nx
from app.graph.graph_schema import (
    NodeType,
    EdgeType,
    VehicleNode,
    CameraNodeGraph,
    RoadNode,
    LaneNode,
    JunctionNode,
    SignalNode,
    EdgeAttributes,
)


class SpatioTemporalGraphBuilder:
    """
    Heterogeneous Spatio-Temporal Traffic Graph Manager powered by NetworkX.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.initialize_city_topology()

    def initialize_city_topology(self):
        """
        Initializes base urban traffic topology matching Phase 1 & 5 camera/junction networks.
        """
        # 1. Junctions
        junctions = [
            JunctionNode(junction_id="JUNC_A", latitude=12.9716, longitude=77.5946, incoming_roads=["ROAD_R_BA"], outgoing_roads=["ROAD_R_AB"]),
            JunctionNode(junction_id="JUNC_B", latitude=12.9716, longitude=77.5973, incoming_roads=["ROAD_R_AB"], outgoing_roads=["ROAD_R_BA"]),
            JunctionNode(junction_id="JUNC_C", latitude=12.9738, longitude=77.5946, incoming_roads=["ROAD_R_DC"], outgoing_roads=["ROAD_R_CD"]),
            JunctionNode(junction_id="JUNC_D", latitude=12.9738, longitude=77.5973, incoming_roads=["ROAD_R_CD"], outgoing_roads=["ROAD_R_DC"]),
        ]
        for j in junctions:
            self.add_junction_node(j)

        # 2. Signals
        signals = [
            SignalNode(signal_id="SIG_A", junction_id="JUNC_A", active_phase="GREEN_NS", mode="FIXED"),
            SignalNode(signal_id="SIG_B", junction_id="JUNC_B", active_phase="GREEN_EW", mode="REACTIVE"),
            SignalNode(signal_id="SIG_C", junction_id="JUNC_C", active_phase="GREEN_NS", mode="FIXED"),
            SignalNode(signal_id="SIG_D", junction_id="JUNC_D", active_phase="GREEN_EW", mode="PREDICTIVE"),
        ]
        for s in signals:
            self.add_signal_node(s)

        # 3. Roads & Lanes
        roads = [
            RoadNode(road_id="ROAD_R_AB", road_name="R_AB", length_meters=300.0, speed_limit_kmh=60.0, capacity=100),
            RoadNode(road_id="ROAD_R_BA", road_name="R_BA", length_meters=300.0, speed_limit_kmh=60.0, capacity=100),
            RoadNode(road_id="ROAD_R_CD", road_name="R_CD", length_meters=250.0, speed_limit_kmh=50.0, capacity=80),
            RoadNode(road_id="ROAD_R_DC", road_name="R_DC", length_meters=250.0, speed_limit_kmh=50.0, capacity=80),
        ]
        for r in roads:
            self.add_road_node(r)

        lanes = [
            LaneNode(lane_id="LANE_R_AB_01", road_id="ROAD_R_AB", direction="EAST"),
            LaneNode(lane_id="LANE_R_BA_01", road_id="ROAD_R_BA", direction="WEST"),
            LaneNode(lane_id="LANE_R_CD_01", road_id="ROAD_R_CD", direction="NORTH"),
            LaneNode(lane_id="LANE_R_DC_01", road_id="ROAD_R_DC", direction="SOUTH"),
        ]
        for l in lanes:
            self.add_lane_node(l)

        # 4. Cameras
        cameras = [
            CameraNodeGraph(camera_id="CAM_A_EAST", latitude=12.9716, longitude=77.5946, road_name="R_AB", direction="EAST"),
            CameraNodeGraph(camera_id="CAM_B_WEST", latitude=12.9716, longitude=77.5973, road_name="R_BA", direction="WEST"),
            CameraNodeGraph(camera_id="CAM_C_NORTH", latitude=12.9738, longitude=77.5946, road_name="R_CD", direction="NORTH"),
            CameraNodeGraph(camera_id="CAM_D_NORTH", latitude=12.9738, longitude=77.5973, road_name="R_DC", direction="NORTH"),
        ]
        for c in cameras:
            self.add_camera_node(c)

        # 5. Connect Topology Edges
        self.add_typed_edge("LANE_R_AB_01", "ROAD_R_AB", EdgeType.LANE_BELONGS_TO_ROAD)
        self.add_typed_edge("LANE_R_BA_01", "ROAD_R_BA", EdgeType.LANE_BELONGS_TO_ROAD)
        self.add_typed_edge("LANE_R_CD_01", "ROAD_R_CD", EdgeType.LANE_BELONGS_TO_ROAD)
        self.add_typed_edge("LANE_R_DC_01", "ROAD_R_DC", EdgeType.LANE_BELONGS_TO_ROAD)

        self.add_typed_edge("ROAD_R_AB", "JUNC_B", EdgeType.ROAD_CONNECTS_TO_JUNCTION)
        self.add_typed_edge("ROAD_R_BA", "JUNC_A", EdgeType.ROAD_CONNECTS_TO_JUNCTION)
        self.add_typed_edge("ROAD_R_CD", "JUNC_D", EdgeType.ROAD_CONNECTS_TO_JUNCTION)
        self.add_typed_edge("ROAD_R_DC", "JUNC_C", EdgeType.ROAD_CONNECTS_TO_JUNCTION)

        self.add_typed_edge("JUNC_A", "SIG_A", EdgeType.JUNCTION_CONTROLLED_BY_SIGNAL)
        self.add_typed_edge("JUNC_B", "SIG_B", EdgeType.JUNCTION_CONTROLLED_BY_SIGNAL)
        self.add_typed_edge("JUNC_C", "SIG_C", EdgeType.JUNCTION_CONTROLLED_BY_SIGNAL)
        self.add_typed_edge("JUNC_D", "SIG_D", EdgeType.JUNCTION_CONTROLLED_BY_SIGNAL)

        self.add_typed_edge("JUNC_A", "JUNC_B", EdgeType.JUNCTION_CONNECTS_TO_JUNCTION, distance=300.0)
        self.add_typed_edge("JUNC_B", "JUNC_D", EdgeType.JUNCTION_CONNECTS_TO_JUNCTION, distance=250.0)

    def add_vehicle_node(self, node_or_id: Union[VehicleNode, str]):
        if isinstance(node_or_id, VehicleNode):
            node = node_or_id
        else:
            node = VehicleNode(vehicle_id=str(node_or_id))
        self.graph.add_node(
            node.vehicle_id,
            node_type=NodeType.VEHICLE.value,
            data=node.model_dump(),
        )

    def add_camera_node(
        self,
        node_or_id: Union[CameraNodeGraph, str],
        latitude: float = 0.0,
        longitude: float = 0.0,
        road_name: str = "",
        direction: str = "EAST",
    ):
        if isinstance(node_or_id, CameraNodeGraph):
            node = node_or_id
        else:
            node = CameraNodeGraph(
                camera_id=str(node_or_id),
                latitude=latitude,
                longitude=longitude,
                road_name=road_name,
                direction=direction,
            )
        self.graph.add_node(
            node.camera_id,
            node_type=NodeType.CAMERA.value,
            data=node.model_dump(),
        )

    def add_road_node(
        self,
        node_or_id: Union[RoadNode, str],
        road_name: str = "",
        length_meters: float = 500.0,
        speed_limit_kmh: float = 60.0,
        capacity: int = 100,
    ):
        if isinstance(node_or_id, RoadNode):
            node = node_or_id
        else:
            node = RoadNode(
                road_id=str(node_or_id),
                road_name=road_name or str(node_or_id),
                length_meters=length_meters,
                speed_limit_kmh=speed_limit_kmh,
                capacity=capacity,
            )
        self.graph.add_node(
            node.road_id,
            node_type=NodeType.ROAD.value,
            data=node.model_dump(),
        )

    def add_lane_node(self, node_or_id: Union[LaneNode, str], road_id: str = "", direction: str = "FORWARD"):
        if isinstance(node_or_id, LaneNode):
            node = node_or_id
        else:
            node = LaneNode(lane_id=str(node_or_id), road_id=road_id, direction=direction)
        self.graph.add_node(
            node.lane_id,
            node_type=NodeType.LANE.value,
            data=node.model_dump(),
        )

    def add_junction_node(
        self,
        node_or_id: Union[JunctionNode, str],
        latitude: float = 0.0,
        longitude: float = 0.0,
        incoming_roads: Optional[List[str]] = None,
        outgoing_roads: Optional[List[str]] = None,
    ):
        if isinstance(node_or_id, JunctionNode):
            node = node_or_id
        else:
            node = JunctionNode(
                junction_id=str(node_or_id),
                latitude=latitude,
                longitude=longitude,
                incoming_roads=incoming_roads or [],
                outgoing_roads=outgoing_roads or [],
            )
        self.graph.add_node(
            node.junction_id,
            node_type=NodeType.JUNCTION.value,
            data=node.model_dump(),
        )

    def add_signal_node(
        self,
        node_or_id: Union[SignalNode, str],
        junction_id: str = "",
        active_phase: str = "GREEN_NS",
        mode: str = "FIXED",
    ):
        if isinstance(node_or_id, SignalNode):
            node = node_or_id
        else:
            node = SignalNode(
                signal_id=str(node_or_id),
                junction_id=junction_id,
                active_phase=active_phase,
                mode=mode,
            )
        self.graph.add_node(
            node.signal_id,
            node_type=NodeType.SIGNAL.value,
            data=node.model_dump(),
        )

    def add_road_edge(
        self,
        road_id: str,
        from_junction: str,
        to_junction: str,
        length_meters: float = 500.0,
        travel_time: float = 30.0,
        speed_limit_kmh: float = 60.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        edge_meta = metadata or {}
        edge_meta["road_id"] = road_id
        edge_meta["speed_limit_kmh"] = speed_limit_kmh

        self.add_typed_edge(
            source=from_junction,
            target=to_junction,
            edge_type=EdgeType.JUNCTION_CONNECTS_TO_JUNCTION,
            distance=length_meters,
            travel_time=travel_time,
            metadata=edge_meta,
        )

    def add_typed_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        timestamp: float = 0.0,
        distance: float = 0.0,
        travel_time: float = 0.0,
        speed: float = 0.0,
        direction: str = "UNKNOWN",
        camera_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        edge_meta = metadata or {}
        attrs = EdgeAttributes(
            edge_type=edge_type,
            timestamp=timestamp,
            distance=distance,
            travel_time=travel_time,
            speed=speed,
            direction=direction,
            camera_id=camera_id,
            source=source,
            target=target,
            metadata=edge_meta,
        )
        road_id = edge_meta.get("road_id", f"ROAD_{source}_{target}")
        edge_key = 0 if timestamp == 0.0 else f"{edge_type.value}_{timestamp}"
        self.graph.add_edge(
            source,
            target,
            key=edge_key,
            edge_type=edge_type.value,
            timestamp=timestamp,
            distance=distance,
            length_meters=distance if distance > 0 else 500.0,
            travel_time=travel_time if travel_time > 0 else 30.0,
            road_id=road_id,
            data=attrs.model_dump(),
        )

    def get_nodes_by_type(self, node_type: NodeType) -> List[str]:
        return [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == node_type.value
        ]

