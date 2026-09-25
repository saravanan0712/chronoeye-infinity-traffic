"""
ChronoEye Infinity - Phase 6: Temporal Graph Update Engine
Consumes Phase 5 vehicle identities & continuous journeys, applies incremental graph updates,
calculates transition features, and captures historical graph snapshots.
"""

from typing import Dict, List, Optional, Tuple, Any
from app.schemas.tracking import TrackState
from app.schemas.plate import VehicleIdentityEvidence
from app.schemas.reid import VehicleJourney, JourneySegment
from app.graph.graph_schema import (
    NodeType,
    EdgeType,
    VehicleNode,
    CameraNodeGraph,
    RoadNode,
    EdgeAttributes,
)
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.graph.graph_metrics import TrafficMetricsEngine


class TemporalTrafficGraphEngine:
    """
    Incremental Spatio-Temporal Traffic Graph Update Engine with Snapshot Retention.
    """

    def __init__(self, builder: Optional[SpatioTemporalGraphBuilder] = None):
        self.builder = builder or SpatioTemporalGraphBuilder()
        self.snapshots: Dict[float, Dict[str, Any]] = {}  # timestamp -> graph_state

    def update_vehicle_observation(
        self,
        evidence: VehicleIdentityEvidence,
        track: TrackState,
        timestamp: float,
        road_id: str = "ROAD_R_AB",
        lane_id: str = "LANE_R_AB_01",
    ):
        """
        Updates graph with single vehicle observation at timestamp.
        """
        v_id = track.track_id  # Uses camera-local or global ID
        dir_str = track.direction.value if track.direction else "UNKNOWN"
        plate_str = evidence.associated_plate.best_plate_number if evidence.associated_plate else None

        # 1. Upsert Vehicle Node
        if v_id in self.builder.graph.nodes:
            # Update existing vehicle node
            v_data = self.builder.graph.nodes[v_id].get("data", {})
            v_data["last_seen"] = timestamp
            v_data["observation_count"] = v_data.get("observation_count", 0) + 1
            v_data["current_camera"] = evidence.camera_id
            v_data["current_road"] = road_id
            v_data["current_lane"] = lane_id
            v_data["current_speed"] = track.speed_estimate
            v_data["current_direction"] = dir_str
            if plate_str:
                v_data["plate_number"] = plate_str
        else:
            node = VehicleNode(
                vehicle_id=v_id,
                plate_number=plate_str,
                vehicle_type=evidence.vehicle_type,
                first_seen=timestamp,
                last_seen=timestamp,
                observation_count=1,
                current_camera=evidence.camera_id,
                current_road=road_id,
                current_lane=lane_id,
                current_speed=track.speed_estimate,
                current_direction=dir_str,
            )
            self.builder.add_vehicle_node(node)

        # 2. Add Temporal Edges
        self.builder.add_typed_edge(
            v_id,
            evidence.camera_id,
            EdgeType.VEHICLE_OBSERVED_BY_CAMERA,
            timestamp=timestamp,
            speed=track.speed_estimate,
            direction=dir_str,
            camera_id=evidence.camera_id,
        )

        self.builder.add_typed_edge(
            v_id,
            road_id,
            EdgeType.VEHICLE_TRANSITIONS_TO_ROAD,
            timestamp=timestamp,
            speed=track.speed_estimate,
            direction=dir_str,
            camera_id=evidence.camera_id,
        )

        self.builder.add_typed_edge(
            v_id,
            lane_id,
            EdgeType.VEHICLE_TRAVELS_ON_LANE,
            timestamp=timestamp,
            speed=track.speed_estimate,
            direction=dir_str,
            camera_id=evidence.camera_id,
        )

    def update_journey(self, journey: VehicleJourney):
        """
        Consumes complete Phase 5 VehicleJourney output and incorporates its multi-camera temporal transitions.
        """
        g_veh_id = journey.global_vehicle_id

        # 1. Upsert Global Vehicle Node
        if g_veh_id in self.builder.graph.nodes:
            v_data = self.builder.graph.nodes[g_veh_id].get("data", {})
            v_data["last_seen"] = journey.last_seen
            v_data["observation_count"] = len(journey.segments)
            if journey.plate_number:
                v_data["plate_number"] = journey.plate_number
        else:
            node = VehicleNode(
                vehicle_id=g_veh_id,
                plate_number=journey.plate_number,
                vehicle_type=journey.vehicle_type,
                first_seen=journey.first_seen,
                last_seen=journey.last_seen,
                observation_count=len(journey.segments),
                current_camera=journey.segments[-1].camera_id if journey.segments else None,
                current_speed=journey.segments[-1].speed_estimate if journey.segments else 0.0,
                current_direction=journey.segments[-1].direction if journey.segments else "UNKNOWN",
            )
            self.builder.add_vehicle_node(node)

        # 2. Add Multi-Camera Temporal Transition Edges
        segments = journey.segments
        for i in range(len(segments)):
            seg = segments[i]
            if seg.camera_id and seg.camera_id not in self.builder.graph.nodes:
                self.builder.add_camera_node(seg.camera_id)
            # Observed by Camera edge
            self.builder.add_typed_edge(
                g_veh_id,
                seg.camera_id,
                EdgeType.VEHICLE_OBSERVED_BY_CAMERA,
                timestamp=seg.timestamp,
                speed=seg.speed_estimate,
                direction=seg.direction,
                camera_id=seg.camera_id,
                metadata={"local_track_id": seg.track_id},
            )

            # Camera-to-Camera Transition edge
            if i > 0:
                prev_seg = segments[i - 1]
                delta_t = seg.timestamp - prev_seg.timestamp
                dist_m = 300.0  # Topology distance
                est_speed = (dist_m / max(0.1, delta_t) * 3.6) if delta_t > 0 else seg.speed_estimate

                self.builder.add_typed_edge(
                    prev_seg.camera_id,
                    seg.camera_id,
                    EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA,
                    timestamp=seg.timestamp,
                    distance=dist_m,
                    travel_time=delta_t,
                    speed=round(est_speed, 2),
                    direction=seg.direction,
                    metadata={"global_vehicle_id": g_veh_id, "journey_id": journey.journey_id},
                )

    def snapshot(self, timestamp: float) -> Dict[str, Any]:
        """
        Captures an immutable traffic graph state snapshot at timestamp.
        """
        metrics = TrafficMetricsEngine.get_network_metrics(self.builder)
        snap = {
            "timestamp": timestamp,
            "metrics": metrics,
            "node_count": len(self.builder.graph.nodes),
            "edge_count": len(self.builder.graph.edges),
            "vehicles": self.builder.get_nodes_by_type(NodeType.VEHICLE),
            "cameras": self.builder.get_nodes_by_type(NodeType.CAMERA),
            "roads": self.builder.get_nodes_by_type(NodeType.ROAD),
        }
        self.snapshots[timestamp] = snap
        return snap

    def get_historical_snapshot(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Retrieves historical snapshot if available."""
        return self.snapshots.get(timestamp)
