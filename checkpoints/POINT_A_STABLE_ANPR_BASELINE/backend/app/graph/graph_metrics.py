"""
ChronoEye Infinity - Phase 6: Spatio-Temporal Traffic Graph Metrics Engine
Calculates graph-level & road-level metrics: vehicle count, density, flow, average speed,
queue length estimation, and normalized congestion scores.
"""

from typing import Dict, List, Any
from app.graph.graph_schema import NodeType, EdgeType
from app.graph.graph_builder import SpatioTemporalGraphBuilder


class TrafficMetricsEngine:
    """
    Traffic Graph Analytics & Congestion Metrics Engine.
    """

    @staticmethod
    def calculate_vehicle_count(builder: SpatioTemporalGraphBuilder, road_id: str) -> int:
        """
        Calculates active vehicle count currently observed on a specific road segment.
        """
        if road_id not in builder.graph.nodes:
            return 0

        count = 0
        for u, v, k, d in builder.graph.edges(data=True, keys=True):
            if d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_ROAD.value and v == road_id:
                count += 1
        return count

    @staticmethod
    def calculate_density(builder: SpatioTemporalGraphBuilder, road_id: str) -> float:
        """
        Calculates traffic density (vehicles per kilometer).
        """
        if road_id not in builder.graph.nodes:
            return 0.0

        road_data = builder.graph.nodes[road_id].get("data", {})
        length_m = road_data.get("length_meters", 500.0)
        length_km = max(0.05, length_m / 1000.0)

        vehicle_count = TrafficMetricsEngine.calculate_vehicle_count(builder, road_id)
        return round(vehicle_count / length_km, 2)

    @staticmethod
    def calculate_average_speed(builder: SpatioTemporalGraphBuilder, road_id: str) -> float:
        """
        Calculates average speed of active vehicles on a road segment in km/h.
        """
        if road_id not in builder.graph.nodes:
            return 60.0

        speeds = []
        for u, v, k, d in builder.graph.edges(data=True, keys=True):
            if d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_ROAD.value and v == road_id:
                edge_data = d.get("data", {})
                speed = edge_data.get("speed", 60.0)
                if speed > 0:
                    speeds.append(speed)

        if not speeds:
            road_data = builder.graph.nodes[road_id].get("data", {})
            return road_data.get("speed_limit_kmh", 60.0)

        return round(sum(speeds) / float(len(speeds)), 2)

    @staticmethod
    def estimate_queue_length(
        builder: SpatioTemporalGraphBuilder, road_id: str, low_speed_thresh_kmh: float = 5.0
    ) -> int:
        """
        Estimates queued/stopped vehicles on road segment (speed < low_speed_thresh_kmh).
        """
        if road_id not in builder.graph.nodes:
            return 0

        queue_count = 0
        for u, v, k, d in builder.graph.edges(data=True, keys=True):
            if d.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_ROAD.value and v == road_id:
                edge_data = d.get("data", {})
                speed = edge_data.get("speed", 60.0)
                if speed < low_speed_thresh_kmh:
                    queue_count += 1

        return queue_count

    @staticmethod
    def calculate_congestion_score(builder: SpatioTemporalGraphBuilder, road_id: str) -> float:
        """
        Calculates normalized 0.0 (free flow) to 1.0 (heavy congestion) score.
        Formula: 0.4 * (density / capacity_density) + 0.4 * max(0, 1 - avg_speed/speed_limit) + 0.2 * (queue / capacity)
        """
        if road_id not in builder.graph.nodes:
            return 0.0

        road_data = builder.graph.nodes[road_id].get("data", {})
        length_km = max(0.05, road_data.get("length_meters", 500.0) / 1000.0)
        speed_limit = road_data.get("speed_limit_kmh", 60.0)
        capacity = max(10, road_data.get("capacity", 100))
        capacity_density = capacity / length_km

        density = TrafficMetricsEngine.calculate_density(builder, road_id)
        avg_speed = TrafficMetricsEngine.calculate_average_speed(builder, road_id)
        queue = TrafficMetricsEngine.estimate_queue_length(builder, road_id)

        density_ratio = min(1.0, density / max(1.0, capacity_density))
        speed_deficit = max(0.0, 1.0 - (avg_speed / max(1.0, speed_limit)))
        queue_ratio = min(1.0, queue / float(capacity))

        congestion = 0.4 * density_ratio + 0.4 * speed_deficit + 0.2 * queue_ratio
        return round(min(1.0, max(0.0, congestion)), 3)

    @classmethod
    def get_network_metrics(cls, builder: SpatioTemporalGraphBuilder) -> Dict[str, Any]:
        """
        Aggregates graph-level traffic network statistics across all road nodes.
        """
        road_ids = builder.get_nodes_by_type(NodeType.ROAD)
        vehicle_ids = builder.get_nodes_by_type(NodeType.VEHICLE)

        total_vehicles = len(vehicle_ids)
        road_metrics = {}
        total_congestion = 0.0

        for r_id in road_ids:
            v_count = cls.calculate_vehicle_count(builder, r_id)
            density = cls.calculate_density(builder, r_id)
            avg_speed = cls.calculate_average_speed(builder, r_id)
            queue = cls.estimate_queue_length(builder, r_id)
            cong_score = cls.calculate_congestion_score(builder, r_id)

            total_congestion += cong_score
            road_metrics[r_id] = {
                "vehicle_count": v_count,
                "density": density,
                "average_speed": avg_speed,
                "queue_length": queue,
                "congestion_score": cong_score,
            }

        avg_network_congestion = (
            round(total_congestion / float(len(road_ids)), 3)
            if road_ids else 0.0
        )

        return {
            "total_nodes": len(builder.graph.nodes),
            "total_edges": len(builder.graph.edges),
            "active_vehicles": total_vehicles,
            "total_roads": len(road_ids),
            "average_network_congestion": avg_network_congestion,
            "roads": road_metrics,
        }
