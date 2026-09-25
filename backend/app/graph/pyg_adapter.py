"""
ChronoEye Infinity - Phase 6: PyTorch Geometric (PyG) HeteroData Graph Adapter
Converts NetworkX spatio-temporal traffic graph into PyTorch Geometric HeteroData structures
for ST-GNN forecasting model training. Provides structured feature fallback if PyG is uninstalled.
"""

from typing import Dict, List, Any, Optional
from app.graph.graph_schema import NodeType, EdgeType
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.graph.graph_metrics import TrafficMetricsEngine


class PyGGraphAdapter:
    """
    Adapter converting NetworkX spatio-temporal graph to PyTorch Geometric HeteroData.
    """

    @staticmethod
    def to_hetero_data(builder: SpatioTemporalGraphBuilder) -> Dict[str, Any]:
        """
        Converts graph to PyTorch Geometric HeteroData instance if PyG is installed,
        or returns structured feature matrices dictionary if PyG is absent.
        """
        try:
            import torch
            from torch_geometric.data import HeteroData

            data = HeteroData()

            # 1. Vehicle Node Features: [speed, vehicle_type_id, observation_count]
            vehicle_nodes = builder.get_nodes_by_type(NodeType.VEHICLE)
            v_map = {v_id: idx for idx, v_id in enumerate(vehicle_nodes)}
            v_feats = []
            for v_id in vehicle_nodes:
                d = builder.graph.nodes[v_id].get("data", {})
                v_type_id = 1.0 if d.get("vehicle_type") == "car" else 2.0
                v_feats.append([
                    float(d.get("current_speed", 0.0)),
                    v_type_id,
                    float(d.get("observation_count", 1)),
                ])

            if v_feats:
                data["vehicle"].x = torch.tensor(v_feats, dtype=torch.float)
            else:
                data["vehicle"].x = torch.empty((0, 3), dtype=torch.float)

            # 2. Road Node Features: [vehicle_count, density, avg_speed, queue, capacity, congestion_score]
            road_nodes = builder.get_nodes_by_type(NodeType.ROAD)
            r_map = {r_id: idx for idx, r_id in enumerate(road_nodes)}
            r_feats = []
            for r_id in road_nodes:
                v_count = TrafficMetricsEngine.calculate_vehicle_count(builder, r_id)
                density = TrafficMetricsEngine.calculate_density(builder, r_id)
                avg_speed = TrafficMetricsEngine.calculate_average_speed(builder, r_id)
                queue = TrafficMetricsEngine.estimate_queue_length(builder, r_id)
                cong = TrafficMetricsEngine.calculate_congestion_score(builder, r_id)
                road_data = builder.graph.nodes[r_id].get("data", {})
                cap = float(road_data.get("capacity", 100))

                r_feats.append([float(v_count), float(density), float(avg_speed), float(queue), cap, float(cong)])

            if r_feats:
                data["road"].x = torch.tensor(r_feats, dtype=torch.float)
            else:
                data["road"].x = torch.empty((0, 6), dtype=torch.float)

            # 3. Camera Node Features: [observation_count, active_vehicle_count]
            cam_nodes = builder.get_nodes_by_type(NodeType.CAMERA)
            c_map = {c_id: idx for idx, c_id in enumerate(cam_nodes)}
            c_feats = []
            for c_id in cam_nodes:
                d = builder.graph.nodes[c_id].get("data", {})
                c_feats.append([float(d.get("observation_count", 0)), float(d.get("active_vehicle_count", 0))])

            if c_feats:
                data["camera"].x = torch.tensor(c_feats, dtype=torch.float)
            else:
                data["camera"].x = torch.empty((0, 2), dtype=torch.float)

            # 4. Vehicle -> Camera Observation Edge Index
            edge_src, edge_dst = [], []
            for u, v, k, d in builder.graph.edges(data=True, keys=True):
                if d.get("edge_type") == EdgeType.VEHICLE_OBSERVED_BY_CAMERA.value:
                    if u in v_map and v in c_map:
                        edge_src.append(v_map[u])
                        edge_dst.append(c_map[v])

            if edge_src:
                data["vehicle", "observed_by", "camera"].edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

            return {"pyg_available": True, "hetero_data": data}

        except ImportError:
            # Fallback structured feature dictionary if PyG/Torch is unavailable
            return PyGGraphAdapter._build_fallback_feature_dict(builder)

    @staticmethod
    def _build_fallback_feature_dict(builder: SpatioTemporalGraphBuilder) -> Dict[str, Any]:
        """Structured dictionary feature matrix representation when PyG is unavailable."""
        vehicle_nodes = builder.get_nodes_by_type(NodeType.VEHICLE)
        road_nodes = builder.get_nodes_by_type(NodeType.ROAD)
        camera_nodes = builder.get_nodes_by_type(NodeType.CAMERA)

        road_features = {}
        for r_id in road_nodes:
            road_features[r_id] = {
                "vehicle_count": TrafficMetricsEngine.calculate_vehicle_count(builder, r_id),
                "density": TrafficMetricsEngine.calculate_density(builder, r_id),
                "average_speed": TrafficMetricsEngine.calculate_average_speed(builder, r_id),
                "queue_length": TrafficMetricsEngine.estimate_queue_length(builder, r_id),
                "congestion_score": TrafficMetricsEngine.calculate_congestion_score(builder, r_id),
            }

        return {
            "pyg_available": False,
            "vehicle_nodes_count": len(vehicle_nodes),
            "road_nodes_count": len(road_nodes),
            "camera_nodes_count": len(camera_nodes),
            "road_features": road_features,
        }
