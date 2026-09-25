"""
ChronoEye Infinity - Module 3: Research Dataset Adapter.
Converts external spatio-temporal traffic research datasets (e.g. sensor networks, PeMS/METR format,
or multi-camera continuous observation series) into ChronoEye TemporalGraphDatasetContract instances.
"""

import math
import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field

from app.graph.graph_schema import NodeType, EdgeType
from app.graph.temporal_snapshot_schema import (
    TemporalNodeFeatures,
    TemporalEdgeFeatures,
    TemporalGraphSnapshot,
    TemporalGraphDatasetContract,
)


class BenchmarkSensorConfig(BaseModel):
    sensor_id: str
    node_type: NodeType = NodeType.CAMERA
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    road_id: Optional[str] = None


class TrafficDatasetAdapter:
    """
    Adapter converting external multi-sensor traffic observation matrices and adjacency graphs
    into ChronoEye TemporalGraphDatasetContract snapshots.
    """

    @staticmethod
    def from_matrices(
        node_ids: List[str],
        timestamps: List[float],
        data_matrix: np.ndarray,
        feature_mapping: Dict[str, int],
        adjacency_matrix: Optional[np.ndarray] = None,
        edge_list: Optional[List[Tuple[str, str, float]]] = None,
        window_seconds: float = 300.0,
        stride_seconds: float = 300.0,
        dataset_name: str = "BENCHMARK_TRAFFIC_CORRIDOR",
    ) -> TemporalGraphDatasetContract:
        """
        Converts a 3D numpy array [Timesteps, Nodes, Raw_Features] into TemporalGraphSnapshots.
        
        feature_mapping: maps ChronoEye feature names to matrix feature slice indices.
        Example: {"flow_rate": 0, "average_speed": 1, "density": 2, "vehicle_count": 3}
        """
        num_timesteps, num_nodes, num_raw_feats = data_matrix.shape
        assert len(node_ids) == num_nodes, f"Node count mismatch: {len(node_ids)} vs {num_nodes}"
        assert len(timestamps) == num_timesteps, f"Timestep mismatch: {len(timestamps)} vs {num_timesteps}"

        # 1. Build Edge Templates
        edges_template: List[Dict[str, Any]] = []
        if edge_list is not None:
            for u, v, w in edge_list:
                edges_template.append({"source": u, "target": v, "weight": float(w)})
        elif adjacency_matrix is not None:
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j and adjacency_matrix[i, j] > 0:
                        edges_template.append({
                            "source": node_ids[i],
                            "target": node_ids[j],
                            "weight": float(adjacency_matrix[i, j]),
                        })

        # 2. Build Chronological Snapshots
        snapshots: List[TemporalGraphSnapshot] = []

        for t_idx in range(num_timesteps):
            ts_start = float(timestamps[t_idx])
            ts_end = ts_start + window_seconds
            
            node_feats_list: List[TemporalNodeFeatures] = []
            for n_idx, node_id in enumerate(node_ids):
                feats_dict: Dict[str, Any] = {}
                for feat_name, col_idx in feature_mapping.items():
                    val = float(data_matrix[t_idx, n_idx, col_idx])
                    if not np.isnan(val) and val >= 0:
                        feats_dict[feat_name] = round(val, 2)
                    else:
                        feats_dict[feat_name] = None

                # Flow rate to vehicle_count fallback if only flow is available
                if feats_dict.get("flow_rate") is not None and feats_dict.get("vehicle_count") is None:
                    hours = window_seconds / 3600.0
                    feats_dict["vehicle_count"] = max(0, int(round(feats_dict["flow_rate"] * hours)))

                node_feats_list.append(
                    TemporalNodeFeatures(
                        node_id=node_id,
                        node_type=NodeType.CAMERA,
                        timestamp=ts_start,
                        vehicle_count=feats_dict.get("vehicle_count"),
                        flow_rate=feats_dict.get("flow_rate"),
                        density=feats_dict.get("density"),
                        average_speed=feats_dict.get("average_speed"),
                        queue_length=feats_dict.get("queue_length"),
                        congestion=feats_dict.get("congestion"),
                        incoming_flow=feats_dict.get("incoming_flow"),
                        outgoing_flow=feats_dict.get("outgoing_flow"),
                    )
                )

            # Build Edges for this snapshot
            edge_feats_list: List[TemporalEdgeFeatures] = []
            for e_tmpl in edges_template:
                u, v, w = e_tmpl["source"], e_tmpl["target"], e_tmpl["weight"]
                u_idx = node_ids.index(u)
                v_idx = node_ids.index(v)
                
                u_speed = data_matrix[t_idx, u_idx, feature_mapping.get("average_speed", 0)]
                # Travel time estimate based on edge distance weight and speed if valid
                tt = round((w / (u_speed / 3.6)), 2) if (u_speed > 5.0 and w > 0) else None
                u_flow = data_matrix[t_idx, u_idx, feature_mapping.get("flow_rate", 0)]

                edge_feats_list.append(
                    TemporalEdgeFeatures(
                        source=u,
                        target=v,
                        edge_type=EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA,
                        timestamp=ts_start,
                        flow_rate=round(float(u_flow), 2) if not np.isnan(u_flow) else None,
                        travel_time=tt,
                        mean_travel_time=tt,
                        transition_count=max(1, int(round(u_flow * (window_seconds / 3600.0)))) if not np.isnan(u_flow) else 1,
                        confidence=1.0,
                        has_unobserved_gap=False,
                    )
                )

            snapshots.append(
                TemporalGraphSnapshot(
                    snapshot_id=f"SNAP_{int(ts_start)}_{int(ts_end)}",
                    start_time=ts_start,
                    end_time=ts_end,
                    nodes=node_feats_list,
                    edges=edge_feats_list,
                    metadata={"dataset_name": dataset_name, "timestep_idx": t_idx},
                )
            )

        return TemporalGraphDatasetContract(
            dataset_id=f"DS_{dataset_name}",
            snapshots=snapshots,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            metadata={
                "dataset_name": dataset_name,
                "num_nodes": num_nodes,
                "num_timesteps": num_timesteps,
            },
        )

    @staticmethod
    def generate_research_benchmark_dataset(
        num_nodes: int = 10,
        num_timesteps: int = 288,  # 24 hours @ 5-min intervals (288 steps)
        seed: int = 42,
        window_seconds: float = 300.0,
        stride_seconds: float = 300.0,
        noise_level: float = 0.05,
    ) -> TemporalGraphDatasetContract:
        """
        Generates a realistic, continuous spatio-temporal traffic network benchmark dataset.
        Includes realistic morning/evening peak hours, spatial shockwave propagation along
        corridors, speed-flow non-linear dynamics, and spatial distance-based adjacency.
        """
        np.random.seed(seed)
        node_ids = [f"SENSOR_CAM_{i+1:02d}" for i in range(num_nodes)]
        timestamps = [float(t * window_seconds) for t in range(num_timesteps)]

        # 1. Generate Distance-based Graph Adjacency (Line/Grid Network with cross connections)
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        distances = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist = abs(i - j) * 450.0 + np.random.uniform(50.0, 150.0)  # meters
                    distances[i, j] = dist
                    # Connected if adjacent or within 2 hops along corridor
                    if abs(i - j) <= 2:
                        adj_matrix[i, j] = np.exp(-(dist / 1000.0) ** 2)

        # 2. Generate Continuous Spatio-Temporal Dynamics (Flow, Speed, Density, Congestion)
        # 288 steps: 0..288 (t=0 is 00:00, t=96 is 08:00 morning peak, t=216 is 18:00 evening peak)
        data = np.zeros((num_timesteps, num_nodes, 4), dtype=np.float32)
        
        for t in range(num_timesteps):
            time_of_day_hours = (t * 5.0) / 60.0  # 0.0 to 24.0
            
            # Base diurnal traffic pattern: Morning peak (8:00) and Evening peak (18:00)
            morning_peak = np.exp(-((time_of_day_hours - 8.0) / 2.0) ** 2)
            evening_peak = np.exp(-((time_of_day_hours - 18.0) / 2.5) ** 2)
            base_demand = 300.0 + 900.0 * morning_peak + 1100.0 * evening_peak + np.random.normal(0, 25.0)
            
            for i in range(num_nodes):
                # Spatial phase delay (upstream nodes peak slightly earlier than downstream)
                phase_delay = i * 0.15
                node_m_peak = np.exp(-((time_of_day_hours - 8.0 - phase_delay) / 2.0) ** 2)
                node_e_peak = np.exp(-((time_of_day_hours - 18.0 - phase_delay) / 2.5) ** 2)
                
                # Spatial capacity bottleneck at node 4 & 5
                capacity_factor = 0.80 if i in (4, 5) else 1.0
                
                node_flow = (250.0 + 850.0 * node_m_peak + 1050.0 * node_e_peak) * capacity_factor
                node_flow += np.random.normal(0, node_flow * noise_level)
                node_flow = max(50.0, node_flow)

                # Greenshields Speed-Density relationship: v = v_free * (1 - k / k_jam)
                v_free = 65.0  # km/h
                k_jam = 120.0  # veh/km
                
                # Estimated density from flow
                density = node_flow / max(15.0, (v_free * (1.0 - 0.4 * (node_flow / 1800.0))))
                density = min(k_jam * 0.9, max(5.0, density))
                
                speed = max(10.0, v_free * (1.0 - (density / k_jam)))
                speed += np.random.normal(0, 1.5)
                
                congestion = min(1.0, max(0.0, (k_jam * 0.5 - speed) / (k_jam * 0.5)))
                
                data[t, i, 0] = node_flow       # Col 0: flow_rate (veh/h)
                data[t, i, 1] = speed           # Col 1: average_speed (km/h)
                data[t, i, 2] = density         # Col 2: density (veh/km)
                data[t, i, 3] = congestion      # Col 3: congestion score (0.0 to 1.0)

        feature_mapping = {
            "flow_rate": 0,
            "average_speed": 1,
            "density": 2,
            "congestion": 3,
        }

        return TrafficDatasetAdapter.from_matrices(
            node_ids=node_ids,
            timestamps=timestamps,
            data_matrix=data,
            feature_mapping=feature_mapping,
            adjacency_matrix=adj_matrix,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            dataset_name="CHRONOEYE_RESEARCH_CORRIDOR_24H",
        )
