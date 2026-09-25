"""
ChronoEye Infinity - Real/Public Traffic Forecasting Dataset Adapter.
Adapts standard public benchmark datasets (e.g., Caltrans PeMS08 / PeMS04)
into ChronoEye Module 3 TemporalGraphDatasetContract for leakage-free ST-GNN forecasting.
"""

import os
import csv
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from app.graph.temporal_snapshot_schema import (
    TemporalGraphSnapshot,
    TemporalNodeFeatures,
    TemporalEdgeFeatures,
    TemporalGraphDatasetContract,
    ForecastHorizon,
)


class RealTrafficDatasetAdapter:
    """
    Adapter for real-world public traffic benchmarks (Caltrans PeMS08).
    Converts multi-sensor time-series arrays and distance connectivity tables
    into standardized ChronoEye TemporalGraphDatasetContract.
    """

    FEATURE_NAMES = [
        "vehicle_count",
        "flow_rate",
        "density",
        "average_speed",
        "queue_length",
        "congestion",
        "incoming_flow",
        "outgoing_flow",
    ]

    TARGET_NAMES = [
        "flow_rate",
        "density",
        "congestion",
        "travel_time",
    ]

    FREE_FLOW_SPEED_MPH = 65.0  # Standard California highway free-flow speed

    @classmethod
    def load_pems08(
        cls,
        npz_path: str = "data/research/pems08/PEMS08.npz",
        csv_path: str = "data/research/pems08/PEMS08.csv",
        num_timesteps: Optional[int] = None,
        num_nodes: Optional[int] = None,
        start_step: int = 0,
        window_seconds: float = 300.0,  # 5-minute sampling
    ) -> Tuple[TemporalGraphDatasetContract, Dict[str, Any]]:
        """
        Loads the official PeMS08 dataset.
        npz_path: path to PEMS08.npz containing 'data' array [T, N, 3]
        csv_path: path to PEMS08.csv containing (from, to, cost) connectivity
        num_timesteps: subset of timesteps to process (default: all or 2016 steps = 7 days)
        num_nodes: subset of nodes to process (default: all 170 nodes)
        """
        npz_file = Path(npz_path)
        if not npz_file.exists():
            raise FileNotFoundError(f"PeMS08 dataset file not found at {npz_path}")

        raw_data = np.load(npz_file)
        full_tensor = raw_data["data"]  # [T, N, 3]: flow, occupancy, speed
        total_timesteps, total_nodes, num_channels = full_tensor.shape

        # Select temporal slice and spatial slice
        n_steps = num_timesteps if num_timesteps is not None else min(total_timesteps, 2016)
        n_nodes = num_nodes if num_nodes is not None else total_nodes

        tensor_slice = full_tensor[start_step : start_step + n_steps, :n_nodes, :]
        node_ids = [f"PEMS08_SENSOR_{i:03d}" for i in range(n_nodes)]

        # 1. Load Topology and Connectivity Graph from CSV
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        edges_list: List[Tuple[int, int, float]] = []

        if csv_path and Path(csv_path).exists():
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    u = int(row["from"])
                    v = int(row["to"])
                    dist = float(row["cost"])
                    if u < n_nodes and v < n_nodes:
                        edges_list.append((u, v, dist))
                        # Gaussian distance kernel
                        adj_matrix[u, v] = math.exp(-((dist / 1000.0) ** 2))

        # Ensure symmetric or connected topology
        for u, v, dist in edges_list:
            if adj_matrix[v, u] == 0:
                adj_matrix[v, u] = adj_matrix[u, v]

        # 2. Build ChronoEye TemporalGraphSnapshots
        snapshots: List[TemporalGraphSnapshot] = []

        for t in range(n_steps):
            step_idx = start_step + t
            t_start = float(step_idx * window_seconds)
            t_end = t_start + window_seconds

            node_feats: List[TemporalNodeFeatures] = []
            for i in range(n_nodes):
                raw_flow = float(tensor_slice[t, i, 0])
                raw_occ = float(tensor_slice[t, i, 1])
                raw_speed = float(tensor_slice[t, i, 2])

                # Feature mapping:
                # 1. vehicle_count: flow per 5-min
                veh_count = raw_flow
                # 2. flow_rate: veh/5min
                flow_rate = raw_flow
                # 3. density: derived from occupancy (occ * 100) or flow/speed
                density = raw_occ * 100.0
                # 4. average_speed: mph
                avg_speed = raw_speed
                # 5. congestion index: percentage speed degradation from free-flow (65 mph)
                if raw_speed > 0:
                    congestion = max(0.0, min(100.0, (1.0 - (raw_speed / cls.FREE_FLOW_SPEED_MPH)) * 100.0))
                else:
                    congestion = 100.0

                node_feat = TemporalNodeFeatures(
                    node_id=node_ids[i],
                    camera_id=node_ids[i],
                    vehicle_count=veh_count,
                    flow_rate=flow_rate,
                    density=density,
                    average_speed=avg_speed,
                    queue_length=None,  # Unobserved in PeMS loop detectors
                    congestion=congestion,
                    incoming_flow=None,
                    outgoing_flow=None,
                    timestamp_uncertainty_seconds=0.0,
                )
                node_feats.append(node_feat)

            # Spatial Edges for Snapshot
            edge_feats: List[TemporalEdgeFeatures] = []
            for u, v, dist in edges_list:
                edge_feats.append(
                    TemporalEdgeFeatures(
                        source=node_ids[u],
                        target=node_ids[v],
                        transition_flow=None,
                        average_speed=None,
                        travel_time_seconds=float(dist / max(1.0, tensor_slice[t, u, 2] * 0.44704)),  # dist / (speed in m/s)
                        has_unobserved_gap=False,
                    )
                )

            snapshots.append(
                TemporalGraphSnapshot(
                    snapshot_id=f"SNAP_PEMS08_{t:06d}",
                    start_time=t_start,
                    end_time=t_end,
                    nodes=node_feats,
                    edges=edge_feats,
                    metadata={"dataset": "PeMS08", "step_index": step_idx},
                )
            )

        contract = TemporalGraphDatasetContract(
            dataset_id="DS_PEMS08_San_Bernardino_Real",
            snapshots=snapshots,
            feature_names=cls.FEATURE_NAMES,
            target_names=cls.TARGET_NAMES,
            window_seconds=window_seconds,
            stride_seconds=window_seconds,
            forecast_horizons=[
                ForecastHorizon.PLUS_5MIN,
                ForecastHorizon.PLUS_10MIN,
                ForecastHorizon.PLUS_15MIN,
            ],
            metadata={
                "dataset_name": "PeMS08",
                "source": "Caltrans Performance Measurement System (District 8, San Bernardino)",
                "provenance": "Official Zenodo Record 7816008",
                "num_nodes": n_nodes,
                "num_timesteps": n_steps,
                "sampling_interval_minutes": 5,
                "total_edges": len(edges_list),
            },
        )

        metadata = {
            "dataset_name": "Caltrans PeMS08",
            "source_provenance": "California Department of Transportation (District 8 San Bernardino Freeway Network)",
            "official_zenodo_doi": "10.5281/zenodo.7816008",
            "file_npz": str(npz_path),
            "file_csv": str(csv_path),
            "total_available_timesteps": total_timesteps,
            "total_available_nodes": total_nodes,
            "loaded_timesteps": n_steps,
            "loaded_nodes": n_nodes,
            "temporal_resolution_seconds": window_seconds,
            "feature_mapping": {
                "PEMS08_channel_0": "flow_rate (veh/5min) -> ChronoEye flow_rate",
                "PEMS08_channel_1": "occupancy ([0, 1]) -> ChronoEye density (occ * 100)",
                "PEMS08_channel_2": "speed (mph) -> ChronoEye average_speed",
                "derived_congestion": "max(0, 1 - speed/65.0)*100 -> ChronoEye congestion",
                "unobserved_fields": ["queue_length", "incoming_flow", "outgoing_flow"],
            },
            "target_availability": {
                "flow_rate": True,
                "density": True,
                "congestion": True,
                "travel_time": False,
            },
            "node_mapping": {node_ids[i]: i for i in range(n_nodes)},
        }

        return contract, metadata
