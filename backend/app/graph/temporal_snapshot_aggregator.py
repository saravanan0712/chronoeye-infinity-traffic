"""
ChronoEye Infinity - Module 3 Step 3: Temporal Snapshot Window Aggregator
Converts continuous, timestamped vehicle observations and reconstructed journeys
into deterministic, consecutive temporal graph snapshots G(t0), G(t1), G(t2), ...
using half-open intervals [start_time, end_time) and the Step 2 Temporal Graph Data Contract.
"""

import math
import statistics
import uuid
from typing import List, Dict, Optional, Any, Union, Tuple

from app.graph.graph_schema import NodeType, EdgeType
from app.schemas.reid import VehicleJourney, JourneySegment, ReIDScoreBreakdown
from app.graph.temporal_snapshot_schema import (
    TemporalNodeFeatures,
    TemporalEdgeFeatures,
    TemporalGraphSnapshot,
    TemporalGraphDatasetContract,
)


class TemporalSnapshotAggregator:
    """
    Deterministic Sliding Window Aggregator for Spatio-Temporal Traffic Graphs.
    Aggregates continuous Module 2 journeys & vehicle observations into discrete G(t) snapshots.
    """

    def __init__(
        self,
        window_seconds: float = 300.0,
        stride_seconds: float = 300.0,
        include_empty_windows: bool = False,
    ):
        """
        Initializes the aggregator with configurable window duration and stride.
        Default: 300s (5-min) window, 300s (5-min) non-overlapping stride.
        """
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.include_empty_windows = include_empty_windows

    def aggregate(
        self,
        journeys: Optional[List[VehicleJourney]] = None,
        observations: Optional[List[Any]] = None,
        segments: Optional[List[JourneySegment]] = None,
        window_seconds: Optional[float] = None,
        stride_seconds: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> TemporalGraphDatasetContract:
        """
        Aggregates inputs into consecutive chronological TemporalGraphSnapshots.
        Supports journeys, segments, raw observation dictionaries, or arbitrary normalized events.
        """
        win_sec = float(window_seconds) if window_seconds is not None else self.window_seconds
        stride_sec = float(stride_seconds) if stride_seconds is not None else self.stride_seconds

        # 1. Normalize all inputs into structured internal observation & transition records
        obs_records, trans_records = self._extract_records(
            journeys=journeys, observations=observations, segments=segments
        )

        if not obs_records and not trans_records:
            return TemporalGraphDatasetContract(
                dataset_id=f"DS_EMPTY_{int(win_sec)}_{int(stride_sec)}",
                snapshots=[],
                window_seconds=win_sec,
                stride_seconds=stride_sec,
            )

        # 2. Determine time boundary
        all_timestamps = [r["timestamp"] for r in obs_records] + [r["timestamp"] for r in trans_records]
        earliest_ts = min(all_timestamps) if all_timestamps else 0.0
        latest_ts = max(all_timestamps) if all_timestamps else 0.0

        if start_time is not None:
            t_start = float(start_time)
        else:
            t_start = math.floor(earliest_ts / win_sec) * win_sec

        t_end = float(end_time) if end_time is not None else (latest_ts + 0.001)

        # 3. Generate windows [w_start, w_end)
        snapshots: List[TemporalGraphSnapshot] = []
        current_w_start = t_start

        while current_w_start <= latest_ts or (end_time is not None and current_w_start < t_end):
            current_w_end = current_w_start + win_sec

            # Filter records in half-open interval [current_w_start, current_w_end)
            win_obs = [
                r for r in obs_records
                if current_w_start <= r["timestamp"] < current_w_end
            ]
            win_trans = [
                r for r in trans_records
                if current_w_start <= r["timestamp"] < current_w_end
            ]

            if win_obs or win_trans or self.include_empty_windows:
                snap = self._build_snapshot(
                    start_time=current_w_start,
                    end_time=current_w_end,
                    obs_records=win_obs,
                    trans_records=win_trans,
                    window_seconds=win_sec,
                )
                snapshots.append(snap)

            current_w_start += stride_sec
            if stride_sec <= 0:
                break

        # Sort snapshots chronologically
        snapshots.sort(key=lambda s: s.start_time)

        return TemporalGraphDatasetContract(
            dataset_id=f"DS_{int(t_start)}_{int(t_end)}_{int(win_sec)}_{int(stride_sec)}",
            snapshots=snapshots,
            window_seconds=win_sec,
            stride_seconds=stride_sec,
        )

    def _extract_records(
        self,
        journeys: Optional[List[VehicleJourney]] = None,
        observations: Optional[List[Any]] = None,
        segments: Optional[List[JourneySegment]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Normalizes multi-evidence inputs into flat observation and transition records.
        Preserves all Module 2 evidence (plates, 7-signal breakdowns, uncertainty, gap flags).
        """
        obs_records: List[Dict[str, Any]] = []
        trans_records: List[Dict[str, Any]] = []

        # A. Process VehicleJourneys
        if journeys:
            for jrn in journeys:
                g_id = jrn.global_vehicle_id
                j_id = jrn.journey_id
                segs = jrn.segments
                for i, seg in enumerate(segs):
                    obs_records.append({
                        "node_id": seg.camera_id,
                        "timestamp": seg.timestamp,
                        "global_vehicle_id": g_id,
                        "journey_id": j_id,
                        "track_id": seg.track_id,
                        "speed": seg.speed_estimate if seg.speed_estimate > 0 else None,
                        "direction": seg.direction if seg.direction != "UNKNOWN" else None,
                        "vehicle_type": getattr(seg, "vehicle_type", jrn.vehicle_type),
                        "plate_number": seg.plate_number or jrn.plate_number,
                        "plate_confidence": seg.plate_confidence,
                        "plate_status": seg.plate_status,
                        "timestamp_uncertainty_seconds": seg.timestamp_uncertainty_seconds,
                    })

                    # Extract inter-camera transitions
                    if i > 0:
                        prev_seg = segs[i - 1]
                        delta_t = max(0.0, seg.timestamp - prev_seg.timestamp)
                        trans_records.append({
                            "source": prev_seg.camera_id,
                            "target": seg.camera_id,
                            "timestamp": seg.timestamp,
                            "travel_time": delta_t if delta_t > 0 else None,
                            "global_vehicle_id": g_id,
                            "journey_id": j_id,
                            "transition_score": seg.transition_score,
                            "transition_decision": seg.transition_decision,
                            "transition_breakdown": seg.transition_breakdown,
                            "has_unobserved_gap": seg.has_unobserved_gap,
                            "direction": seg.direction if seg.direction != "UNKNOWN" else None,
                            "plate_number": seg.plate_number or jrn.plate_number,
                            "plate_confidence": seg.plate_confidence,
                            "plate_status": seg.plate_status,
                            "timestamp_uncertainty_seconds": seg.timestamp_uncertainty_seconds,
                        })

        # B. Process JourneySegments directly
        if segments:
            for seg in segments:
                obs_records.append({
                    "node_id": seg.camera_id,
                    "timestamp": seg.timestamp,
                    "global_vehicle_id": getattr(seg, "global_vehicle_id", None),
                    "journey_id": getattr(seg, "journey_id", None),
                    "track_id": seg.track_id,
                    "speed": seg.speed_estimate if seg.speed_estimate > 0 else None,
                    "direction": seg.direction if seg.direction != "UNKNOWN" else None,
                    "vehicle_type": getattr(seg, "vehicle_type", "car"),
                    "plate_number": seg.plate_number,
                    "plate_confidence": seg.plate_confidence,
                    "plate_status": seg.plate_status,
                    "timestamp_uncertainty_seconds": seg.timestamp_uncertainty_seconds,
                })

        # C. Process raw observation records / dictionaries
        if observations:
            for item in observations:
                if isinstance(item, dict):
                    obs_records.append({
                        "node_id": item.get("camera_id") or item.get("node_id", "UNKNOWN"),
                        "timestamp": float(item.get("timestamp", 0.0)),
                        "global_vehicle_id": item.get("global_vehicle_id"),
                        "journey_id": item.get("journey_id"),
                        "track_id": item.get("track_id", "TRK_0"),
                        "speed": item.get("speed") if item.get("speed") is not None and item.get("speed") > 0 else None,
                        "direction": item.get("direction"),
                        "vehicle_type": item.get("vehicle_type", "car"),
                        "plate_number": item.get("plate_number"),
                        "plate_confidence": item.get("plate_confidence"),
                        "plate_status": item.get("plate_status"),
                        "timestamp_uncertainty_seconds": item.get("timestamp_uncertainty_seconds"),
                    })
                elif hasattr(item, "camera_id") and hasattr(item, "timestamp"):
                    obs_records.append({
                        "node_id": item.camera_id,
                        "timestamp": float(item.timestamp),
                        "global_vehicle_id": getattr(item, "global_vehicle_id", None),
                        "journey_id": getattr(item, "journey_id", None),
                        "track_id": getattr(item, "track_id", "TRK_0"),
                        "speed": getattr(item, "speed_estimate", None),
                        "direction": getattr(item, "direction", None),
                        "vehicle_type": getattr(item, "vehicle_type", "car"),
                        "plate_number": getattr(item, "plate_number", None),
                        "plate_confidence": getattr(item, "plate_confidence", None),
                        "plate_status": getattr(item, "plate_status", None),
                        "timestamp_uncertainty_seconds": getattr(item, "timestamp_uncertainty_seconds", None),
                    })

        return obs_records, trans_records

    def _build_snapshot(
        self,
        start_time: float,
        end_time: float,
        obs_records: List[Dict[str, Any]],
        trans_records: List[Dict[str, Any]],
        window_seconds: float,
    ) -> TemporalGraphSnapshot:
        """
        Aggregates observations and transitions for a specific window into TemporalNodeFeatures and TemporalEdgeFeatures.
        """
        # 1. Group observations by node_id
        nodes_dict: Dict[str, List[Dict[str, Any]]] = {}
        for r in obs_records:
            n_id = r["node_id"]
            if n_id not in nodes_dict:
                nodes_dict[n_id] = []
            nodes_dict[n_id].append(r)

        # 2. Group transitions by (source, target)
        edges_dict: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for t in trans_records:
            pair = (t["source"], t["target"])
            if pair not in edges_dict:
                edges_dict[pair] = []
            edges_dict[pair].append(t)

        # 3. Aggregate node features
        node_features_list: List[TemporalNodeFeatures] = []
        for n_id in sorted(nodes_dict.keys()):
            records = nodes_dict[n_id]

            # Count distinct vehicles if global IDs or track IDs exist
            unique_veh_ids = {
                r["global_vehicle_id"] if r.get("global_vehicle_id") else r["track_id"]
                for r in records
            }
            veh_count = len(unique_veh_ids)

            # Flow rate = (veh_count / window_hours)
            hours = window_seconds / 3600.0
            flow_rate = round(veh_count / hours, 2) if hours > 0 else None

            # Average speed across valid speed observations
            valid_speeds = [r["speed"] for r in records if r.get("speed") is not None and r["speed"] > 0]
            avg_speed = round(sum(valid_speeds) / len(valid_speeds), 2) if valid_speeds else None

            # Incoming/outgoing transition flow counts for this node
            incoming_count = sum(1 for t in trans_records if t["target"] == n_id)
            outgoing_count = sum(1 for t in trans_records if t["source"] == n_id)
            inc_flow = round(incoming_count / hours, 2) if (hours > 0 and incoming_count > 0) else None
            out_flow = round(outgoing_count / hours, 2) if (hours > 0 and outgoing_count > 0) else None

            # Metadata preservation
            plates = [r["plate_number"] for r in records if r.get("plate_number")]
            uncertainties = [r["timestamp_uncertainty_seconds"] for r in records if r.get("timestamp_uncertainty_seconds") is not None]
            node_meta = {
                "observed_vehicle_ids": sorted(list(unique_veh_ids)),
                "observation_count": len(records),
            }
            if plates:
                node_meta["plates"] = plates
            if uncertainties:
                node_meta["mean_timestamp_uncertainty_seconds"] = round(sum(uncertainties) / len(uncertainties), 4)

            node_features_list.append(
                TemporalNodeFeatures(
                    node_id=n_id,
                    node_type=NodeType.CAMERA,
                    timestamp=start_time,
                    vehicle_count=veh_count,
                    flow_rate=flow_rate,
                    density=None,          # Never fabricate density without physical segment length
                    average_speed=avg_speed,
                    queue_length=None,     # Never fabricate queue length without sensor queue definitions
                    congestion=None,       # Never fabricate research congestion labels
                    incoming_flow=inc_flow,
                    outgoing_flow=out_flow,
                    metadata=node_meta,
                )
            )

        # 4. Aggregate edge features
        edge_features_list: List[TemporalEdgeFeatures] = []
        for pair in sorted(edges_dict.keys()):
            source, target = pair
            records = edges_dict[pair]

            unique_veh_ids = {
                r["global_vehicle_id"] if r.get("global_vehicle_id") else r.get("journey_id", "TRK_0")
                for r in records
            }
            veh_count = len(unique_veh_ids)
            trans_count = len(records)

            hours = window_seconds / 3600.0
            flow_rate = round(trans_count / hours, 2) if hours > 0 else None

            # Travel times
            valid_tt = [r["travel_time"] for r in records if r.get("travel_time") is not None and r["travel_time"] > 0]
            mean_tt = round(sum(valid_tt) / len(valid_tt), 2) if valid_tt else None
            med_tt = round(statistics.median(valid_tt), 2) if valid_tt else None

            # Unobserved gaps
            unobs_count = sum(1 for r in records if r.get("has_unobserved_gap", False))
            obs_count = trans_count - unobs_count
            has_gap = unobs_count > 0

            # Transition confidences / scores
            valid_confs = [r["transition_score"] for r in records if r.get("transition_score") is not None]
            mean_conf = round(sum(valid_confs) / len(valid_confs), 3) if valid_confs else None

            # Directions
            directions = [r["direction"] for r in records if r.get("direction")]
            consensus_dir = max(set(directions), key=directions.count) if directions else None

            # Timestamp uncertainty
            uncertainties = [r["timestamp_uncertainty_seconds"] for r in records if r.get("timestamp_uncertainty_seconds") is not None]
            mean_unc = round(sum(uncertainties) / len(uncertainties), 4) if uncertainties else None

            # Transition breakdowns
            breakdowns = [
                r["transition_breakdown"].model_dump() if hasattr(r["transition_breakdown"], "model_dump")
                else r["transition_breakdown"]
                for r in records if r.get("transition_breakdown") is not None
            ]

            edge_meta = {
                "journey_ids": sorted(list({r["journey_id"] for r in records if r.get("journey_id")})),
                "global_vehicle_ids": sorted(list({r["global_vehicle_id"] for r in records if r.get("global_vehicle_id")})),
                "transition_decisions": [
                    r["transition_decision"].value if hasattr(r["transition_decision"], "value") else str(r["transition_decision"])
                    for r in records if r.get("transition_decision") is not None
                ],
            }
            if breakdowns:
                edge_meta["transition_breakdowns"] = breakdowns

            edge_features_list.append(
                TemporalEdgeFeatures(
                    source=source,
                    target=target,
                    edge_type=EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA,
                    timestamp=start_time,
                    vehicle_count=veh_count,
                    flow_rate=flow_rate,
                    travel_time=mean_tt,
                    mean_travel_time=mean_tt,
                    median_travel_time=med_tt,
                    transition_count=trans_count,
                    observed_transition_count=obs_count,
                    unobserved_gap_count=unobs_count,
                    has_unobserved_gap=has_gap,
                    confidence=mean_conf,
                    mean_transition_confidence=mean_conf,
                    direction=consensus_dir,
                    uncertainty=mean_unc,
                    metadata=edge_meta,
                )
            )

        return TemporalGraphSnapshot(
            snapshot_id=f"SNAP_{int(start_time)}_{int(end_time)}",
            start_time=start_time,
            end_time=end_time,
            nodes=node_features_list,
            edges=edge_features_list,
            metadata={
                "window_seconds": window_seconds,
                "node_count": len(node_features_list),
                "edge_count": len(edge_features_list),
            },
        )
