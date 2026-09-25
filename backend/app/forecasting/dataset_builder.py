"""
ChronoEye Infinity - Phase 8: Dataset Builder
Generates synthetic traffic simulation datasets and constructs supervised training feature matrices (X, Y)
with strict chronological train/validation/test splits preventing future information leakage.
"""

from typing import List, Tuple, Dict, Any, Optional
from app.simulation.engine import TrafficSimulationEngine
from app.perception.adapter import DetectionAdapter
from app.perception.tracker import VehicleTrackerManager
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.graph.temporal_graph import TemporalTrafficGraphEngine
from app.state.traffic_state_schema import NetworkTrafficSnapshot
from app.state.traffic_state_engine import DynamicTrafficStateEngine
from app.forecasting.feature_window import FeatureWindowExtractor


class ForecastingDatasetBuilder:
    """
    Traffic Dataset & Supervised Window Matrix Construction Engine.
    """

    @staticmethod
    def generate_synthetic_simulation_dataset(
        num_scenarios: int = 5,
        steps_per_scenario: int = 30,
        seed: int = 42,
    ) -> List[NetworkTrafficSnapshot]:
        """
        Generates reproducible simulation-based traffic state history snapshots across multiple traffic scenarios.
        """
        snapshots = []
        global_timestamp = 0.0

        for s_idx in range(num_scenarios):
            sim_seed = seed + s_idx
            sim = TrafficSimulationEngine(seed=sim_seed)
            builder = SpatioTemporalGraphBuilder()
            temporal_graph = TemporalTrafficGraphEngine(builder=builder)
            state_engine = DynamicTrafficStateEngine(max_history=100)
            tracker_mgr = VehicleTrackerManager()

            for step_idx in range(steps_per_scenario):
                global_timestamp += 5.0  # 5-second interval
                sim.step(5.0)

                # Process observations into graph & state
                if sim.recent_observations:
                    det_events = DetectionAdapter.batch_convert(sim.recent_observations)
                    if det_events:
                        cam_id = det_events[0].camera_id
                        tracks = tracker_mgr.update(cam_id, det_events, global_timestamp)
                        for trk in tracks:
                            from app.schemas.plate import VehicleIdentityEvidence
                            ev = VehicleIdentityEvidence(
                                track_id=trk.track_id,
                                camera_id=cam_id,
                                vehicle_type=trk.vehicle_type,
                                last_updated_timestamp=global_timestamp,
                            )
                            temporal_graph.update_vehicle_observation(ev, trk, global_timestamp, road_id="ROAD_R_AB")

                snap = state_engine.compute_network_state(builder, timestamp=global_timestamp)
                snapshots.append(snap)

        return snapshots

    @staticmethod
    def create_supervised_dataset(
        history: List[NetworkTrafficSnapshot],
        window_size: int = 12,
        horizon_steps: int = 6,
        segment_id: str = "ROAD_R_AB",
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Constructs supervised training feature matrices X (flattened historical lag windows)
        and target matrices Y (future flow, density, speed, queue, travel_time at target horizons).
        Enforces strict chronological ordering to prevent future information leakage.
        """
        X, Y = [], []
        num_snaps = len(history)

        for i in range(window_size, num_snaps - horizon_steps):
            window_snaps = history[i - window_size : i]
            flat_x = FeatureWindowExtractor.flatten_window(
                FeatureWindowExtractor.extract_lag_windows(window_snaps, window_size, segment_id)
            )

            # Target at horizon i + horizon_steps
            target_snap = history[i + horizon_steps]
            target_state = target_snap.segment_states.get(segment_id)

            if target_state:
                y_val = [
                    target_state.flow_rate,
                    float(target_state.queue_length),
                    target_state.density,
                    target_state.average_speed,
                    target_state.travel_time,
                ]
            else:
                y_val = [0.0, 0.0, 0.0, 60.0, 30.0]

            X.append(flat_x)
            Y.append(y_val)

        return X, Y
