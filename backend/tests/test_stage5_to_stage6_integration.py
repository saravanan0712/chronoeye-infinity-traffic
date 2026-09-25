"""
ChronoEye Infinity - Stage 5 to Stage 6 Integration Test Suite
Verifies the smallest Stage 5 -> Stage 6 integration:
1. JourneyReconstructionEngine outputs VehicleJourney -> TemporalTrafficGraphEngine updates graph nodes & edges.
2. Cross-camera transitions create VEHICLE_OBSERVED_BY_CAMERA and VEHICLE_TRANSITIONS_TO_CAMERA edges.
3. Graph state converts to DynamicTrafficStateEngine network snapshot.
4. TrafficForecastingEngine consumes the snapshot and produces ST-GNN forecasts.
5. Insufficient/empty historical data is handled gracefully (FC_EMPTY).
6. Stage 5 standalone operation remains 100% backward-compatible.
"""

import math
import os
import sys
import unittest
from typing import List, Tuple

# Add backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.detection import BoundingBoxXYXY
from app.schemas.tracking import TrackState, TrackStatus, Direction
from app.schemas.plate import FusedPlateIdentity, VehicleIdentityEvidence
from app.schemas.reid import ReIDConfig, VehicleJourney
from app.perception.camera_topology import CityCameraTopology
from app.perception.reid_engine import ReIDMatchingEngine
from app.perception.journey import JourneyReconstructionEngine
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.graph.temporal_graph import TemporalTrafficGraphEngine
from app.graph.graph_schema import NodeType, EdgeType
from app.state.traffic_state_engine import DynamicTrafficStateEngine
from app.state.traffic_state_schema import NetworkTrafficSnapshot
from app.forecasting.forecasting_engine import TrafficForecastingEngine
from app.forecasting.forecasting_schema import ForecastHorizon, ForecastingModelType, NetworkForecastSnapshot


class TestStage5ToStage6Integration(unittest.TestCase):
    """
    Focused integration tests proving Stage 5 VehicleJourney -> Stage 6 TemporalTrafficGraphEngine & Forecasting.
    """

    def setUp(self):
        self.topology = CityCameraTopology()
        self.config = ReIDConfig()
        self.matching_engine = ReIDMatchingEngine(topology=self.topology, config=self.config)
        self.builder = SpatioTemporalGraphBuilder()
        self.temporal_graph = TemporalTrafficGraphEngine(builder=self.builder)
        self.journey_engine = JourneyReconstructionEngine(
            matching_engine=self.matching_engine,
            config=self.config,
            temporal_graph_engine=self.temporal_graph,
        )

        # Fixed deterministic 128-D normalized embedding vector
        val = 1.0 / math.sqrt(128)
        self.fixed_embedding: List[float] = [val] * 128

        self.state_engine = DynamicTrafficStateEngine()
        self.forecasting_engine = TrafficForecastingEngine()

    def _create_observation(
        self,
        camera_id: str,
        track_id: str,
        timestamp: float,
        plate: str = "TN09AB1234",
        vehicle_type: str = "car",
        speed: float = 48.0,
        direction: Direction = Direction.EAST,
    ) -> Tuple[VehicleIdentityEvidence, TrackState]:
        """Creates a synthetic track and associated plate evidence."""
        track = TrackState(
            track_id=track_id,
            camera_id=camera_id,
            vehicle_type=vehicle_type,
            current_bbox=BoundingBoxXYXY(x1=150.0, y1=200.0, x2=350.0, y2=400.0),
            current_center=(250.0, 300.0),
            confidence=0.95,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
            status=TrackStatus.CONFIRMED,
            confirmed=True,
            direction=direction,
            speed_estimate=speed,
        )

        fused_plate = FusedPlateIdentity(
            track_id=track_id,
            camera_id=camera_id,
            best_plate_number=plate,
            overall_confidence=0.95,
            observation_count=3,
            confirmed=True,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
        )

        evidence = VehicleIdentityEvidence(
            track_id=track_id,
            camera_id=camera_id,
            vehicle_type=vehicle_type,
            associated_plate=fused_plate,
            last_updated_timestamp=timestamp,
        )

        self.journey_engine.track_embeddings[track_id] = (timestamp, self.fixed_embedding)
        return evidence, track

    def test_01_stage5_journey_automatically_updates_stage6_graph(self):
        """
        Verify that when JourneyReconstructionEngine processes track evidence,
        it automatically updates the Spatio-Temporal Graph with vehicle nodes and camera edges.
        """
        # 1. First observation on Camera A
        ev_a, trk_a = self._create_observation("CAM_A_EAST", "TRK_A_01", 10.0, "TN09AB1234", speed=50.0)
        journey_a = self.journey_engine.process_track_evidence(ev_a, trk_a)

        self.assertIsNotNone(journey_a)
        g_veh_id = journey_a.global_vehicle_id

        # Graph node check
        self.assertIn(g_veh_id, self.builder.graph.nodes)
        node_data = self.builder.graph.nodes[g_veh_id]["data"]
        self.assertEqual(node_data["vehicle_id"], g_veh_id)
        self.assertEqual(node_data["plate_number"], "TN09AB1234")
        self.assertEqual(node_data["observation_count"], 1)

        # Camera observation edge check
        self.assertTrue(self.builder.graph.has_edge(g_veh_id, "CAM_A_EAST"))

        # 2. Subsequent observation on Camera B (Cross-camera match)
        ev_b, trk_b = self._create_observation("CAM_B_WEST", "TRK_B_01", 25.0, "TN09AB1234", speed=45.0)
        journey_b = self.journey_engine.process_track_evidence(ev_b, trk_b)

        # Journey was merged
        self.assertEqual(journey_b.global_vehicle_id, g_veh_id)
        self.assertEqual(len(journey_b.segments), 2)

        # Graph node updated
        updated_node_data = self.builder.graph.nodes[g_veh_id]["data"]
        self.assertEqual(updated_node_data["observation_count"], 2)
        self.assertEqual(updated_node_data["last_seen"], 25.0)

        # Check Camera observation edge for Camera B
        self.assertTrue(self.builder.graph.has_edge(g_veh_id, "CAM_B_WEST"))

        # Check Camera-to-Camera transition edge
        self.assertTrue(self.builder.graph.has_edge("CAM_A_EAST", "CAM_B_WEST"))

        # Check edge attributes on transition
        edge_data = self.builder.graph.get_edge_data("CAM_A_EAST", "CAM_B_WEST")
        self.assertIsNotNone(edge_data)
        # Find the VEHICLE_TRANSITIONS_TO_CAMERA edge
        transition_found = False
        for k, v in edge_data.items():
            if v.get("edge_type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA.value:
                transition_found = True
                self.assertEqual(v.get("travel_time"), 15.0)
                self.assertEqual(v.get("data", {}).get("metadata", {}).get("global_vehicle_id"), g_veh_id)
        self.assertTrue(transition_found, "VEHICLE_TRANSITIONS_TO_CAMERA edge was not created")

    def test_02_graph_state_to_dynamic_traffic_state_and_forecasting(self):
        """
        Verify that the graph populated by Stage 5 journeys can be converted to
        NetworkTrafficSnapshot and consumed by TrafficForecastingEngine for ST-GNN forecasting.
        """
        # Populate journey
        ev, trk = self._create_observation("CAM_A_EAST", "TRK_A_02", 30.0, "KA01MJ9999", speed=55.0)
        self.journey_engine.process_track_evidence(ev, trk)

        # Convert graph to NetworkTrafficSnapshot
        net_snapshot = self.state_engine.compute_network_state(self.builder, timestamp=30.0)
        self.assertIsInstance(net_snapshot, NetworkTrafficSnapshot)
        self.assertIn("ROAD_R_AB", net_snapshot.segment_states)

        # Forecast using ST-GNN
        forecast = self.forecasting_engine.forecast_network(
            history=[net_snapshot],
            horizon=ForecastHorizon.PLUS_5MIN,
            model_type=ForecastingModelType.ST_GNN,
            builder=self.builder,
        )

        self.assertIsInstance(forecast, NetworkForecastSnapshot)
        self.assertEqual(forecast.horizon, ForecastHorizon.PLUS_5MIN)
        self.assertEqual(forecast.model_type, ForecastingModelType.ST_GNN)
        self.assertIn("ROAD_R_AB", forecast.segment_forecasts)

        seg_fc = forecast.segment_forecasts["ROAD_R_AB"]
        self.assertGreaterEqual(seg_fc.predicted_speed, 0.0)
        self.assertGreaterEqual(seg_fc.predicted_flow, 0.0)
        self.assertGreaterEqual(seg_fc.predicted_congestion, 0.0)
        self.assertIsNotNone(seg_fc.provenance)

    def test_03_insufficient_or_empty_history_handled_gracefully(self):
        """
        Verify that when history is empty or insufficient, the forecasting engine
        returns an empty forecast snapshot (FC_EMPTY) gracefully without throwing exceptions.
        """
        # Case A: Empty history list
        fc_empty = self.forecasting_engine.forecast_network(history=[])
        self.assertIsInstance(fc_empty, NetworkForecastSnapshot)
        self.assertEqual(fc_empty.snapshot_id, "FC_EMPTY")
        self.assertEqual(len(fc_empty.segment_forecasts), 0)

        # Case B: None history
        fc_none = self.forecasting_engine.forecast_network(history=None)
        self.assertIsInstance(fc_none, NetworkForecastSnapshot)
        self.assertEqual(fc_none.snapshot_id, "FC_EMPTY")
        self.assertEqual(len(fc_none.segment_forecasts), 0)

        # Case C: Single snapshot with empty segment_states
        empty_snap = NetworkTrafficSnapshot(
            snapshot_id="SNAP_EMPTY",
            timestamp=100.0,
            segment_states={},
        )
        fc_empty_snap = self.forecasting_engine.forecast_network(history=[empty_snap])
        self.assertEqual(fc_empty_snap.snapshot_id, "FC_EMPTY")
        self.assertEqual(len(fc_empty_snap.segment_forecasts), 0)

    def test_04_stage5_standalone_backward_compatibility(self):
        """
        Verify that JourneyReconstructionEngine without temporal_graph_engine
        works perfectly as a standalone Stage 5 component without any regressions.
        """
        standalone_engine = JourneyReconstructionEngine(
            matching_engine=self.matching_engine,
            config=self.config,
            temporal_graph_engine=None,
        )
        self.assertIsNone(standalone_engine.temporal_graph_engine)

        ev, trk = self._create_observation("CAM_A_EAST", "TRK_STANDALONE", 10.0, "TN01AA0001")
        journey = standalone_engine.process_track_evidence(ev, trk)

        self.assertIsNotNone(journey)
        self.assertEqual(journey.plate_number, "TN01AA0001")
        self.assertEqual(len(journey.segments), 1)
        self.assertEqual(journey.status, "ACTIVE")


if __name__ == "__main__":
    unittest.main()
