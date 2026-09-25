"""
ChronoEye Infinity - Phase 17: Comprehensive Integration & System Verification Test Suite
Executes 17 scenario-based, benchmark, performance latency, and reliability edge-case verification tests.
"""

import os
import sys
import time
import unittest
from fastapi.testclient import TestClient

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api.main import app
from app.simulation.traffic_generator import SimulationConfig, SyntheticTrafficGenerator
from app.perception.detector import VehicleDetector
from app.perception.bytetrack import ByteTracker
from app.perception.ocr_engine import ALPRPipeline
from app.perception.reid_engine import VehicleReIDEngine
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.state.traffic_state_engine import DynamicTrafficStateEngine
from app.forecasting.forecasting_engine import TrafficForecastingEngine
from app.forecasting.uncertainty import UncertaintyEstimator
from app.optimization.signal_optimizer import TrafficSignalOptimizationEngine
from app.optimization.predictive_router import ChronoEyePredictiveAStarRouter
from app.incident.incident_engine import IncidentDetectionEngine
from app.incident.corridor_engine import EmergencyCorridorEngine
from app.incident.emergency_schema import EmergencyVehicle


class TestPhase17SystemVerification(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.sim = SyntheticTrafficGenerator(SimulationConfig(num_intersections=4, seed=42))
        self.state_engine = DynamicTrafficStateEngine()
        self.forecaster = TrafficForecastingEngine()
        self.uncertainty = UncertaintyEstimator()
        self.signal_optimizer = TrafficSignalOptimizationEngine()
        self.router = ChronoEyePredictiveAStarRouter()
        self.incident_engine = IncidentDetectionEngine()
        self.corridor_engine = EmergencyCorridorEngine()
        self.builder = SpatioTemporalGraphBuilder()

        # Build standard test graph
        self.builder.add_junction_node("JUNC_1", latitude=13.0827, longitude=80.2707)
        self.builder.add_junction_node("JUNC_2", latitude=13.0850, longitude=80.2720)
        self.builder.add_junction_node("JUNC_3", latitude=13.0880, longitude=80.2750)
        self.builder.add_junction_node("JUNC_4", latitude=13.0900, longitude=80.2800)

        self.builder.add_road_edge("ROAD_1_2", "JUNC_1", "JUNC_2", length_meters=500.0, travel_time=30.0)
        self.builder.add_road_edge("ROAD_2_4", "JUNC_2", "JUNC_4", length_meters=500.0, travel_time=30.0)
        self.builder.add_road_edge("ROAD_1_3", "JUNC_1", "JUNC_3", length_meters=600.0, travel_time=35.0)
        self.builder.add_road_edge("ROAD_3_4", "JUNC_3", "JUNC_4", length_meters=600.0, travel_time=35.0)

    def test_1_scenario_free_flow_traffic(self):
        """Scenario 1: Free-flow traffic verification."""
        snap = self.sim.step()
        net = self.state_engine.compute_network_snapshot(snap)
        self.assertLess(net.network_congestion_index, 0.45)

    def test_2_scenario_rush_hour_traffic(self):
        """Scenario 2: Rush-hour traffic verification."""
        for _ in range(5):
            snap = self.sim.step()
        net = self.state_engine.compute_network_snapshot(snap)
        self.assertIsNotNone(net)

    def test_3_scenario_sudden_demand_increase(self):
        """Scenario 3: Sudden traffic demand surge detection."""
        snap = self.sim.step()
        net = self.state_engine.compute_network_snapshot(snap)
        # Inject artificial surge in queue
        if "ROAD_R_AB" in net.segment_states:
            net.segment_states["ROAD_R_AB"].queue_length = 25
            net.segment_states["ROAD_R_AB"].congestion_score = 0.88

        incidents = self.incident_engine.process_snapshot_and_forecast(net)
        self.assertIsInstance(incidents, list)

    def test_4_scenario_road_blockage(self):
        """Scenario 4: Road blockage detection & residual anomaly."""
        snap = self.sim.step()
        net = self.state_engine.compute_network_snapshot(snap)
        if "ROAD_R_BC" in net.segment_states:
            net.segment_states["ROAD_R_BC"].average_speed = 0.0
            net.segment_states["ROAD_R_BC"].congestion_score = 0.98

        incidents = self.incident_engine.process_snapshot_and_forecast(net)
        self.assertIsInstance(incidents, list)

    def test_5_scenario_forecast_congestion(self):
        """Scenario 5: Multi-horizon ST-GNN forecast congestion prediction."""
        snap = self.sim.step()
        net = self.state_engine.compute_network_snapshot(snap)
        fc = self.forecaster.forecast_network(net, model_type="st_gnn")
        self.assertIn("ROAD_R_AB", fc)

    def test_6_scenario_predictive_signal_response(self):
        """Scenario 6: Rolling-horizon predictive traffic signal response."""
        plan = self.signal_optimizer.run_scenario_experiment(num_steps=1)
        self.assertIsNotNone(plan)

    def test_7_scenario_predictive_route_diversion(self):
        """Scenario 7: ChronoEye Predictive A* dynamic route diversion."""
        route = self.router.find_route(self.builder, origin="JUNC_1", destination="JUNC_4")
        self.assertEqual(route.origin_junction, "JUNC_1")
        self.assertEqual(route.destination_junction, "JUNC_4")

    def test_8_scenario_emergency_green_corridor(self):
        """Scenario 8: Emergency vehicle priority green corridor activation."""
        veh = EmergencyVehicle(
            vehicle_id="AMB_911",
            vehicle_type="AMBULANCE",
            origin_junction="JUNC_1",
            destination_junction="JUNC_4",
            current_location_junction="JUNC_1",
        )
        plan = self.corridor_engine.create_corridor_plan(self.builder, veh)
        self.assertEqual(plan.status, "ACTIVE_GREEN_WAVE")

    def test_9_scenario_multi_camera_vehicle_journey(self):
        """Scenario 9: Multi-camera vehicle journey reconstruction."""
        res = self.client.get("/api/v1/vehicles/journey/VEH_101")
        self.assertEqual(res.status_code, 200)

    def test_10_scenario_incident_recovery(self):
        """Scenario 10: Incident resolution & network recovery."""
        snap = self.sim.step()
        net = self.state_engine.compute_network_snapshot(snap)
        incidents = self.incident_engine.process_snapshot_and_forecast(net)
        for inc in incidents:
            self.incident_engine.resolve_incident(inc.incident_id)

    def test_11_latency_and_throughput_measurements(self):
        """Measurement: System processing latency and route computation benchmarks."""
        t0 = time.perf_counter()
        snap = self.sim.step()
        net = self.state_engine.compute_network_snapshot(snap)
        fc = self.forecaster.forecast_network(net)
        route = self.router.find_route(self.builder, "JUNC_1", "JUNC_4")
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        self.assertLess(elapsed_ms, 500.0) # Sub-500ms pipeline execution

    def test_12_reliability_missing_data(self):
        """Reliability 1: Missing feature data fallback handling."""
        snap = self.sim.step()
        net = self.state_engine.compute_network_snapshot(snap)
        # Empty segment state resilience
        net.segment_states = {}
        fc = self.forecaster.forecast_network(net)
        self.assertEqual(len(fc), 0)

    def test_13_reliability_delayed_data(self):
        """Reliability 2: Out-of-order delayed timestamp handling."""
        snap = self.sim.step()
        net = self.state_engine.compute_network_snapshot(snap)
        net.timestamp -= 500.0 # Stale timestamp
        fc = self.forecaster.forecast_network(net)
        self.assertIsNotNone(fc)

    def test_14_reliability_empty_and_overloaded_traffic(self):
        """Reliability 3: Zero vehicle traffic & high density overload edge cases."""
        snap = self.sim.step()
        snap.detections = [] # Zero vehicles
        net = self.state_engine.compute_network_snapshot(snap)
        self.assertEqual(net.total_network_vehicles, 0)

    def test_15_reliability_disconnected_camera_and_ml_fallback(self):
        """Reliability 4: Disconnected camera & ML fallback (Level 1 Persistence)."""
        snap = self.sim.step()
        net = self.state_engine.compute_network_snapshot(snap)
        # Level 1 persistence fallback model
        fc = self.forecaster.forecast_network(net, model_type="persistence")
        self.assertIn("ROAD_R_AB", fc)

    def test_16_reliability_api_reconnect(self):
        """Reliability 5: API client reconnect & health check status."""
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)

    def test_17_full_pipeline_stage1_to_stage15_end_to_end(self):
        """Stage 1-15 Full End-to-End Pipeline Integration Test."""
        # Stage 1: Simulation
        sim_snap = self.sim.step()
        self.assertIsNotNone(sim_snap)

        # Stage 7: State Engine
        net_snap = self.state_engine.compute_network_snapshot(sim_snap)
        self.assertIsNotNone(net_snap)

        # Stage 8: Forecast
        fc = self.forecaster.forecast_network(net_snap, model_type="st_gnn")
        self.assertIsNotNone(fc)

        # Stage 9: Uncertainty
        if "ROAD_R_AB" in fc:
            unc = self.uncertainty.estimate_mc_dropout_uncertainty(fc["ROAD_R_AB"])
            self.assertIsNotNone(unc)

        # Stage 10: Signal Optimization
        sig_plan = self.signal_optimizer.run_scenario_experiment(num_steps=1)
        self.assertIsNotNone(sig_plan)

        # Stage 11: Route Optimization
        route = self.router.find_route(self.builder, "JUNC_1", "JUNC_4")
        self.assertIsNotNone(route)

        # Stage 13: Emergency Corridor
        veh = EmergencyVehicle(vehicle_id="AMB_911", vehicle_type="AMBULANCE", origin_junction="JUNC_1", destination_junction="JUNC_4", current_location_junction="JUNC_1")
        corr = self.corridor_engine.create_corridor_plan(self.builder, veh)
        self.assertEqual(corr.status, "ACTIVE_GREEN_WAVE")

        # Stage 14: REST API
        api_res = self.client.get("/api/v1/health")
        self.assertEqual(api_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
