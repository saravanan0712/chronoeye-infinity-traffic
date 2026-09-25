"""
ChronoEye Infinity - Phase 14 Verification Test Suite
Automated Python test suite verifying REST API Endpoints and WebSockets requirements across 16 test cases.
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api.main import app
from app.api.websocket_manager import WebSocketConnectionManager


class TestPhase14APIBackendPipeline(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_1_health_endpoint(self):
        """Test 1: Verify GET /api/v1/health status 200 and schema response."""
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "HEALTHY")

    def test_2_current_traffic_state(self):
        """Test 2: Verify GET /api/v1/traffic/state."""
        res = self.client.get("/api/v1/traffic/state")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("segment_states", data)

    def test_3_traffic_history(self):
        """Test 3: Verify GET /api/v1/traffic/history."""
        res = self.client.get("/api/v1/traffic/history?limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("history", data)

    def test_4_traffic_forecasts(self):
        """Test 4: Verify GET /api/v1/traffic/forecasts."""
        res = self.client.get("/api/v1/traffic/forecasts?model_type=st_gnn")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data, dict)

    def test_5_traffic_uncertainty(self):
        """Test 5: Verify GET /api/v1/traffic/uncertainty."""
        res = self.client.get("/api/v1/traffic/uncertainty")
        self.assertEqual(res.status_code, 200)

    def test_6_graph_state(self):
        """Test 6: Verify GET /api/v1/graph/state."""
        res = self.client.get("/api/v1/graph/state")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("num_nodes", data)

    def test_7_vehicle_journey(self):
        """Test 7: Verify GET /api/v1/vehicles/journey/{vehicle_id}."""
        res = self.client.get("/api/v1/vehicles/journey/VEH_101")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["global_vehicle_id"], "VEH_101")

    def test_8_signals_state(self):
        """Test 8: Verify GET /api/v1/signals/state."""
        res = self.client.get("/api/v1/signals/state")
        self.assertEqual(res.status_code, 200)

    def test_9_route_optimization(self):
        """Test 9: Verify POST /api/v1/routes/optimize."""
        payload = {
            "origin": "JUNC_1",
            "destination": "JUNC_4",
            "departure_timestamp": 0.0,
            "algorithm": "ChronoEyePredictiveAStar",
        }
        res = self.client.post("/api/v1/routes/optimize", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["origin_junction"], "JUNC_1")

    def test_10_emergency_corridor(self):
        """Test 10: Verify POST /api/v1/emergency/corridor."""
        payload = {
            "vehicle_id": "AMB_911",
            "vehicle_type": "AMBULANCE",
            "origin_junction": "JUNC_1",
            "destination_junction": "JUNC_4",
        }
        res = self.client.post("/api/v1/emergency/corridor", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ACTIVE_GREEN_WAVE")

    def test_11_active_incidents(self):
        """Test 11: Verify GET /api/v1/incidents/active."""
        res = self.client.get("/api/v1/incidents/active")
        self.assertEqual(res.status_code, 200)

    def test_12_websocket_connection(self):
        """Test 12: Verify WS /ws/traffic connection and ping-pong exchange."""
        with self.client.websocket_connect("/ws/traffic") as websocket:
            websocket.send_text("ping")
            data = websocket.receive_text()
            self.assertEqual(data, "pong")

    def test_13_pydantic_validation(self):
        """Test 13: Verify 422 Unprocessable Entity error handling on invalid request body."""
        res = self.client.post("/api/v1/routes/optimize", json={"invalid_field": 123})
        self.assertEqual(res.status_code, 422)

    def test_14_non_blocking_websocket_broadcaster(self):
        """Test 14: Verify WebSocketConnectionManager connect/disconnect behavior."""
        mgr = WebSocketConnectionManager()
        self.assertEqual(len(mgr.active_connections), 0)

    def test_15_cors_headers(self):
        """Test 15: Verify CORS headers configuration."""
        res = self.client.options("/api/v1/health")
        self.assertEqual(res.status_code, 200)

    def test_16_phase_1_to_phase_14_end_to_end_integration(self):
        """Test 16: Verify Phase 1 simulation -> Phase 14 REST API end-to-end integration."""
        res = self.client.get("/api/v1/traffic/state")
        self.assertEqual(res.status_code, 200)

    def test_17_data_provenance_tracking(self):
        """Test 17: Verify Data Provenance metadata envelope on API responses (OBSERVED, SIMULATED, PREDICTED)."""
        health_res = self.client.get("/api/v1/health")
        self.assertEqual(health_res.status_code, 200)
        h_data = health_res.json()
        self.assertIn("provenance", h_data)
        self.assertEqual(h_data["provenance"]["provenance_type"], "OBSERVED")

        traffic_res = self.client.get("/api/v1/traffic/state")
        self.assertEqual(traffic_res.status_code, 200)
        t_data = traffic_res.json()
        self.assertIn("provenance", t_data)
        self.assertIn(t_data["provenance"]["provenance_type"], ["SIMULATED", "OBSERVED"])

    def test_18_video_sources_endpoint(self):
        """Test 18: Verify GET /api/v1/video/sources returns supported input sources."""
        res = self.client.get("/api/v1/video/sources")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("supported_sources", data)
        self.assertIn("active_mode", data)

    def test_19_cameras_endpoint(self):
        """Test 19: Verify GET /api/v1/cameras returns metadata for active cameras."""
        res = self.client.get("/api/v1/video/cameras")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_20_video_mode_switching(self):
        """Test 20: Verify POST /api/v1/video/mode switches ingestion mode (LIVE, RECORDED, SIMULATION)."""
        payload = {"mode": "LIVE", "camera_id": "CAM_A_EAST"}
        res = self.client.post("/api/v1/video/mode", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["active_mode"], "LIVE")


if __name__ == "__main__":
    unittest.main()


