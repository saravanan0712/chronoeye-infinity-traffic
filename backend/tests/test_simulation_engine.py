"""
ChronoEye Infinity - Phase 1 Verification Test Suite
Automated Python test runner verifying Phase 1 Traffic Simulation Engine functionality.
"""

import os
import sys
import unittest

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.simulation import (
    SignalMode,
    SignalPhase,
    VehicleType,
    VehicleStatus,
)
from app.simulation.engine import TrafficSimulationEngine


class TestPhase1TrafficSimulationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = TrafficSimulationEngine(seed=42)

    def test_simulation_reset_and_initialization(self):
        """1. Verify initial network topology, junctions, roads, signals, and cameras."""
        snapshot = self.engine.get_snapshot()
        self.assertEqual(snapshot["seed"], 42)
        self.assertEqual(len(snapshot["junctions"]), 4)  # J_A, J_B, J_C, J_D
        self.assertEqual(len(snapshot["roads"]), 8)      # Bi-directional segments
        self.assertEqual(len(snapshot["signals"]), 4)    # SIG_J_A .. SIG_J_D
        self.assertEqual(len(snapshot["cameras"]), 4)    # CAM_A_EAST .. CAM_D_NORTH
        self.assertEqual(snapshot["active_vehicle_count"], 0)
        self.assertEqual(snapshot["simulation_time"], 0.0)

    def test_vehicle_spawning_and_movement(self):
        """2. Verify vehicle generation, kinematic advancement, and status updates."""
        # Step simulation for 30 seconds
        for _ in range(30):
            snapshot = self.engine.step(dt_seconds=1.0)

        self.assertGreater(snapshot["active_vehicle_count"], 0)
        self.assertGreater(snapshot["simulation_time"], 0.0)

        # Check at least one vehicle has moved along its distance_on_road
        active_vehs = snapshot["vehicles"]
        has_moving_vehicle = any(v["speed_kmh"] > 0.0 for v in active_vehs)
        self.assertTrue(has_moving_vehicle)

    def test_signal_mode_switching_and_transitions(self):
        """3. Verify signal modes (FIXED, REACTIVE, PREDICTIVE, EMERGENCY)."""
        self.engine.set_signal_mode(SignalMode.REACTIVE)
        for sig in self.engine.topology.signals.values():
            self.assertEqual(sig.mode, SignalMode.REACTIVE)

        # Run step to update timers
        self.engine.step(dt_seconds=10.0)

        # Test EMERGENCY mode override
        self.engine.signal_manager.set_emergency_corridor(["J_A", "J_B"], active=True)
        sig_a = self.engine.topology.signals["SIG_J_A"]
        self.assertEqual(sig_a.current_phase, SignalPhase.EMERGENCY_OVERRIDE)

    def test_camera_fov_captures(self):
        """4. Verify camera FOV sensors capture observations with bounding boxes."""
        # Force spawn vehicle near camera on R_AB
        route = ["R_AB", "R_BD"]
        v = self.engine.generator.create_vehicle(
            road_id="R_AB",
            destination_junction_id="J_D",
            route=route,
            timestamp=self.engine.sim_time,
        )
        v.distance_on_road = 60.0  # Inside CAM_A_EAST FOV range (50-110m)
        self.engine.vehicles[v.vehicle_id] = v

        snapshot = self.engine.step(dt_seconds=1.0)
        obs_list = snapshot["recent_camera_observations"]

        self.assertGreater(len(obs_list), 0)
        cam_obs = obs_list[0]
        self.assertEqual(cam_obs["camera_id"], "CAM_A_EAST")
        self.assertEqual(cam_obs["plate_number"], v.plate_number)
        self.assertIn("bbox", cam_obs)
        self.assertGreater(cam_obs["detection_confidence"], 0.9)

    def test_incident_injection_and_metrics(self):
        """5. Verify incident injection creates blockage and updates congestion metrics."""
        self.engine.inject_incident("R_AB", severity=0.9)
        road_ab = self.engine.topology.roads["R_AB"]
        self.assertTrue(road_ab.is_blocked)
        self.assertEqual(road_ab.blockage_severity, 0.9)

        metrics = self.engine.compute_traffic_metrics()
        r_ab_metric = next(m for m in metrics if m.road_id == "R_AB")
        self.assertGreaterEqual(r_ab_metric.congestion_score, 0.45)

        self.engine.resolve_incident("R_AB")
        self.assertFalse(road_ab.is_blocked)

    def test_emergency_vehicle_dispatch_and_green_wave(self):
        """6. Verify emergency vehicle dispatching activates green wave corridor."""
        emg = self.engine.dispatch_emergency_vehicle(
            origin_road_id="R_AB",
            destination_junction_id="J_D",
            route=["R_AB", "R_BD"],
        )
        self.assertEqual(emg.vehicle_type, VehicleType.EMERGENCY)
        self.assertEqual(emg.plate_number, "EMG9999")

        sig_b = self.engine.topology.signals["SIG_J_B"]
        sig_d = self.engine.topology.signals["SIG_J_D"]
        self.assertEqual(sig_b.current_phase, SignalPhase.EMERGENCY_OVERRIDE)
        self.assertEqual(sig_d.current_phase, SignalPhase.EMERGENCY_OVERRIDE)

    def test_reproducibility_with_seed(self):
        """7. Verify deterministic reproducibility given identical seeds."""
        engine1 = TrafficSimulationEngine(seed=123)
        for _ in range(20):
            engine1.step(1.0)
        snap1 = engine1.get_snapshot()

        engine2 = TrafficSimulationEngine(seed=123)
        for _ in range(20):
            engine2.step(1.0)
        snap2 = engine2.get_snapshot()

        self.assertEqual(snap1["active_vehicle_count"], snap2["active_vehicle_count"])
        self.assertEqual(
            [v["vehicle_id"] for v in snap1["vehicles"]],
            [v["vehicle_id"] for v in snap2["vehicles"]],
        )


if __name__ == "__main__":
    unittest.main()
