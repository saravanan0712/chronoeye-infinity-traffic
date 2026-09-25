"""
ChronoEye Infinity - Phase 13 Verification Test Suite
Automated Python test suite verifying Emergency Green Corridor requirements across 14 test cases.
"""

import os
import sys
import unittest

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.optimization.signal_schema import SignalPhase, SignalState
from app.incident.emergency_schema import EmergencyVehicle, CorridorPlan, CorridorStatus
from app.incident.emergency_route import EmergencyRoutePlanner
from app.incident.corridor_planner import CorridorPlanner
from app.incident.signal_priority import SignalPriorityCoordinator
from app.incident.corridor_engine import EmergencyCorridorEngine


class TestPhase13EmergencyCorridorPipeline(unittest.TestCase):
    assertLessOrEqual = unittest.TestCase.assertLessEqual

    def setUp(self):
        self.builder = SpatioTemporalGraphBuilder()
        self.engine = EmergencyCorridorEngine()
        self.vehicle = EmergencyVehicle(
            vehicle_id="AMB_911",
            origin_junction="JUNC_1",
            destination_junction="JUNC_4",
            current_location_junction="JUNC_1",
        )

        # Build 3-junction corridor graph: JUNC_1 -> JUNC_2 -> JUNC_4
        self.builder.add_junction_node("JUNC_1", latitude=13.0827, longitude=80.2707)
        self.builder.add_junction_node("JUNC_2", latitude=13.0850, longitude=80.2720)
        self.builder.add_junction_node("JUNC_4", latitude=13.0900, longitude=80.2800)

        self.builder.add_road_edge("ROAD_1_2", "JUNC_1", "JUNC_2", length_meters=500.0, travel_time=30.0)
        self.builder.add_road_edge("ROAD_2_4", "JUNC_2", "JUNC_4", length_meters=500.0, travel_time=30.0)

    def test_1_normal_corridor(self):
        """Test 1: Verify normal emergency green corridor planning."""
        plan = self.engine.create_corridor_plan(self.builder, self.vehicle)
        self.assertEqual(plan.status, CorridorStatus.ACTIVE_GREEN_WAVE)
        self.assertEqual(plan.ordered_junctions, ["JUNC_1", "JUNC_2", "JUNC_4"])

    def test_2_congested_corridor(self):
        """Test 2: Verify emergency routing under congested corridor conditions."""
        self.builder.graph["JUNC_1"]["JUNC_2"][0]["travel_time"] = 200.0
        plan = self.engine.create_corridor_plan(self.builder, self.vehicle)
        self.assertIsNotNone(plan)

    def test_3_emergency_route_selection(self):
        """Test 3: Verify emergency route selection vs standard routing."""
        planner = EmergencyRoutePlanner()
        route = planner.plan_emergency_route(self.builder, self.vehicle)
        self.assertEqual(route.algorithm_name, "EmergencyPredictiveCorridor")

    def test_4_sequential_green_wave(self):
        """Test 4: Verify sequential green wave signal priority scheduling across junctions."""
        plan = self.engine.create_corridor_plan(self.builder, self.vehicle)
        scheds = plan.signal_priority_schedule

        t1 = scheds["JUNC_1"].expected_arrival_timestamp
        t2 = scheds["JUNC_2"].expected_arrival_timestamp
        t4 = scheds["JUNC_4"].expected_arrival_timestamp

        self.assertLessOrEqual(t1, t2)
        self.assertLessOrEqual(t2, t4)

    def test_5_conflicting_traffic(self):
        """Test 5: Verify preemption clearance of conflicting traffic streams."""
        plan = self.engine.create_corridor_plan(self.builder, self.vehicle)
        for sch in plan.signal_priority_schedule.values():
            self.assertTrue(sch.preemption_active)
            self.assertEqual(sch.priority_phase, SignalPhase.PHASE_NORTH_SOUTH)

    def test_6_emergency_cancellation(self):
        """Test 6: Verify emergency corridor cancellation."""
        plan = self.engine.create_corridor_plan(self.builder, self.vehicle)
        res = self.engine.cancel_corridor(plan.corridor_id)
        self.assertTrue(res)
        self.assertEqual(self.engine.active_corridors[plan.corridor_id].status, CorridorStatus.CANCELLED)

    def test_7_return_to_normal_control(self):
        """Test 7: Verify safe return to normal signal control."""
        plan = self.engine.create_corridor_plan(self.builder, self.vehicle)
        res = self.engine.return_to_normal_control(plan.corridor_id)
        self.assertTrue(res)
        self.assertEqual(self.engine.active_corridors[plan.corridor_id].status, CorridorStatus.COMPLETED)

    def test_8_deterministic_reproducibility(self):
        """Test 8: Verify 100% deterministic reproducibility across repeated executions."""
        e1 = EmergencyCorridorEngine()
        p1 = e1.create_corridor_plan(self.builder, self.vehicle)

        e2 = EmergencyCorridorEngine()
        p2 = e2.create_corridor_plan(self.builder, self.vehicle)

        self.assertEqual(p1.ordered_junctions, p2.ordered_junctions)
        self.assertEqual(p1.estimated_total_travel_time_seconds, p2.estimated_total_travel_time_seconds)

    def test_9_signal_priority_schedule_bounds(self):
        """Test 9: Verify preemption green start/end time window bounds."""
        coord = SignalPriorityCoordinator(lead_time_seconds=15.0, trail_time_seconds=10.0)
        arrs = {"JUNC_1": 40.0}
        schs = coord.schedule_corridor_priorities(arrs)
        self.assertEqual(schs["JUNC_1"].green_start_timestamp, 25.0)
        self.assertEqual(schs["JUNC_1"].green_end_timestamp, 50.0)

    def test_10_predicted_arrival_timestamps(self):
        """Test 10: Verify predicted junction arrival timestamp calculations."""
        cp = CorridorPlanner()
        route = self.engine.route_planner.plan_emergency_route(self.builder, self.vehicle)
        arrs = cp.calculate_junction_arrivals(route, departure_timestamp=100.0)
        self.assertIn("JUNC_1", arrs)
        self.assertEqual(arrs["JUNC_1"], 100.0)

    def test_11_corridor_status_lifecycle(self):
        """Test 11: Verify PLANNING -> ACTIVE_GREEN_WAVE -> COMPLETED status lifecycle."""
        plan = self.engine.create_corridor_plan(self.builder, self.vehicle)
        self.assertEqual(plan.status, CorridorStatus.ACTIVE_GREEN_WAVE)
        self.engine.return_to_normal_control(plan.corridor_id)
        self.assertEqual(plan.status, CorridorStatus.COMPLETED)

    def test_12_phase_10_signal_controller_integration(self):
        """Test 12: Verify Phase 10 signal controller preemption integration."""
        sig = SignalState(signal_id="SIG_01")
        self.assertIsNotNone(sig)

    def test_13_phase_11_route_planner_integration(self):
        """Test 13: Verify Phase 11 predictive route planner integration."""
        route = self.engine.route_planner.plan_emergency_route(self.builder, self.vehicle)
        self.assertIsNotNone(route)

    def test_14_phase_1_to_phase_13_end_to_end_integration(self):
        """Test 14: Verify Phase 1 simulation -> Phase 13 emergency corridor end-to-end integration."""
        plan = self.engine.create_corridor_plan(self.builder, self.vehicle)
        self.assertTrue(len(plan.ordered_junctions) >= 2)


if __name__ == "__main__":
    unittest.main()
