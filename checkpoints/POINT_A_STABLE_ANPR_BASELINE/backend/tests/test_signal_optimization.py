"""
ChronoEye Infinity - Phase 10 Verification Test Suite
Automated Python test suite verifying Traffic Signal Control Optimization requirements across 17 test cases.
"""

import os
import sys
import unittest

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.optimization.signal_schema import (
    SignalPhase,
    SignalState,
    IntersectionState,
    SignalOptimizationPlan,
)
from app.optimization.objective import SignalObjectiveFunction
from app.optimization.fixed_controller import FixedController
from app.optimization.reactive_controller import ReactiveController
from app.optimization.predictive_controller import ChronoEyePredictiveController
from app.optimization.signal_optimizer import TrafficSignalOptimizationEngine


class TestPhase10SignalOptimizationPipeline(unittest.TestCase):

    def setUp(self):
        self.engine = TrafficSignalOptimizationEngine()
        self.signal_state = SignalState(signal_id="SIG_01", current_phase=SignalPhase.PHASE_NORTH_SOUTH)
        self.inter_state = IntersectionState(
            intersection_id="JUNC_A", queue_ns=10, queue_ew=10, flow_ns=300.0, flow_ew=300.0
        )

    def test_1_initialization(self):
        """Test 1: Verify TrafficSignalOptimizationEngine initialization."""
        self.assertIsNotNone(self.engine.fixed_controller)
        self.assertIsNotNone(self.engine.reactive_controller)
        self.assertIsNotNone(self.engine.predictive_controller)

    def test_2_safety_min_green_constraint(self):
        """Test 2: Verify enforcement of minimum green clearance safety constraint."""
        state = SignalState(signal_id="SIG_01", current_phase=SignalPhase.PHASE_NORTH_SOUTH, elapsed_green=5.0, min_green=10.0)
        plan = self.engine.fixed_controller.decide_signal_plan(state, self.inter_state)
        self.assertFalse(plan.is_phase_switch)
        self.assertEqual(plan.recommended_phase, SignalPhase.PHASE_NORTH_SOUTH)

    def test_3_safety_max_green_constraint(self):
        """Test 3: Verify enforcement of maximum green clearance safety constraint."""
        state = SignalState(signal_id="SIG_01", current_phase=SignalPhase.PHASE_NORTH_SOUTH, elapsed_green=60.0, max_green=60.0)
        plan = self.engine.reactive_controller.decide_signal_plan(state, self.inter_state)
        self.assertTrue(plan.is_phase_switch)
        self.assertEqual(plan.recommended_phase, SignalPhase.PHASE_EAST_WEST)

    def test_4_safety_yellow_clearance(self):
        """Test 4: Verify yellow and all-red clearance interval specifications."""
        state = SignalState(signal_id="SIG_01")
        self.assertEqual(state.yellow_time, 3.0)
        self.assertEqual(state.all_red_time, 2.0)

    def test_5_fixed_controller_decision(self):
        """Test 5: Verify FixedController pre-timed phase transitions."""
        state = SignalState(signal_id="SIG_01", current_phase=SignalPhase.PHASE_NORTH_SOUTH, elapsed_green=30.0)
        plan = self.engine.fixed_controller.decide_signal_plan(state, self.inter_state)
        self.assertTrue(plan.is_phase_switch)
        self.assertEqual(plan.recommended_phase, SignalPhase.PHASE_EAST_WEST)

    def test_6_reactive_controller_decision(self):
        """Test 6: Verify ReactiveController queue-actuated switching."""
        state = SignalState(signal_id="SIG_01", current_phase=SignalPhase.PHASE_NORTH_SOUTH, elapsed_green=15.0)
        imbalanced = IntersectionState(intersection_id="JUNC_A", queue_ns=2, queue_ew=20)
        plan = self.engine.reactive_controller.decide_signal_plan(state, imbalanced)
        self.assertTrue(plan.is_phase_switch)
        self.assertEqual(plan.recommended_phase, SignalPhase.PHASE_EAST_WEST)

    def test_7_predictive_controller_decision(self):
        """Test 7: Verify ChronoEyePredictiveController rolling-horizon plan decision."""
        state = SignalState(signal_id="SIG_01", current_phase=SignalPhase.PHASE_NORTH_SOUTH, elapsed_green=15.0)
        plan = self.engine.predictive_controller.decide_signal_plan(
            state, self.inter_state, forecast_queue_ns=5.0, forecast_queue_ew=25.0
        )
        self.assertIsNotNone(plan.recommended_phase)
        self.assertGreater(plan.recommended_green_duration, 0.0)

    def test_8_objective_function_evaluation(self):
        """Test 8: Verify SignalObjectiveFunction scalar cost evaluation."""
        obj = SignalObjectiveFunction()
        cost = obj.evaluate_cost(self.signal_state, self.inter_state, SignalPhase.PHASE_NORTH_SOUTH, 30.0)
        self.assertIsInstance(cost, float)

    def test_9_scenario_a_low_traffic(self):
        """Test 9: Verify Scenario A (Low traffic conditions)."""
        inter = IntersectionState(intersection_id="JUNC_A", queue_ns=2, queue_ew=2, flow_ns=100.0, flow_ew=100.0)
        res = self.engine.run_scenario_experiment("Scenario_A_Low_Traffic", num_steps=20, initial_inter_state=inter)
        self.assertIn("controllers", res)

    def test_10_scenario_b_heavy_traffic(self):
        """Test 10: Verify Scenario B (Heavy traffic conditions)."""
        inter = IntersectionState(intersection_id="JUNC_A", queue_ns=25, queue_ew=25, flow_ns=800.0, flow_ew=800.0)
        res = self.engine.run_scenario_experiment("Scenario_B_Heavy_Traffic", num_steps=20, initial_inter_state=inter)
        self.assertIn("controllers", res)

    def test_11_scenario_c_sudden_demand_increase(self):
        """Test 11: Verify Scenario C (Sudden traffic demand spike)."""
        inter = IntersectionState(intersection_id="JUNC_A", queue_ns=5, queue_ew=35, flow_ns=200.0, flow_ew=1200.0)
        res = self.engine.run_scenario_experiment("Scenario_C_Demand_Spike", num_steps=20, initial_inter_state=inter)
        self.assertIn("controllers", res)

    def test_12_scenario_d_unequal_directional_demand(self):
        """Test 12: Verify Scenario D (Unequal directional demand imbalance)."""
        inter = IntersectionState(intersection_id="JUNC_A", queue_ns=30, queue_ew=5, flow_ns=900.0, flow_ew=100.0)
        res = self.engine.run_scenario_experiment("Scenario_D_Directional_Imbalance", num_steps=20, initial_inter_state=inter)
        self.assertIn("controllers", res)

    def test_13_scenario_e_incident_queue(self):
        """Test 13: Verify Scenario E (Incident-induced queue congestion)."""
        inter = IntersectionState(intersection_id="JUNC_A", queue_ns=40, queue_ew=10, flow_ns=1000.0, flow_ew=200.0)
        res = self.engine.run_scenario_experiment("Scenario_E_Incident_Queue", num_steps=20, initial_inter_state=inter)
        self.assertIn("controllers", res)

    def test_14_scenario_f_emergency_vehicle(self):
        """Test 14: Verify Scenario F (Emergency vehicle green corridor priority override)."""
        inter = IntersectionState(
            intersection_id="JUNC_A", queue_ns=10, queue_ew=10, flow_ns=300.0, flow_ew=300.0, has_emergency_vehicle_ns=True
        )
        state = SignalState(signal_id="SIG_01", current_phase=SignalPhase.PHASE_EAST_WEST, elapsed_green=15.0)
        plan = self.engine.predictive_controller.decide_signal_plan(state, inter)
        self.assertTrue(plan.is_phase_switch)
        self.assertEqual(plan.recommended_phase, SignalPhase.PHASE_NORTH_SOUTH)

    def test_15_performance_metrics_measurement(self):
        """Test 15: Verify delay, queue, throughput, and travel time metric calculation."""
        res = self.engine.run_scenario_experiment("Metrics_Test", num_steps=10)
        pred_res = res["controllers"]["ChronoEye_Predictive"]
        self.assertIn("average_delay", pred_res)
        self.assertIn("maximum_queue", pred_res)
        self.assertIn("average_queue", pred_res)
        self.assertIn("throughput", pred_res)
        self.assertIn("travel_time", pred_res)

    def test_16_phase_8_9_forecast_uncertainty_integration(self):
        """Test 16: Verify Phase 8/9 ST-GNN forecast & uncertainty integration with signal controller."""
        state = SignalState(signal_id="SIG_01", current_phase=SignalPhase.PHASE_NORTH_SOUTH, elapsed_green=15.0)
        plan = self.engine.predictive_controller.decide_signal_plan(
            state, self.inter_state, forecast_queue_ns=10.0, forecast_queue_ew=20.0, uncertainty_std_ns=2.0, uncertainty_std_ew=4.0
        )
        self.assertIsNotNone(plan)

    def test_17_phase_1_to_phase_10_end_to_end_integration(self):
        """Test 17: Verify Phase 1 simulation -> Phase 10 signal control end-to-end integration."""
        res = self.engine.run_scenario_experiment("End_to_End", num_steps=30)
        self.assertIn("controllers", res)


if __name__ == "__main__":
    unittest.main()
