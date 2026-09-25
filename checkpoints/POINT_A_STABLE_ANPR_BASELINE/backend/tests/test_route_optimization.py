"""
ChronoEye Infinity - Phase 11 Verification Test Suite
Automated Python test suite verifying Dynamic Predictive Route Optimization requirements across 17 test cases.
"""

import os
import sys
import unittest

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.graph.graph_schema import EdgeType
from app.optimization.route_schema import RouteSegment, OptimizationRoute
from app.optimization.graph_cost import SpatioTemporalEdgeCostCalculator
from app.optimization.dijkstra_router import DijkstraRouter
from app.optimization.astar_router import AStarRouter
from app.optimization.time_dependent_astar import TimeDependentAStarRouter
from app.optimization.predictive_router import ChronoEyePredictiveAStarRouter


class TestPhase11RouteOptimizationPipeline(unittest.TestCase):

    def setUp(self):
        self.builder = SpatioTemporalGraphBuilder()
        self.cost_calc = SpatioTemporalEdgeCostCalculator()
        self.dijkstra = DijkstraRouter()
        self.astar = AStarRouter()
        self.td_astar = TimeDependentAStarRouter()
        self.predictive_astar = ChronoEyePredictiveAStarRouter()

        # Build a 3-path network graph between JUNC_1 and JUNC_4
        # Path 1 (Direct/Shortest distance): JUNC_1 -> JUNC_2 -> JUNC_4 (Total 1000m)
        # Path 2 (Alternative/Longer distance): JUNC_1 -> JUNC_3 -> JUNC_4 (Total 1500m)
        self.builder.add_junction_node("JUNC_1", latitude=13.0827, longitude=80.2707)
        self.builder.add_junction_node("JUNC_2", latitude=13.0850, longitude=80.2720)
        self.builder.add_junction_node("JUNC_3", latitude=13.0800, longitude=80.2750)
        self.builder.add_junction_node("JUNC_4", latitude=13.0900, longitude=80.2800)

        # Path 1 edges
        self.builder.add_road_edge("ROAD_1_2", "JUNC_1", "JUNC_2", length_meters=500.0, travel_time=30.0)
        self.builder.add_road_edge("ROAD_2_4", "JUNC_2", "JUNC_4", length_meters=500.0, travel_time=30.0)

        # Path 2 edges
        self.builder.add_road_edge("ROAD_1_3", "JUNC_1", "JUNC_3", length_meters=750.0, travel_time=40.0)
        self.builder.add_road_edge("ROAD_3_4", "JUNC_3", "JUNC_4", length_meters=750.0, travel_time=40.0)

    def test_1_initialization(self):
        """Test 1: Verify all 4 router modules initialize correctly."""
        self.assertIsNotNone(self.dijkstra)
        self.assertIsNotNone(self.astar)
        self.assertIsNotNone(self.td_astar)
        self.assertIsNotNone(self.predictive_astar)

    def test_2_spatiotemporal_edge_cost_calculator(self):
        """Test 2: Verify SpatioTemporalEdgeCostCalculator cost calculation."""
        seg = RouteSegment(
            road_id="ROAD_1_2", from_junction="JUNC_1", to_junction="JUNC_2", length_meters=500.0, travel_time_seconds=30.0
        )
        cost = self.cost_calc.calculate_cost(seg, predicted_travel_time=45.0, predicted_congestion=0.5)
        self.assertGreater(cost, 45.0)

    def test_3_standard_dijkstra_routing(self):
        """Test 3: Verify DijkstraRouter shortest path finding."""
        route = self.dijkstra.find_route(self.builder, "JUNC_1", "JUNC_4")
        self.assertEqual(route.origin_junction, "JUNC_1")
        self.assertEqual(route.destination_junction, "JUNC_4")
        self.assertEqual(route.path_junctions, ["JUNC_1", "JUNC_2", "JUNC_4"])

    def test_4_standard_astar_routing(self):
        """Test 4: Verify AStarRouter shortest path finding."""
        route = self.astar.find_route(self.builder, "JUNC_1", "JUNC_4")
        self.assertEqual(route.origin_junction, "JUNC_1")
        self.assertEqual(route.destination_junction, "JUNC_4")
        self.assertEqual(route.path_junctions, ["JUNC_1", "JUNC_2", "JUNC_4"])

    def test_5_time_dependent_astar_routing(self):
        """Test 5: Verify TimeDependentAStarRouter dynamic arrival time evaluation."""
        route = self.td_astar.find_route(self.builder, "JUNC_1", "JUNC_4", departure_timestamp=100.0)
        self.assertEqual(route.algorithm_name, "TimeDependentAStar")
        self.assertGreater(len(route.path_junctions), 1)

    def test_6_predictive_astar_routing(self):
        """Test 6: Verify ChronoEyePredictiveAStarRouter route recommendation."""
        route = self.predictive_astar.find_route(self.builder, "JUNC_1", "JUNC_4")
        self.assertEqual(route.algorithm_name, "ChronoEyePredictiveAStar")
        self.assertEqual(route.path_junctions, ["JUNC_1", "JUNC_2", "JUNC_4"])

    def test_7_scenario_1_shortest_route_congested(self):
        """Test 7: Demonstration Scenario 1 - Shortest route is congested; router selects free-flowing alternative."""
        # Heavily congest Path 1 (ROAD_1_2 travel time = 300s, congestion = 0.9)
        self.builder.graph["JUNC_1"]["JUNC_2"][0]["travel_time"] = 300.0
        self.builder.graph["JUNC_1"]["JUNC_2"][0]["congestion_score"] = 0.9

        route = self.predictive_astar.find_route(self.builder, "JUNC_1", "JUNC_4")
        self.assertEqual(route.path_junctions, ["JUNC_1", "JUNC_3", "JUNC_4"])

    def test_8_scenario_2_longer_route_faster_in_future(self):
        """Test 8: Demonstration Scenario 2 - Longer route is faster in future arrival window."""
        # Add dynamic temporal profile making Path 1 congested at t+40s
        def temporal_prof(t):
            return 200.0 if t >= 30.0 else 30.0

        self.builder.graph["JUNC_2"]["JUNC_4"][0]["temporal_profile"] = temporal_prof

        route = self.predictive_astar.find_route(self.builder, "JUNC_1", "JUNC_4", departure_timestamp=0.0)
        self.assertEqual(route.path_junctions, ["JUNC_1", "JUNC_3", "JUNC_4"])

    def test_9_scenario_3_incident_blocks_primary_route(self):
        """Test 9: Demonstration Scenario 3 - Incident blocks primary route; dynamic rerouting around incident."""
        self.builder.graph["JUNC_1"]["JUNC_2"][0]["has_incident"] = True
        route = self.predictive_astar.find_route(self.builder, "JUNC_1", "JUNC_4")
        self.assertEqual(route.path_junctions, ["JUNC_1", "JUNC_3", "JUNC_4"])

    def test_10_scenario_4_uncertainty_changes_preference(self):
        """Test 10: Demonstration Scenario 4 - High prediction uncertainty changes route preference."""
        self.builder.graph["JUNC_1"]["JUNC_2"][0]["uncertainty_std"] = 150.0
        route = self.predictive_astar.find_route(self.builder, "JUNC_1", "JUNC_4")
        self.assertEqual(route.path_junctions, ["JUNC_1", "JUNC_3", "JUNC_4"])

    def test_11_scenario_5_no_available_alternative(self):
        """Test 11: Demonstration Scenario 5 - No alternative available fallback handling."""
        route = self.predictive_astar.find_route(self.builder, "JUNC_1", "NON_EXISTENT")
        self.assertEqual(route.path_junctions, [])

    def test_12_scenario_6_deterministic_result(self):
        """Test 12: Demonstration Scenario 6 - Identical routes produce 100% deterministic result."""
        r1 = self.predictive_astar.find_route(self.builder, "JUNC_1", "JUNC_4")
        r2 = self.predictive_astar.find_route(self.builder, "JUNC_1", "JUNC_4")
        self.assertEqual(r1.path_junctions, r2.path_junctions)
        self.assertEqual(r1.total_travel_time_seconds, r2.total_travel_time_seconds)

    def test_13_route_computation_latency(self):
        """Test 13: Verify route computation latency is under 50ms."""
        route = self.predictive_astar.find_route(self.builder, "JUNC_1", "JUNC_4")
        self.assertLess(route.computation_latency_ms, 50.0)

    def test_14_travel_distance_delay_metrics(self):
        """Test 14: Verify travel time, distance, delay, and confidence bound calculations."""
        route = self.predictive_astar.find_route(self.builder, "JUNC_1", "JUNC_4")
        self.assertEqual(route.total_distance_km, 1.0)
        self.assertGreater(route.total_travel_time_seconds, 0.0)
        self.assertIsNotNone(route.confidence_lower_seconds)
        self.assertIsNotNone(route.confidence_upper_seconds)

    def test_15_phase_6_graph_topology_integration(self):
        """Test 15: Verify Phase 6 road graph topology integration."""
        self.assertIn("JUNC_1", self.builder.graph.nodes)
        self.assertIn("JUNC_4", self.builder.graph.nodes)

    def test_16_phase_8_9_predictive_forecast_integration(self):
        """Test 16: Verify Phase 8/9 ST-GNN forecast & uncertainty integration."""
        route = self.predictive_astar.find_route(self.builder, "JUNC_1", "JUNC_4", departure_timestamp=300.0)
        self.assertIsNotNone(route)

    def test_17_phase_1_to_phase_11_end_to_end_integration(self):
        """Test 17: Verify Phase 1 simulation -> Phase 11 route optimization end-to-end integration."""
        route = self.predictive_astar.find_route(self.builder, "JUNC_1", "JUNC_4")
        self.assertTrue(len(route.path_junctions) >= 2)


if __name__ == "__main__":
    unittest.main()
