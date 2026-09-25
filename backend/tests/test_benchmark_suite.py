"""
ChronoEye Infinity - Phase 18 Research Benchmark Test Suite
Automated Python test suite verifying benchmark experiments, CSV exports, ablation studies, and research findings.
"""

import os
import sys
import unittest

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.benchmark.benchmark_runner import ResearchBenchmarkRunner


class TestPhase18ResearchBenchmarkSuite(unittest.TestCase):

    def setUp(self):
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "benchmark", "results"))
        self.runner = ResearchBenchmarkRunner(output_dir=self.output_dir)

    def test_1_forecasting_benchmark_results(self):
        """Test 1: Verify ST-GNN forecasting model outperforming baselines."""
        res = self.runner.run_forecasting_benchmark()
        st_gnn_5m = next(r for r in res if r["model"] == "ChronoEye ST-GNN" and r["horizon"] == "+5m")
        ma_5m = next(r for r in res if r["model"] == "Moving Average" and r["horizon"] == "+5m")
        self.assertLess(st_gnn_5m["mae"], ma_5m["mae"])

    def test_2_signal_benchmark_results(self):
        """Test 2: Verify ChronoEye Predictive signal control delay reduction."""
        res = self.runner.run_signal_benchmark()
        pred = next(r for r in res if r["controller"] == "ChronoEye Predictive")
        fixed = next(r for r in res if r["controller"] == "Fixed Timing")
        self.assertLess(pred["avg_delay_s"], fixed["avg_delay_s"])

    def test_3_routing_benchmark_results(self):
        """Test 3: Verify ChronoEye Predictive A* travel time speedup."""
        res = self.runner.run_routing_benchmark()
        pred = next(r for r in res if r["algorithm"] == "ChronoEye Predictive A*")
        dijk = next(r for r in res if r["algorithm"] == "Dijkstra")
        self.assertLess(pred["travel_time_s"], dijk["travel_time_s"])

    def test_4_incident_benchmark_results(self):
        """Test 4: Verify residual incident detection precision and F1-score."""
        res = self.runner.run_incident_benchmark()
        f1 = next(r for r in res if r["metric"] == "F1-Score")["value"]
        self.assertGreater(f1, 0.90)

    def test_5_uncertainty_benchmark_results(self):
        """Test 5: Verify Monte Carlo Dropout predictive uncertainty bounds."""
        res = self.runner.run_uncertainty_benchmark()
        cov90 = next(r for r in res if r["interval_level"] == "90%")["coverage_pct"]
        self.assertGreater(cov90, 85.0)

    def test_6_emergency_benchmark_results(self):
        """Test 6: Verify Emergency Green Corridor travel time reduction."""
        res = self.runner.run_emergency_benchmark()
        corr = next(r for r in res if r["operation_mode"] == "ChronoEye Green Wave Corridor")
        norm = next(r for r in res if r["operation_mode"] == "Normal Signal Operation")
        self.assertLess(corr["emergency_travel_time_s"], norm["emergency_travel_time_s"])

    def test_7_ablation_study_results(self):
        """Test 7: Verify component-wise removal ablation performance drop."""
        res = self.runner.run_ablation_study()
        full = next(r for r in res if r["ablation_configuration"] == "Full System (Complete Pipeline)")
        no_gnn = next(r for r in res if r["ablation_configuration"] == "Without ST-GNN Forecasting")
        self.assertLess(full["prediction_mae"], no_gnn["prediction_mae"])

    def test_8_export_csv_data_files(self):
        """Test 8: Verify CSV data export creation."""
        self.runner.export_all_results_to_csv()
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "benchmark_summary.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "forecasting_results.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "signal_results.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "routing_results.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "incident_results.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "uncertainty_results.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "ablation_results.csv")))


if __name__ == "__main__":
    unittest.main()
