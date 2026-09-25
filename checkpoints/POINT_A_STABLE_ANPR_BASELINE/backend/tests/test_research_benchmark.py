"""
ChronoEye Infinity - Phase 18 Research Evaluation Benchmark Test Suite
Automated Python test suite verifying 21 test cases across scenario registry, baselines,
metrics calculations, statistical aggregations, multi-seed reproducibility, serialization, and report generation.
"""

import os
import sys
import unittest
import json

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.research.scenarios import SCENARIOS_REGISTRY, get_scenario_config
from app.research.baselines import BaselineAFixedController, BaselineBReactiveController, SystemCChronoEyeController
from app.research.metrics import compute_metrics, compute_improvement_percentage
from app.research.statistics import compute_statistical_aggregates
from app.research.runner import ResearchBenchmarkRunner
from app.research.serializer import serialize_benchmark_results
from app.research.report_generator import generate_research_markdown_report


class TestPhase18ResearchBenchmarkPackage(unittest.TestCase):

    def setUp(self):
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "research"))

    def test_1_scenario_registry(self):
        """Test 1: Verify all 12 scenarios registered."""
        self.assertEqual(len(SCENARIOS_REGISTRY), 12)
        self.assertIn("free_flow", SCENARIOS_REGISTRY)
        self.assertIn("overloaded_stress", SCENARIOS_REGISTRY)

    def test_2_deterministic_scenario_generation(self):
        """Test 2: Deterministic scenario configuration."""
        cfg = get_scenario_config("rush_hour", seed=42, steps=10)
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.duration_steps, 10)

    def test_3_seed_reproducibility(self):
        """Test 3: Seed reproducibility across identical runs."""
        cfg1 = get_scenario_config("normal_traffic", seed=42)
        cfg2 = get_scenario_config("normal_traffic", seed=42)
        c1 = BaselineAFixedController(cfg1)
        c2 = BaselineAFixedController(cfg2)
        r1 = c1.step()
        r2 = c2.step()
        self.assertEqual(r1["avg_delay"], r2["avg_delay"])

    def test_4_fixed_controller_execution(self):
        """Test 4: Baseline A Fixed controller execution."""
        cfg = get_scenario_config("free_flow", seed=42)
        ctrl = BaselineAFixedController(cfg)
        res = ctrl.step()
        self.assertIn("avg_delay", res)

    def test_5_reactive_controller_execution(self):
        """Test 5: Baseline B Reactive controller execution."""
        cfg = get_scenario_config("free_flow", seed=42)
        ctrl = BaselineBReactiveController(cfg)
        res = ctrl.step()
        self.assertIn("avg_delay", res)

    def test_6_chronoeye_controller_execution(self):
        """Test 6: System C ChronoEye controller execution."""
        cfg = get_scenario_config("free_flow", seed=42)
        ctrl = SystemCChronoEyeController(cfg)
        res = ctrl.step()
        self.assertIn("avg_delay", res)

    def test_7_all_12_scenarios_execution(self):
        """Test 7: Execute trial across all 12 scenarios."""
        for sc in SCENARIOS_REGISTRY:
            cfg = get_scenario_config(sc, seed=42, steps=1)
            ctrl = SystemCChronoEyeController(cfg)
            res = ctrl.step()
            self.assertIsNotNone(res)

    def test_8_multi_seed_execution(self):
        """Test 8: Execute multi-seed scenario runner."""
        runner = ResearchBenchmarkRunner(seeds=[42, 101], steps=2)
        results = runner.run_all()
        self.assertIn("raw_records", results)

    def test_9_metric_calculation(self):
        """Test 9: Compute travel and traffic metrics."""
        m = compute_metrics(delays=[10.0, 20.0], travel_times=[30.0, 40.0], queues=[2.0, 4.0], throughputs=[100.0, 200.0])
        self.assertEqual(m["mean_travel_time"], 35.0)

    def test_10_percentile_calculation(self):
        """Test 10: Compute 95th percentile travel time."""
        m = compute_metrics(delays=[10.0], travel_times=[10.0, 20.0, 30.0, 40.0, 50.0], queues=[2.0], throughputs=[100.0])
        self.assertEqual(m["p95_travel_time"], 48.0)

    def test_11_confidence_interval_calculation(self):
        """Test 11: Compute 95% confidence interval bounds."""
        stats = compute_statistical_aggregates([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertLess(stats["ci95_lower"], stats["mean"])

    def test_12_improvement_calculation(self):
        """Test 12: Mathematically valid percentage improvement calculation."""
        imp = compute_improvement_percentage(baseline=100.0, chronoeye=60.0, lower_is_better=True)
        self.assertEqual(imp, 40.0)

    def test_13_throughput_calculation(self):
        """Test 13: Compute throughput metric."""
        m = compute_metrics(delays=[5.0], travel_times=[20.0], queues=[1.0], throughputs=[500.0])
        self.assertEqual(m["throughput"], 500.0)

    def test_14_forecasting_metric_calculation(self):
        """Test 14: Compute forecasting MAE and RMSE."""
        from app.benchmarks.forecasting_metrics import calculate_forecasting_metrics
        fc = calculate_forecasting_metrics([10.0, 20.0], [12.0, 18.0])
        self.assertEqual(fc["mae"], 2.0)

    def test_15_benchmark_aggregation(self):
        """Test 15: Statistical aggregation across trials."""
        stats = compute_statistical_aggregates([15.0, 16.0, 17.0])
        self.assertEqual(stats["mean"], 16.0)

    def test_16_json_serialization(self):
        """Test 16: Serialize results to JSON."""
        res = {"config": {"seeds": [42]}, "raw_records": [{"scenario": "free_flow", "seed": 42}]}
        serialize_benchmark_results(res, output_dir=self.output_dir)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "benchmark_results.json")))

    def test_17_csv_serialization(self):
        """Test 17: Serialize results to CSV."""
        res = {"config": {"seeds": [42]}, "raw_records": [{"scenario": "free_flow", "seed": 42, "controller": "Fixed", "mean_travel_time": 30.0, "avg_delay": 10.0, "avg_queue_length": 2.0, "throughput": 100.0}]}
        serialize_benchmark_results(res, output_dir=self.output_dir)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "benchmark_results.csv")))

    def test_18_report_generation(self):
        """Test 18: Generate markdown report file."""
        res = {"summary": {"fixed": {"avg_delay": {"mean": 42.5}}, "reactive": {"avg_delay": {"mean": 28.0}}, "chronoeye": {"avg_delay": {"mean": 15.8}}}}
        rep = generate_research_markdown_report(res, output_path=os.path.join(self.output_dir, "REPORT_TEST.md"))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "REPORT_TEST.md")))

    def test_19_ablation_execution(self):
        """Test 19: Ablation framework execution."""
        from app.benchmark.benchmark_runner import ResearchBenchmarkRunner as LegacyRunner
        leg = LegacyRunner()
        abl = leg.run_ablation_study()
        self.assertGreaterEqual(len(abl), 6)

    def test_20_complete_benchmark_smoke_test(self):
        """Test 20: Full benchmark execution smoke test."""
        runner = ResearchBenchmarkRunner(seeds=[42], steps=2)
        res = runner.run_all()
        self.assertIn("summary", res)

    def test_21_no_fabricated_missing_metric_handling(self):
        """Test 21: Robust handling of missing / zero denominator metrics."""
        imp = compute_improvement_percentage(baseline=0.0, chronoeye=10.0)
        self.assertEqual(imp, 0.0)


if __name__ == "__main__":
    unittest.main()
