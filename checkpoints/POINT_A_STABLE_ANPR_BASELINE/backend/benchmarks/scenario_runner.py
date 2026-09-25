"""
ChronoEye Infinity - Benchmark Scenario Runner
Orchestrates scenarios across multi-seed runs for Fixed, Reactive, and ChronoEye runners.
"""

import os
import json
from typing import Dict, List, Any
from app.benchmarks.benchmark_config import BenchmarkConfig
from app.benchmarks.baseline_fixed import BaselineFixedRunner
from app.benchmarks.baseline_reactive import BaselineReactiveRunner
from app.benchmarks.chronoeye_runner import ChronoEyeRunner
from app.benchmarks.statistics import compute_summary_statistics
from app.benchmarks.metrics import calculate_percentage_improvement


class ScenarioRunner:
    """Orchestrates scenario evaluation across Fixed, Reactive, and ChronoEye systems."""

    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def run_benchmark_suite((self) -> Dict[str, Any]:
        """Executes complete multi-scenario, multi-seed benchmark evaluation."""
        suite_results = {
            "config": {
                "scenarios": self.config.scenarios,
                "seeds": self.config.seeds,
                "duration_steps": self.config.simulation_duration_steps,
            },
            "fixed": [],
            "reactive": [],
            "chronoeye": [],
            "comparisons": {},
        }

        fixed_delays = []
        reactive_delays = []
        chronoeye_delays = []

        for seed in self.config.seeds:
            # Fixed
            fixed_runner = BaselineFixedRunner(seed=seed, num_intersections=self.config.num_intersections)
            m_fixed = fixed_runner.run_scenario(steps=self.config.simulation_duration_steps)
            suite_results["fixed"].append(m_fixed)
            fixed_delays.append(m_fixed["avg_delay"])

            # Reactive
            react_runner = BaselineReactiveRunner(seed=seed, num_intersections=self.config.num_intersections)
            m_react = react_runner.run_scenario(steps=self.config.simulation_duration_steps)
            suite_results["reactive"].append(m_react)
            reactive_delays.append(m_react["avg_delay"])

            # ChronoEye
            ce_runner = ChronoEyeRunner(seed=seed, num_intersections=self.config.num_intersections)
            m_ce = ce_runner.run_scenario(steps=self.config.simulation_duration_steps)
            suite_results["chronoeye"].append(m_ce)
            chronoeye_delays.append(m_ce["avg_delay"])

        # Compute summary statistics
        stat_fixed = compute_summary_statistics(fixed_delays)
        stat_react = compute_summary_statistics(reactive_delays)
        stat_ce = compute_summary_statistics(chronoeye_delays)

        imp_vs_fixed = calculate_percentage_improvement(stat_fixed["mean"], stat_ce["mean"], lower_is_better=True)
        imp_vs_react = calculate_percentage_improvement(stat_react["mean"], stat_ce["mean"], lower_is_better=True)

        suite_results["comparisons"] = {
            "fixed_stats": stat_fixed,
            "reactive_stats": stat_react,
            "chronoeye_stats": stat_ce,
            "improvement_vs_fixed_pct": imp_vs_fixed,
            "improvement_vs_reactive_pct": imp_vs_react,
        }

        # Save JSON output
        output_file = os.path.join(self.config.output_dir, "benchmark_execution_results.json")
        with open(output_file, "w") as f:
            json.dump(suite_results, f, indent=2)

        return suite_results
