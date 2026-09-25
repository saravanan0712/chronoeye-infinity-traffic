"""
ChronoEye Infinity - Benchmark Multi-Seed Scenario Runner Engine
Orchestrates multi-seed simulation trials for 12 scenarios across Fixed, Reactive, and ChronoEye controllers.
"""

from typing import Dict, List, Any
from app.research.scenarios import SCENARIOS_REGISTRY, get_scenario_config
from app.research.baselines import BaselineAFixedController, BaselineBReactiveController, SystemCChronoEyeController
from app.research.metrics import compute_metrics
from app.research.statistics import compute_statistical_aggregates


class ResearchBenchmarkRunner:
    """Executes empirical multi-seed benchmark scenarios across Fixed, Reactive, and ChronoEye controllers."""

    def __init__(self, seeds: List[int] = None, steps: int = 20):
        self.seeds = seeds or [42, 101, 202, 303, 404]
        self.steps = steps

    def run_all(self) -> Dict[str, Any]:
        raw_records = []
        fixed_delays, reactive_delays, chronoeye_delays = [], [], []

        for name in SCENARIOS_REGISTRY:
            for seed in self.seeds:
                cfg = get_scenario_config(name, seed=seed, steps=self.steps)

                # Fixed
                ctrl_fixed = BaselineAFixedController(cfg)
                res_f = ctrl_fixed.step()
                res_f.update({"scenario": name, "seed": seed, "controller": "Fixed"})
                raw_records.append(res_f)
                fixed_delays.append(res_f["avg_delay"])

                # Reactive
                ctrl_react = BaselineBReactiveController(cfg)
                res_r = ctrl_react.step()
                res_r.update({"scenario": name, "seed": seed, "controller": "Reactive"})
                raw_records.append(res_r)
                reactive_delays.append(res_r["avg_delay"])

                # ChronoEye
                ctrl_ce = SystemCChronoEyeController(cfg)
                res_ce = ctrl_ce.step()
                res_ce.update({"scenario": name, "seed": seed, "controller": "ChronoEye"})
                raw_records.append(res_ce)
                chronoeye_delays.append(res_ce["avg_delay"])

        summary = {
            "fixed": {"avg_delay": compute_statistical_aggregates(fixed_delays)},
            "reactive": {"avg_delay": compute_statistical_aggregates(reactive_delays)},
            "chronoeye": {"avg_delay": compute_statistical_aggregates(chronoeye_delays)},
        }

        return {
            "config": {"seeds": self.seeds, "steps": self.steps, "num_scenarios": len(SCENARIOS_REGISTRY)},
            "raw_records": raw_records,
            "summary": summary,
        }
