# ChronoEye Infinity - Research Evaluation Benchmark Suite & Baseline Comparisons

## Executive Summary
This document specifies the research evaluation benchmark suite for **ChronoEye Infinity**, comparing the full predictive spatio-temporal traffic intelligence pipeline against **Baseline A (Fixed Timing)** and **Baseline B (Reactive Control)**.

---

## 1. Experimental Methodology
- **Scenarios**: 12 standard urban traffic scenarios (`free_flow`, `normal_traffic`, `rush_hour`, `demand_surge`, `persistent_congestion`, `road_blockage`, `multi_junction_propagation`, `forecasted_congestion`, `emergency_corridor`, `incident_recovery`, `mixed_traffic_demand`, `overloaded_stress`).
- **Seeds**: Multi-seed deterministic evaluation ($S \in \{42, 101, 202, 303, 404\}$).
- **Simulation Duration**: 20 steps per trial run.

---

## 2. Baselines & Evaluation Modes
1. **Baseline A — Fixed Timing**: Static green phase allocation, no future traffic forecasting, no predictive routing.
2. **Baseline B — Reactive Control**: Actuated signal control responding strictly to current queue observations.
3. **System C — ChronoEye Predictive**: Unified pipeline (State Engine $\to$ ST-GNN Forecasting $\to$ Monte Carlo Uncertainty $\to$ Predictive Signal Optimization $\to$ Time-Dependent Predictive A* Routing $\to$ Incident Intelligence $\to$ Green Wave Emergency Corridor).

---

## 3. Metrics & Statistical Formulae
- **Traffic Metrics**: Mean, median, and 95th percentile travel time, average/max delay, queue length, network throughput (veh/hr), average speed.
- **Percentage Improvement**:
$$\text{Improvement \%} = \frac{\text{Baseline} - \text{ChronoEye}}{\text{Baseline}} \times 100$$

---

## 4. Benchmark Architecture
```text
Scenario Engine
      ↓
Scenario Runner
      ↓
┌────────────────────┬────────────────────┬─────────────────────┐
│ Baseline Fixed     │ Baseline Reactive  │ ChronoEye Predictive│
└────────────────────┴────────────────────┴─────────────────────┘
      ↓
Metrics Engine
      ↓
Statistical Aggregation & 95% Confidence Intervals
      ↓
JSON / CSV Serialization & Research Report Generator
```

---

## 5. Verification Test Suite
The benchmark suite is verified by 21 automated Python tests in `backend/tests/test_research_benchmark.py`.
Total system test count: **265 / 265 PASS**.
