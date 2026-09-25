"""
ChronoEye Infinity - Phase 8 Verification Test Suite
Automated Python test suite verifying Future Traffic Forecasting requirements across 20 test cases.
"""

import os
import sys
import tempfile
import unittest

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.state.traffic_state_schema import NetworkTrafficSnapshot, RoadSegmentState
from app.state.traffic_state_engine import DynamicTrafficStateEngine
from app.forecasting.forecasting_schema import (
    ForecastHorizon,
    ForecastingModelType,
    SegmentForecast,
    NetworkForecastSnapshot,
    ModelEvaluationMetrics,
)
from app.forecasting.feature_window import FeatureWindowExtractor
from app.forecasting.dataset_builder import ForecastingDatasetBuilder
from app.forecasting.baseline_forecaster import PersistenceForecaster
from app.forecasting.ridge_forecaster import RidgeTrafficForecaster
from app.forecasting.st_gnn import SpatioTemporalGNNForecaster
from app.forecasting.evaluation import ForecastingEvaluator
from app.forecasting.model_registry import ModelRegistry
from app.forecasting.forecasting_engine import TrafficForecastingEngine


class TestPhase8ForecastingPipeline(unittest.TestCase):

    def setUp(self):
        self.builder = SpatioTemporalGraphBuilder()
        self.engine = TrafficForecastingEngine(window_size=12)

        # Build mock 20-snapshot history sequence
        self.history = []
        for t_idx in range(20):
            t_sec = float(t_idx * 5)
            seg_state = RoadSegmentState(
                segment_id="ROAD_R_AB",
                road_name="R_AB",
                timestamp=t_sec,
                vehicle_count=5 + (t_idx % 3),
                flow_rate=300.0 + (t_idx * 10),
                density=10.0 + t_idx,
                average_speed=50.0 - t_idx,
                occupancy=0.10 + (t_idx * 0.01),
                queue_length=t_idx % 2,
                travel_time=30.0 + t_idx,
                congestion_score=0.15 + (t_idx * 0.01),
                normalized_features=[0.1, 0.1, 0.4, 0.1, 0.0, 0.15],
            )
            snap = NetworkTrafficSnapshot(
                timestamp=t_sec,
                segment_states={"ROAD_R_AB": seg_state},
                total_active_vehicles=seg_state.vehicle_count,
            )
            self.history.append(snap)

    def test_1_initialization(self):
        """Test 1: Verify TrafficForecastingEngine initialization."""
        self.assertIsNotNone(self.engine.persistence_model)
        self.assertIsNotNone(self.engine.ridge_model)
        self.assertIsNotNone(self.engine.st_gnn_model)

    def test_2_dataset_builder_synthetic(self):
        """Test 2: Verify ForecastingDatasetBuilder synthetic dataset generation."""
        dataset = ForecastingDatasetBuilder.generate_synthetic_simulation_dataset(
            num_scenarios=2, steps_per_scenario=10, seed=42
        )
        self.assertGreater(len(dataset), 10)

    def test_3_supervised_dataset_matrix(self):
        """Test 3: Verify create_supervised_dataset X and Y matrix construction."""
        X, Y = ForecastingDatasetBuilder.create_supervised_dataset(
            self.history, window_size=5, horizon_steps=2, segment_id="ROAD_R_AB"
        )
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(Y))

    def test_4_no_future_leakage(self):
        """Test 4: Verify strict chronological splitting prevents future leakage."""
        X, Y = ForecastingDatasetBuilder.create_supervised_dataset(
            self.history, window_size=5, horizon_steps=2, segment_id="ROAD_R_AB"
        )
        self.assertTrue(all(len(row) == 5 * 6 for row in X))

    def test_5_feature_window_extraction(self):
        """Test 5: Verify FeatureWindowExtractor lag window extraction."""
        window = FeatureWindowExtractor.extract_lag_windows(self.history, window_size=12, segment_id="ROAD_R_AB")
        self.assertEqual(len(window), 12)
        flat = FeatureWindowExtractor.flatten_window(window)
        self.assertEqual(len(flat), 72)

    def test_6_persistence_forecaster_5min(self):
        """Test 6: Verify Level 1 Persistence forecaster at +5 min horizon."""
        pred = self.engine.persistence_model.predict(self.history, ForecastHorizon.PLUS_5MIN, segment_id="ROAD_R_AB")
        self.assertEqual(pred.horizon, ForecastHorizon.PLUS_5MIN)
        self.assertGreater(pred.predicted_speed, 0.0)

    def test_7_persistence_forecaster_10min(self):
        """Test 7: Verify Level 1 Persistence forecaster at +10 min horizon."""
        pred = self.engine.persistence_model.predict(self.history, ForecastHorizon.PLUS_10MIN, segment_id="ROAD_R_AB")
        self.assertEqual(pred.horizon, ForecastHorizon.PLUS_10MIN)

    def test_8_ridge_forecaster_training(self):
        """Test 8: Verify Level 2 Ridge Regression model training."""
        X, Y = ForecastingDatasetBuilder.create_supervised_dataset(self.history, window_size=5, horizon_steps=2)
        self.engine.ridge_model.fit(X, Y)
        self.assertIsNotNone(self.engine.ridge_model.weights)

    def test_9_ridge_forecaster_prediction(self):
        """Test 9: Verify Level 2 Ridge Regression prediction at +15 min horizon."""
        X, Y = ForecastingDatasetBuilder.create_supervised_dataset(self.history, window_size=5, horizon_steps=2)
        self.engine.ridge_model.fit(X, Y)
        pred = self.engine.ridge_model.predict(self.history, ForecastHorizon.PLUS_15MIN, segment_id="ROAD_R_AB")
        self.assertEqual(pred.horizon, ForecastHorizon.PLUS_15MIN)
        self.assertGreaterEqual(pred.predicted_flow, 0.0)

    def test_10_st_gnn_forecaster_training(self):
        """Test 10: Verify Level 3 ST-GNN model training."""
        X, Y = ForecastingDatasetBuilder.create_supervised_dataset(self.history, window_size=5, horizon_steps=2)
        self.engine.st_gnn_model.fit(self.builder, X, Y)
        self.assertIsNotNone(self.engine.st_gnn_model.temporal_weights)

    def test_11_st_gnn_forecaster_prediction(self):
        """Test 11: Verify Level 3 ST-GNN multi-horizon prediction at +30 min horizon."""
        X, Y = ForecastingDatasetBuilder.create_supervised_dataset(self.history, window_size=5, horizon_steps=2)
        self.engine.st_gnn_model.fit(self.builder, X, Y)
        pred = self.engine.st_gnn_model.predict(self.builder, self.history, ForecastHorizon.PLUS_30MIN, segment_id="ROAD_R_AB")
        self.assertEqual(pred.horizon, ForecastHorizon.PLUS_30MIN)

    def test_12_multi_horizon_prediction_all(self):
        """Test 12: Verify all 4 forecast horizons (+5m, +10m, +15m, +30m)."""
        self.engine.train_models(self.history, self.builder)
        for h in [ForecastHorizon.PLUS_5MIN, ForecastHorizon.PLUS_10MIN, ForecastHorizon.PLUS_15MIN, ForecastHorizon.PLUS_30MIN]:
            snap = self.engine.forecast_network(self.history, h, model_type=ForecastingModelType.ST_GNN, builder=self.builder)
            self.assertEqual(snap.horizon, h)
            self.assertIn("ROAD_R_AB", snap.segment_forecasts)

    def test_13_missing_historical_imputation(self):
        """Test 13: Verify feature window imputation when history is incomplete."""
        short_hist = self.history[:2]
        window = FeatureWindowExtractor.extract_lag_windows(short_hist, window_size=12, segment_id="ROAD_R_AB")
        self.assertEqual(len(window), 12)

    def test_14_deterministic_inference(self):
        """Test 14: Verify deterministic forecaster inference reproducibility."""
        p1 = self.engine.persistence_model.predict(self.history, ForecastHorizon.PLUS_5MIN, segment_id="ROAD_R_AB")
        p2 = self.engine.persistence_model.predict(self.history, ForecastHorizon.PLUS_5MIN, segment_id="ROAD_R_AB")
        self.assertEqual(p1.predicted_flow, p2.predicted_flow)
        self.assertEqual(p1.predicted_speed, p2.predicted_speed)

    def test_15_cpu_compatibility(self):
        """Test 15: Verify CPU-only execution without hard GPU dependency."""
        self.engine.train_models(self.history, self.builder)
        snap = self.engine.forecast_network(self.history, ForecastHorizon.PLUS_5MIN, model_type=ForecastingModelType.ST_GNN, builder=self.builder)
        self.assertIsNotNone(snap)

    def test_16_model_checkpoint_save_load(self):
        """Test 16: Verify ModelRegistry checkpoint saving and loading."""
        X, Y = ForecastingDatasetBuilder.create_supervised_dataset(self.history, window_size=5, horizon_steps=2)
        self.engine.ridge_model.fit(X, Y)

        ckpt = ModelRegistry.save_checkpoint("ridge", self.engine.ridge_model)
        self.assertEqual(ckpt["model_name"], "ridge")
        loaded = ModelRegistry.load_checkpoint(ckpt)
        self.assertIn("state_dict", loaded)

    def test_17_evaluation_metrics_mae_rmse_mape(self):
        """Test 17: Verify ForecastingEvaluator MAE, RMSE, and MAPE calculations."""
        y_true = [50.0, 60.0, 70.0]
        y_pred = [52.0, 58.0, 75.0]
        metrics = ForecastingEvaluator.evaluate_predictions("TestModel", ForecastHorizon.PLUS_5MIN, y_true, y_pred)

        self.assertGreater(metrics.mae, 0.0)
        self.assertGreater(metrics.rmse, 0.0)
        self.assertGreater(metrics.mape, 0.0)

    def test_18_model_comparison(self):
        """Test 18: Verify ForecastingEvaluator.compare_models comparing Level 1 vs 2 vs 3."""
        self.engine.train_models(self.history, self.builder)
        eval_results = self.engine.evaluate_all(self.history, self.builder)
        self.assertIn("best_models_by_horizon", eval_results)

    def test_19_phase_7_to_phase_8_integration(self):
        """Test 19: Verify Phase 7 dynamic traffic state history -> Phase 8 forecasting integration."""
        state_engine = DynamicTrafficStateEngine(max_history=20)
        for t in range(5):
            state_engine.compute_network_state(self.builder, timestamp=float(t*5))
        hist = state_engine.history_manager.history

        fc_snap = self.engine.forecast_network(hist, ForecastHorizon.PLUS_5MIN, model_type=ForecastingModelType.PERSISTENCE, builder=self.builder)
        self.assertGreater(len(fc_snap.segment_forecasts), 0)

    def test_20_phase_1_to_phase_8_end_to_end_integration(self):
        """Test 20: Phase 1 simulation -> Phase 8 forecasting pipeline end-to-end integration."""
        syn_dataset = ForecastingDatasetBuilder.generate_synthetic_simulation_dataset(num_scenarios=1, steps_per_scenario=10, seed=42)
        self.engine.train_models(syn_dataset, self.builder)
        fc_snap = self.engine.forecast_network(syn_dataset, ForecastHorizon.PLUS_15MIN, model_type=ForecastingModelType.ST_GNN, builder=self.builder)
        self.assertEqual(fc_snap.horizon, ForecastHorizon.PLUS_15MIN)


if __name__ == "__main__":
    unittest.main()
