"""
ChronoEye Infinity - Phase 9 Verification Test Suite
Automated Python test suite verifying Uncertainty Estimation Engine requirements across 16 test cases.
"""

import os
import sys
import unittest

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.state.traffic_state_schema import NetworkTrafficSnapshot, RoadSegmentState
from app.forecasting.forecasting_schema import ForecastHorizon, SegmentForecast
from app.forecasting.st_gnn import SpatioTemporalGNNForecaster
from app.forecasting.forecasting_engine import TrafficForecastingEngine
from app.forecasting.prediction_interval import PredictionIntervalCalculator
from app.forecasting.calibration import UncertaintyCalibrator
from app.forecasting.uncertainty import UncertaintyEstimator, UncertainPrediction


class TestPhase9UncertaintyPipeline(unittest.TestCase):

    def setUp(self):
        self.builder = SpatioTemporalGraphBuilder()
        self.estimator = UncertaintyEstimator(num_mc_samples=20, dropout_rate=0.1)
        self.fc_engine = TrafficForecastingEngine(window_size=12)

        # Mock 15-snapshot history
        self.history = []
        for t in range(15):
            seg = RoadSegmentState(
                segment_id="ROAD_R_AB",
                road_name="R_AB",
                timestamp=float(t * 5),
                vehicle_count=5,
                flow_rate=300.0,
                density=10.0,
                average_speed=50.0,
                occupancy=0.1,
                queue_length=0,
                travel_time=30.0,
                congestion_score=0.1,
                normalized_features=[0.1, 0.1, 0.4, 0.1, 0.0, 0.1],
            )
            snap = NetworkTrafficSnapshot(timestamp=float(t * 5), segment_states={"ROAD_R_AB": seg})
            self.history.append(snap)

    def test_1_initialization(self):
        """Test 1: Verify UncertaintyEstimator initialization."""
        self.assertEqual(self.estimator.num_mc_samples, 20)
        self.assertEqual(self.estimator.dropout_rate, 0.1)

    def test_2_monte_carlo_dropout_sampling(self):
        """Test 2: Verify Monte Carlo Dropout sampling across multiple forward passes."""
        unc = self.estimator.estimate_mc_dropout_uncertainty(
            self.fc_engine.st_gnn_model, self.builder, self.history, ForecastHorizon.PLUS_5MIN, segment_id="ROAD_R_AB"
        )
        self.assertEqual(unc.num_samples, 20)
        self.assertEqual(unc.estimation_method, "MonteCarloDropout")

    def test_3_predictive_mean_calculation(self):
        """Test 3: Verify predictive mean calculation."""
        unc = self.estimator.estimate_mc_dropout_uncertainty(
            self.fc_engine.st_gnn_model, self.builder, self.history, ForecastHorizon.PLUS_5MIN, segment_id="ROAD_R_AB"
        )
        self.assertGreater(unc.mean, 0.0)

    def test_4_predictive_std_variance(self):
        """Test 4: Verify predictive standard deviation calculation."""
        unc = self.estimator.estimate_mc_dropout_uncertainty(
            self.fc_engine.st_gnn_model, self.builder, self.history, ForecastHorizon.PLUS_5MIN, segment_id="ROAD_R_AB"
        )
        self.assertGreaterEqual(unc.std, 0.0)

    def test_5_prediction_interval_80pct(self):
        """Test 5: Verify 80% prediction interval calculation (z = 1.282)."""
        low, high = PredictionIntervalCalculator.calculate_interval(mean=50.0, std=5.0, confidence_level=0.80)
        self.assertAlmostEqual(low, 50.0 - 1.282 * 5.0, delta=0.1)

    def test_6_prediction_interval_90pct(self):
        """Test 6: Verify 90% prediction interval calculation (z = 1.645)."""
        low, high = PredictionIntervalCalculator.calculate_interval(mean=50.0, std=5.0, confidence_level=0.90)
        self.assertAlmostEqual(low, 50.0 - 1.645 * 5.0, delta=0.1)

    def test_7_prediction_interval_95pct(self):
        """Test 7: Verify 95% prediction interval calculation (z = 1.960)."""
        low, high = PredictionIntervalCalculator.calculate_interval(mean=50.0, std=5.0, confidence_level=0.95)
        self.assertAlmostEqual(low, 50.0 - 1.960 * 5.0, delta=0.1)

    def test_8_physical_min_bound_enforcement(self):
        """Test 8: Verify non-negative physical lower bound constraint (lower >= 0.0)."""
        low, high = PredictionIntervalCalculator.calculate_interval(mean=2.0, std=10.0, confidence_level=0.95, min_bound=0.0)
        self.assertEqual(low, 0.0)

    def test_9_empirical_coverage_picp(self):
        """Test 9: Verify UncertaintyCalibrator PICP calculation."""
        y_true = [45.0, 50.0, 55.0]
        lows = [40.0, 45.0, 50.0]
        highs = [60.0, 60.0, 60.0]
        picp = UncertaintyCalibrator.calculate_picp(y_true, lows, highs)
        self.assertEqual(picp, 1.0)

    def test_10_mean_interval_width_mpiw(self):
        """Test 10: Verify UncertaintyCalibrator MPIW calculation."""
        lows = [40.0, 45.0]
        highs = [60.0, 55.0]
        mpiw = UncertaintyCalibrator.calculate_mpiw(lows, highs)
        self.assertEqual(mpiw, 15.0)

    def test_11_calibration_error_ece(self):
        """Test 11: Verify UncertaintyCalibrator ECE calculation."""
        ece = UncertaintyCalibrator.calculate_ece([50.0, 60.0], [52.0, 58.0], [2.0, 2.0])
        self.assertEqual(ece, 0.0)

    def test_12_deterministic_testing_mode(self):
        """Test 12: Verify deterministic testing mode (zero variance)."""
        unc = self.estimator.estimate_mc_dropout_uncertainty(
            self.fc_engine.st_gnn_model, self.builder, self.history, ForecastHorizon.PLUS_5MIN, segment_id="ROAD_R_AB", deterministic=True
        )
        self.assertEqual(unc.std, 0.0)
        self.assertEqual(unc.confidence_score, 1.0)

    def test_13_deep_ensemble_interface(self):
        """Test 13: Verify estimate_ensemble_uncertainty Deep Ensemble interface."""
        m1 = SpatioTemporalGNNForecaster()
        m2 = SpatioTemporalGNNForecaster()
        unc = self.estimator.estimate_ensemble_uncertainty([m1, m2], self.builder, self.history, ForecastHorizon.PLUS_10MIN)
        self.assertEqual(unc.estimation_method, "DeepEnsemble")

    def test_14_no_future_leakage(self):
        """Test 14: Verify zero future information leakage in uncertainty estimation."""
        unc = self.estimator.estimate_mc_dropout_uncertainty(
            self.fc_engine.st_gnn_model, self.builder, self.history[:5], ForecastHorizon.PLUS_5MIN
        )
        self.assertEqual(unc.target_timestamp, 20.0 + 300.0)

    def test_15_phase_8_forecasting_integration(self):
        """Test 15: Verify Phase 8 forecasting -> Phase 9 uncertainty integration."""
        self.fc_engine.train_models(self.history, self.builder)
        unc = self.estimator.estimate_mc_dropout_uncertainty(
            self.fc_engine.st_gnn_model, self.builder, self.history, ForecastHorizon.PLUS_15MIN, segment_id="ROAD_R_AB"
        )
        self.assertEqual(unc.horizon, ForecastHorizon.PLUS_15MIN)
        self.assertGreater(unc.upper_bound, unc.lower_bound)

    def test_16_phase_1_to_phase_9_end_to_end_integration(self):
        """Test 16: Verify Phase 1 simulation -> Phase 9 uncertainty end-to-end integration."""
        from app.forecasting.dataset_builder import ForecastingDatasetBuilder
        dataset = ForecastingDatasetBuilder.generate_synthetic_simulation_dataset(num_scenarios=1, steps_per_scenario=10, seed=42)
        unc = self.estimator.estimate_mc_dropout_uncertainty(
            self.fc_engine.st_gnn_model, self.builder, dataset, ForecastHorizon.PLUS_30MIN, segment_id="ROAD_R_AB"
        )
        self.assertEqual(unc.horizon, ForecastHorizon.PLUS_30MIN)


if __name__ == "__main__":
    unittest.main()
