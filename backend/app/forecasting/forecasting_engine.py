"""
ChronoEye Infinity - Phase 8: Traffic Forecasting Engine
Master orchestrator connecting Phase 6 spatio-temporal graph topology & Phase 7 historical state sequences
with multi-level forecasters (Level 1 Persistence, Level 2 Ridge, Level 3 ST-GNN).
"""

import os
from typing import Dict, List, Optional, Any
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.state.traffic_state_schema import NetworkTrafficSnapshot
from app.forecasting.forecasting_schema import (
    ForecastHorizon,
    ForecastingModelType,
    SegmentForecast,
    NetworkForecastSnapshot,
    ModelEvaluationMetrics,
)
from app.forecasting.baseline_forecaster import PersistenceForecaster
from app.forecasting.ridge_forecaster import RidgeTrafficForecaster
from app.forecasting.st_gnn import SpatioTemporalGNNForecaster
from app.forecasting.dataset_builder import ForecastingDatasetBuilder
from app.forecasting.evaluation import ForecastingEvaluator
from app.forecasting.model_registry import ModelRegistry
from app.models.stgnn.config import STGNNConfig
from app.models.stgnn.model import SpatioTemporalGNN


class TrafficForecastingEngine:
    """
    Multi-Horizon Spatio-Temporal Traffic Forecasting Engine.
    Unifies Level 1 Persistence, Level 2 Ridge Regression, and Level 3 Spatio-Temporal Graph Neural Network
    with canonical PyTorch ST-GNN checkpoint loading and inference.
    """

    def __init__(self, window_size: int = 12, checkpoint_path: Optional[str] = None):
        self.window_size = window_size
        self.persistence_model = PersistenceForecaster(moving_avg_window=3)
        self.ridge_model = RidgeTrafficForecaster(alpha=1.0, window_size=window_size)
        self.st_gnn_model = SpatioTemporalGNNForecaster(window_size=window_size, checkpoint_path=checkpoint_path)
        self.registry = ModelRegistry()

        self.registry.register_model("persistence", self.persistence_model)
        self.registry.register_model("ridge", self.ridge_model)
        self.registry.register_model("st_gnn", self.st_gnn_model)

    def load_canonical_stgnn(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[STGNNConfig] = None,
    ) -> bool:
        """
        Loads the modern canonical PyTorch SpatioTemporalGNN model checkpoint into the forecasting engine.
        """
        candidate_paths = [
            checkpoint_path,
            os.path.join("experiments", "module3", "research_exp_01", "checkpoints", "best_stgnn_model.pt"),
            os.path.join("experiments", "module3", "research_exp_02", "checkpoints", "best_stgnn_model.pt"),
            os.path.join("checkpoints", "stgnn", "best_stgnn_model.pt"),
        ]
        for p in candidate_paths:
            if p and os.path.exists(p):
                success = self.st_gnn_model.load_pytorch_checkpoint(p, config=config)
                if success:
                    return True
        return False

    def get_canonical_stgnn(self) -> Optional[SpatioTemporalGNN]:
        """
        Retrieves the active PyTorch SpatioTemporalGNN neural network instance if loaded.
        """
        return self.st_gnn_model.pytorch_model

    def train_models(
        self,
        history: List[NetworkTrafficSnapshot],
        builder: Optional[SpatioTemporalGraphBuilder] = None,
        segment_id: str = "ROAD_R_AB",
    ):
        """
        Trains Ridge Regression and ST-GNN forecaster models using historical snapshot sequence.
        If history is small, generates synthetic simulation training scenario dataset.
        """
        train_data = history
        if len(history) < (self.window_size + 10):
            train_data = ForecastingDatasetBuilder.generate_synthetic_simulation_dataset(
                num_scenarios=5, steps_per_scenario=30, seed=42
            )

        X, Y = ForecastingDatasetBuilder.create_supervised_dataset(
            train_data, window_size=self.window_size, horizon_steps=6, segment_id=segment_id
        )

        if X and Y:
            self.ridge_model.fit(X, Y)
            b = builder or SpatioTemporalGraphBuilder()
            self.st_gnn_model.fit(b, X, Y)

    def forecast_network(
        self,
        history,
        horizon: ForecastHorizon = ForecastHorizon.PLUS_5MIN,
        model_type: ForecastingModelType = ForecastingModelType.ST_GNN,
        builder: Optional[SpatioTemporalGraphBuilder] = None,
    ) -> NetworkForecastSnapshot:
        """
        Generates multi-horizon prediction snapshot across all network road segments.
        Accepts:
          - history: List[NetworkTrafficSnapshot]  (canonical form)
          - history: NetworkTrafficSnapshot          (single snapshot — wrapped into [snapshot])
        horizon defaults to PLUS_5MIN for backward compatibility with callers that omit it.
        Explicit calls forecast_network(history, horizon) work exactly as before.
        """
        # Normalize: single snapshot → one-element list (valid one-observation history)
        if isinstance(history, NetworkTrafficSnapshot):
            history = [history]

        if not history:
            return NetworkForecastSnapshot(
                snapshot_id="FC_EMPTY",
                timestamp=0.0,
                horizon=horizon,
                model_type=model_type,
                segment_forecasts={},
            )

        # Empty segment states check for missing data forecasting semantics
        if hasattr(history[-1], "segment_states") and isinstance(history[-1].segment_states, dict):
            if len(history[-1].segment_states) == 0:
                return NetworkForecastSnapshot(
                    snapshot_id="FC_EMPTY",
                    timestamp=history[-1].timestamp if history else 0.0,
                    horizon=horizon,
                    model_type=model_type,
                    segment_forecasts={},
                )

        b = builder or SpatioTemporalGraphBuilder()
        from app.graph.graph_schema import NodeType
        road_ids = b.get_nodes_by_type(NodeType.ROAD)

        if not road_ids:
            road_ids = ["ROAD_R_AB", "ROAD_R_BA", "ROAD_R_CD", "ROAD_R_DC"]

        timestamp = history[-1].timestamp if history else 0.0
        segment_forecasts: Dict[str, SegmentForecast] = {}

        for r_id in road_ids:
            if model_type == ForecastingModelType.PERSISTENCE:
                pred = self.persistence_model.predict(history, horizon, segment_id=r_id)
            elif model_type == ForecastingModelType.RIDGE:
                pred = self.ridge_model.predict(history, horizon, segment_id=r_id)
            else:
                pred = self.st_gnn_model.predict(b, history, horizon, segment_id=r_id)

            segment_forecasts[r_id] = pred

        return NetworkForecastSnapshot(
            timestamp=timestamp,
            horizon=horizon,
            model_type=model_type,
            segment_forecasts=segment_forecasts,
        )


    def evaluate_all(
        self,
        test_history: List[NetworkTrafficSnapshot],
        builder: Optional[SpatioTemporalGraphBuilder] = None,
        segment_id: str = "ROAD_R_AB",
    ) -> Dict[str, Any]:
        """
        Evaluates Persistence vs Ridge vs ST-GNN models across all 4 horizons (+5m, +10m, +15m, +30m).
        Returns MAE, RMSE, and MAPE evaluation metrics comparison.
        """
        metrics_list: List[ModelEvaluationMetrics] = []
        b = builder or SpatioTemporalGraphBuilder()

        for horizon in [
            ForecastHorizon.PLUS_5MIN,
            ForecastHorizon.PLUS_10MIN,
            ForecastHorizon.PLUS_15MIN,
            ForecastHorizon.PLUS_30MIN,
        ]:
            # Extract actual future speed targets for horizon
            actuals = [snap.segment_states[segment_id].average_speed for snap in test_history if segment_id in snap.segment_states]
            if not actuals:
                actuals = [60.0 for _ in range(10)]

            # 1. Persistence
            p_preds = [self.persistence_model.predict(test_history[:i+1], horizon, segment_id).predicted_speed for i in range(len(test_history)-1)]
            p_actuals = actuals[1:len(p_preds)+1]
            if p_actuals and p_preds:
                metrics_list.append(ForecastingEvaluator.evaluate_predictions("Level_1_Persistence", horizon, p_actuals, p_preds))

            # 2. Ridge
            r_preds = [self.ridge_model.predict(test_history[:i+1], horizon, segment_id).predicted_speed for i in range(len(test_history)-1)]
            if p_actuals and r_preds:
                metrics_list.append(ForecastingEvaluator.evaluate_predictions("Level_2_Ridge", horizon, p_actuals, r_preds))

            # 3. ST-GNN
            g_preds = [self.st_gnn_model.predict(b, test_history[:i+1], horizon, segment_id).predicted_speed for i in range(len(test_history)-1)]
            if p_actuals and g_preds:
                metrics_list.append(ForecastingEvaluator.evaluate_predictions("Level_3_ST_GNN", horizon, p_actuals, g_preds))

        comparison = ForecastingEvaluator.compare_models(metrics_list)
        return {
            "all_metrics": [m.model_dump() for m in metrics_list],
            "best_models_by_horizon": comparison,
        }
