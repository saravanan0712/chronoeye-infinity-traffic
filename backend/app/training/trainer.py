"""
ChronoEye Infinity - ST-GNN Training & Evaluation Pipeline.
Manages batch tensor collation, masked loss optimization, early stopping,
checkpoint saving/loading, and multi-horizon test evaluation.
"""

import os
import json
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from app.models.stgnn.config import STGNNConfig
from app.models.stgnn.model import SpatioTemporalGNN
from app.models.stgnn.loss import MaskedLoss, compute_all_metrics
from app.graph.temporal_dataset_slicer import SpatioTemporalDataset, SpatioTemporalSample


class TrainingResult(BaseModel):
    """
    Detailed training execution results and evaluation metrics.
    """
    model_name: str
    parameter_count: int
    train_samples_count: int
    val_samples_count: int
    test_samples_count: int
    epochs_trained: int
    best_epoch: int
    best_val_loss: float
    training_time_seconds: float
    history: Dict[str, List[float]] = Field(default_factory=dict)
    test_metrics: Dict[str, Any] = Field(default_factory=dict)  # model -> horizon -> target -> metrics
    metadata: Dict[str, Any] = Field(default_factory=dict)


class STGNNTrainer:
    """
    PyTorch Trainer for Spatio-Temporal Graph Neural Networks.
    """

    def __init__(
        self,
        model: SpatioTemporalGNN,
        config: Optional[STGNNConfig] = None,
        loss_type: str = "mae",
        checkpoint_dir: Optional[str] = None,
    ):
        self.model = model
        self.config = config or getattr(model, "config", STGNNConfig())
        self.device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device != "cpu" else "cpu")
        self.model.to(self.device)
        self.loss_fn = MaskedLoss(loss_type=loss_type)
        self.checkpoint_dir = checkpoint_dir or "checkpoints/stgnn"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _dataset_to_tensors(
        self, samples: List[SpatioTemporalSample], use_normalized: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts list of SpatioTemporalSample into 4D PyTorch tensors:
        X: [B, T_in, N, F]
        X_mask: [B, T_in, N, F]
        Y: [B, H, N, T_target]
        Y_mask: [B, H, N, T_target]
        """
        if not samples:
            return (
                torch.empty((0, self.config.input_sequence_length, 1, self.config.input_dim)),
                torch.empty((0, self.config.input_sequence_length, 1, self.config.input_dim)),
                torch.empty((0, self.config.num_horizons, 1, self.config.output_dim)),
                torch.empty((0, self.config.num_horizons, 1, self.config.output_dim)),
            )

        x_list, x_mask_list = [], []
        y_list, y_mask_list = [], []

        for s in samples:
            x_data = s.x_normalized if use_normalized else s.x_raw
            y_data = s.y_normalized if use_normalized else s.y_raw

            # Convert None to 0.0 for tensor conversion
            x_tensor = torch.tensor(
                [[[val if val is not None else 0.0 for val in node] for node in step] for step in x_data],
                dtype=torch.float32,
            )
            x_mask_tensor = torch.tensor(s.x_mask, dtype=torch.bool)

            y_tensor = torch.tensor(
                [[[val if val is not None else 0.0 for val in node] for node in step] for step in y_data],
                dtype=torch.float32,
            )
            y_mask_tensor = torch.tensor(s.y_mask, dtype=torch.bool)

            x_list.append(x_tensor)
            x_mask_list.append(x_mask_tensor)
            y_list.append(y_tensor)
            y_mask_list.append(y_mask_tensor)

        return (
            torch.stack(x_list).to(self.device),
            torch.stack(x_mask_list).to(self.device),
            torch.stack(y_list).to(self.device),
            torch.stack(y_mask_list).to(self.device),
        )

    def train(
        self,
        dataset: SpatioTemporalDataset,
        epochs: int = 30,
        batch_size: int = 16,
        patience: int = 8,
        lr: Optional[float] = None,
        save_checkpoint_name: str = "best_stgnn_model.pt",
    ) -> TrainingResult:
        """
        Trains ST-GNN using chronological train/val splits with early stopping.
        """
        start_time = time.time()
        learning_rate = lr or self.config.learning_rate
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Build Tensors
        x_train, x_mask_train, y_train, y_mask_train = self._dataset_to_tensors(dataset.train_samples)
        x_val, x_mask_val, y_val, y_mask_val = self._dataset_to_tensors(dataset.val_samples)
        x_test, x_mask_test, y_test, y_mask_test = self._dataset_to_tensors(dataset.test_samples)

        # Graph Adjacency Matrix
        adj_matrix = torch.tensor(dataset.adjacency.adjacency_matrix, dtype=torch.float32).to(self.device)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_model_weights = None

        num_train_samples = x_train.size(0)

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            # Batch iteration
            indices = list(range(0, num_train_samples, batch_size))
            for start_idx in indices:
                end_idx = min(start_idx + batch_size, num_train_samples)
                bx = x_train[start_idx:end_idx]
                bx_mask = x_mask_train[start_idx:end_idx]
                by = y_train[start_idx:end_idx]
                by_mask = y_mask_train[start_idx:end_idx]

                optimizer.zero_grad()
                pred = self.model(bx, adj_matrix, feature_mask=bx_mask)
                loss = self.loss_fn(pred, by, mask=by_mask)

                if not torch.isnan(loss) and loss.item() > 0:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1

            avg_train_loss = round(epoch_loss / max(1, num_batches), 4)
            history["train_loss"].append(avg_train_loss)

            # Validation Evaluation
            self.model.eval()
            with torch.no_grad():
                if x_val.size(0) > 0:
                    val_pred = self.model(x_val, adj_matrix, feature_mask=x_mask_val)
                    val_loss_val = self.loss_fn(val_pred, y_val, mask=y_mask_val).item()
                else:
                    val_loss_val = avg_train_loss

            val_loss_val = round(float(val_loss_val), 4)
            history["val_loss"].append(val_loss_val)

            # Model Selection via Validation Loss
            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                best_epoch = epoch
                patience_counter = 0
                best_model_weights = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Load best checkpoint weights
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            checkpoint_path = os.path.join(self.checkpoint_dir, save_checkpoint_name)
            torch.save(best_model_weights, checkpoint_path)

        # Final Evaluation on TEST SET ONLY
        test_metrics = self.evaluate_all_models(
            dataset=dataset,
            x_test=x_test,
            x_mask_test=x_mask_test,
            y_test=y_test,
            y_mask_test=y_mask_test,
            adj_matrix=adj_matrix,
        )

        total_time = round(time.time() - start_time, 2)

        return TrainingResult(
            model_name=self.config.model_name,
            parameter_count=self.model.get_parameter_count(),
            train_samples_count=len(dataset.train_samples),
            val_samples_count=len(dataset.val_samples),
            test_samples_count=len(dataset.test_samples),
            epochs_trained=len(history["train_loss"]),
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            training_time_seconds=total_time,
            history=history,
            test_metrics=test_metrics,
            metadata={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "hidden_dim": self.config.hidden_dim,
                "horizons": self.config.forecast_horizons,
            },
        )

    def evaluate_all_models(
        self,
        dataset: SpatioTemporalDataset,
        x_test: Optional[torch.Tensor] = None,
        x_mask_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        y_mask_test: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates ST-GNN and baseline models on the test split.
        Returns: {model_name: {horizon: {target_name: {mae, rmse, mape}}}}
        """
        if x_test is None or y_test is None:
            x_test, x_mask_test, y_test, y_mask_test = self._dataset_to_tensors(dataset.test_samples)
        if adj_matrix is None:
            adj_matrix = torch.tensor(dataset.adjacency.adjacency_matrix, dtype=torch.float32).to(self.device)

        results: Dict[str, Any] = {}
        if x_test.size(0) == 0:
            return results

        self.model.eval()
        with torch.no_grad():
            stgnn_pred = self.model(x_test, adj_matrix, feature_mask=x_mask_test)

        # Baseline Predictions
        from app.training.baselines import PersistenceBaseline, MovingAverageBaseline, LinearRegressionBaseline

        # Map target indices
        feat_to_target = []
        for tname in dataset.target_names:
            if tname in dataset.feature_names:
                feat_to_target.append(dataset.feature_names.index(tname))
            else:
                feat_to_target.append(0)

        pers_baseline = PersistenceBaseline()
        pers_pred = pers_baseline.predict(
            x_test, self.config.num_horizons, self.config.output_dim, feature_to_target_map=feat_to_target
        )

        ma_baseline = MovingAverageBaseline()
        ma_pred = ma_baseline.predict(
            x_test, self.config.num_horizons, self.config.output_dim, feature_to_target_map=feat_to_target, x_mask=x_mask_test
        )

        lr_baseline = LinearRegressionBaseline(alpha=1.0)
        x_tr, _, y_tr, _ = self._dataset_to_tensors(dataset.train_samples)
        if x_tr.size(0) > 0 and y_tr.size(0) > 0:
            lr_baseline.fit(x_tr, y_tr)
        lr_pred = lr_baseline.predict(x_test, self.config.num_horizons, self.config.output_dim)

        # Map raw vs denormalized targets if needed, compute metrics on test set
        models_preds = {
            "ST-GNN": stgnn_pred,
            "Persistence": pers_pred,
            "Moving_Average": ma_pred,
            "Linear_Regression": lr_pred,
        }

        for m_name, pred_tensor in models_preds.items():
            results[m_name] = {}
            for h_idx, h_name in enumerate(self.config.forecast_horizons):
                results[m_name][h_name] = {}
                for t_idx, t_name in enumerate(dataset.target_names):
                    if t_idx < pred_tensor.size(-1) and t_idx < y_test.size(-1):
                        p_slice = pred_tensor[:, h_idx, :, t_idx]
                        y_slice = y_test[:, h_idx, :, t_idx]
                        m_slice = y_mask_test[:, h_idx, :, t_idx]
                        
                        metrics = compute_all_metrics(p_slice, y_slice, mask=m_slice)
                        results[m_name][h_name][t_name] = metrics

        return results
