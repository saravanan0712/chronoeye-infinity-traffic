import os
import json
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple, Union, Sequence
from pydantic import BaseModel, Field
from torch.utils.data import Dataset, DataLoader

from app.models.stgnn.config import STGNNConfig
from app.models.stgnn.model import SpatioTemporalGNN
from app.models.stgnn.loss import MaskedLoss, compute_all_metrics
from app.graph.temporal_dataset_slicer import SpatioTemporalDataset, SpatioTemporalSample


class _LegacySampleDataset(Dataset):
    """Fallback Dataset wrapper for legacy SpatioTemporalSample lists."""
    def __init__(self, samples: Sequence[SpatioTemporalSample], use_normalized: bool = True):
        self.samples = samples
        self.use_normalized = use_normalized

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        x_data = s.x_normalized if self.use_normalized else s.x_raw
        y_data = s.y_normalized if self.use_normalized else s.y_raw

        x = torch.tensor(
            [[[val if val is not None else 0.0 for val in node] for node in step] for step in x_data],
            dtype=torch.float32,
        )
        x_mask = torch.tensor(s.x_mask, dtype=torch.bool)
        y = torch.tensor(
            [[[val if val is not None else 0.0 for val in node] for node in step] for step in y_data],
            dtype=torch.float32,
        )
        y_mask = torch.tensor(s.y_mask, dtype=torch.bool)
        return x, x_mask, y, y_mask


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
    PyTorch Trainer for Spatio-Temporal Graph Neural Networks with true mini-batching on GPU.
    """

    def __init__(
        self,
        model: SpatioTemporalGNN,
        config: Optional[STGNNConfig] = None,
        loss_type: str = "mae",
        checkpoint_dir: Optional[str] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.model = model
        self.config = config or getattr(model, "config", STGNNConfig())
        if device is not None:
            self.device = torch.device(device)
        elif self.config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.loss_fn = MaskedLoss(loss_type=loss_type)
        self.checkpoint_dir = checkpoint_dir or "checkpoints/stgnn"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _get_dataloader(
        self,
        dataset: Any,
        split: str,
        batch_size: int = 16,
        shuffle: bool = False,
    ) -> DataLoader:
        """Constructs DataLoader for given dataset and split."""
        if hasattr(dataset, "get_dataloader"):
            return dataset.get_dataloader(split=split, batch_size=batch_size, shuffle=shuffle)

        # Check for proxy or list samples
        samples = getattr(dataset, f"{split}_samples", [])
        if hasattr(samples, "split_dataset"):
            return DataLoader(samples.split_dataset, batch_size=batch_size, shuffle=shuffle)

        if len(samples) > 0:
            return DataLoader(_LegacySampleDataset(samples), batch_size=batch_size, shuffle=shuffle)

        # Empty dataset fallback
        empty_ds = _LegacySampleDataset([])
        return DataLoader(empty_ds, batch_size=batch_size, shuffle=False)

    def _dataset_to_tensors(
        self, samples: Sequence[SpatioTemporalSample], use_normalized: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts list of SpatioTemporalSample into 4D PyTorch tensors:
        X: [B, T_in, N, F]
        X_mask: [B, T_in, N, F]
        Y: [B, H, N, T_target]
        Y_mask: [B, H, N, T_target]
        """
        if not samples or len(samples) == 0:
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
        dataset: Any,
        epochs: int = 30,
        batch_size: int = 16,
        patience: int = 8,
        lr: Optional[float] = None,
        save_checkpoint_name: str = "best_stgnn_model.pt",
    ) -> TrainingResult:
        """
        Trains ST-GNN using chronological train/val splits with early stopping and mini-batch streaming.
        Only mini-batches are transferred to GPU, ensuring constant memory footprint.
        """
        start_time = time.time()
        learning_rate = lr or self.config.learning_rate
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay,
        )

        train_loader = self._get_dataloader(dataset, "train", batch_size=batch_size, shuffle=True)
        val_loader = self._get_dataloader(dataset, "val", batch_size=batch_size, shuffle=False)

        # Graph Adjacency Matrix on device
        adj_matrix = torch.tensor(dataset.adjacency.adjacency_matrix, dtype=torch.float32).to(self.device)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_model_weights = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for bx, bx_mask, by, by_mask in train_loader:
                bx = bx.to(self.device)
                bx_mask = bx_mask.to(self.device)
                by = by.to(self.device)
                by_mask = by_mask.to(self.device)

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

            # Validation Evaluation (batch-by-batch)
            self.model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for bx, bx_mask, by, by_mask in val_loader:
                    bx = bx.to(self.device)
                    bx_mask = bx_mask.to(self.device)
                    by = by.to(self.device)
                    by_mask = by_mask.to(self.device)

                    val_pred = self.model(bx, adj_matrix, feature_mask=bx_mask)
                    v_loss = self.loss_fn(val_pred, by, mask=by_mask)
                    if not torch.isnan(v_loss):
                        val_loss_sum += v_loss.item()
                        val_batches += 1

            val_loss_val = round(val_loss_sum / max(1, val_batches), 4) if val_batches > 0 else avg_train_loss
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

        # Final Evaluation on TEST SET ONLY (mini-batch evaluation)
        test_metrics = self.evaluate_all_models(
            dataset=dataset,
            adj_matrix=adj_matrix,
            batch_size=batch_size,
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
        dataset: Any,
        x_test: Optional[torch.Tensor] = None,
        x_mask_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        y_mask_test: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Evaluates ST-GNN and baseline models on the test split.
        Mini-batches are streamed to GPU and collected on CPU for metric calculation.
        """
        if adj_matrix is None:
            adj_matrix = torch.tensor(dataset.adjacency.adjacency_matrix, dtype=torch.float32).to(self.device)

        from app.training.baselines import PersistenceBaseline, MovingAverageBaseline, LinearRegressionBaseline

        feat_to_target = []
        for tname in dataset.target_names:
            if tname in dataset.feature_names:
                feat_to_target.append(dataset.feature_names.index(tname))
            else:
                feat_to_target.append(0)

        pers_baseline = PersistenceBaseline()
        ma_baseline = MovingAverageBaseline()
        lr_baseline = LinearRegressionBaseline(alpha=1.0)

        # If x_test is provided directly (legacy path)
        if x_test is not None and y_test is not None:
            if x_test.size(0) == 0:
                return {}
            self.model.eval()
            with torch.no_grad():
                stgnn_pred = self.model(x_test, adj_matrix, feature_mask=x_mask_test).cpu()
            pers_pred = pers_baseline.predict(
                x_test, self.config.num_horizons, self.config.output_dim, feature_to_target_map=feat_to_target
            ).cpu()
            ma_pred = ma_baseline.predict(
                x_test, self.config.num_horizons, self.config.output_dim, feature_to_target_map=feat_to_target, x_mask=x_mask_test
            ).cpu()
            x_tr, _, y_tr, _ = self._dataset_to_tensors(dataset.train_samples)
            if x_tr.size(0) > 0 and y_tr.size(0) > 0:
                lr_baseline.fit(x_tr, y_tr)
            lr_pred = lr_baseline.predict(x_test, self.config.num_horizons, self.config.output_dim).cpu()

            models_preds = {
                "ST-GNN": stgnn_pred,
                "Persistence": pers_pred,
                "Moving_Average": ma_pred,
                "Linear_Regression": lr_pred,
            }
            y_test_cpu = y_test.cpu()
            y_mask_test_cpu = y_mask_test.cpu() if y_mask_test is not None else None
        else:
            # Memory-safe DataLoader streaming evaluation
            test_loader = self._get_dataloader(dataset, "test", batch_size=batch_size, shuffle=False)
            train_loader = self._get_dataloader(dataset, "train", batch_size=batch_size, shuffle=False)

            # Fit Ridge on training batches
            if len(dataset.train_samples) > 0:
                lr_baseline.fit_loader(train_loader, device=torch.device("cpu"))

            stgnn_preds_list = []
            pers_preds_list = []
            ma_preds_list = []
            lr_preds_list = []
            y_test_list = []
            y_mask_test_list = []

            self.model.eval()
            with torch.no_grad():
                for bx, bx_mask, by, by_mask in test_loader:
                    bx_dev = bx.to(self.device)
                    bx_mask_dev = bx_mask.to(self.device)
                    pred_batch = self.model(bx_dev, adj_matrix, feature_mask=bx_mask_dev)
                    stgnn_preds_list.append(pred_batch.cpu())

                    pers_batch = pers_baseline.predict(
                        bx, self.config.num_horizons, self.config.output_dim, feature_to_target_map=feat_to_target
                    )
                    pers_preds_list.append(pers_batch.cpu())

                    ma_batch = ma_baseline.predict(
                        bx, self.config.num_horizons, self.config.output_dim, feature_to_target_map=feat_to_target, x_mask=bx_mask
                    )
                    ma_preds_list.append(ma_batch.cpu())

                    lr_batch = lr_baseline.predict(
                        bx, self.config.num_horizons, self.config.output_dim
                    )
                    lr_preds_list.append(lr_batch.cpu())

                    y_test_list.append(by.cpu())
                    y_mask_test_list.append(by_mask.cpu())

            if not stgnn_preds_list:
                return {}

            models_preds = {
                "ST-GNN": torch.cat(stgnn_preds_list, dim=0),
                "Persistence": torch.cat(pers_preds_list, dim=0),
                "Moving_Average": torch.cat(ma_preds_list, dim=0),
                "Linear_Regression": torch.cat(lr_preds_list, dim=0),
            }
            y_test_cpu = torch.cat(y_test_list, dim=0)
            y_mask_test_cpu = torch.cat(y_mask_test_list, dim=0)

        results: Dict[str, Any] = {}
        for m_name, pred_tensor in models_preds.items():
            results[m_name] = {}
            for h_idx, h_name in enumerate(self.config.forecast_horizons):
                results[m_name][h_name] = {}
                for t_idx, t_name in enumerate(dataset.target_names):
                    if t_idx < pred_tensor.size(-1) and t_idx < y_test_cpu.size(-1):
                        p_slice = pred_tensor[:, h_idx, :, t_idx]
                        y_slice = y_test_cpu[:, h_idx, :, t_idx]
                        m_slice = y_mask_test_cpu[:, h_idx, :, t_idx] if y_mask_test_cpu is not None else None

                        metrics = compute_all_metrics(p_slice, y_slice, mask=m_slice)
                        results[m_name][h_name][t_name] = metrics

        return results

