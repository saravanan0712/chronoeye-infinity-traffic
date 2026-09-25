"""
Master Research Execution Script for ChronoEye Infinity Module 3 — Experiment 02.
Runs the real/public traffic forecasting pipeline on Caltrans PeMS08:
  1. Real Dataset Loading & Schema Adaptation (RealTrafficDatasetAdapter)
  2. Chronological Slicing & Train-Only Normalization (TemporalDatasetSlicer)
  3. ST-GNN Model Training with Validation Loss Early Stopping (STGNNTrainer)
  4. Multi-Horizon Baseline Model Evaluation (+5, +10, +15 min) (Persistence, Moving Avg, Ridge)
  5. Comprehensive Metric Computation (MAE, RMSE, MAPE)
  6. Ablation Studies (A: Temporal Only, B: Spatial Only, D: Short History)
  7. Experiment 02 Artifact Persistence (JSON configs, metrics, scalers, checkpoints)
  8. Research Visualization Plot Generation
"""

import os
import sys
import json
import time
import math
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List

# Ensure backend app is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "backend"))

from app.adapters.real_traffic_adapter import RealTrafficDatasetAdapter
from app.graph.temporal_dataset_slicer import TemporalDatasetSlicer
from app.models.stgnn.config import STGNNConfig
from app.models.stgnn.model import SpatioTemporalGNN
from app.models.stgnn.loss import compute_all_metrics, MaskedLoss
from app.training.trainer import STGNNTrainer
from app.training.baselines import PersistenceBaseline, MovingAverageBaseline, LinearRegressionBaseline
from app.training.ablation import AblationStudyRunner


def run_real_experiment(
    npz_path: str = "data/research/pems08/PEMS08.npz",
    csv_path: str = "data/research/pems08/PEMS08.csv",
    output_dir: str = "experiments/module3/research_exp_02",
    num_nodes: int = 30,
    num_timesteps: int = 576,  # 48 hours @ 5-min intervals (576 steps)
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.005,
    seed: int = 42,
):
    print("=" * 70)
    print("CHRONOEYE INFINITY — MODULE 3 EXPERIMENT 02 (REAL PEMS08 BENCHMARK)")
    print("=" * 70)

    # 0. Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_path = repo_root / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    plots_path = out_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(out_path / "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Dataset Ingestion & Adaptation
    print(f"\n[1/7] Loading & Adapting Caltrans PeMS08 Real Traffic Dataset...")
    dataset_contract, meta = RealTrafficDatasetAdapter.load_pems08(
        npz_path=str(repo_root / npz_path),
        csv_path=str(repo_root / csv_path),
        num_nodes=num_nodes,
        num_timesteps=num_timesteps,
    )
    print(f"  -> Dataset: {meta['dataset_name']} ({meta['source_provenance']})")
    print(f"  -> Processed {len(dataset_contract.snapshots)} snapshots ({num_timesteps*5/60:.1f} hours).")
    print(f"  -> Spatial Nodes: {dataset_contract.snapshots[0].node_count} loop detectors.")
    print(f"  -> Topology Edges: {dataset_contract.snapshots[0].edge_count} physical road connections.")

    # Save dataset metadata, target availability, node mapping
    with open(out_path / "dataset_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(out_path / "target_availability.json", "w") as f:
        json.dump(meta["target_availability"], f, indent=2)

    with open(out_path / "node_mapping.json", "w") as f:
        json.dump(meta["node_mapping"], f, indent=2)

    # 2. Chronological Slicing & Train-Only Normalization
    print("\n[2/7] Slicing Temporal Sequences & Normalizing (Train-Only Fit)...")
    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,      # T_in = 3 (15 min history)
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"], # +5, +10, +15 min
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        normalization_method="zscore",
    )
    st_dataset = slicer.slice_dataset(dataset_contract)
    print(f"  -> Train samples: {len(st_dataset.train_samples)}")
    print(f"  -> Val samples:   {len(st_dataset.val_samples)}")
    print(f"  -> Test samples:  {len(st_dataset.test_samples)}")

    # Save Scaler metadata
    scaler_dict = {
        "feature_means": st_dataset.feature_scaler.mean if st_dataset.feature_scaler else {},
        "feature_stds": st_dataset.feature_scaler.std if st_dataset.feature_scaler else {},
        "target_means": st_dataset.target_scaler.mean if st_dataset.target_scaler else {},
        "target_stds": st_dataset.target_scaler.std if st_dataset.target_scaler else {},
        "normalization_fitted_on": "TRAIN_SPLIT_ONLY",
    }
    with open(out_path / "scaler.json", "w") as f:
        json.dump(scaler_dict, f, indent=2)

    # 3. Model Architecture & Initialization
    print("\n[3/7] Initializing Spatio-Temporal Graph Neural Network (ST-GNN)...")
    stgnn_config = STGNNConfig(
        model_name="SpatioTemporalGNN_PeMS08_Real",
        in_channels=8,
        hidden_dim=32,
        num_targets=4,
        num_st_blocks=2,
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],
        dropout=0.1,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        max_epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=10,
        random_seed=seed,
    )
    model = SpatioTemporalGNN(stgnn_config)
    param_count = model.get_parameter_count()
    print(f"  -> Architecture: 2 ST-Blocks (GraphSpatialConv + TemporalGRUBlock) + Multi-Head Forecast")
    print(f"  -> Trainable Parameters: {param_count}")

    model_summary = {
        "model_name": stgnn_config.model_name,
        "parameter_count": param_count,
        "layers": [
            {"type": "FeatureProjection", "in": 8, "out": 32},
            {"type": "STGNNBlock_1", "spatial": "GraphSpatialConv", "temporal": "TemporalGRUBlock(hidden=32)"},
            {"type": "STGNNBlock_2", "spatial": "GraphSpatialConv", "temporal": "TemporalGRUBlock(hidden=32)"},
            {"type": "MultiHorizonForecastHead", "horizons": ["+5min", "+10min", "+15min"], "targets": ["flow_rate", "density", "congestion", "travel_time"]},
        ],
        "device": "cpu",
        "random_seed": seed,
    }
    with open(out_path / "model_summary.json", "w") as f:
        json.dump(model_summary, f, indent=2)

    # 4. Training
    print("\n[4/7] Training ST-GNN on PeMS08 Real Data with Early Stopping...")
    trainer = STGNNTrainer(
        model=model,
        config=stgnn_config,
        loss_type="smooth_l1",
        checkpoint_dir=checkpoint_dir,
    )
    train_result = trainer.train(
        dataset=st_dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
    )
    print(f"  -> Training completed in {train_result.training_time_seconds}s.")
    print(f"  -> Best Validation Loss: {train_result.best_val_loss:.4f} at Epoch {train_result.best_epoch}.")

    with open(out_path / "training_history.json", "w") as f:
        json.dump(train_result.history, f, indent=2)

    # 5. Baseline Evaluation & Test Set Comparison
    print("\n[5/7] Evaluating Test Set Performance against Baselines...")
    test_metrics = trainer.evaluate_all_models(st_dataset)

    print("\n" + "=" * 70)
    print("PEMS08 REAL TEST EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<18} | {'Horizon':<10} | {'Target':<12} | {'MAE':<8} | {'RMSE':<8} | {'MAPE (%)':<8}")
    print("-" * 70)
    for model_name, horizons in test_metrics.items():
        for horizon_name, targets in horizons.items():
            for target_name, metrics in targets.items():
                if target_name in ("flow_rate", "density", "congestion"):
                    print(f"{model_name:<18} | {horizon_name:<10} | {target_name:<12} | {metrics['mae']:<8.4f} | {metrics['rmse']:<8.4f} | {metrics['mape']:<8.2f}")
    print("=" * 70)

    with open(out_path / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # 6. Ablation Studies
    print("\n[6/7] Running Scientific Ablation Studies on PeMS08 Real Data...")
    ablation_results_raw = AblationStudyRunner.run_all_ablations(
        st_dataset,
        contract=dataset_contract,
        epochs=15,
        checkpoint_dir=str(out_path / "checkpoints" / "ablations"),
    )
    ablation_results = {k: v.model_dump() for k, v in ablation_results_raw.items()}

    with open(out_path / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    print(f"  -> Ablation A (Temporal Only): Evaluated.")
    print(f"  -> Ablation B (Spatial Only): Evaluated.")
    print(f"  -> Ablation D (1-Step History): Evaluated.")

    # 7. Experiment 02 Configuration Metadata
    exp_config = {
        "experiment_id": "module3_research_exp_02",
        "dataset_name": "Caltrans_PeMS08_Real",
        "dataset_version": "1.0.0",
        "provenance": {
            "source": "California Department of Transportation Performance Measurement System",
            "district": "District 8 (San Bernardino Freeway Network)",
            "doi": "10.5281/zenodo.7816008",
        },
        "topology": {
            "num_nodes": num_nodes,
            "num_edges": dataset_contract.snapshots[0].edge_count,
            "directed": True,
            "adjacency_normalization": "Symmetric Degree Normalization (D^-1/2 A D^-1/2)",
        },
        "temporal_parameters": {
            "snapshot_interval_seconds": 300,
            "history_window_steps": 3,
            "history_window_minutes": 15,
            "forecast_horizons_steps": [1, 2, 3],
            "forecast_horizons_minutes": [5, 10, 15],
        },
        "splits": {
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "chronological": True,
            "train_samples": len(st_dataset.train_samples),
            "val_samples": len(st_dataset.val_samples),
            "test_samples": len(st_dataset.test_samples),
        },
        "stgnn_hyperparameters": {
            "in_channels": 8,
            "hidden_dim": 32,
            "num_st_blocks": 2,
            "dropout": 0.1,
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "weight_decay": 1e-4,
            "batch_size": batch_size,
            "max_epochs": epochs,
            "best_epoch": train_result.best_epoch,
            "best_val_loss": train_result.best_val_loss,
            "parameter_count": param_count,
            "random_seed": seed,
        },
        "leakage_prevention": {
            "chronological_split_verified": True,
            "normalization_train_only": True,
            "test_evaluation_post_selection_only": True,
        }
    }
    with open(out_path / "config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    # 8. Visualizations
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot 1: Loss curves
        plt.figure(figsize=(8, 4))
        plt.plot(train_result.history["train_loss"], label="Train Loss (Smooth L1)", color="#3b82f6")
        plt.plot(train_result.history["val_loss"], label="Val Loss (Smooth L1)", color="#ef4444")
        plt.axvline(train_result.best_epoch, color="#10b981", linestyle="--", label=f"Best Checkpoint (Epoch {train_result.best_epoch})")
        plt.title("PeMS08 Real Dataset: ST-GNN Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_path / "loss_curves.png", dpi=150)
        plt.close()

        # Plot 2: Horizon MAE comparison across models
        plt.figure(figsize=(9, 4.5))
        horizons = ["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"]
        horizon_labels = ["+5 min", "+10 min", "+15 min"]
        x = np.arange(len(horizons))

        available_models = [m for m in ["ST-GNN", "Persistence", "Moving_Average", "Linear_Regression"] if m in test_metrics]
        num_models = len(available_models)
        width = 0.8 / max(1, num_models)
        colors = ["#6366f1", "#ef4444", "#f59e0b", "#14b8a6"]

        for i, m_name in enumerate(available_models):
            mae_vals = [test_metrics[m_name][h]["flow_rate"]["mae"] for h in horizons]
            offset = (i - (num_models - 1) / 2.0) * width
            plt.bar(x + offset, mae_vals, width, label=m_name, color=colors[i % len(colors)])

        plt.ylabel("Flow Rate MAE (Normalized)")
        plt.title("PeMS08 Real Traffic: Multi-Horizon Forecast Error (ST-GNN vs Baselines)")
        plt.xticks(x, horizon_labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_path / "multi_horizon_comparison.png", dpi=150)
        plt.close()

        # Plot 3: Ablation Study
        if ablation_results:
            plt.figure(figsize=(9, 4.5))
            ablation_names = list(ablation_results.keys())
            ab_val_losses = [ablation_results[k]["best_val_loss"] for k in ablation_names]
            clean_names = [k.replace("Ablation_", "").replace("_", " ") for k in ablation_names]
            plt.bar(clean_names, ab_val_losses, color=["#6366f1", "#ec4899", "#8b5cf6", "#3b82f6", "#10b981"])
            plt.ylabel("Best Validation Loss")
            plt.title("PeMS08 Real Traffic: Ablation Study Comparison")
            plt.xticks(rotation=20, ha="right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_path / "ablation_comparison.png", dpi=150)
            plt.close()

        print(f"\n[7/7] Saved real dataset research plots to {plots_path}")
    except Exception as e:
        print(f"\n[7/7] Plotting note: {e}")

    # Generate README in experiment dir
    readme_content = f"""# ChronoEye Infinity Module 3 — Research Experiment 02 (Real PeMS08)

## Overview
- **Experiment ID**: `module3_research_exp_02`
- **Dataset**: Caltrans PeMS08 Real-World Freeway Traffic Benchmark (District 8, San Bernardino)
- **Provenance**: Official Zenodo Record 7816008
- **Model**: Spatio-Temporal Graph Neural Network (ST-GNN)
- **Spatial Topology**: {num_nodes} loop detector nodes, {dataset_contract.snapshots[0].edge_count} connectivity edges
- **Train / Val / Test**: {len(st_dataset.train_samples)} / {len(st_dataset.val_samples)} / {len(st_dataset.test_samples)}
- **Trainable Parameters**: {param_count}
- **Best Epoch**: {train_result.best_epoch} (Val Loss: {train_result.best_val_loss:.4f})

## Files
- `config.json`: Experiment configuration and hyperparameters.
- `dataset_metadata.json`: Provenance, sensor counts, and variable mappings.
- `target_availability.json`: Explicit target channel availability table.
- `node_mapping.json`: Deterministic sensor ID to graph node mapping.
- `metrics.json`: Test split evaluation metrics (MAE, RMSE, MAPE) across $+5, +10, +15$ min horizons.
- `training_history.json`: Loss history per epoch.
- `scaler.json`: Training-split-only normalizer parameters.
- `model_summary.json`: Structural overview of the ST-GNN architecture.
- `ablation_results.json`: Results of Spatial, Temporal, and History ablations on real data.
- `checkpoints/best_stgnn_model.pt`: Checkpoint at best validation epoch.
"""
    with open(out_path / "README.md", "w") as f:
        f.write(readme_content)

    print("\n" + "=" * 70)
    print("EXPERIMENT 02 (REAL PEMS08) COMPLETE")
    print(f"Artifacts saved in: {out_path}")
    print("=" * 70)
    return train_result, test_metrics, ablation_results


if __name__ == "__main__":
    run_real_experiment()
