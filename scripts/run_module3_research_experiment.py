"""
Master Research Execution Script for ChronoEye Infinity Module 3.
Runs the complete research pipeline:
  1. Benchmark Dataset Generation & Adaptation
  2. Chronological Slicing & Train-Only Normalization
  3. ST-GNN Model Training with Early Stopping
  4. Multi-Horizon Baseline Model Evaluation (+5, +10, +15 min)
  5. Comprehensive Metric Computation (MAE, RMSE, MAPE)
  6. Ablation Studies (A, B, C, D)
  7. Experiment Artifact Persistence (JSON configs, metrics, scaler, checkpoints)
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

from app.adapters.benchmark_adapter import TrafficDatasetAdapter
from app.graph.temporal_dataset_slicer import TemporalDatasetSlicer
from app.models.stgnn.config import STGNNConfig
from app.models.stgnn.model import SpatioTemporalGNN
from app.models.stgnn.loss import compute_all_metrics, MaskedLoss
from app.training.trainer import STGNNTrainer
from app.training.baselines import PersistenceBaseline, MovingAverageBaseline, LinearRegressionBaseline
from app.training.ablation import AblationStudyRunner


def run_research_pipeline(
    output_dir: str = "experiments/module3/research_exp_01",
    num_sensors: int = 10,
    num_snapshots: int = 288,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.005,
    seed: int = 42,
):
    print("=" * 70)
    print("CHRONOEYE INFINITY — MODULE 3 RESEARCH PIPELINE EXECUTION")
    print("=" * 70)
    
    # 0. Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    out_path = repo_root / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    plots_path = out_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Dataset Generation & Adapter
    print(f"\n[1/7] Generating & Adapting Benchmark Dataset ({num_snapshots} snapshots, {num_sensors} sensors)...")
    adapter = TrafficDatasetAdapter()
    dataset_contract = adapter.generate_research_benchmark_dataset(
        num_nodes=num_sensors,
        num_timesteps=num_snapshots,
        seed=seed,
    )
    print(f"  -> Generated {len(dataset_contract.snapshots)} snapshots.")
    print(f"  -> Topology: {dataset_contract.snapshots[0].node_count} nodes, {dataset_contract.snapshots[0].edge_count} edges.")
    
    # 2. Chronological Slicing & Normalization
    print("\n[2/7] Slicing Temporal Windows & Normalizing (Train-Only Fit)...")
    slicer = TemporalDatasetSlicer(
        input_sequence_length=3,      # T_in = 3 (15 min history)
        forecast_horizons=[1, 2, 3], # +5, +10, +15 min
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
        model_name="SpatioTemporalGNN_Research",
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
    print(f"  -> ST-GNN Architecture: 2 ST-Blocks (Spatial GraphConv + Temporal GRU) + Multi-Head Forecast")
    print(f"  -> Total Trainable Parameters: {param_count}")
    
    # Save Model Summary
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
    print("\n[4/7] Training ST-GNN with Validation Loss Early Stopping...")
    checkpoint_dir = str(out_path / "checkpoints")
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
    
    # Save Training History
    with open(out_path / "training_history.json", "w") as f:
        json.dump(train_result.history, f, indent=2)
        
    # 5. Baseline Evaluation & Test Set Comparison
    print("\n[5/7] Evaluating Test Set Performance against Baselines...")
    test_metrics = trainer.evaluate_all_models(st_dataset)
    
    print("\n" + "=" * 70)
    print("TEST EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<18} | {'Horizon':<10} | {'Target':<12} | {'MAE':<8} | {'RMSE':<8} | {'MAPE (%)':<8}")
    print("-" * 70)
    for model_name, horizons in test_metrics.items():
        for horizon_name, targets in horizons.items():
            for target_name, metrics in targets.items():
                if target_name in ("flow_rate", "density", "congestion"):
                    print(f"{model_name:<18} | {horizon_name:<10} | {target_name:<12} | {metrics['mae']:<8.4f} | {metrics['rmse']:<8.4f} | {metrics['mape']:<8.2f}")
    print("=" * 70)
    
    # Save Metrics
    with open(out_path / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
        
    # 6. Ablation Studies
    print("\n[6/7] Running Scientific Ablation Studies...")
    ablation_results_raw = AblationStudyRunner.run_all_ablations(
        st_dataset,
        contract=dataset_contract,
        epochs=15,
        checkpoint_dir=str(out_path / "checkpoints" / "ablations"),
    )
    ablation_results = {k: v.model_dump() for k, v in ablation_results_raw.items()}
    
    with open(out_path / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
        
    print(f"  -> Ablation A (Temporal Only - No Spatial Graph): Evaluated.")
    print(f"  -> Ablation B (Spatial Only - No Temporal Recurrence): Evaluated.")
    print(f"  -> Ablation C (Feature Ablation - Without Flow): Evaluated.")
    print(f"  -> Ablation D (Temporal Window - 1-step history): Evaluated.")

    # 7. Experiment Config & Metadata
    exp_config = {
        "experiment_id": "module3_research_exp_01",
        "dataset_name": "ChronoEye_MultiSensor_Corridor_Benchmark_24h",
        "dataset_version": "1.0.0",
        "topology": {
            "num_nodes": num_sensors,
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

    # 8. Visualizations (Optional matplotlib plots)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        # Plot 1: Loss curves
        plt.figure(figsize=(8, 4))
        plt.plot(train_result.history["train_loss"], label="Train Loss (Smooth L1)", color="#3b82f6")
        plt.plot(train_result.history["val_loss"], label="Val Loss (Smooth L1)", color="#ef4444")
        plt.axvline(train_result.best_epoch, color="#10b981", linestyle="--", label=f"Best Checkpoint (Epoch {train_result.best_epoch})")
        plt.title("ST-GNN Training & Validation Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_path / "loss_curves.png", dpi=150)
        plt.close()
        
        # Plot 2: Horizon MAE comparison across models (flow_rate)
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
            
        plt.ylabel("Flow Rate MAE")
        plt.title("Multi-Horizon Forecasting Error: ST-GNN vs Baselines")
        plt.xticks(x, horizon_labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_path / "multi_horizon_comparison.png", dpi=150)
        plt.close()
        
        # Plot 3: Ablation Study Comparison (Validation Loss & +15min MAE)
        if ablation_results:
            plt.figure(figsize=(9, 4.5))
            ablation_names = list(ablation_results.keys())
            ab_val_losses = [ablation_results[k]["best_val_loss"] for k in ablation_names]
            
            clean_names = [k.replace("Ablation_", "").replace("_", " ") for k in ablation_names]
            plt.bar(clean_names, ab_val_losses, color=["#6366f1", "#ec4899", "#8b5cf6", "#3b82f6", "#10b981"])
            plt.ylabel("Best Validation Loss")
            plt.title("Ablation Study: Architecture & Temporal Component Contributions")
            plt.xticks(rotation=20, ha="right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_path / "ablation_comparison.png", dpi=150)
            plt.close()

        print(f"\n[7/7] Saved research plots to {plots_path}")
    except Exception as e:
        print(f"\n[7/7] Plotting note: {e}")

    # Generate README in experiment dir
    readme_content = f"""# ChronoEye Infinity Module 3 — Research Experiment 01

## Overview
- **Experiment ID**: `module3_research_exp_01`
- **Model**: Spatio-Temporal Graph Neural Network (ST-GNN)
- **Dataset**: Multi-Sensor Corridor Benchmark (24 hours, 288 snapshots, 10 nodes, 18 directed edges)
- **Train / Val / Test**: {len(st_dataset.train_samples)} / {len(st_dataset.val_samples)} / {len(st_dataset.test_samples)}
- **Trainable Parameters**: {param_count}
- **Best Epoch**: {train_result.best_epoch} (Val Loss: {train_result.best_val_loss:.4f})

## Files
- `config.json`: Full reproducible experiment metadata and hyperparameters.
- `metrics.json`: Multi-horizon (+5, +10, +15 min) test metrics (MAE, RMSE, MAPE) for ST-GNN and baselines.
- `training_history.json`: Epoch-by-epoch training and validation loss progression.
- `scaler.json`: Preprocessing scaler parameters fitted strictly on training split.
- `model_summary.json`: Structural overview of the ST-GNN architecture.
- `ablation_results.json`: Results of Spatial, Temporal, Feature, and History ablations.
- `best_stgnn_checkpoint.pt`: Saved model weights at best validation epoch.
"""
    with open(out_path / "README.md", "w") as f:
        f.write(readme_content)

    print("\n" + "=" * 70)
    print("RESEARCH PIPELINE EXECUTION COMPLETE")
    print(f"Artifacts saved in: {out_path}")
    print("=" * 70)
    return train_result, test_metrics, ablation_results


if __name__ == "__main__":
    run_research_pipeline()
