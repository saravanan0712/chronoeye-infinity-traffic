"""
Master Research Execution Script for ChronoEye Infinity Module 3 — Experiment 03 (Full PeMS08 Benchmark).
Processes the full Caltrans PeMS08 research dataset (17,856 timesteps x 170 nodes x 8 features)
with memory-hardened lazy spatio-temporal slicing and streaming GPU mini-batching.

Pipeline Steps:
  1. Compact Real Dataset Ingestion & Topology Loading (RealTrafficDatasetAdapter)
  2. Lazy Chronological Spatio-Temporal Slicing & Train-Only Scaler Fitting (TemporalDatasetSlicer)
  3. ST-GNN Model Architecture Initialization on Target Device (CUDA/CPU)
  4. Memory-Safe Mini-Batch ST-GNN Training with Early Stopping (STGNNTrainer)
  5. Multi-Horizon Baseline Model Evaluation (+5, +10, +15 min)
  6. Scientific Ablation Studies on Full Topology
  7. Experiment 03 Artifact Persistence (JSON configs, metrics, scalers, checkpoints)
  8. Research Visualization Plot Generation
"""

import os
import sys
import json
import time
import argparse
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


def run_full_pems08_experiment(
    npz_path: str = "data/research/pems08/PEMS08.npz",
    csv_path: str = "data/research/pems08/PEMS08.csv",
    output_dir: str = "experiments/module3/research_exp_03_full_pems08",
    num_nodes: int = 170,
    num_timesteps: int = 17856,
    input_sequence_length: int = 12,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.003,
    seed: int = 42,
    run_ablations: bool = True,
):
    print("=" * 75)
    print("CHRONOEYE INFINITY — MODULE 3 EXPERIMENT 03 (FULL PEMS08 MEMORY-SCALED)")
    print("=" * 75)

    # 0. Reproducibility & Device Selection
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Target Device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Model: {torch.cuda.get_device_name(0)}")

    out_path = repo_root / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    plots_path = out_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(out_path / "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Dataset Ingestion & Adaptation (Compact Representation)
    print(f"\n[1/7] Ingesting Caltrans PeMS08 Real Traffic Dataset (Compact Array Mode)...")
    compact_dataset, meta = RealTrafficDatasetAdapter.load_pems08_compact(
        npz_path=str(repo_root / npz_path),
        csv_path=str(repo_root / csv_path),
        num_nodes=num_nodes,
        num_timesteps=num_timesteps,
    )
    print(f"  -> Dataset: {meta['dataset_name']} ({meta['source_provenance']})")
    print(f"  -> Total Timesteps: {compact_dataset.features.shape[0]} ({compact_dataset.features.shape[0]*5/60:.1f} hours / ~62 days).")
    print(f"  -> Spatial Nodes: {compact_dataset.node_count} loop detectors.")
    print(f"  -> Topology Edges: {compact_dataset.edge_count} physical road connections.")
    print(f"  -> Features Array Shape: {compact_dataset.features.shape} ({compact_dataset.features.nbytes / 1024 / 1024:.2f} MB)")
    print(f"  -> Targets Array Shape:  {compact_dataset.targets.shape} ({compact_dataset.targets.nbytes / 1024 / 1024:.2f} MB)")

    # Save dataset metadata, target availability, node mapping
    with open(out_path / "dataset_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(out_path / "target_availability.json", "w") as f:
        json.dump(meta["target_availability"], f, indent=2)

    with open(out_path / "node_mapping.json", "w") as f:
        json.dump(meta["node_mapping"], f, indent=2)

    # 2. Chronological Slicing & Train-Only Normalization
    print("\n[2/7] Slicing Temporal Windows (Lazy Spatio-Temporal Dataset, Train-Only Scaler)...")
    slicer = TemporalDatasetSlicer(
        input_sequence_length=input_sequence_length,  # T_in = 12 (60 min history)
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],  # +5, +10, +15 min
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        normalization_method="zscore",
    )
    st_dataset = slicer.slice_compact_dataset(compact_dataset)
    print(f"  -> Train samples: {len(st_dataset.train_samples)}")
    print(f"  -> Val samples:   {len(st_dataset.val_samples)}")
    print(f"  -> Test samples:  {len(st_dataset.test_samples)}")
    print(f"  -> Total samples: {len(st_dataset.train_samples) + len(st_dataset.val_samples) + len(st_dataset.test_samples)}")

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
        model_name="SpatioTemporalGNN_PeMS08_Full_Scaled",
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
        "device": str(device),
        "random_seed": seed,
    }
    with open(out_path / "model_summary.json", "w") as f:
        json.dump(model_summary, f, indent=2)

    # 4. Training with DataLoader Streaming
    print(f"\n[4/7] Training ST-GNN on Full PeMS08 Real Data ({epochs} epochs, batch_size={batch_size})...")
    trainer = STGNNTrainer(
        model=model,
        config=stgnn_config,
        loss_type="smooth_l1",
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    train_result = trainer.train(
        dataset=st_dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
    )
    print(f"  -> Training completed in {train_result.training_time_seconds:.2f}s.")
    print(f"  -> Best Validation Loss: {train_result.best_val_loss:.4f} at Epoch {train_result.best_epoch}.")

    with open(out_path / "training_history.json", "w") as f:
        json.dump(train_result.history, f, indent=2)

    # 5. Baseline Evaluation & Test Set Comparison
    print("\n[5/7] Evaluating Test Set Performance against Baselines...")
    test_metrics = trainer.evaluate_all_models(st_dataset)

    print("\n" + "=" * 75)
    print("FULL PEMS08 REAL TEST EVALUATION RESULTS SUMMARY")
    print("=" * 75)
    print(f"{'Model':<18} | {'Horizon':<10} | {'Target':<12} | {'MAE':<8} | {'RMSE':<8} | {'MAPE (%)':<8}")
    print("-" * 75)
    for model_name, horizons in test_metrics.items():
        for horizon_name, targets in horizons.items():
            for target_name, metrics in targets.items():
                if target_name in ("flow_rate", "density", "congestion"):
                    print(f"{model_name:<18} | {horizon_name:<10} | {target_name:<12} | {metrics['mae']:<8.4f} | {metrics['rmse']:<8.4f} | {metrics['mape']:<8.2f}")
    print("=" * 75)

    with open(out_path / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # 6. Ablation Studies
    ablation_results = {}
    if run_ablations:
        print("\n[6/7] Running Scientific Ablation Studies on PeMS08 Full Topology...")
        try:
            ablation_results_raw = AblationStudyRunner.run_all_ablations(
                st_dataset,
                compact_dataset=compact_dataset,
                epochs=min(epochs, 10),
                checkpoint_dir=str(out_path / "checkpoints" / "ablations"),
                device=device,
            )
            ablation_results = {k: v.model_dump() for k, v in ablation_results_raw.items()}
            print(f"  -> Ablation A (Temporal Only): Evaluated.")
            print(f"  -> Ablation B (Spatial Only): Evaluated.")
            print(f"  -> Ablation D (1-Step History): Evaluated.")
        except Exception as e:
            print(f"  -> Ablation runner notice: {e}")
            ablation_results = {"status": "skipped", "reason": str(e)}

    with open(out_path / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    # 7. Experiment 03 Configuration Metadata
    exp_config = {
        "experiment_id": "module3_research_exp_03_full_pems08",
        "dataset_name": "Caltrans_PeMS08_Full_Real",
        "dataset_version": "1.0.0",
        "provenance": {
            "source": "California Department of Transportation Performance Measurement System",
            "district": "District 8 (San Bernardino Freeway Network)",
            "doi": "10.5281/zenodo.7816008",
        },
        "topology": {
            "num_nodes": compact_dataset.node_count,
            "num_edges": compact_dataset.edge_count,
            "directed": True,
            "adjacency_normalization": "Symmetric Degree Normalization (D^-1/2 A D^-1/2)",
        },
        "temporal_parameters": {
            "snapshot_interval_seconds": 300,
            "total_timesteps": compact_dataset.features.shape[0],
            "history_window_steps": input_sequence_length,
            "history_window_minutes": input_sequence_length * 5,
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
            "total_samples": len(st_dataset.train_samples) + len(st_dataset.val_samples) + len(st_dataset.test_samples),
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
            "device": str(device),
        },
        "memory_optimization": {
            "lazy_dataset_loading": True,
            "gpu_streaming_minibatch": True,
            "train_only_normalization": True,
            "compact_numpy_source_arrays": True,
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
        plt.title("Full PeMS08 (17,856 steps): ST-GNN Training & Validation Loss")
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
        plt.title("Full PeMS08 Real Traffic: Multi-Horizon Forecast Error (ST-GNN vs Baselines)")
        plt.xticks(x, horizon_labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_path / "multi_horizon_comparison.png", dpi=150)
        plt.close()

        # Plot 3: Ablation Study
        if ablation_results and "status" not in ablation_results:
            plt.figure(figsize=(9, 4.5))
            ablation_names = list(ablation_results.keys())
            ab_val_losses = [ablation_results[k]["best_val_loss"] for k in ablation_names]
            clean_names = [k.replace("Ablation_", "").replace("_", " ") for k in ablation_names]
            plt.bar(clean_names, ab_val_losses, color=["#6366f1", "#ec4899", "#8b5cf6", "#3b82f6", "#10b981"])
            plt.ylabel("Best Validation Loss")
            plt.title("Full PeMS08 Real Traffic: Ablation Study Comparison")
            plt.xticks(rotation=20, ha="right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_path / "ablation_comparison.png", dpi=150)
            plt.close()

        print(f"\n[7/7] Saved full dataset research plots to {plots_path}")
    except Exception as e:
        print(f"\n[7/7] Plotting note: {e}")

    # Generate README in experiment dir
    best_model_chk = os.path.join(checkpoint_dir, "best_stgnn_model.pt")
    readme_content = f"""# ChronoEye Infinity Module 3 — Research Experiment 03 (Full PeMS08 Benchmark)

## Executive Summary
- **Experiment ID**: `module3_research_exp_03_full_pems08`
- **Dataset**: Full Caltrans PeMS08 Real-World Freeway Traffic Benchmark (District 8, San Bernardino)
- **Provenance**: Official Zenodo Record 7816008
- **Model**: Spatio-Temporal Graph Neural Network (ST-GNN)
- **Spatial Topology**: {compact_dataset.node_count} loop detector nodes, {compact_dataset.edge_count} physical road connections
- **Temporal Resolution**: 5-minute intervals across {compact_dataset.features.shape[0]} timesteps (~62 consecutive days)
- **Input History Window**: {input_sequence_length} steps (60 minutes)
- **Forecast Horizons**: +5 min, +10 min, +15 min
- **Dataset Splits (Chronological)**:
  - Train: {len(st_dataset.train_samples)} samples (70%)
  - Validation: {len(st_dataset.val_samples)} samples (15%)
  - Test: {len(st_dataset.test_samples)} samples (15%)
  - Total: {len(st_dataset.train_samples) + len(st_dataset.val_samples) + len(st_dataset.test_samples)} samples
- **Memory Scaling Architecture**: Lazy Spatio-Temporal Dataset with True GPU Mini-Batching (DataLoader streaming)
- **Trainable Parameters**: {param_count}
- **Device**: {device}
- **Best Validation Loss**: {train_result.best_val_loss:.4f} (Epoch {train_result.best_epoch})

## Experiment Artifacts
- `config.json`: Master hyperparameter, topology, and split configuration.
- `dataset_metadata.json`: Dataset provenance, feature dimensions, and target channels.
- `target_availability.json`: Explicit target availability across flow rate, density, congestion, travel time.
- `node_mapping.json`: Deterministic sensor ID to graph node index mapping.
- `metrics.json`: Final test evaluation metrics across ST-GNN and baselines.
- `training_history.json`: Per-epoch train and validation loss curves.
- `scaler.json`: Train-split-only normalization parameters (feature and target mean/std).
- `model_summary.json`: Structural overview of ST-GNN architecture and layer shapes.
- `ablation_results.json`: Results of Spatial, Temporal, and History ablations on the full PeMS08 network.
- `checkpoints/best_stgnn_model.pt`: Saved model weights at the optimal validation epoch.
"""
    with open(out_path / "README.md", "w") as f:
        f.write(readme_content)

    print("\n" + "=" * 75)
    print("EXPERIMENT 03 (FULL PEMS08 MEMORY-SCALED) EXECUTION COMPLETE")
    print(f"Artifacts saved in: {out_path}")
    print("=" * 75)
    return train_result, test_metrics, ablation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ChronoEye Module 3 Experiment 03 (Full PeMS08)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--timesteps", type=int, default=17856, help="Number of timesteps to process")
    parser.add_argument("--nodes", type=int, default=170, help="Number of nodes")
    parser.add_argument("--history", type=int, default=12, help="Input history sequence length")
    parser.add_argument("--skip_ablations", action="store_true", help="Skip ablation studies")

    args = parser.parse_args()

    run_full_pems08_experiment(
        num_nodes=args.nodes,
        num_timesteps=args.timesteps,
        input_sequence_length=args.history,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        run_ablations=not args.skip_ablations,
    )
