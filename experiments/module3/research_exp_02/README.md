# ChronoEye Infinity Module 3 — Research Experiment 02 (Real PeMS08)

## Overview
- **Experiment ID**: `module3_research_exp_02`
- **Dataset**: Caltrans PeMS08 Real-World Freeway Traffic Benchmark (District 8, San Bernardino)
- **Provenance**: Official Zenodo Record 7816008
- **Model**: Spatio-Temporal Graph Neural Network (ST-GNN)
- **Spatial Topology**: 30 loop detector nodes, 21 connectivity edges
- **Train / Val / Test**: 399 / 85 / 87
- **Trainable Parameters**: 17116
- **Best Epoch**: 16 (Val Loss: 0.0654)

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
