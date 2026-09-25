# ChronoEye Infinity Module 3 — Research Experiment 01

## Overview
- **Experiment ID**: `module3_research_exp_01`
- **Model**: Spatio-Temporal Graph Neural Network (ST-GNN)
- **Dataset**: Multi-Sensor Corridor Benchmark (24 hours, 288 snapshots, 10 nodes, 18 directed edges)
- **Train / Val / Test**: 198 / 42 / 43
- **Trainable Parameters**: 17116
- **Best Epoch**: 5 (Val Loss: 0.3921)

## Files
- `config.json`: Full reproducible experiment metadata and hyperparameters.
- `metrics.json`: Multi-horizon (+5, +10, +15 min) test metrics (MAE, RMSE, MAPE) for ST-GNN and baselines.
- `training_history.json`: Epoch-by-epoch training and validation loss progression.
- `scaler.json`: Preprocessing scaler parameters fitted strictly on training split.
- `model_summary.json`: Structural overview of the ST-GNN architecture.
- `ablation_results.json`: Results of Spatial, Temporal, Feature, and History ablations.
- `best_stgnn_checkpoint.pt`: Saved model weights at best validation epoch.
