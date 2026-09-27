# ChronoEye Infinity Module 3 — Research Experiment 03 (Full PeMS08 Benchmark)

## Executive Summary
- **Experiment ID**: `module3_research_exp_03_full_pems08`
- **Dataset**: Full Caltrans PeMS08 Real-World Freeway Traffic Benchmark (District 8, San Bernardino)
- **Provenance**: Official Zenodo Record 7816008
- **Model**: Spatio-Temporal Graph Neural Network (ST-GNN)
- **Spatial Topology**: 170 loop detector nodes, 295 physical road connections
- **Temporal Resolution**: 5-minute intervals across 576 timesteps (~62 consecutive days)
- **Input History Window**: 12 steps (60 minutes)
- **Forecast Horizons**: +5 min, +10 min, +15 min
- **Dataset Splits (Chronological)**:
  - Train: 393 samples (70%)
  - Validation: 84 samples (15%)
  - Test: 85 samples (15%)
  - Total: 562 samples
- **Memory Scaling Architecture**: Lazy Spatio-Temporal Dataset with True GPU Mini-Batching (DataLoader streaming)
- **Trainable Parameters**: 17116
- **Device**: cpu
- **Best Validation Loss**: 0.1829 (Epoch 2)

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
