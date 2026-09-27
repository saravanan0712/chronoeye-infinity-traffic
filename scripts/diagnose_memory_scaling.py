"""
ChronoEye Infinity - Module 3 Memory Hardening Diagnostic Script.
Measures process RSS memory and GPU memory footprint during:
1. Process initialization
2. Full PeMS08 compact loading (17,856 timesteps, 170 nodes)
3. SpatioTemporalDataset sequence slicing & train-only normalization
4. PyTorch DataLoader instantiation
5. First mini-batch creation & GPU transfer
6. ST-GNN forward and loss pass
"""

import os
import sys
import gc
import psutil
import torch
import numpy as np
from pathlib import Path

# Add backend to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "backend"))

from app.adapters.real_traffic_adapter import RealTrafficDatasetAdapter
from app.graph.temporal_dataset_slicer import TemporalDatasetSlicer
from app.models.stgnn.config import STGNNConfig
from app.models.stgnn.model import SpatioTemporalGNN
from app.models.stgnn.loss import MaskedLoss


def get_ram_mb() -> float:
    """Returns current process Resident Set Size (RSS) in Megabytes."""
    gc.collect()
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_mb() -> float:
    """Returns current GPU allocated memory in Megabytes."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def run_diagnostic():
    print("=" * 70)
    print("CHRONOEYE INFINITY — MODULE 3 FULL PEMS08 MEMORY DIAGNOSTIC")
    print("=" * 70)

    # 1. Baseline Memory
    ram_init = get_ram_mb()
    gpu_init = get_gpu_mb()
    print(f"\n[1] Baseline RAM before loading:       {ram_init:8.2f} MB")
    print(f"    Baseline GPU memory:              {gpu_init:8.2f} MB")

    # 2. Compact PeMS08 Loading
    npz_path = str(repo_root / "data/research/pems08/PEMS08.npz")
    csv_path = str(repo_root / "data/research/pems08/PEMS08.csv")

    if not Path(npz_path).exists():
        print(f"Error: {npz_path} not found.")
        return

    compact_data, meta = RealTrafficDatasetAdapter.load_pems08_compact(
        npz_path=npz_path,
        csv_path=csv_path,
        num_timesteps=17856,
        num_nodes=170,
    )
    ram_after_load = get_ram_mb()
    delta_load = ram_after_load - ram_init
    print(f"\n[2] RAM after Full PeMS08 compact load:{ram_after_load:8.2f} MB (+{delta_load:.2f} MB)")
    print(f"    Loaded Timesteps:                 {compact_data.num_timesteps}")
    print(f"    Loaded Nodes:                     {compact_data.num_nodes}")
    print(f"    Features Array Shape:             {compact_data.features.shape} ({compact_data.features.nbytes / (1024*1024):.2f} MB)")
    print(f"    Targets Array Shape:              {compact_data.targets.shape} ({compact_data.targets.nbytes / (1024*1024):.2f} MB)")

    # 3. Slicing & Train-Only Normalization
    slicer = TemporalDatasetSlicer(
        input_sequence_length=12,
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        normalization_method="zscore",
    )
    dataset = slicer.slice_dataset(compact_data)
    ram_after_slicing = get_ram_mb()
    delta_slice = ram_after_slicing - ram_after_load
    print(f"\n[3] RAM after Lazy Dataset creation:   {ram_after_slicing:8.2f} MB (+{delta_slice:.2f} MB)")
    print(f"    Total Supervised Samples:         {dataset.total_samples}")
    print(f"    Train Split Samples:              {len(dataset.train_samples)}")
    print(f"    Val Split Samples:                {len(dataset.val_samples)}")
    print(f"    Test Split Samples:               {len(dataset.test_samples)}")

    # 4. DataLoader & First Batch
    train_loader = dataset.get_dataloader("train", batch_size=16, shuffle=True)
    batch = next(iter(train_loader))
    bx, bx_mask, by, by_mask = batch
    ram_during_batch = get_ram_mb()

    print(f"\n[4] RAM during first DataLoader batch: {ram_during_batch:8.2f} MB")
    print(f"    Batch X shape:                    {list(bx.shape)}")
    print(f"    Batch X_mask shape:               {list(bx_mask.shape)}")
    print(f"    Batch Y shape:                    {list(by.shape)}")
    print(f"    Batch Y_mask shape:               {list(by_mask.shape)}")

    # 5. Model Forward Pass & GPU Memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = STGNNConfig(
        input_dim=8,
        output_dim=4,
        hidden_dim=32,
        forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],
        num_st_blocks=2,
    )
    model = SpatioTemporalGNN(config).to(device)
    loss_fn = MaskedLoss(loss_type="smooth_l1")

    gpu_before_batch = get_gpu_mb()
    adj_matrix = torch.tensor(dataset.adjacency.adjacency_matrix, dtype=torch.float32).to(device)

    bx_dev = bx.to(device)
    bx_mask_dev = bx_mask.to(device)
    by_dev = by.to(device)
    by_mask_dev = by_mask.to(device)

    pred = model(bx_dev, adj_matrix, feature_mask=bx_mask_dev)
    loss = loss_fn(pred, by_dev, mask=by_mask_dev)

    gpu_after_batch = get_gpu_mb()
    ram_after_forward = get_ram_mb()

    print(f"\n[5] Target Device:                    {device}")
    print(f"    GPU memory before batch transfer: {gpu_before_batch:8.2f} MB")
    print(f"    GPU memory after forward pass:    {gpu_after_batch:8.2f} MB")
    print(f"    Model Predictions Shape:          {list(pred.shape)}")
    print(f"    Batch Loss Value:                 {loss.item():.4f}")
    print(f"    Final Process RAM:                {ram_after_forward:8.2f} MB")

    print("\n" + "=" * 70)
    print("MEMORY HARDENING DIAGNOSTIC PASSED: NO RUNTIME EXPLOSION")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostic()
