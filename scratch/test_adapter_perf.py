import time
import sys
from pathlib import Path

repo_root = Path(".").resolve()
sys.path.insert(0, str(repo_root / "backend"))

from app.adapters.real_traffic_adapter import RealTrafficDatasetAdapter
from app.graph.temporal_dataset_slicer import TemporalDatasetSlicer

print("Testing loading speed on 1000 timesteps...")
t0 = time.time()
contract, meta = RealTrafficDatasetAdapter.load_pems08(
    npz_path="data/research/pems08/PEMS08.npz",
    csv_path="data/research/pems08/PEMS08.csv",
    num_nodes=170,
    num_timesteps=1000,
)
t1 = time.time()
print(f"Loaded 1000 snapshots in {t1 - t0:.2f}s. Number of snapshots: {len(contract.snapshots)}")

t0 = time.time()
slicer = TemporalDatasetSlicer(
    input_sequence_length=12,
    forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    normalization_method="zscore",
)
st_dataset = slicer.slice_dataset(contract)
t1 = time.time()
print(f"Sliced in {t1 - t0:.2f}s: Train={len(st_dataset.train_samples)}, Val={len(st_dataset.val_samples)}, Test={len(st_dataset.test_samples)}")
