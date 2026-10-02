"""
ChronoEye Infinity - Module 3 Ablation Study Runner
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.adapters.real_traffic_adapter import RealTrafficDatasetAdapter
from app.graph.temporal_dataset_slicer import TemporalDatasetSlicer
from app.training.ablation import AblationStudyRunner


EPOCHS = 10

PEMS08_PATH = ROOT / "data/research/pems08/PEMS08.npz"
PEMS08_CSV = ROOT / "data/research/pems08/PEMS08.csv"

RESULT_DIR = ROOT / "experiments/module3/research_exp_03_full_pems08"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints/ablations"
RESULT_FILE = RESULT_DIR / "ablation_results.json"


def main():

    print("=" * 70)
    print("ChronoEye Infinity - Module 3 Ablation Study")
    print("=" * 70)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    adapter = RealTrafficDatasetAdapter()

    # Compact arrays for memory-safe loading.
    compact_dataset, compact_metadata = adapter.load_pems08_compact(
        npz_path=str(PEMS08_PATH),
        csv_path=str(PEMS08_CSV),
    )

    print()
    print("=== COMPACT DATASET ===")
    print("Timesteps :", compact_dataset.features.shape[0])
    print("Nodes     :", compact_dataset.features.shape[1])
    print("Features  :", compact_dataset.features.shape)
    print("Targets   :", compact_dataset.targets.shape)

    # Full lazy dataset used by the standard ablation models.
    slicer = TemporalDatasetSlicer(
        input_sequence_length=12,
        forecast_horizons=compact_dataset.forecast_horizons,
        train_ratio=compact_dataset.metadata.get("train_ratio", 0.70),
        val_ratio=compact_dataset.metadata.get("val_ratio", 0.15),
        test_ratio=compact_dataset.metadata.get("test_ratio", 0.15),
    )

    dataset_full = slicer.slice_compact_dataset(compact_dataset)

    print()
    print("=== FULL LAZY DATASET ===")
    print("Train samples:", len(dataset_full.train_samples))
    print("Val samples  :", len(dataset_full.val_samples))
    print("Test samples :", len(dataset_full.test_samples))

    print()
    print("=== RUNNING 4 ABLATIONS ===")
    print("Epochs:", EPOCHS)

    results = AblationStudyRunner.run_all_ablations(
        dataset=dataset_full,
        compact_dataset=compact_dataset,
        epochs=EPOCHS,
        checkpoint_dir=str(CHECKPOINT_DIR),
    )

    serialized = {
        name: result.model_dump()
        for name, result in results.items()
    }

    RESULT_FILE.write_text(
        json.dumps(serialized, indent=2)
    )

    print()
    print("=== RESULTS ===")

    for name, result in results.items():
        print(
            name,
            "best_val_loss=",
            result.best_val_loss,
        )

    print()
    print("Results saved:")
    print(RESULT_FILE)

    print()
    print("Checkpoints:")

    for checkpoint in sorted(CHECKPOINT_DIR.glob("*.pt")):
        print(
            checkpoint.name,
            f"{checkpoint.stat().st_size / 1024:.1f} KB"
        )


if __name__ == "__main__":
    main()
