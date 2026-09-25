"""
ChronoEye Infinity - Benchmark Result Serialization Engine
Serializes raw and summary benchmark results to JSON and CSV formats under data/research/.
"""

import os
import json
import csv
from typing import Dict, Any


def serialize_benchmark_results(results: Dict[str, Any], output_dir: str = "e:/chronoeye/data/research") -> None:
    """Serializes benchmark results to JSON and CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save JSON
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # 2. Save CSV Summary
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "seed", "controller", "mean_travel_time", "avg_delay", "avg_queue", "throughput"])
        for record in results.get("raw_records", []):
            writer.writerow([
                record.get("scenario"),
                record.get("seed"),
                record.get("controller"),
                record.get("mean_travel_time"),
                record.get("avg_delay"),
                record.get("avg_queue_length"),
                record.get("throughput"),
            ])
