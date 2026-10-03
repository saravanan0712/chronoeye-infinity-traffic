# ChronoEye Infinity — Vehicle-Rear Set01 Cross-Camera Evaluation Harness

## 1. Overview
This experiment directory houses the quantitative evaluation pipeline for ChronoEye Infinity Module 2 on the **Vehicle-Rear Set01** two-camera benchmark.

The pipeline measures how the existing production perception, tracking, ANPR, and Module 2 Re-ID journey reconstruction stack performs on real cross-camera traffic footage.

> **STRICT RESEARCH PROTOCOL**:
> - Production algorithms, seven-signal weights, formulas, thresholds, and filters are **FROZEN**.
> - Ground-truth annotations are strictly isolated and evaluated **post-inference only**.

---

## 2. Directory Layout
```
experiments/vehicle_rear_set01/
├── README.md               # Documentation and protocol specification
├── config.json             # Configuration parameters and paths
├── ground_truth/           # Normalized GT artifacts (set01_ground_truth.json, set01_cross_camera_gt.json)
├── predictions/            # Raw pipeline inference output (set01_predictions.json)
├── evaluation/             # Final evaluation reports, summaries, and CSV tables
│   ├── set01_evaluation_report.json
│   ├── set01_evaluation_summary.md
│   ├── set01_association_results.csv
│   ├── set01_plate_results.csv
│   └── set01_signal_analysis.csv
└── logs/                   # Execution logs
```

---

## 3. Running the Evaluation

### Step 1: Ground-Truth Preparation
Extracts and normalizes annotations from Camera 1 and Camera 2 `vehicles.xml`:
```bash
python scripts/prepare_vehicle_rear_set01_gt.py \
    --dataset-root /path/to/Vehicle-Rear/extracted \
    --output-dir experiments/vehicle_rear_set01/ground_truth
```

### Step 2: Evaluation Run
Executes inference via `MultiCameraRunner` and calculates all cross-camera metrics against ground truth:
```bash
python scripts/evaluate_vehicle_rear_set01.py \
    --config experiments/vehicle_rear_set01/config.json \
    --device cpu
```
