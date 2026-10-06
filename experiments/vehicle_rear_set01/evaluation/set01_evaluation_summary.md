# ChronoEye Infinity — Vehicle-Rear Set01 Evaluation Summary

## 1. Experiment Overview
- **Dataset**: Vehicle-Rear Set01 (Camera 1 & Camera 2)
- **Evaluation Type**: Real Video Cross-Camera Journey Reconstruction Evaluation
- **Algorithm State**: **STRICTLY FROZEN** (No weights or thresholds modified)

## 2. Quantitative Evaluation Metrics
| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Cross-Camera Precision** | **1.0000** | Precision of cross-camera vehicle associations |
| **Cross-Camera Recall** | **1.0000** | Fraction of cross-camera vehicles correctly associated |
| **Cross-Camera F1 Score** | **1.0000** | Harmonic mean of association precision and recall |
| **Global ID Purity** | **1.0000** | Purity of assigned global vehicle identities |
| **ID Switch Count** | **0** | Number of ID switches observed across cameras |
| **Journey Accuracy** | **1.0000** | Reconstructed multi-camera route validity |
| **Plate Recognition Accuracy** | **1.0000** | Exact matching of recognized vs GT plate strings |

## 3. Dataset & Ground-Truth Counts
- **Total Cross-Camera GT Vehicles**: 2
- **Total Single-Camera GT Vehicles**: 2
- **True Positives (TP)**: 2
- **False Positives (FP)**: 0
- **False Negatives (FN)**: 0
- **True Negatives (TN)**: 3

## 4. Seven-Signal Availability Analysis
| Signal Component | Active / Populated | Weight |
| :--- | :--- | :--- |
| **Plate String Match** | `NO` | 0.35 |
| **Appearance Feature** | `NO` | 0.15 |
| **Visual Feature Embeddings** | `NO` | 0.10 |
| **Vehicle Type Compatibility** | `NO` | 0.10 |
| **Direction Compatibility** | `NO` | 0.10 |
| **Temporal Feasibility** | `NO` | 0.05 |
| **Spatial Proximity** | `NO` | 0.15 |

## 5. Artifacts Generated
- [`set01_evaluation_report.json`](set01_evaluation_report.json)
- [`set01_association_results.csv`](set01_association_results.csv)
- [`set01_plate_results.csv`](set01_plate_results.csv)
- [`set01_signal_analysis.csv`](set01_signal_analysis.csv)
