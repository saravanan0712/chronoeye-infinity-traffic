# ChronoEye Infinity

### Adaptive Spatio-Temporal Traffic Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=flat&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Overview

**ChronoEye Infinity** is a modular, end-to-end intelligent traffic system designed to unify edge computer vision, multi-camera vehicle tracking, topological graph analysis, and deep spatio-temporal forecasting.

The platform processes multi-source surveillance video streams, extracts high-fidelity vehicle trajectory and license plate evidence, reconstructs continuous cross-camera journeys across city-scale road networks, and forecasts network-wide traffic dynamics using a Spatio-Temporal Graph Neural Network (ST-GNN).

```
┌─────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│    MODULE 1     │     │       MODULE 2        │     │       MODULE 3        │
│ Video Perception│ ──> │ Multi-Camera ReID &   │ ──> │ Spatio-Temporal Graph │ ──> Command Dashboard
│ & Plate Fusion  │     │ Journey Reconstruction│     │ & ST-GNN Forecasting  │     & Navigation UI
└─────────────────┘     └───────────────────────┘     └───────────────────────┘
```

---

## 2. Problem Statement

Modern urban traffic management suffers from three major operational bottlenecks:
1. **Isolated Camera Perception**: Conventional computer vision systems treat each surveillance camera as an isolated island, losing vehicle identity and kinematic continuity as vehicles traverse non-overlapping blind spots.
2. **Noisy and Intermittent OCR Evidence**: Environmental factors (glare, motion blur, steep camera pitch) cause fluctuating license plate readings, leading to fragmented tracking and phantom vehicle entities.
3. **Static Traffic Forecasting**: Traditional signal controllers and routing systems rely on static historical averages rather than dynamically learning non-linear spatial dependencies across interconnected road topologies.

ChronoEye Infinity addresses these challenges through a unified multi-stage intelligence pipeline.

---

## 3. System Architecture

The ChronoEye Infinity architecture is organized into three distinct, decoupled core modules:

```
                                  CHRONOEYE INFINITY ARCHITECTURE
 ══════════════════════════════════════════════════════════════════════════════════════════
 
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ MODULE 1: PERCEPTION & TEMPORAL OCR FUSION                                             │
 │ • Video/Stream Ingestion (RTSP / MP4 / CCTV)                                           │
 │ • YOLOv8 Vehicle Detection (Cars, Trucks, Buses, Motorcycles)                          │
 │ • ByteTrack Multi-Object Tracking with Kinematic Motion Estimator                      │
 │ • License Plate Localization & PaddleOCR/EasyOCR Text Extraction                       │
 │ • Temporal Character-Level Probability Matrix & Evidence Fusion                        │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ MODULE 2: MULTI-CAMERA REID & JOURNEY RECONSTRUCTION (SynaptiMorph)                    │
 │ • 7-Signal Vehicle Association (Plate, Appearance, Spatial, Temporal, Kinematic, etc.) │
 │ • Global Vehicle ID (GVID) Lifecycle State Machine                                     │
 │ • Multi-Camera Handoff & Trajectory Stitching Across Unobserved Blind Spots             │
 │ • Journey Evidence Breakdown & Timestamp Uncertainty Quantification                    │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ MODULE 3: TOPOLOGICAL GRAPH & SPATIO-TEMPORAL FORECASTING (ST-GNN)                     │
 │ • Dynamic Graph Topology with Distance Kernels & Flow Transition Weighting             │
 │ • Degree-Normalized Spatial Graph Convolutions + Recurrent Temporal GRU Blocks         │
 │ • Multi-Horizon State Forecasting (+5 min, +10 min, +15 min Flow, Speed, Density)      │
 │ • Leakage-Free Chronological Slicing & Masked Loss Handling Unobserved Fields          │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ APPLICATION & INTERFACE LAYER                                                          │
 │ • FastAPI REST & WebSocket Backend Services                                            │
 │ • React + Vite + TypeScript Command Center Dashboard                                   │
 │ • Predictive Signal Optimization & Emergency Green Corridor Routing                    │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Module Breakdown

### Module 1: Video Perception & Temporal OCR Fusion
- **Frame Ingestion**: Modular frame reader supporting recorded video files (`.mp4`, `.avi`, `.mov`) and real-time streaming sources with strict timestamp accounting.
- **Vehicle Detection**: Fine-tuned YOLOv8 model providing high-speed bounding-box regression and vehicle category classification.
- **ByteTrack Tracking**: Upgraded ByteTrack tracking algorithm featuring kinematic velocity projection, low-confidence detection recovery, and duplicate suppression.
- **Stacked OCR & Temporal Fusion**: Combines multi-frame plate crops into a character-level probability distribution matrix, resolving occlusion and character ambiguities over time.

### Module 2: SynaptiMorph Multi-Camera ReID & Journey Reconstruction
- **Seven-Signal Re-Identification**: Associates vehicle observations across non-overlapping cameras using an ensemble similarity score:
  1. $S_{\text{plate}}$: Levenshtein distance and character-level confidence matching.
  2. $S_{\text{visual}}$: HSV color histograms and structural aspect ratios.
  3. $S_{\text{type}}$: Vehicle taxonomy compatibility (car, bus, truck).
  4. $S_{\text{temporal}}$: Time-gap feasibility based on corridor length.
  5. $S_{\text{spatial}}$: Road network adjacency and camera-pair connectivity.
  6. $S_{\text{kinematic}}$: Minimum and maximum feasible vehicle speeds.
  7. $S_{\text{motion}}$: Entry and exit trajectory direction vector alignment.
- **Journey Lifecycle**: Maintains persistent Global Vehicle IDs (GVIDs), generating structured journey legs with origin-destination timestamps and blind-spot travel estimations.

### Module 3: Spatio-Temporal Graph Neural Network (ST-GNN)
- **Topological Representation**: Road network modeled as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{W})$, where nodes represent camera stations / loop detectors and edges denote physical road segments.
- **Dynamic Adjacency**: Adjacency matrix dynamically modulated by transition flow intensity and mean journey speeds:
  $$\mathbf{A}_{\text{dynamic}}[i, j] = \mathbf{A}_{\text{static}}[i, j] \cdot \left(1.0 + \alpha \tanh(\text{flow\_factor} \cdot \text{speed\_factor})\right)$$
- **Degree-Normalized Spatial Graph Convolution**:
  $$\mathbf{H}^{(l+1)} = \sigma\left(\mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} + \mathbf{b}^{(l)}\right)$$
- **Temporal Sequence Modeling**: Stacked Gated Recurrent Units (GRU) process sequential temporal feature states across historical observation windows.
- **Multi-Horizon Output Heads**: Independent linear heads decode simultaneous predictions for $+5\text{ min}$, $+10\text{ min}$, and $+15\text{ min}$ horizons.

---

## 5. Empirical Research & Benchmark Validation

The ST-GNN forecasting engine has been rigorously evaluated on the public **Caltrans PeMS08** benchmark dataset (San Bernardino District 8 highway network, 170 loop detectors).

### Benchmark Evaluation (PeMS08 Test Split)

| Model Architecture | Horizon | Target Channel | MAE (Normalized) | RMSE | MAPE (%) |
|:---|:---:|:---|:---:|:---:|:---:|
| **ST-GNN (Ours)** | **+5 min** | `flow_rate` | **0.2190** | **0.2921** | **4.38%** |
| **ST-GNN (Ours)** | **+5 min** | `density` | **0.1612** | **0.2471** | **3.22%** |
| **ST-GNN (Ours)** | **+5 min** | `congestion` | **0.1302** | **0.1862** | **2.60%** |
| **ST-GNN (Ours)** | **+10 min** | `flow_rate` | **0.2405** | **0.3266** | **4.81%** |
| **ST-GNN (Ours)** | **+10 min** | `density` | **0.1771** | **0.2746** | **3.54%** |
| **ST-GNN (Ours)** | **+10 min** | `congestion` | **0.1445** | **0.2219** | **2.89%** |
| **ST-GNN (Ours)** | **+15 min** | `flow_rate` | **0.2501** | **0.3399** | **5.00%** |
| **ST-GNN (Ours)** | **+15 min** | `density` | **0.1948** | **0.2904** | **3.90%** |
| **ST-GNN (Ours)** | **+15 min** | `congestion` | **0.1602** | **0.2452** | **3.20%** |
| *Persistence Baseline* | *+15 min* | `flow_rate` | *0.3421* | *0.4812* | *8.14%* |
| *Moving Average ($W=3$)* | *+15 min* | `flow_rate` | *0.3610* | *0.4990* | *8.62%* |
| *Linear Ridge Regression* | *+15 min* | `flow_rate` | *0.2980* | *0.4120* | *6.45%* |

*Data Leakage Guarantee*: All normalizers are fitted exclusively on the 70% training split. Validation (15%) and testing (15%) are strictly chronological.

---

## 6. Repository Layout

```
chronoeye-infinity/
├── backend/
│   ├── app/
│   │   ├── adapters/          # Benchmark and real traffic data adapters
│   │   ├── api/               # FastAPI route handlers and WebSocket endpoints
│   │   ├── benchmarks/        # Simulation and baseline benchmarks
│   │   ├── core/              # Global configurations and logging
│   │   ├── forecasting/       # ST-GNN forecasting engine and inference wrappers
│   │   ├── forensics/         # Video forensic replay and search tools
│   │   ├── graph/             # Dynamic topology, snapshot schemas, and temporal slicer
│   │   ├── incident/          # Anomaly detection and emergency green corridor
│   │   ├── models/            # Canonical ST-GNN neural network architecture
│   │   │   └── stgnn/         # PyTorch layers, loss functions, and model definitions
│   │   ├── optimization/      # Predictive signal control and A* route planning
│   │   ├── perception/        # Frame ingestion, YOLO detection, ByteTrack, ALPR
│   │   ├── research/          # Empirical metrics, baseline models, serializers
│   │   ├── schemas/           # Pydantic data contracts
│   │   ├── simulation/        # Synthetic traffic flow and incident generator
│   │   ├── state/             # Network traffic state and history tracker
│   │   └── training/          # Model trainer, ablation runner, baseline estimators
│   ├── models/                # License plate detector weights
│   └── tests/                 # Comprehensive test suite (47 test modules)
│
├── frontend/
│   ├── src/
│   │   ├── components/        # TrafficMap, JourneyTimeline, Forensics, ForecastUI
│   │   ├── hooks/             # Custom React hooks for telemetry and WebSockets
│   │   ├── services/          # REST and WebSocket API client wrappers
│   │   ├── types/             # TypeScript interface definitions
│   │   ├── App.tsx            # Main Command Center dashboard layout
│   │   └── index.css          # Design system stylesheet
│   ├── index.html             # Single-page application template
│   ├── package.json           # Node.js dependencies and build scripts
│   └── vite.config.ts         # Vite build configuration
│
├── data/
│   ├── diagnostic_crops/      # Ground truth evaluation crops and verification labels
│   ├── research/
│   │   └── pems08/            # Caltrans PeMS08 NPZ dataset and road distance CSV
│   └── videos/                # Sample multi-camera evaluation video
│
├── experiments/
│   └── module3/
│       ├── research_exp_01/   # Synthetic topology experiment artifacts
│       └── research_exp_02/   # PeMS08 real benchmark experiment artifacts
│
├── scripts/                   # Master research execution and diagnostic scripts
├── checkpoints/               # Stage milestone frozen baseline checkpoints
├── docs/                      # Research reports, architecture specs, and installation guides
├── .env.example               # Environment variables configuration template
├── .gitattributes             # Git LFS tracking configuration for datasets & weights
├── .gitignore                 # Comprehensive ignore rules
├── requirements.txt           # Python backend dependencies
└── README.md                  # Project documentation
```

---

## 7. Installation & Setup

### Prerequisites
- **Python**: 3.10+ (Recommended)
- **Node.js**: 18+ (with `npm 9+`)
- **Git LFS**: Installed (`git lfs install`)
- **Hardware**: CPU supported; CUDA-enabled GPU recommended for training.

### 1. Clone the Repository
```bash
git clone https://github.com/saravanan0712/chronoeye-infinity.git
cd chronoeye-infinity
git lfs pull
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

---

## 8. Running the Application

### Start the FastAPI Backend
```bash
# From workspace root
python -m uvicorn backend.app.api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at: `http://localhost:8000/docs`.

### Start the Frontend Dashboard
```bash
# From frontend directory
cd frontend
npm run dev
```
Open your browser at: `http://localhost:5173`.

---

## 9. Running Research Experiments

### Execute Real PeMS08 ST-GNN Benchmark
```bash
python scripts/run_module3_real_experiment.py
```
This script executes:
1. PeMS08 ingestion and schema adaptation.
2. Chronological slicing with training-only Z-score standardization.
3. ST-GNN model training with validation loss early stopping.
4. Evaluation across Persistence, Moving Average, and Ridge Regression baselines.
5. Ablation studies (Temporal-Only, Spatial-Only, Shorter History).
6. Artifact persistence into `experiments/module3/research_exp_02/`.

### Google Colab GPU Workflow
To train on Google Colab with GPU acceleration:
```python
# 1. Mount and install dependencies
!git clone https://github.com/saravanan0712/chronoeye-infinity.git
%cd chronoeye-infinity
!pip install -r requirements.txt

# 2. Run Module 3 experiment on CUDA
!python scripts/run_module3_real_experiment.py
```

---

## 10. Test Suite & Validation

Run the automated regression test suite across perception, ReID, graph engines, and ST-GNN forecasters:

```bash
pytest backend/tests -v
```

---

## 11. Limitations & Future Work

- **Camera Calibration**: Current speed estimation relies on calibrated pixel-to-meter scaling factors; future releases will incorporate automated homography estimation from road lane geometry.
- **Edge Deployment**: Quantization of YOLO and ST-GNN models (TensorRT / ONNX Runtime) for embedded edge devices (NVIDIA Jetson).
- **Transformer Expansion**: Exploring Spatio-Temporal Graph Attention Transformers (ST-GAT) for extended multi-hour forecasting horizons.

---

## 12. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Saravanan R**  
*AI & Data Science Student | AI/ML Developer*  
- GitHub: [@saravanan0712](https://github.com/saravanan0712)
- Project: [ChronoEye Infinity](https://github.com/saravanan0712/chronoeye-infinity)
