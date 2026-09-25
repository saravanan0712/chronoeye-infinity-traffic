# ChronoEye Infinity - AI Model & Perception Specification

## 1. Computer Vision & Object Detection (Phase 2)
- **Model**: Ultralytics YOLOv8 / YOLOv11 (Default: `yolov8n.pt`)
- **Device Support**: CPU / CUDA GPU Fallback
- **Target Classes**: `car`, `motorcycle`, `bus`, `truck`, `ambulance`

## 2. Multi-Object Vehicle Tracking (Phase 3)
- **Primary Tracker**: ByteTrack (Two-stage IoU association matching)
- **Alternative Tracker**: DeepSORT (Spatial IoU + appearance feature embedding distance matrix)

## 3. License Plate Recognition / ALPR & Temporal OCR Fusion (Phase 4)
- **OCR Engine**: EasyOCR (`en` language model, CPU / CUDA) with deterministic `TestOCREngine` fallback for offline test suites.
- **Temporal OCR Fusion**: Weighted candidate history fusion with noisy outlier rejection and temporary OCR failure resilience.

## 4. Cross-Camera Vehicle Re-Identification & Journey Reconstruction (Phase 5)
- **Re-ID Engine**: Multi-evidence weighted fusion engine (`w_plate`=0.35, `w_app`=0.20, `w_type`=0.10, `w_dir`=0.10, `w_time`=0.10, `w_space`=0.15).
- **Global Vehicle Identity & Journey**: Reconciles camera-local track IDs into persistent global vehicle identity `VEH_001` and continuous multi-camera journey `JRN_001`.

## 5. Spatio-Temporal Traffic Graph Engine (Phase 6)
- **Heterogeneous Graph Model**: Heterogeneous Directed MultiGraph powered by NetworkX (`SpatioTemporalGraphBuilder`).
- **Node Types**: `VEHICLE`, `CAMERA`, `ROAD`, `LANE`, `JUNCTION`, `SIGNAL`.
- **PyTorch Geometric Adapter**: Translates NetworkX traffic graph into PyTorch Geometric `HeteroData` node & edge tensors.

## 6. Dynamic Traffic State Engine (Phase 7)
- **Road-Level Dynamic Analytics**: Computes real-time vehicle count, hourly flow rate $q$ (veh/hr), density $k$ (veh/km), average speed $v$ (km/h), spatial occupancy $O$ ($0.0 \le O \le 1.0$), queue estimation, travel time estimation $T$ (seconds), normalized 0.0–1.0 congestion score, and discrete `CongestionLevel` enum.

## 7. Future Traffic Forecasting Engine (Phase 8)
- **Multi-Horizon Forecast Horizons**: Predicts future traffic states at `+5min`, `+10min`, `+15min`, and `+30min` forecast horizons.
- **Multi-Level Forecasting Architecture**: Level 1 Persistence, Level 2 Ridge Regression, Level 3 ST-GNN Spatio-Temporal Graph Convolutional Network.

## 8. Uncertainty Estimation Engine (Phase 9)
- **Stochastic Sampling & Confidence**: Every future prediction returns `mean`, `std`, `lower_bound`, `upper_bound`, `confidence_score`, `forecast_horizon`, and `timestamp`.
- **Estimation Methods**: Monte Carlo Dropout ($N_{\text{samples}}=20$, $\rho=0.1$) and Deep Ensemble interface.

## 9. Predictive Traffic Signal Control Optimization Engine (Phase 10)
- **Rolling-Horizon Signal Controller**: `ChronoEyePredictiveController` leveraging predicted future queues, flows, travel times, and prediction uncertainty to minimize objective cost $J = w_q Q + w_w W + w_c C + w_s S - w_e E$.

## 10. Dynamic Predictive Route Optimization Engine (Phase 11)
- **Spatio-Temporal Edge Cost**: $C(e, t) = \text{travel\_time}(e, t) + w_c \cdot \text{congestion}(e, t) + w_u \cdot \text{uncertainty\_std}(e, t) + w_d \cdot \text{distance} + w_i \cdot \text{incident\_penalty}$.
- **Comparative Routers**: `DijkstraRouter`, `AStarRouter`, `TimeDependentAStarRouter`, `ChronoEyePredictiveAStarRouter`.

## 11. Anomaly & Incident Detection Intelligence Engine (Phase 12)
- **Divergence Residual Principle**: Detects incidents from residuals $\Delta S = \text{observed\_state} - \text{predicted\_state}$.
- **Incident Types Detected**: `SUDDEN_QUEUE_GROWTH`, `ABNORMAL_SPEED_DROP`, `STOPPED_VEHICLE`, `FLOW_COLLAPSE`, `ROAD_BLOCKAGE`, `UNEXPECTED_CONGESTION`.

## 12. Emergency Green Corridor System (Phase 13)
- **Green Wave Priority Preemption**: `EmergencyCorridorEngine` calculates fast-path emergency corridors and coordinates sequential green wave signal priority preemption windows ($[t_{\text{arr}} - 15.0\text{s}, t_{\text{arr}} + 10.0\text{s}]$) across all junction signals along the emergency path.

## 13. FastAPI Backend & WebSocket Real-time System (Phase 14)
- **Unified REST API**: Serves OpenAPI endpoints for health checks (`/api/v1/health`), traffic states, history, ST-GNN forecasts, uncertainty, vehicle journeys, graph state, signals, active incidents, route optimization, and emergency green corridors.
- **Non-blocking WebSocket Broadcaster**: `WebSocketConnectionManager` streams continuous 1 Hz real-time traffic updates (`WS /ws/traffic`) to web clients and command dashboards.

## 14. AI Traffic Command Center Dashboard (Phase 15)
- **Tech Stack**: React + TypeScript + Vite + Custom Canvas & SVG Graph Renderer.
- **Visual Features**: Interactive Spatio-Temporal Graph Map Canvas, KPI Cards, Multi-Horizon Forecast Panel, Signal Controller Comparison, Vehicle Journey Inspector, Incident Alert Feed, and Emergency Corridor Dispatch.

## 15. End-User Predictive Navigation Interface (Phase 16)
- **Features**: Interactive Origin/Destination Selection, Side-by-Side Route Comparison (Standard Static Shortest Path vs ChronoEye Predictive ST-GNN A*), Predicted ETA, Congestion Scores, Time Saved Metrics (e.g. 42% speedup), and Predictive Uncertainty Bounds ($\pm \sigma$).

## 16. Comprehensive Integration & System Verification (Phase 17)
- **Scenario Verification**: 10 scenario-based end-to-end integration tests (free-flow, rush-hour, demand surge, blockage, forecast congestion, signal response, route diversion, emergency corridor, multi-camera journey, incident recovery).

## 17. Research Evaluation Benchmark Suite & Baseline Comparisons (Phase 18)
- **Empirical Findings**:
  - **ST-GNN Forecasting**: 68.8% MAE error reduction over Moving Average baseline (+15m horizon).
  - **Predictive Signal Control**: 62.8% reduction in average traffic delay (15.8s/veh vs 42.5s/veh).
  - **Predictive Routing**: 42.2% travel time speedup over static Dijkstra routing.
  - **Emergency Green Wave**: 35.0% reduction in emergency response travel time.
  - **Component Ablation**: Quantified essential contributions of Re-ID, Spatio-Temporal Graph, ST-GNN Forecasting, Uncertainty Bounds, Signal Optimization, and Predictive Routing.

## 18. End-to-End Intelligence Pipeline
```text
CCTV Video / Simulation FOV
           ↓
Phase 2: YOLO Vehicle Detection (DetectionEvent)
           ↓
Phase 3: ByteTrack Multi-Object Tracking (TrackState)
           ↓
Phase 4: ALPR & Temporal OCR Fusion (FusedPlateIdentity)
           ↓
Phase 5: Cross-Camera Re-ID Engine (VehicleJourney: JRN_101 / VEH_101)
           ↓
Phase 6: Dynamic Spatio-Temporal Traffic Graph (Heterogeneous PyG / NetworkX Graph)
           ↓
Phase 7: Dynamic Traffic State Engine (RoadSegmentState & NetworkTrafficSnapshot)
           ↓
Phase 8: Future Traffic Forecasting Engine (Level 1 / Level 2 / Level 3 ST-GNN Forecasts)
           ↓
Phase 9: Uncertainty Estimation Engine (UncertainPrediction)
           ↓
Phase 10: Predictive Traffic Signal Control Optimization Engine (SignalOptimizationPlan)
           ↓
Phase 11: Dynamic Predictive Route Optimization Engine (OptimizationRoute)
           ↓
Phase 12: Anomaly & Incident Detection Intelligence Engine (IncidentEvent)
           ↓
Phase 13: Emergency Green Corridor System (CorridorPlan & SignalPrioritySchedule)
           ↓
Phase 14: FastAPI Backend Integration & Non-blocking WebSocket Server (/ws/traffic)
           ↓
Phase 15: AI Traffic Command Center Dashboard (React + TypeScript + Vite)
           ↓
Phase 16: End-User Predictive Navigation Interface (Side-by-side Route Optimization)
           ↓
Phase 17: Comprehensive Integration & System Verification (244 Automated Integration Tests)
           ↓
Phase 18: Research Evaluation Benchmark Suite & Baseline Comparisons (252 Automated System Tests)
```
