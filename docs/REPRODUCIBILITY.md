# ChronoEye Infinity - Reproducibility & Deployment Guide

## 1. System Requirements
- **Python**: 3.9+
- **Node.js**: 18+ (npm 9+)
- **OS**: Windows, macOS, or Linux
- **Hardware**: CPU execution host (GPU optional, zero CUDA requirement)

## 2. Backend Setup & Run
```bash
# 1. Navigate to backend directory
cd e:\chronoeye\backend

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run FastAPI Backend Server
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 4. Frontend Setup & Run
```bash
# 1. Navigate to frontend directory
cd e:\chronoeye\frontend

# 2. Install dependencies
npm install

# 3. Start Vite Development Server
npm run dev
```

## 5. Running Full Test & Benchmark Suite
```bash
# Run pytest regression suite
cd e:\chronoeye\backend
pytest

# Run Empirical Research Benchmark
python -m backend.app.research.benchmark
```
