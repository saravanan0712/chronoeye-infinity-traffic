# ChronoEye Infinity - Setup & Installation Guide

## Requirements
- Python 3.10+
- PyTorch & Ultralytics YOLO (`pip install ultralytics opencv-python pydantic`)
- EasyOCR (`pip install easyocr`) for real video ALPR/OCR
- CPU or CUDA-enabled GPU

## Environment Variables
```bash
MODEL_PATH=yolov8n.pt
DEVICE=cpu  # or cuda
CONFIDENCE_THRESHOLD=0.40
IOU_THRESHOLD=0.50
EASYOCR_GPU=False  # or True
```

## Running Tests
```bash
python -m unittest discover -s backend/tests -p "test_*.py"
```
