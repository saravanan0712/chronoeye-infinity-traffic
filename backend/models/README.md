# ChronoEye Infinity - Perception Models Directory

Place dedicated YOLO License Plate Detector PyTorch model weight files here:
- `backend/models/license_plate_detector.pt`
- `backend/models/yolov8n-plate.pt`

When present, `PlateDetector` automatically loads this dedicated model to localize license plate bounding boxes on vehicle crops before running EasyOCR.
