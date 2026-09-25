import cv2
import json
import os
import sys

sys.path.insert(0, r"E:\chronoeye\backend")

from ultralytics import YOLO
from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.ocr_engine import OCREngineFactory

preprocessor = PlatePreprocessor()
ocr_engine = OCREngineFactory.create_engine(prefer_real=True)

yolo_vehicle = YOLO(r"E:\chronoeye\yolov8n.pt")
yolo_plate = YOLO(r"E:\chronoeye\backend\models\license_plate_detector.pt")

key_frames = [1, 6, 8, 13, 23, 31, 36, 41, 46, 56, 68, 73, 80, 93, 98]
video_path = r"E:\chronoeye\data\videos\traffic_video_modified.mp4"
out_dir = r"E:\chronoeye\outputs\diagnostic_inspection"

cap = cv2.VideoCapture(video_path)
frame_data = {}

for fid in key_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Run vehicle detection
    v_res = yolo_vehicle(frame, classes=[2, 3, 5, 7], verbose=False)[0]
    p_res = yolo_plate(frame, verbose=False)[0]
    
    print(f"\n================ FRAME {fid:04d} ================")
    print(f"Vehicle detections: {len(v_res.boxes)}")
    print(f"Plate detections: {len(p_res.boxes)}")
    
    for i, pbox in enumerate(p_res.boxes):
        pxy = pbox.xyxy[0].cpu().numpy().astype(int)
        pconf = float(pbox.conf[0])
        pw = pxy[2] - pxy[0]
        ph = pxy[3] - pxy[1]
        pcrop = frame[pxy[1]:pxy[3], pxy[0]:pxy[2]]
        
        # Check aspect and OCR
        is_stacked = preprocessor.is_likely_stacked_plate(pcrop)
        
        # Single line OCR
        variants = preprocessor.preprocess_plate_roi(pcrop)
        ocr_single = ocr_engine.recognize_text(variants)
        
        # Stacked OCR
        ocr_stacked = preprocessor.process_stacked_plate(pcrop, ocr_engine)
        
        crop_fname = f"frame_{fid:04d}_plate_{i}_{pw}x{ph}.png"
        cv2.imwrite(os.path.join(out_dir, crop_fname), pcrop)
        
        print(f"  Plate {i}: bbox=({pxy[0]},{pxy[1]},{pxy[2]},{pxy[3]}), size=({pw}x{ph}), aspect={pw/float(max(1,ph)):.2f}, conf={pconf:.3f}")
        print(f"    Single OCR:  {ocr_single[:3]}")
        print(f"    Stacked OCR: {ocr_stacked}")

cap.release()
