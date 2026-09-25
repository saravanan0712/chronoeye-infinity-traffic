import cv2
import os
import sys

sys.path.insert(0, r"E:\chronoeye\backend")

from ultralytics import YOLO
from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.ocr_engine import OCREngineFactory
from app.schemas.detection import BoundingBoxXYXY

video_path = r"E:\chronoeye\data\videos\traffic_video_modified.mp4"
yolo_vehicle = YOLO(r"E:\chronoeye\yolov8n.pt")
yolo_plate = YOLO(r"E:\chronoeye\backend\models\license_plate_detector.pt")
preprocessor = PlatePreprocessor()
ocr_engine = OCREngineFactory.create_engine(prefer_real=True)

# Test on frame 31, 36, 41, 46, 56 (TRK_108)
cap = cv2.VideoCapture(video_path)

frames_to_test = [31, 36, 41, 46, 56]

for fid in frames_to_test:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Detect vehicles
    v_res = yolo_vehicle(frame, classes=[2, 3, 5, 7], verbose=False)[0]
    for v_box in v_res.boxes:
        v_xy = v_box.xyxy[0].cpu().numpy().astype(int)
        vw = v_xy[2] - v_xy[0]
        vh = v_xy[3] - v_xy[1]
        
        # Check if this vehicle is on the left lane (TRK_108 position: x around 100-500, y around 600-900)
        if 50 < v_xy[0] < 500 and 500 < v_xy[1] < 900:
            vcrop = frame[v_xy[1]:v_xy[3], v_xy[0]:v_xy[2]]
            if vcrop.size == 0:
                continue
            
            print(f"\n--- Frame {fid}: Vehicle bbox={v_xy}, size={vw}x{vh} ---")
            
            # 1. Primary YOLO (conf=0.25)
            p_res1 = yolo_plate(vcrop, conf=0.25, verbose=False)[0]
            print(f"  Primary YOLO (0.25) boxes: {len(p_res1.boxes)}")
            for b in p_res1.boxes:
                b_xy = b.xyxy[0].cpu().numpy().astype(int)
                print(f"    box: {b_xy}, conf={float(b.conf[0]):.3f}")
            
            # 2. Recovery YOLO (conf=0.15 on lower ROI)
            lower_roi = vcrop[int(vh * 0.40):, :]
            p_res2 = yolo_plate(lower_roi, conf=0.15, verbose=False)[0]
            print(f"  Recovery YOLO (0.15 on lower ROI) boxes: {len(p_res2.boxes)}")
            for b in p_res2.boxes:
                b_xy = b.xyxy[0].cpu().numpy().astype(int)
                # map back
                b_xy[1] += int(vh * 0.40)
                b_xy[3] += int(vh * 0.40)
                bw = b_xy[2] - b_xy[0]
                bh = b_xy[3] - b_xy[1]
                pcrop = vcrop[b_xy[1]:b_xy[3], b_xy[0]:b_xy[2]]
                res_stk = preprocessor.process_stacked_plate(pcrop, ocr_engine)
                res_sng = ocr_engine.recognize_text(preprocessor.preprocess_plate_roi(pcrop))
                print(f"    Recovery box: {b_xy}, size={bw}x{bh}, conf={float(b.conf[0]):.3f}")
                print(f"      Single OCR:  {res_sng[:3]}")
                print(f"      Stacked OCR: {res_stk}")

cap.release()
