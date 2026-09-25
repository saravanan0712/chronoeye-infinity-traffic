import cv2, os, sys
sys.path.insert(0, os.path.abspath("backend"))

from ultralytics import YOLO
from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.ocr_engine import OCREngineFactory

preprocessor = PlatePreprocessor()
ocr_engine = OCREngineFactory.create_engine(prefer_real=True)

yolo_vehicle = YOLO(r"E:\chronoeye\yolov8n.pt")
yolo_plate = YOLO(r"E:\chronoeye\backend\models\license_plate_detector.pt")

video_path = r"E:\chronoeye\data\videos\traffic_video_modified.mp4"
out_dir = r"E:\chronoeye\outputs\point_e_inspection"
os.makedirs(out_dir, exist_ok=True)

# Frames of interest for the key tracks
frames_of_interest = {
    "TRK_104": [3, 7, 8, 13, 18, 23, 28, 33],
    "TRK_114": [191, 196, 203, 208, 213, 216, 228],
    "TRK_116": [199, 204, 207, 211],
    "TRK_117": [232, 237, 248, 273, 278, 300],
    "TRK_125": [317, 332, 337, 342, 355, 366, 375],
    "TRK_127": [338, 348, 366, 373],
    "TRK_106": [23, 28, 53, 68, 80, 85],
}

all_frames = sorted(list(set(f for flist in frames_of_interest.values() for f in flist)))

cap = cv2.VideoCapture(video_path)

for fid in all_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Run vehicle & plate detection
    v_res = yolo_vehicle(frame, classes=[2, 3, 5, 7], verbose=False)[0]
    p_res = yolo_plate(frame, verbose=False)[0]
    
    print(f"\n================ FRAME {fid:04d} ================")
    print(f"Vehicles: {len(v_res.boxes)}, Plates: {len(p_res.boxes)}")
    
    # Save vehicle bboxes and plate detections
    for i, pbox in enumerate(p_res.boxes):
        pxy = pbox.xyxy[0].cpu().numpy().astype(int)
        pconf = float(pbox.conf[0])
        pw = pxy[2] - pxy[0]
        ph = pxy[3] - pxy[1]
        pcrop = frame[max(0, pxy[1]):min(frame.shape[0], pxy[3]), max(0, pxy[0]):min(frame.shape[1], pxy[2])]
        
        if pcrop.size == 0:
            continue
        
        # Check aspect and OCR
        is_stacked = preprocessor.is_likely_stacked_plate(pcrop)
        ocr_single = ocr_engine.recognize_text(preprocessor.preprocess_plate_roi(pcrop))
        ocr_stacked = preprocessor.process_stacked_plate(pcrop, ocr_engine) if is_stacked else None
        
        crop_fname = f"f{fid:04d}_p{i}_{pw}x{ph}_conf{pconf:.2f}.png"
        cv2.imwrite(os.path.join(out_dir, crop_fname), pcrop)
        
        print(f"  Plate {i} at ({pxy[0]},{pxy[1]},{pxy[2]},{pxy[3]}), size {pw}x{ph}, aspect {pw/float(max(1,ph)):.2f}, conf {pconf:.3f}")
        print(f"    Single OCR:  {ocr_single[:3]}")
        print(f"    Stacked OCR: {ocr_stacked}")
    
    # Also save full frame with bboxes annotated
    annotated = frame.copy()
    for v in v_res.boxes:
        vxy = v.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(annotated, (vxy[0], vxy[1]), (vxy[2], vxy[3]), (0, 255, 0), 2)
    for p in p_res.boxes:
        pxy = p.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(annotated, (pxy[0], pxy[1]), (pxy[2], pxy[3]), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out_dir, f"frame_{fid:04d}_annotated.jpg"), annotated)

cap.release()
print("\nExtraction finished.")
