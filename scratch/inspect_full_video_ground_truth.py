import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath("backend"))

from app.perception.ocr_engine import OCREngineFactory, IndianPlateValidator
from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.plate_detector import PlateDetector
from app.schemas.detection import BoundingBoxXYXY

preprocessor = PlatePreprocessor()
plate_detector = PlateDetector()
ocr = OCREngineFactory.create_engine(prefer_real=True)

base_dir = r"E:\chronoeye\outputs\full_video_audit"
veh_crops = sorted([f for f in os.listdir(base_dir) if "_best_" in f])

print(f"================ FULL VIDEO 36-TRACK VISUAL AUDIT ================")
print(f"Total vehicle tracks to inspect: {len(veh_crops)}\n")

for vf in veh_crops:
    vpath = os.path.join(base_dir, vf)
    img = cv2.imread(vpath)
    if img is None:
        continue
    h, w = img.shape[:2]
    
    parts = vf.split("_")
    tid = parts[0] + "_" + parts[1]
    bf = parts[3]
    
    # Extract candidates
    dummy_bbox = BoundingBoxXYXY(x1=0, y1=0, x2=w, y2=h)
    cands = plate_detector.extract_plate_candidates(img, dummy_bbox)
    
    # Also check direct YOLO
    direct_yolo = []
    if plate_detector.plate_model:
        yres = plate_detector.plate_model(img, verbose=False)[0]
        if yres.boxes:
            for b in yres.boxes:
                bxy = b.xyxy[0].cpu().numpy().astype(int)
                direct_yolo.append((bxy, float(b.conf[0])))
                
    print(f"------------------------------------------------------------------")
    print(f"Track: {tid:7s} | Best Frame: {bf} | Veh Size: {w}x{h} | Direct YOLO Plates: {len(direct_yolo)} | Candidates: {len(cands)}")
    
    for c_idx, (pcrop, pbox) in enumerate(cands):
        cw, ch = int(pbox.width), int(pbox.height)
        is_stacked = preprocessor.is_likely_stacked_plate(pcrop)
        
        # Single OCR
        variants = preprocessor.preprocess_plate_roi(pcrop)
        res_single = ocr.recognize_text(variants)
        
        # Stacked OCR
        res_stacked = None
        if is_stacked:
            res_stacked = preprocessor.process_stacked_plate(pcrop, ocr)
            
        val_single, _ = IndianPlateValidator.validate_format(res_single[1])
        st_val = IndianPlateValidator.validate_format(res_stacked[1])[0] if res_stacked else None
        
        st_str = f" | Stacked: raw='{res_stacked[0]}', norm='{res_stacked[1]}', conf={res_stacked[2]:.2f}, val={st_val.value if hasattr(st_val, 'value') else st_val}" if res_stacked else ""
        print(f"  Cand {c_idx} ({cw}x{ch}, aspect={cw/max(1,ch):.2f}, stacked={is_stacked}):")
        print(f"    Single:  raw='{res_single[0]}', norm='{res_single[1]}', conf={res_single[2]:.2f}, val={val_single.value if hasattr(val_single, 'value') else val_single}{st_str}")
