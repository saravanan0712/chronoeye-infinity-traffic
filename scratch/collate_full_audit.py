import cv2
import numpy as np
import os
import sys
import json

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

# Load results from run_full_video_audit
audit_json_path = os.path.join(base_dir, "full_video_audit_results.json")

track_summary = {}

for vf in veh_crops:
    vpath = os.path.join(base_dir, vf)
    img = cv2.imread(vpath)
    if img is None:
        continue
    h, w = img.shape[:2]
    
    parts = vf.split("_")
    tid = parts[0] + "_" + parts[1]
    bf = parts[3].replace("f", "")
    
    # Extract candidates
    dummy_bbox = BoundingBoxXYXY(x1=0, y1=0, x2=w, y2=h)
    cands = plate_detector.extract_plate_candidates(img, dummy_bbox)
    
    cand_res = []
    for c_idx, (pcrop, pbox) in enumerate(cands):
        is_stacked = preprocessor.is_likely_stacked_plate(pcrop)
        
        # Single OCR
        variants = preprocessor.preprocess_plate_roi(pcrop)
        res_single = ocr.recognize_text(variants)
        
        # Stacked OCR
        res_stacked = None
        if is_stacked:
            res_stacked = preprocessor.process_stacked_plate(pcrop, ocr)
            
        cand_res.append({
            "single": res_single,
            "stacked": res_stacked,
            "is_stacked": is_stacked,
            "crop_shape": pcrop.shape
        })
        
    track_summary[tid] = {
        "best_frame": int(bf),
        "veh_size": (w, h),
        "candidates": cand_res
    }

print("\n================ COMPREHENSIVE 36-TRACK AUDIT REPORT ================\n")

for tid, info in sorted(track_summary.items()):
    bf = info["best_frame"]
    vw, vh = info["veh_size"]
    cands = info["candidates"]
    c_str = []
    for c in cands:
        if c["stacked"]:
            c_str.append(f"Stacked='{c['stacked'][1]}' (conf={c['stacked'][2]:.2f})")
        else:
            c_str.append(f"Single='{c['single'][1]}' (conf={c['single'][2]:.2f})")
    c_out = ", ".join(c_str) if c_str else "None"
    print(f"Track: {tid:7s} | Best Frame: {bf:03d} | Size: {vw}x{vh} | OCR at Best Frame: {c_out}")
