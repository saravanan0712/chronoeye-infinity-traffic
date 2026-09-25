import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath("backend"))

from app.perception.ocr_engine import OCREngineFactory, IndianPlateValidator
from app.perception.plate_preprocessor import PlatePreprocessor

preprocessor = PlatePreprocessor()
ocr = OCREngineFactory.create_engine(prefer_real=True)

base_dir = r"E:\chronoeye\outputs\target_track_deep_diagnosis"
target_tracks = ["TRK_104", "TRK_105", "TRK_107", "TRK_109", "TRK_110", "TRK_111"]

print("================ TRACK-BY-TRACK GROUND TRUTH AUDIT ================")

for tid in target_tracks:
    tdir = os.path.join(base_dir, tid)
    cand_files = sorted([f for f in os.listdir(tdir) if "_cand_" in f])
    veh_files = sorted([f for f in os.listdir(tdir) if "_veh_" in f])
    
    print(f"\n=======================================================")
    print(f"TRACK: {tid}")
    print(f"  Available Veh Crops: {len(veh_files)}")
    print(f"  Available Cand Crops: {len(cand_files)}")
    
    # Run high-res OCR with all enhancements on available candidate crops
    for cf in cand_files:
        cpath = os.path.join(tdir, cf)
        img = cv2.imread(cpath)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Single line OCR
        variants = preprocessor.preprocess_plate_roi(img)
        res_single = ocr.recognize_text(variants)
        
        # Stacked OCR
        res_stacked = preprocessor.process_stacked_plate(img, ocr)
        
        print(f"  Crop: {cf} ({w}x{h})")
        print(f"    Single:  raw='{res_single[0]}', norm='{res_single[1]}', conf={res_single[2]:.3f}, variant={res_single[3] if len(res_single)>3 else 'ORIGINAL'}")
        if res_stacked:
            print(f"    Stacked: raw='{res_stacked[0]}', norm='{res_stacked[1]}', conf={res_stacked[2]:.3f}, variant={res_stacked[3]}")
