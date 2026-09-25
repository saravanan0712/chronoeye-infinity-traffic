import cv2
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.abspath("backend"))

from app.perception.ocr_engine import OCREngineFactory, IndianPlateValidator
from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.plate_detector import PlateDetector

with open("outputs/failure_audit_crops/audit_inspection_data.json") as f:
    data = json.load(f)

# Also load full video audit results
with open("outputs/full_video_audit_fix1/full_video_audit_results_fix1.json") as f:
    results = json.load(f)

res_by_id = {r["track_id"]: r for r in results}

ocr = OCREngineFactory.create_engine(prefer_real=True)
preprocessor = PlatePreprocessor()

print("=" * 80)
print("DETAILED VISUAL & OCR AUDIT OF PENDING AND UNRESOLVED TRACKS")
print("=" * 80)

for tid, info in data.items():
    print(f"\n>>> TRACK {tid} <<<")
    print(f"  Status:          {info['status']} (confirmed={info['confirmed']})")
    print(f"  Best Plate:      {info['best_plate']}")
    print(f"  Pending Plate:   {info['pending_plate']}")
    print(f"  Confidence:      {info['overall_conf']:.3f}")
    print(f"  Frames:          {info['frame_range'][0]} - {info['frame_range'][1]} (total {info['frame_count']} frames)")
    print(f"  Max Vehicle:     {info['max_size'][0]}x{info['max_size'][1]} at frame {info['best_frame']}")
    print(f"  Observations in Pipeline ({len(info['observations'])}):")
    for o in info['observations']:
        print(f"    - f{o['frame_id']}: raw='{o['raw_text']}', norm='{o['normalized_text']}', conf={o['ocr_confidence']:.3f}, val={o['validation_status']}")
    
    # Check saved crop file
    bf = info['best_frame']
    crop_path = None
    for fn in os.listdir("outputs/failure_audit_crops"):
        if fn.startswith(f"{tid}_f{bf}_cand0_"):
            crop_path = os.path.join("outputs/failure_audit_crops", fn)
            break
            
    if crop_path and os.path.exists(crop_path):
        crop = cv2.imread(crop_path)
        if crop is not None:
            ch, cw = crop.shape[:2]
            mean_intensity = float(np.mean(crop))
            contrast = float(np.std(crop))
            # Test multiple preprocessors
            vars = preprocessor.preprocess_plate_roi(crop)
            res_single = ocr.recognize_text(vars)
            is_stacked = preprocessor.is_likely_stacked_plate(crop)
            res_stacked = preprocessor.process_stacked_plate(crop, ocr) if is_stacked else None
            
            print(f"  Crop at best frame {bf}: {cw}x{ch} (aspect={cw/max(1,ch):.2f}, mean={mean_intensity:.1f}, contrast={contrast:.1f})")
            print(f"    Single OCR:  raw='{res_single[0]}', norm='{res_single[1]}', conf={res_single[2]:.3f}, variant={res_single[3]}")
            if is_stacked and res_stacked:
                print(f"    Stacked OCR: raw='{res_stacked[0]}', norm='{res_stacked[1]}', conf={res_stacked[2]:.3f}, variant={res_stacked[3]}")
    else:
        print(f"  No crop file found at best frame {bf}")

