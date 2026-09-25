import cv2
import sys
import os

sys.path.insert(0, r"E:\chronoeye\backend")

from app.perception.ocr_engine import PaddleOCREngine
from app.perception.plate_preprocessor import PlatePreprocessor

engine = PaddleOCREngine()
prep = PlatePreprocessor()

test_images = [
    ("TRK_102 F1 (PSX)", "outputs/anpr_debug/frame_0001_track_TRK_102_PSX8525.jpg"),
    ("TRK_102 F6 (8525)", "outputs/anpr_debug/frame_0006_track_TRK_102_8525.jpg"),
    ("TRK_102 F8 (SX8525)", "outputs/anpr_debug/frame_0008_track_TRK_102_SX8525.jpg"),
    ("TRK_106 F80 (TS5330)", "outputs/anpr_debug/frame_0080_track_TRK_106_5330.jpg"),
    ("TRK_101 F1 (KW527)", "outputs/anpr_debug/frame_0001_track_TRK_101_KW527.jpg"),
]

for label, path in test_images:
    img = cv2.imread(path)
    if img is None:
        print(f"Could not load {path}")
        continue
    h, w = img.shape[:2]
    aspect = w / float(h)
    print(f"\n--- {label} ({path}) shape=({h}, {w}) aspect={aspect:.2f} ---")
    
    # Baseline (no trim)
    top_crop, bot_crop = prep.split_stacked_plate(img)
    top_vars = prep.preprocess_plate_roi(top_crop)
    res_base = engine.recognize_text(top_vars)
    print(f"  Top crop base (0% trim): raw='{res_base[0]}' norm='{res_base[1]}' conf={res_base[2]:.3f}")
    
    for trim in [0.05, 0.06, 0.07, 0.08]:
        trim_px = int(round(w * trim))
        trimmed_top = top_crop[:, trim_px:w - trim_px]
        top_vars_trim = prep.preprocess_plate_roi(trimmed_top)
        res_trim = engine.recognize_text(top_vars_trim)
        print(f"  Top crop ({int(trim*100)}% trim, {trim_px}px): raw='{res_trim[0]}' norm='{res_trim[1]}' conf={res_trim[2]:.3f}")
