import cv2
import numpy as np
from app.perception.ocr_engine import PaddleOCREngine, TextNormalizer
from app.perception.plate_preprocessor import PlatePreprocessor

engine = PaddleOCREngine()
preprocessor = PlatePreprocessor()

def test_crop(img_path, desc):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        return
    h, w = img.shape[:2]
    aspect = w / float(max(1, h))
    print(f"\n--- Testing {desc}: {img_path} (w={w}, h={h}, aspect={aspect:.2f}) ---")
    
    # 1. Whole crop
    var_whole = preprocessor.preprocess_plate_roi(img)
    raw_w, norm_w, conf_w, var_name_w = engine.recognize_text(var_whole)
    print(f"Whole crop: raw='{raw_w}', norm='{norm_w}', conf={conf_w:.3f}")
    
    # 2. Split top/bottom with 8% overlap
    mid = h / 2.0
    overlap = int(round(h * 0.08))
    top = img[0:int(mid + overlap), :]
    bottom = img[int(mid - overlap):h, :]
    
    var_top = preprocessor.preprocess_plate_roi(top)
    raw_t, norm_t, conf_t, _ = engine.recognize_text(var_top)
    
    var_bot = preprocessor.preprocess_plate_roi(bottom)
    raw_b, norm_b, conf_b, _ = engine.recognize_text(var_bot)
    
    print(f"Top half: raw='{raw_t}', norm='{norm_t}', conf={conf_t:.3f}")
    print(f"Bottom half: raw='{raw_b}', norm='{norm_b}', conf={conf_b:.3f}")
    combined_raw = f"{raw_t}{raw_b}"
    combined_norm = f"{norm_t}{norm_b}"
    print(f"Combined: raw='{combined_raw}', norm='{combined_norm}'")

test_crop("outputs/anpr_debug/frame_0006_track_TRK_102_8525.jpg", "TRK_102 Frame 6 (SX8525)")
test_crop("outputs/anpr_debug/frame_0080_track_TRK_106_5330.jpg", "TRK_106 Frame 80 (TS5330)")
test_crop("outputs/anpr_debug/frame_0001_track_TRK_101_KW527.jpg", "TRK_101 Frame 1 (KW527)")
