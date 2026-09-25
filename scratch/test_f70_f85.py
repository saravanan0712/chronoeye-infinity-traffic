import cv2
import sys
sys.path.insert(0, r"E:\chronoeye\backend")
from app.perception.ocr_engine import PaddleOCREngine
from app.perception.plate_preprocessor import PlatePreprocessor

engine = PaddleOCREngine()
prep = PlatePreprocessor()

for p in ["outputs/anpr_debug/frame_0070_track_TRK_106_530.jpg", "outputs/anpr_debug/frame_0085_track_TRK_106_5330.jpg"]:
    img = cv2.imread(p)
    h, w = img.shape[:2]
    top_crop, bot_crop = prep.split_stacked_plate(img)
    trim_px = int(round(w * 0.08))
    trimmed_top = top_crop[:, trim_px:w - trim_px]
    
    res_top = engine.recognize_text(prep.preprocess_plate_roi(trimmed_top))
    res_bot = engine.recognize_text(prep.preprocess_plate_roi(bot_crop))
    print(p)
    print("  Top:", res_top[:3])
    print("  Bot:", res_bot[:3])
    print("  Combined:", f"{res_top[1]}{res_bot[1]}")
