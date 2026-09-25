import cv2
import glob
from app.perception.ocr_engine import PaddleOCREngine
from app.perception.plate_preprocessor import PlatePreprocessor

engine = PaddleOCREngine()
preprocessor = PlatePreprocessor()

crops = [
    "outputs/anpr_debug/frame_0006_track_TRK_102_8525.jpg",
    "outputs/anpr_debug/frame_0011_track_TRK_102_8525.jpg",
    "outputs/anpr_debug/frame_0016_track_TRK_102_8525.jpg",
    "outputs/anpr_debug/frame_0021_track_TRK_102_8525.jpg",
    "outputs/anpr_debug/frame_0026_track_TRK_102_8525.jpg",
    "outputs/anpr_debug/frame_0080_track_TRK_106_5330.jpg",
    "outputs/anpr_debug/frame_0085_track_TRK_106_5330.jpg",
    "outputs/anpr_debug/frame_0095_track_TRK_106_PTS.jpg",
]

for p in crops:
    img = cv2.imread(p)
    if img is None:
        continue
    h, w = img.shape[:2]
    # Test split at 50%
    top = img[0:int(h * 0.52), :]
    bot = img[int(h * 0.48):h, :]
    
    var_top = preprocessor.preprocess_plate_roi(top)
    raw_t, norm_t, conf_t, _ = engine.recognize_text(var_top)
    
    var_bot = preprocessor.preprocess_plate_roi(bot)
    raw_b, norm_b, conf_b, _ = engine.recognize_text(var_bot)
    
    print(f"{p}: TOP='{raw_t}' ({conf_t:.2f}) + BOT='{raw_b}' ({conf_b:.2f}) -> '{raw_t}{raw_b}'")
