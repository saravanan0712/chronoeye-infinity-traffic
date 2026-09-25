import cv2
from app.perception.ocr_engine import PaddleOCREngine
from app.perception.plate_preprocessor import PlatePreprocessor

engine = PaddleOCREngine()
preprocessor = PlatePreprocessor()

# Check frame 1 track TRK_102
crop = cv2.imread("outputs/anpr_debug/frame_0001_track_TRK_102_08525.jpg")
if crop is None:
    # Try other frame 1 crops for TRK_102
    import glob
    matches = glob.glob("outputs/anpr_debug/frame_0001_track_TRK_102_*.jpg")
    print("Matches:", matches)
    if matches:
        crop = cv2.imread(matches[0])

if crop is not None:
    h, w = crop.shape[:2]
    print(f"Crop shape: {crop.shape}, aspect: {w/float(h):.2f}")
    top, bot = preprocessor.split_stacked_plate(crop)
    top_var = preprocessor.preprocess_plate_roi(top)
    bot_var = preprocessor.preprocess_plate_roi(bot)
    
    t_res = engine.recognize_text(top_var)
    b_res = engine.recognize_text(bot_var)
    print("Top res:", t_res)
    print("Bot res:", b_res)
    print("Stacked res:", preprocessor.process_stacked_plate(crop, engine))
