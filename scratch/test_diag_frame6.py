import cv2
from app.perception.ocr_engine import PaddleOCREngine
from app.perception.plate_preprocessor import PlatePreprocessor

engine = PaddleOCREngine()
preprocessor = PlatePreprocessor()

crop = cv2.imread("outputs/anpr_debug/frame_0006_track_TRK_102_8525.jpg")
print(f"Crop shape: {crop.shape}")
h, w = crop.shape[:2]
aspect = w / float(h)
print(f"Aspect: {aspect:.2f}")
print(f"is_likely_stacked: {preprocessor.is_likely_stacked_plate(crop)}")

top, bot = preprocessor.split_stacked_plate(crop)
print(f"Top shape: {top.shape if top is not None else None}, Bot shape: {bot.shape if bot is not None else None}")

top_variants = preprocessor.preprocess_plate_roi(top)
print(f"Top variants count: {len(top_variants)}")
top_res = engine.recognize_text(top_variants)
print(f"Top res: {top_res}")

bot_variants = preprocessor.preprocess_plate_roi(bot)
print(f"Bot variants count: {len(bot_variants)}")
bot_res = engine.recognize_text(bot_variants)
print(f"Bot res: {bot_res}")

res = preprocessor.process_stacked_plate(crop, engine)
print(f"process_stacked_plate result: {res}")
