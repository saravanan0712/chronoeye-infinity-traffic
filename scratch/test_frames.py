import cv2
from app.perception.ocr_engine import PaddleOCREngine
from app.perception.plate_preprocessor import PlatePreprocessor

engine = PaddleOCREngine()
prep = PlatePreprocessor()

for p in ['outputs/anpr_debug/frame_0006_track_TRK_102_8525.jpg', 'outputs/anpr_debug/frame_0080_track_TRK_106_5330.jpg']:
    img = cv2.imread(p)
    if img is not None:
        print(p, 'is_stacked:', prep.is_likely_stacked_plate(img))
        res = prep.process_stacked_plate(img, engine)
        print('  Result:', res)
    else:
        print('Could not load', p)
