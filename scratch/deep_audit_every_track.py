import cv2
import numpy as np
import os
import sys

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

# Fused identities from the run:
# Confirmed:
# 'TRK_101': 'KW527'
# 'TRK_102': 'SX8525'
# 'TRK_108': 'WG2119'
# 'TRK_106': 'TS5330'
# 'TRK_110': 'WG2119'
# 'TRK_112': 'VN4712'
# 'TRK_111': 'KT4540'
# 'TRK_119': 'CC435'
# 'TRK_121': 'YB6433'
# 'TRK_128': 'NN773'
# 'TRK_129': 'NN773'
# 'TRK_131': 'SB422'
# 'TRK_132': 'YA8262'
# 'TRK_134': 'YA8262'
# 'TRK_135': 'YA8262'

confirmed_map = {
    'TRK_101': 'KW527',
    'TRK_102': 'SX8525',
    'TRK_106': 'TS5330',
    'TRK_108': 'WG2119',
    'TRK_110': 'WG2119',
    'TRK_111': 'KT4540',
    'TRK_112': 'VN4712',
    'TRK_119': 'CC435',
    'TRK_121': 'YB6433',
    'TRK_128': 'NN773',
    'TRK_129': 'NN773',
    'TRK_131': 'SB422',
    'TRK_132': 'YA8262',
    'TRK_134': 'YA8262',
    'TRK_135': 'YA8262',
}

pending_map = {
    'TRK_104': 'X0147',
    'TRK_109': 'VV51N',
    'TRK_114': 'WG2119',
    'TRK_116': 'VN472',
    'TRK_117': 'WG219',
    'TRK_120': '75BEAYS',
    'TRK_125': 'PIND7823',
    'TRK_126': 'TN782',
    'TRK_127': 'NN773',
    'TRK_133': 'YA8262',
}

print("\n================ DETAILED AUDIT FOR ALL 36 TRACKS ================\n")

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
    
    cand_info = []
    for c_idx, (pcrop, pbox) in enumerate(cands):
        is_stacked = preprocessor.is_likely_stacked_plate(pcrop)
        
        # Single OCR
        variants = preprocessor.preprocess_plate_roi(pcrop)
        res_single = ocr.recognize_text(variants)
        
        # Stacked OCR
        res_stacked = None
        if is_stacked:
            res_stacked = preprocessor.process_stacked_plate(pcrop, ocr)
            
        cand_info.append({
            "single": res_single,
            "stacked": res_stacked,
            "is_stacked": is_stacked,
            "crop_size": (int(pbox.width), int(pbox.height))
        })
        
    status = "CONFIRMED" if tid in confirmed_map else ("PENDING" if tid in pending_map else "NONE")
    plate_val = confirmed_map.get(tid, pending_map.get(tid, "None"))
    
    print(f"------------------------------------------------------------------")
    print(f"TRACK: {tid} | Best Frame: {bf} | Veh Size: {w}x{h}")
    print(f"  ChronoEye Fusion: Status={status}, Plate='{plate_val}'")
    for idx, c in enumerate(cand_info):
        csz = c["crop_size"]
        if c["stacked"]:
            print(f"  Cand {idx} ({csz[0]}x{csz[1]}): Stacked='{c['stacked'][1]}' (raw='{c['stacked'][0]}', conf={c['stacked'][2]:.2f}) | Single='{c['single'][1]}'")
        else:
            print(f"  Cand {idx} ({csz[0]}x{csz[1]}): Single='{c['single'][1]}' (raw='{c['single'][0]}', conf={c['single'][2]:.2f})")
