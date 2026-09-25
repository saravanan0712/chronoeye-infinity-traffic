import cv2
import sys
import os
sys.path.insert(0, os.path.abspath('backend'))
from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.ocr_engine import OCREngineFactory
from app.perception.ocr_normalizer import generate_normalized_plate_candidates

pre = PlatePreprocessor()
ocr = OCREngineFactory.create_engine(prefer_real=True)
print('OCR backend:', type(ocr).__name__)

INSPECT_DIR = r'E:\chronoeye\outputs\point_e_inspection'

# Crops to inspect:
# TRK_106 at frame 68: 97x54 (asp=1.80 -> stacked trigger) raw='PTS5330' in anpr_debug
# TRK_106 at frame 80: 100x58 (asp=1.72 -> stacked trigger) raw='5330'
# TRK_106 at frame 85: 106x60 (asp=1.77 -> stacked trigger) raw='5330'
# TRK_117 at frame 237: 3 candidates detected
# TRK_117 at frame 248: 2 candidates detected (single-line 91x28 asp=3.25)
CROPS = [
    ('TRK106_f68',   os.path.join(INSPECT_DIR, 'f0068_p0_97x54_conf0.76.png')),
    ('TRK106_f80',   os.path.join(INSPECT_DIR, 'f0080_p0_100x58_conf0.55.png')),
    ('TRK106_f85',   os.path.join(INSPECT_DIR, 'f0085_p0_106x60_conf0.73.png')),
    ('TRK117_f237_p0', os.path.join(INSPECT_DIR, 'f0237_p0_68x46_conf0.55.png')),
    ('TRK117_f237_p1', os.path.join(INSPECT_DIR, 'f0237_p1_85x30_conf0.42.png')),
    ('TRK117_f237_p2', os.path.join(INSPECT_DIR, 'f0237_p2_101x62_conf0.32.png')),
    ('TRK117_f248_p0', os.path.join(INSPECT_DIR, 'f0248_p0_91x28_conf0.78.png')),
    ('TRK117_f248_p1', os.path.join(INSPECT_DIR, 'f0248_p1_83x48_conf0.63.png')),
]

for label, path in CROPS:
    img = cv2.imread(path)
    if img is None:
        print(label + ': FILE NOT FOUND - ' + path)
        continue
    h, w = img.shape[:2]
    asp = w / max(1, h)
    stacked = pre.is_likely_stacked_plate(img)
    print('\n' + '='*60)
    print('CROP: ' + label)
    print('  File: ' + os.path.basename(path))
    print('  Size: ' + str(w) + 'x' + str(h) + '  aspect=' + str(round(asp, 2)))
    print('  Stacked trigger: ' + str(stacked) + ' (threshold=2.2)')
    print('  Per-variant OCR:')
    variants = pre.preprocess_plate_roi(img)
    for v in variants:
        vt = v.get('variant_type', '?') if isinstance(v, dict) else 'RAW'
        try:
            raw, norm, conf, _ = ocr.recognize_text([v])
            clean = raw.upper().replace(' ', '').replace('-', '').strip() if raw else ''
            cands = generate_normalized_plate_candidates(clean, norm)
            print('    ' + vt.ljust(22) + ': raw=' + repr(clean).ljust(14) +
                  ' norm=' + repr(norm).ljust(14) + ' conf=' + str(round(conf, 3)) +
                  ' cands=' + str(cands))
        except Exception as exc:
            print('    ' + vt.ljust(22) + ': ERR ' + str(exc)[:60])
    if stacked:
        top, bot = pre.split_stacked_plate(img)
        if top is not None and bot is not None:
            th, tw = top.shape[:2]
            bh, bw = bot.shape[:2]
            tv = pre.preprocess_plate_roi(top)
            bv = pre.preprocess_plate_roi(bot)
            try:
                tr, tn, tc, _ = ocr.recognize_text(tv)
                br, bn, bc, _ = ocr.recognize_text(bv)
                print('  STACKED SPLIT: top=' + str(tw) + 'x' + str(th) + '  bot=' + str(bw) + 'x' + str(bh))
                print('    TOP  raw=' + repr(tr) + ' norm=' + repr(tn) + ' conf=' + str(round(tc, 3)))
                print('    BOT  raw=' + repr(br) + ' norm=' + repr(bn) + ' conf=' + str(round(bc, 3)))
                print('    COMBINED: ' + repr(tn + bn) + '  avg_conf=' + str(round((tc + bc) / 2, 3)))
                top_cands = generate_normalized_plate_candidates(tr, tn)
                print('    TOP candidates: ' + str(top_cands))
                print('  Top per-variant breakdown:')
                for v in tv[:5]:
                    vt = v.get('variant_type', '?') if isinstance(v, dict) else 'RAW'
                    try:
                        r, n, c, _ = ocr.recognize_text([v])
                        print('    TOP ' + vt.ljust(18) + ': raw=' + repr(r) + ' norm=' + repr(n) + ' conf=' + str(round(c, 3)))
                    except Exception:
                        print('    TOP ' + vt.ljust(18) + ': ERR')
            except Exception as exc:
                print('  Stacked OCR error: ' + str(exc)[:80])

print('\nDIAGNOSTIC COMPLETE')
