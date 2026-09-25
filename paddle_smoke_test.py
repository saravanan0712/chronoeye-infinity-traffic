"""
ChronoEye PaddleOCR Smoke Test
Compares PaddleOCR TextRecognition and EasyOCR on the known KW527 plate crop.

Usage:
    E:\chronoeye\chronoeye\Scripts\python.exe E:\chronoeye\paddle_smoke_test.py
"""

import os
import re
import sys
import cv2

# ─────────────────────────────────────────
# Crop selection
# ─────────────────────────────────────────
PRIMARY_CROP = r"E:\chronoeye\data\diagnostic_crops\plate_crop_005.png"
FALLBACK_CROP = r"E:\chronoeye\data\diagnostic_crops\plate_crop_001.png"

if os.path.exists(PRIMARY_CROP):
    crop_path = PRIMARY_CROP
    crop_label = "plate_crop_005.png (KW527 benchmark crop)"
elif os.path.exists(FALLBACK_CROP):
    crop_path = FALLBACK_CROP
    crop_label = "plate_crop_001.png (fallback)"
else:
    print("ERROR: No crop found in data/diagnostic_crops/")
    sys.exit(1)

print("=" * 60)
print(f"Crop file  : {crop_label}")
print(f"Full path  : {crop_path}")

img = cv2.imread(crop_path)
if img is None:
    print(f"ERROR: cv2.imread could not load: {crop_path}")
    sys.exit(1)

h, w = img.shape[:2]
print(f"Dimensions : {w}x{h} px")
print("=" * 60)

# ─────────────────────────────────────────
# [1] PaddleOCR TextRecognition (primary)
# ─────────────────────────────────────────
print("\n[1] PaddleOCR TextRecognition (paddleocr 3.7)")
paddle_ok = False
paddle_text = ""
paddle_conf = 0.0

try:
    from paddleocr import TextRecognition

    recognizer = TextRecognition(device="cpu")
    paddle_ok = True
    print("  Initialization : SUCCESS")

    results = list(recognizer.predict(img))

    if results:
        result = results[0]
        # Attribute-style access (standard paddleocr 3.7 result object)
        text = getattr(result, "rec_text", None)
        score = getattr(result, "rec_score", None)
        # Dict-style fallback
        if text is None and isinstance(result, dict):
            text = result.get("rec_text", result.get("text", ""))
            score = result.get("rec_score", result.get("confidence", 0.0))
        paddle_text = str(text or "")
        paddle_conf = float(score or 0.0)
    else:
        paddle_text = ""
        paddle_conf = 0.0

    clean = re.sub(r"[^A-Z0-9]", "", paddle_text.upper())
    print(f"  Raw text       : {paddle_text!r}")
    print(f"  Alphanumeric   : {clean!r}")
    print(f"  Confidence     : {paddle_conf:.4f}")
    print(f"  Backend        : PaddleOCR")
    print(f"  Model          : PP-OCRv6_medium_rec (default)")
    print(f"  Variant        : ORIGINAL")

except Exception as exc:
    print(f"  Initialization : FAILED — {exc}")
    paddle_ok = False

# ─────────────────────────────────────────
# [2] EasyOCR (fallback comparison)
# ─────────────────────────────────────────
print("\n[2] EasyOCR (fallback comparison)")
easy_ok = False

try:
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False)
    easy_ok = True
    print("  Initialization : SUCCESS")

    results = reader.readtext(
        img,
        detail=1,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )

    if results:
        sorted_res = sorted(
            results,
            key=lambda r: (
                min(pt[1] for pt in r[0]),
                min(pt[0] for pt in r[0]),
            ),
        )
        print(f"  Boxes detected : {len(sorted_res)}")
        for i, (bbox, text, conf) in enumerate(sorted_res):
            print(f"    Box {i + 1}: text={text!r}  conf={conf:.4f}")

        if len(sorted_res) >= 2:
            combined = "".join(t.strip() for _, t, _ in sorted_res)
            avg_c = sum(float(c) for _, _, c in sorted_res) / len(sorted_res)
            clean_comb = re.sub(r"[^A-Z0-9]", "", combined.upper())
            print(f"  Multibox concat: {clean_comb!r}  avg_conf={avg_c:.4f}")

        best = max(sorted_res, key=lambda r: r[2])
        best_clean = re.sub(r"[^A-Z0-9]", "", best[1].upper())
        print(f"  Best single box: {best_clean!r}  conf={best[2]:.4f}")
    else:
        print("  No boxes detected.")

except Exception as exc:
    print(f"  Initialization : FAILED — {exc}")
    easy_ok = False

# ─────────────────────────────────────────
# Summary
# ─────────────────────────────────────────
paddle_clean = re.sub(r"[^A-Z0-9]", "", paddle_text.upper())

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Crop              : {crop_label}")
print(f"  PaddleOCR init    : {'OK' if paddle_ok else 'FAILED'}")
print(f"  PaddleOCR result  : {paddle_clean!r}  conf={paddle_conf:.4f}")
print(f"  EasyOCR init      : {'OK' if easy_ok else 'FAILED'}")
print("=" * 60)
