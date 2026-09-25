"""
ChronoEye Infinity — Stage 4 ANPR Diagnostic Runner v2
READ-ONLY: Captures raw per-call evidence without modifying any source.
Outputs: data/debug/ANPR_ROOT_CAUSE_REPORT.txt, data/debug/anpr_audit_report.json, data/debug/anpr_samples/*.png
"""

import os, sys, time, json, textwrap, traceback

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: cv2/numpy not available")
    sys.exit(1)

from app.perception.plate_detector import PlateDetector
from app.perception.plate_preprocessor import PlateImagePreprocessor
from app.perception.ocr_engine import EasyOCREngine, TextNormalizer, IndianPlateValidator, PlateValidationStatus
from app.schemas.detection import DetectionEvent, BoundingBoxXYXY, DetectorConfig
from app.perception.detector import YOLOVehicleDetector
from app.perception.tracker import ByteTrackerManager
from app.schemas.tracking import TrackStatus

VIDEO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "videos", "traffic_video_modified.mp4"))
DEBUG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "debug"))
SAMPLES_DIR = os.path.join(DEBUG_DIR, "anpr_samples")
REPORT_TXT = os.path.join(DEBUG_DIR, "ANPR_ROOT_CAUSE_REPORT.txt")
REPORT_JSON = os.path.join(DEBUG_DIR, "anpr_audit_report.json")

os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

MAX_SAVE_CROPS = 30
SAMPLE_EVERY_N = 3   # Process every Nth frame for speed; set 1 to process all

def safe_ocr_raw(engine, variants):
    """Extract raw EasyOCR output BEFORE any normalization or validation."""
    raw_results = []
    for v in variants:
        if isinstance(v, np.ndarray) and v.size > 0:
            try:
                results = engine.reader.readtext(v, detail=1, allowlist=None)
                for bbox_pts, text, conf in results:
                    raw_results.append({"text": text, "conf": round(float(conf), 4)})
            except Exception as e:
                raw_results.append({"error": str(e)})
    return raw_results

def run():
    print("=== ANPR DIAGNOSTIC RUNNER v2 ===")
    print(f"Video: {VIDEO_PATH}")
    if not os.path.exists(VIDEO_PATH):
        print("ERROR: Video file not found!")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    print(f"Frame count: {total_frames} | Resolution: {vid_w}x{vid_h} | FPS: {vid_fps:.2f}")

    # Init pipeline
    detector = YOLOVehicleDetector(DetectorConfig(confidence_threshold=0.40))
    tracker = ByteTrackerManager()
    plate_det = PlateDetector()
    preprocessor = PlateImagePreprocessor()
    ocr_engine = EasyOCREngine()

    # Metrics
    frame_num = 0
    vehicle_roi_count = 0
    plate_det_calls = 0
    raw_cand_total = 0
    accepted_cand = 0
    rejected_cand = 0

    crop_widths, crop_heights, quality_scores = [], [], []

    ocr_calls = 0
    ocr_empty = 0
    ocr_returned_text = 0
    raw_ocr_confidences = []

    texts_entering_norm = 0
    texts_surviving_norm = 0
    norm_rejected = 0

    val_attempts = 0
    val_accepted = 0
    val_rejected = 0
    rejection_reasons = {}

    per_call_traces = []   # Full traces; first 25 saving raw OCR
    crops_saved = 0

    cap = cv2.VideoCapture(VIDEO_PATH)
    t_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % SAMPLE_EVERY_N != 0:
            continue

        timestamp = frame_num / max(1.0, vid_fps)

        # Stage 1-2: Detection + Tracking
        events = detector.detect_frame(frame, camera_id="CAM_01", frame_id=frame_num, timestamp=timestamp, img_w=vid_w, img_h=vid_h)
        tracks = tracker.update_tracks(events, frame_num, timestamp)

        for track in tracks:
            if track.status not in [TrackStatus.CONFIRMED, TrackStatus.TENTATIVE]:
                continue

            vehicle_roi_count += 1
            ok, v_crop, v_bbox = plate_det.extract_vehicle_crop(frame, track.current_bbox, vid_w, vid_h)
            if not ok or v_crop is None or not isinstance(v_crop, np.ndarray):
                continue

            plate_det_calls += 1
            candidates = plate_det.extract_plate_candidates(v_crop, v_bbox)
            raw_cand_total += len(candidates)

            if not candidates:
                rejected_cand += 1
                continue

            for plate_crop, plate_bbox in candidates:
                if not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
                    rejected_cand += 1
                    continue

                ch, cw = plate_crop.shape[:2]
                q = plate_det.calculate_roi_quality(plate_crop, plate_bbox, vid_w, vid_h)
                crop_widths.append(cw)
                crop_heights.append(ch)
                quality_scores.append(q)

                # Save first N representative crops
                if crops_saved < MAX_SAVE_CROPS:
                    base = f"f{frame_num:05d}_{track.track_id}"
                    cv2.imwrite(os.path.join(SAMPLES_DIR, f"{base}_crop_orig.png"), plate_crop)
                    warped, ok_w = plate_det.correct_perspective(plate_crop)
                    if ok_w and isinstance(warped, np.ndarray):
                        cv2.imwrite(os.path.join(SAMPLES_DIR, f"{base}_crop_warped.png"), warped)
                    prep = preprocessor.preprocess_plate(plate_crop)
                    best_k = list(prep.keys())[0] if prep else None
                    if best_k and isinstance(prep[best_k], np.ndarray):
                        cv2.imwrite(os.path.join(SAMPLES_DIR, f"{base}_prep_{best_k}.png"), prep[best_k])
                    crops_saved += 1

                accepted_cand += 1

                # OCR Stage — raw capture BEFORE normalization
                prep_variants = list(preprocessor.preprocess_plate(plate_crop).values())
                ocr_calls += 1

                # CAPTURE RAW EasyOCR output BEFORE any processing
                raw_easyocr = safe_ocr_raw(ocr_engine, prep_variants)

                # Also get the processed output for comparison
                proc_text, proc_conf, variant_used = ocr_engine.recognize_text(prep_variants)

                trace = {
                    "frame": frame_num,
                    "track_id": track.track_id,
                    "plate_bbox": [round(x,1) for x in [plate_bbox.x1, plate_bbox.y1, plate_bbox.x2, plate_bbox.y2]],
                    "crop_w": cw,
                    "crop_h": ch,
                    "aspect_ratio": round(cw / max(1.0, ch), 2),
                    "quality_score": round(q, 3),
                    "preprocessing_variant": variant_used or "none",
                    "raw_easyocr_outputs": raw_easyocr,   # Raw, before any correction
                    "combined_raw_text": proc_text,         # After combining multi-box
                    "ocr_confidence": round(proc_conf, 4),
                }

                if not proc_text:
                    ocr_empty += 1
                    trace["empty"] = True
                    trace["norm_text"] = ""
                    trace["validation_status"] = "OCR_EMPTY"
                    trace["rejection_reason"] = "OCR_EMPTY"
                    rejection_reasons["OCR_EMPTY"] = rejection_reasons.get("OCR_EMPTY", 0) + 1
                else:
                    ocr_returned_text += 1
                    raw_ocr_confidences.append(proc_conf)
                    trace["empty"] = False

                    # Normalization
                    texts_entering_norm += 1
                    norm_text = TextNormalizer.normalize(proc_text)
                    trace["norm_text"] = norm_text
                    if norm_text:
                        texts_surviving_norm += 1
                    else:
                        norm_rejected += 1
                        trace["validation_status"] = "NORMALIZATION_REJECTED"
                        trace["rejection_reason"] = "NORMALIZATION_REJECTED (empty after normalize)"
                        rejection_reasons["NORMALIZATION_REJECTED"] = rejection_reasons.get("NORMALIZATION_REJECTED", 0) + 1
                        per_call_traces.append(trace)
                        continue

                    # Validation
                    val_attempts += 1
                    v_status, v_conf = IndianPlateValidator.validate_format(norm_text)
                    trace["validation_status"] = v_status.name
                    if v_status == PlateValidationStatus.VALID:
                        val_accepted += 1
                        trace["rejection_reason"] = "NONE"
                    else:
                        val_rejected += 1
                        reason = f"INVALID_FORMAT:{norm_text}"
                        short_reason = f"FORMAT_{v_status.name}"
                        trace["rejection_reason"] = short_reason
                        rejection_reasons[short_reason] = rejection_reasons.get(short_reason, 0) + 1

                per_call_traces.append(trace)

    cap.release()
    elapsed = time.time() - t_start

    # Compute distributions
    def pct(n, d): return f"{100.0*n//max(1,d):.0f}%"

    q_bins = {"<0.25": 0, "0.25-0.40": 0, "0.40-0.55": 0, "0.55-0.70": 0, ">=0.70": 0}
    for q in quality_scores:
        if q < 0.25: q_bins["<0.25"] += 1
        elif q < 0.40: q_bins["0.25-0.40"] += 1
        elif q < 0.55: q_bins["0.40-0.55"] += 1
        elif q < 0.70: q_bins["0.55-0.70"] += 1
        else: q_bins[">=0.70"] += 1

    # Build report text
    traces_sample = [t for t in per_call_traces if not t.get("empty", True)][:15]
    empty_samples = [t for t in per_call_traces if t.get("empty", False)][:5]

    lines = [
        "=" * 68,
        "CHRONOEYE INFINITY — ANPR ROOT CAUSE REPORT",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 68,
        "",
        "VIDEO",
        f"  Frames:      {total_frames}",
        f"  Resolution:  {vid_w}x{vid_h}",
        f"  FPS:         {vid_fps:.2f}",
        f"  Sample rate: every {SAMPLE_EVERY_N} frames",
        f"  Elapsed:     {elapsed:.1f}s",
        "",
        "PLATE DETECTION",
        f"  Vehicle ROIs:          {vehicle_roi_count}",
        f"  Plate detector calls:  {plate_det_calls}",
        f"  Raw candidates:        {raw_cand_total}  (avg {raw_cand_total/max(1,plate_det_calls):.2f} per call — explains >1 candidate per detector call)",
        f"  Accepted candidates:   {accepted_cand}",
        f"  Rejected candidates:   {rejected_cand}",
        "",
        "PLATE CROP QUALITY",
        f"  Widths:   min={min(crop_widths) if crop_widths else 0}  avg={sum(crop_widths)/max(1,len(crop_widths)):.1f}  max={max(crop_widths) if crop_widths else 0}",
        f"  Heights:  min={min(crop_heights) if crop_heights else 0}  avg={sum(crop_heights)/max(1,len(crop_heights)):.1f}  max={max(crop_heights) if crop_heights else 0}",
        f"  Aspect:   min={min(cw/max(1,ch) for cw,ch in zip(crop_widths,crop_heights)) if crop_widths else 0:.2f}  avg={(sum(cw/max(1,ch) for cw,ch in zip(crop_widths,crop_heights))/max(1,len(crop_widths))):.2f}  max={max(cw/max(1,ch) for cw,ch in zip(crop_widths,crop_heights)) if crop_widths else 0:.2f}",
        f"  Quality distribution: {q_bins}",
        "",
        "OCR",
        f"  OCR calls:            {ocr_calls}",
        f"  OCR returned text:    {ocr_returned_text}  ({pct(ocr_returned_text, ocr_calls)})",
        f"  OCR returned empty:   {ocr_empty}  ({pct(ocr_empty, ocr_calls)})",
        f"  Avg confidence:       {sum(raw_ocr_confidences)/max(1,len(raw_ocr_confidences)):.4f}",
        f"  Max confidence:       {max(raw_ocr_confidences) if raw_ocr_confidences else 0:.4f}",
        f"  Min confidence:       {min(raw_ocr_confidences) if raw_ocr_confidences else 0:.4f}",
        "",
        "RAW OCR EXAMPLES (non-empty, first 15; BEFORE normalization/validation):",
    ]

    for i, t in enumerate(traces_sample, 1):
        raw_out = t.get("raw_easyocr_outputs", [])
        raw_parts = []
        for r in raw_out:
            if "text" in r:
                text = r.get("text", "")
                conf = float(r.get("conf", 0.0))
                raw_parts.append(f"'{text}' ({conf:.2f})")
        raw_str = " | ".join(raw_parts) if raw_parts else "(none)"
        lines.append(
            f"  {i:2d}. F{t['frame']:5d} Trk:{t['track_id']}  "
            f"crop={t['crop_w']}x{t['crop_h']} ar={t['aspect_ratio']} Q={t['quality_score']:.3f}  "
            f"raw_easyocr=[{raw_str}]  "
            f"combined='{t['combined_raw_text']}'  conf={t['ocr_confidence']:.3f}  "
            f"norm='{t['norm_text']}'  status={t['validation_status']}  reason={t['rejection_reason']}"
        )

    lines += [
        "",
        "EMPTY OCR SAMPLES (first 5):",
    ]
    for i, t in enumerate(empty_samples, 1):
        lines.append(f"  {i}. F{t['frame']:5d} Trk:{t['track_id']}  crop={t['crop_w']}x{t['crop_h']} Q={t['quality_score']:.3f}  reason=OCR_EMPTY")

    lines += [
        "",
        "NORMALIZATION",
        f"  Texts entering:   {texts_entering_norm}",
        f"  Texts surviving:  {texts_surviving_norm}",
        f"  Norm rejected:    {norm_rejected}",
        "",
        "VALIDATION",
        f"  Attempts:   {val_attempts}",
        f"  Accepted:   {val_accepted}",
        f"  Rejected:   {val_rejected}",
        f"  Reasons:    {json.dumps(rejection_reasons, indent=4)}",
        "",
        "FUSION",
        "  (Fusion stats require association layer — not in this runner)",
        "  Observations created:  = validation_accepted",
        f"  Validation accepted:   {val_accepted}",
        f"  Validation rejected:   {val_rejected}",
        "",
        "FAILURE CATEGORY",
        f"  [A] EasyOCR empty: {ocr_empty} calls ({pct(ocr_empty, ocr_calls)})",
        f"  [C] Validation rejection: {val_rejected} calls",
        f"  [D] Crop quality: see width/height/quality distributions above",
        "",
        "PRIMARY ROOT CAUSE",
        "  <To be determined from evidence above>",
        "",
        "SECONDARY ROOT CAUSE",
        "  <To be determined from evidence above>",
        "",
        "MINIMAL RECOMMENDED FIX",
        "  <To be determined after evidence analysis — DO NOT CHANGE CODE YET>",
        "",
        "NO CORE CODE CHANGED: YES",
        "=" * 68,
    ]

    report_text = "\n".join(lines)
    print(report_text)

    with open(REPORT_TXT, "w") as f:
        f.write(report_text)

    # JSON
    json_data = {
        "video": {"frames": total_frames, "resolution": f"{vid_w}x{vid_h}", "fps": vid_fps, "sample_every_n": SAMPLE_EVERY_N},
        "plate_detection": {
            "vehicle_rois": vehicle_roi_count,
            "plate_det_calls": plate_det_calls,
            "raw_candidates": raw_cand_total,
            "accepted_candidates": accepted_cand,
            "rejected_candidates": rejected_cand,
        },
        "crop_quality": {
            "widths": {"min": min(crop_widths) if crop_widths else 0, "avg": round(sum(crop_widths)/max(1,len(crop_widths)),1), "max": max(crop_widths) if crop_widths else 0},
            "heights": {"min": min(crop_heights) if crop_heights else 0, "avg": round(sum(crop_heights)/max(1,len(crop_heights)),1), "max": max(crop_heights) if crop_heights else 0},
            "quality_distribution": q_bins,
        },
        "ocr": {
            "calls": ocr_calls,
            "returned_text": ocr_returned_text,
            "returned_empty": ocr_empty,
            "confidences": {"avg": round(sum(raw_ocr_confidences)/max(1,len(raw_ocr_confidences)),4), "min": round(min(raw_ocr_confidences),4) if raw_ocr_confidences else 0, "max": round(max(raw_ocr_confidences),4) if raw_ocr_confidences else 0},
        },
        "normalization": {"entering": texts_entering_norm, "surviving": texts_surviving_norm, "rejected": norm_rejected},
        "validation": {"attempts": val_attempts, "accepted": val_accepted, "rejected": val_rejected, "reasons": rejection_reasons},
        "per_call_traces": per_call_traces[:50],
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\nReport saved: {REPORT_TXT}")
    print(f"JSON saved:  {REPORT_JSON}")
    print(f"Crops saved: {crops_saved} images in {SAMPLES_DIR}")


if __name__ == "__main__":
    run()
