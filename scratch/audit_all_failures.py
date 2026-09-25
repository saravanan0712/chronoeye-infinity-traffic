import json
import os
import sys
import cv2

sys.path.insert(0, os.path.abspath("backend"))

from app.perception.plate_detector import PlateDetector
from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.ocr_engine import OCREngineFactory
from app.schemas.detection import BoundingBoxXYXY

with open("outputs/full_video_audit_fix1/full_video_audit_results_fix1.json") as f:
    results = json.load(f)

res_by_id = {r["track_id"]: r for r in results}

out_dir = "outputs/failure_audit_crops"
os.makedirs(out_dir, exist_ok=True)

plate_detector = PlateDetector()
preprocessor = PlatePreprocessor()
ocr = OCREngineFactory.create_engine(prefer_real=True)

# Tracks to audit: 6 pending + 12 unresolved + special attention tracks (TRK_117, TRK_125, TRK_131)
pending_tracks = [r["track_id"] for r in results if not r["confirmed"] and (r["best_plate"] or r["pending_plate"])]
unresolved_tracks = [r["track_id"] for r in results if not r["confirmed"] and not r["best_plate"] and not r["pending_plate"]]
special_tracks = ["TRK_117", "TRK_125", "TRK_131"]

audit_track_ids = sorted(list(set(pending_tracks + unresolved_tracks + special_tracks)), key=lambda x: int(x.split("_")[1]))

print(f"Total Pending Tracks:    {len(pending_tracks)} -> {pending_tracks}")
print(f"Total Unresolved Tracks: {len(unresolved_tracks)} -> {unresolved_tracks}")
print(f"Special Confirmed Tracks: {special_tracks}")
print(f"Total Tracks to Audit:   {len(audit_track_ids)}\n")

# Open video
cap = cv2.VideoCapture("data/videos/traffic_video_modified.mp4")

track_inspection_data = {}

for tid in audit_track_ids:
    r = res_by_id[tid]
    bf = r["best_frame"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, bf - 1)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {bf} for {tid}")
        continue
    
    # Check if there is already a saved vehicle crop from full_video_audit
    veh_crop_file = None
    for fn in os.listdir("outputs/full_video_audit"):
        if fn.startswith(f"{tid}_best_"):
            veh_crop_file = os.path.join("outputs/full_video_audit", fn)
            break
            
    veh_img = cv2.imread(veh_crop_file) if veh_crop_file else None
    if veh_img is None:
        veh_img = frame
        
    vh, vw = veh_img.shape[:2]
    dummy_bbox = BoundingBoxXYXY(x1=0, y1=0, x2=vw, y2=vh)
    cands = plate_detector.extract_plate_candidates(veh_img, dummy_bbox)
    
    cand_info = []
    for c_idx, (pcrop, pbox) in enumerate(cands):
        ch, cw = pcrop.shape[:2]
        is_stacked = preprocessor.is_likely_stacked_plate(pcrop)
        vars = preprocessor.preprocess_plate_roi(pcrop)
        raw, norm, conf, var_name = ocr.recognize_text(vars)
        stacked_res = None
        if is_stacked:
            stacked_res = preprocessor.process_stacked_plate(pcrop, ocr)
        
        cand_info.append({
            "size": (cw, ch),
            "aspect": round(cw / max(1, ch), 2),
            "is_stacked": is_stacked,
            "raw": raw,
            "norm": norm,
            "conf": conf,
            "stacked": stacked_res
        })
        
        # Save plate crop for visual inspection
        cv2.imwrite(os.path.join(out_dir, f"{tid}_f{bf}_cand{c_idx}_{cw}x{ch}.png"), pcrop)

    track_inspection_data[tid] = {
        "status": r["fused_status"],
        "confirmed": r["confirmed"],
        "best_plate": r["best_plate"],
        "pending_plate": r["pending_plate"],
        "overall_conf": r["overall_conf"],
        "frame_range": (r["start_frame"], r["end_frame"]),
        "frame_count": r["frame_count"],
        "best_frame": bf,
        "max_size": r["max_size"],
        "observations": r["observations"],
        "cands": cand_info
    }

cap.release()

with open(os.path.join(out_dir, "audit_inspection_data.json"), "w") as f:
    json.dump(track_inspection_data, f, indent=2)

print("Visual extraction complete. Saved to outputs/failure_audit_crops.")
