import sys
import os
import time
import cv2
import json

sys.path.insert(0, os.path.abspath("backend"))

from app.schemas.detection import DetectorConfig
from app.perception.detector import YOLOVehicleDetector
from app.perception.bytetrack import ByteTracker
from app.perception.plate_association import PlateTrackerAssociationManager
from app.perception.frame_source import VideoFrameSource

video_path = r"data/videos/traffic_video_modified.mp4"
src = VideoFrameSource(video_path)
detector = YOLOVehicleDetector(DetectorConfig(model_path="yolov8n.pt", device="cpu", confidence_threshold=0.35))
tracker = ByteTracker(camera_id="CAM_A_EAST")
alpr = PlateTrackerAssociationManager(ocr_frame_interval=5)

out_dir = r"E:\chronoeye\outputs\full_video_audit"
os.makedirs(out_dir, exist_ok=True)

track_best_crops = {}
track_history = {}

t_start = time.time()
frame_idx = 0

print(f"Starting FULL VIDEO AUDIT on {video_path} (Total frames: 715)...", flush=True)

while True:
    success, frame, meta = src.read_frame()
    if not success:
        break
    frame_idx += 1
    
    dets = detector.detect_frame(frame, "CAM_A_EAST", frame_idx, meta.timestamp, meta.width, meta.height)
    tracks = tracker.update(dets, meta.timestamp)
    
    for trk in tracks:
        tid = trk.track_id
        if tid not in track_history:
            track_history[tid] = {
                "start_frame": frame_idx,
                "end_frame": frame_idx,
                "frame_count": 0,
                "max_size": (0, 0),
                "best_frame": frame_idx,
            }
        
        hist = track_history[tid]
        hist["end_frame"] = frame_idx
        hist["frame_count"] += 1
        
        vx1, vy1, vx2, vy2 = int(trk.bbox.x1), int(trk.bbox.y1), int(trk.bbox.x2), int(trk.bbox.y2)
        vw, vh = vx2 - vx1, vy2 - vy1
        
        if (vw * vh) > (hist["max_size"][0] * hist["max_size"][1]):
            hist["max_size"] = (vw, vh)
            hist["best_frame"] = frame_idx
            # Save best vehicle crop
            vcrop = frame[max(0, vy1):min(meta.height, vy2), max(0, vx1):min(meta.width, vx2)]
            if vcrop.size > 0:
                track_best_crops[tid] = vcrop.copy()

        # Run standard ALPR
        alpr.process_track_frame(trk, frame, meta.timestamp, frame_idx, meta.width, meta.height)
        
    if frame_idx % 50 == 0 or frame_idx in (1, 6, 8, 41, 80, 100, 200, 300, 400, 500, 600, 700, 715):
        confirmed = {t: alpr.fusion_engine.get_fused_identity(t).best_plate_number 
                     for t in alpr.fusion_engine.fused_identities 
                     if alpr.fusion_engine.get_fused_identity(t) and alpr.fusion_engine.get_fused_identity(t).confirmed}
        pending = {t: alpr.fusion_engine.get_fused_identity(t).best_plate_number or alpr.fusion_engine.get_fused_identity(t).pending_plate_number
                   for t in alpr.fusion_engine.fused_identities 
                   if alpr.fusion_engine.get_fused_identity(t) and not alpr.fusion_engine.get_fused_identity(t).confirmed and (alpr.fusion_engine.get_fused_identity(t).best_plate_number or alpr.fusion_engine.get_fused_identity(t).pending_plate_number)}
        elapsed = time.time() - t_start
        print(f"Frame {frame_idx:03d}/715 ({elapsed:.1f}s) | Active Tracks: {len(tracks)} | Confirmed ({len(confirmed)}): {confirmed} | Pending ({len(pending)}): {pending}", flush=True)

src.release()
total_runtime = time.time() - t_start

print(f"\nCompleted {frame_idx} frames in {total_runtime:.2f}s ({frame_idx/max(0.1, total_runtime):.2f} FPS).", flush=True)

# Save all best vehicle crops for visual audit
for tid, crop in track_best_crops.items():
    hist = track_history.get(tid, {})
    bf = hist.get("best_frame", 0)
    ms = hist.get("max_size", (0, 0))
    cv2.imwrite(os.path.join(out_dir, f"{tid}_best_f{bf:03d}_{ms[0]}x{ms[1]}.png"), crop)

# Collate results
results = []
for tid, hist in sorted(track_history.items(), key=lambda x: x[1]["start_frame"]):
    fused = alpr.fusion_engine.get_fused_identity(tid)
    res = {
        "track_id": tid,
        "start_frame": hist["start_frame"],
        "end_frame": hist["end_frame"],
        "frame_count": hist["frame_count"],
        "max_size": hist["max_size"],
        "best_frame": hist["best_frame"],
        "fused_status": fused.status if fused else "NONE",
        "confirmed": fused.confirmed if fused else False,
        "best_plate": fused.best_plate_number if fused else None,
        "pending_plate": fused.pending_plate_number if fused else None,
        "overall_conf": fused.overall_confidence if fused else 0.0,
        "observations_count": fused.observation_count if fused else 0,
        "observations": [
            {
                "frame_id": o["frame_id"],
                "raw_text": o["raw_text"],
                "normalized_text": o["normalized_text"],
                "ocr_confidence": o["ocr_confidence"],
                "overall_confidence": o.get("score", 0.0),
                "validation_status": o.get("validation_status", ""),
            } for o in (fused.raw_observations_audit if fused else [])
        ]
    }
    results.append(res)

with open(os.path.join(out_dir, "full_video_audit_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n================== FULL VIDEO AUDIT SUMMARY ==================")
print(f"Total Frames Processed: {frame_idx}")
print(f"Total Tracks Detected:  {len(track_history)}")
print(f"Total Runtime:          {total_runtime:.2f}s")
print(f"\nTracks Summary:")
for r in results:
    status_str = f"CONFIRMED '{r['best_plate']}' (conf={r['overall_conf']:.2f})" if r['confirmed'] else (f"PENDING '{r['pending_plate'] or r['best_plate']}' (conf={r['overall_conf']:.2f})" if (r['pending_plate'] or r['best_plate']) else "NONE")
    obs_summary = ", ".join([f"f{o['frame_id']}='{o['normalized_text']}'({o['ocr_confidence']:.2f})" for o in r['observations']])
    print(f"  {r['track_id']} (Frames {r['start_frame']:03d}-{r['end_frame']:03d}, {r['frame_count']} appearances, max {r['max_size'][0]}x{r['max_size'][1]}): {status_str} | Obs ({r['observations_count']}): [{obs_summary}]")
