"""
ChronoEye Point C — Full 715-frame validation run.
Uses the existing ANPR pipeline with the single-observation fusion fast-path enabled.
Also runs target-plate evaluator at the end.
No source code changes — this script simply exercises the pipeline as-is.

NOTE: This is a Python 3.10-compatible copy of run_pointc_audit.py.
The only change from the original is line 148: the f-string containing a
backslash (not valid in Python < 3.12) is rewritten as a semantically
identical expression using a helper variable.  All pipeline logic, thresholds,
and output paths are 100% unchanged.
"""
import sys, os, time, cv2, json

sys.path.insert(0, os.path.abspath("backend"))

from app.schemas.detection import DetectorConfig
from app.perception.detector import YOLOVehicleDetector
from app.perception.bytetrack import ByteTracker
from app.perception.plate_association import PlateTrackerAssociationManager
from app.perception.frame_source import VideoFrameSource
from app.perception.target_plate_evaluator import TargetPlateEvaluator

video_path = r"data/videos/traffic_video_modified.mp4"
src = VideoFrameSource(video_path)
detector = YOLOVehicleDetector(DetectorConfig(model_path="yolov8n.pt", device="cpu", confidence_threshold=0.35))
tracker = ByteTracker(camera_id="CAM_A_EAST")
alpr = PlateTrackerAssociationManager(ocr_frame_interval=5)

out_dir = r"E:\chronoeye\outputs\full_video_audit_pointc"
os.makedirs(out_dir, exist_ok=True)

track_best_crops = {}
track_history = {}

t_start = time.time()
frame_idx = 0

print(f"Starting POINT C FULL VIDEO AUDIT on {video_path}...", flush=True)

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
            vcrop = frame[max(0, vy1):min(meta.height, vy2), max(0, vx1):min(meta.width, vx2)]
            if vcrop.size > 0:
                track_best_crops[tid] = vcrop.copy()

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

# Save best vehicle crops
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

with open(os.path.join(out_dir, "full_video_audit_results_pointc.json"), "w") as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n================== POINT C FULL VIDEO AUDIT SUMMARY ==================")
print(f"Total Frames Processed: {frame_idx}")
print(f"Total Tracks Detected:  {len(track_history)}")
print(f"Total Runtime:          {total_runtime:.2f}s")
print(f"\nTracks Summary:")
for r in results:
    status_str = f"CONFIRMED '{r['best_plate']}' (conf={r['overall_conf']:.2f})" if r['confirmed'] else (f"PENDING '{r['pending_plate'] or r['best_plate']}' (conf={r['overall_conf']:.2f})" if (r['pending_plate'] or r['best_plate']) else "NONE")
    obs_summary = ", ".join([f"f{o['frame_id']}='{o['normalized_text']}'({o['ocr_confidence']:.2f})" for o in r['observations']])
    print(f"  {r['track_id']} (Frames {r['start_frame']:03d}-{r['end_frame']:03d}, {r['frame_count']} appearances, max {r['max_size'][0]}x{r['max_size'][1]}): {status_str} | Obs ({r['observations_count']}): [{obs_summary}]")

# Count categories
confirmed_tracks = [r for r in results if r['confirmed']]
pending_tracks = [r for r in results if not r['confirmed'] and (r['best_plate'] or r['pending_plate'])]
zero_obs_tracks = [r for r in results if r['observations_count'] == 0]
total_obs = sum(r['observations_count'] for r in results)

print(f"\n=================== COUNTS ===================")
print(f"Confirmed:        {len(confirmed_tracks)}")
print(f"Pending:          {len(pending_tracks)}")
print(f"Zero-observation: {len(zero_obs_tracks)}")
print(f"Total observations: {total_obs}")
# Python 3.10-compatible version of the original line 148 (no backslash inside f-string)
confirmed_plate_pairs = ", ".join(str(r['track_id']) + "=" + str(r['best_plate']) for r in confirmed_tracks)
print(f"Confirmed plates: {confirmed_plate_pairs}")

# -----------------------------------------------------------------------
# TARGET PLATE EVALUATION
# -----------------------------------------------------------------------
print("\n\n================== TARGET PLATE EVALUATION ==================")
evaluator = TargetPlateEvaluator()
target_tests = ["SX8525", "KW527", "TS5330", "ZZ9999"]
for target in target_tests:
    result = evaluator.evaluate(target, alpr.fusion_engine)
    evaluator.print_report(result)
    print(f"[TARGET_PLATE_JSON] {json.dumps(result.to_dict())}")

print("\n================== AUDIT COMPLETE ==================")
