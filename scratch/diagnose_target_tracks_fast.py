import sys
import os
import cv2
import json
import numpy as np

sys.path.insert(0, os.path.abspath("backend"))

from app.schemas.detection import DetectorConfig
from app.perception.detector import YOLOVehicleDetector
from app.perception.bytetrack import ByteTracker
from app.perception.plate_association import PlateTrackerAssociationManager
from app.perception.frame_source import VideoFrameSource
from app.perception.ocr_engine import IndianPlateValidator

target_tracks = ["TRK_104", "TRK_105", "TRK_107", "TRK_109", "TRK_110", "TRK_111"]

src = VideoFrameSource("data/videos/traffic_video_modified.mp4")
detector = YOLOVehicleDetector(DetectorConfig(model_path="yolov8n.pt", device="cpu", confidence_threshold=0.35))
tracker = ByteTracker(camera_id="CAM_A_EAST")
alpr = PlateTrackerAssociationManager(ocr_frame_interval=5)

out_dir = r"E:\chronoeye\outputs\target_track_diagnosis"
os.makedirs(out_dir, exist_ok=True)
for tid in target_tracks:
    os.makedirs(os.path.join(out_dir, tid), exist_ok=True)

track_presence = {tid: [] for tid in target_tracks}

print("Running 100-frame pipeline...", flush=True)

for frame_idx in range(1, 101):
    success, frame, meta = src.read_frame()
    if not success:
        break
    
    dets = detector.detect_frame(frame, "CAM_A_EAST", frame_idx, meta.timestamp, meta.width, meta.height)
    tracks = tracker.update(dets, meta.timestamp)
    
    # Save vehicle crop for target tracks if present
    for trk in tracks:
        tid = trk.track_id
        if tid in target_tracks:
            vx1, vy1, vx2, vy2 = int(trk.bbox.x1), int(trk.bbox.y1), int(trk.bbox.x2), int(trk.bbox.y2)
            vw, vh = vx2 - vx1, vy2 - vy1
            track_presence[tid].append({
                "frame_id": frame_idx,
                "bbox": [vx1, vy1, vx2, vy2],
                "size": [vw, vh]
            })
            # Save periodic/key vehicle crop
            if frame_idx % 5 == 0 or len(track_presence[tid]) == 1:
                vcrop = frame[max(0, vy1):min(meta.height, vy2), max(0, vx1):min(meta.width, vx2)]
                if vcrop.size > 0:
                    cv2.imwrite(os.path.join(out_dir, tid, f"frame_{frame_idx:03d}_veh_{vw}x{vh}.png"), vcrop)
                    
        # ALPR process
        alpr.process_track_frame(trk, frame, meta.timestamp, frame_idx, meta.width, meta.height)

src.release()

print("\nPipeline completed across 100 frames.", flush=True)

# Print Summary for each target track
for tid in target_tracks:
    fused = alpr.fusion_engine.get_fused_identity(tid)
    pres = track_presence[tid]
    print(f"\n=======================================================")
    print(f"TRACK: {tid}")
    print(f"Appearances: {len(pres)} frames (Frames: {[p['frame_id'] for p in pres]})")
    if pres:
        print(f"Vehicle sizes: min={min([p['size'] for p in pres])}, max={max([p['size'] for p in pres])}")
    if fused:
        print(f"Fusion State: Status={fused.status}, Best='{fused.best_plate_number}', Pending='{fused.pending_plate_number}', Conf={fused.overall_confidence:.3f}")
        print(f"Total Observation Candidates: {len(fused.candidates)}")
        for c in fused.candidates:
            print(f"  Frame {c.frame_id:03d}: text='{c.plate_text}', norm='{c.cleaned_text}', conf={c.confidence:.3f}, source={c.source}, crop={c.crop_width}x{c.crop_height}")
    else:
        print("Fusion State: None (No observations registered)")
