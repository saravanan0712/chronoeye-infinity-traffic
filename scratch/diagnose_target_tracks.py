import sys
import os
import cv2
import json
import time

sys.path.insert(0, os.path.abspath("backend"))

from app.schemas.detection import DetectorConfig
from app.perception.detector import YOLOVehicleDetector
from app.perception.bytetrack import ByteTracker
from app.perception.plate_association import PlateTrackerAssociationManager
from app.perception.frame_source import VideoFrameSource

target_tracks = ["TRK_104", "TRK_105", "TRK_107", "TRK_109", "TRK_110", "TRK_111"]

src = VideoFrameSource("data/videos/traffic_video_modified.mp4")
detector = YOLOVehicleDetector(DetectorConfig(model_path="yolov8n.pt", device="cpu", confidence_threshold=0.35))
tracker = ByteTracker(camera_id="CAM_A_EAST")
alpr = PlateTrackerAssociationManager(ocr_frame_interval=5)

out_dir = r"E:\chronoeye\outputs\target_track_diagnosis"
os.makedirs(out_dir, exist_ok=True)

track_frame_history = {tid: [] for tid in target_tracks}

print("Running 100 frames and collecting deep diagnosis data...", flush=True)

for i in range(1, 101):
    success, frame, meta = src.read_frame()
    if not success:
        break
    
    dets = detector.detect_frame(frame, "CAM_A_EAST", i, meta.timestamp, meta.width, meta.height)
    tracks = tracker.update(dets, meta.timestamp)
    
    # Check each track
    for trk in tracks:
        tid = trk.track_id
        if tid in target_tracks:
            vx1 = int(trk.bbox.x1)
            vy1 = int(trk.bbox.y1)
            vx2 = int(trk.bbox.x2)
            vy2 = int(trk.bbox.y2)
            
            # Save vehicle crop
            veh_crop = frame[max(0, vy1):min(meta.height, vy2), max(0, vx1):min(meta.width, vx2)]
            
            history_entry = {
                "frame_id": i,
                "veh_bbox": (vx1, vy1, vx2, vy2),
                "veh_size": (vx2 - vx1, vy2 - vy1),
            }
            
            # Save vehicle crop image
            v_crop_name = f"{tid}_frame_{i:03d}_veh_{vx2-vx1}x{vy2-vy1}.png"
            if veh_crop.size > 0:
                cv2.imwrite(os.path.join(out_dir, v_crop_name), veh_crop)
            
            # Check what plate detector detects inside/around this vehicle
            p_res = alpr.plate_detector(frame, verbose=False)[0]
            detected_plates_in_veh = []
            for pb in p_res.boxes:
                pxy = pb.xyxy[0].cpu().numpy().astype(int)
                # check overlap with veh
                if pxy[0] >= vx1 - 30 and pxy[2] <= vx2 + 30 and pxy[1] >= vy1 - 30 and pxy[3] <= vy2 + 30:
                    detected_plates_in_veh.append((pxy, float(pb.conf[0])))
            
            history_entry["detected_plates_in_veh"] = [
                {"bbox": p[0].tolist(), "conf": p[1], "size": (int(p[0][2]-p[0][0]), int(p[0][3]-p[0][1]))} 
                for p in detected_plates_in_veh
            ]
            
            track_frame_history[tid].append(history_entry)

        alpr.process_track_frame(trk, frame, meta.timestamp, i, meta.width, meta.height)

src.release()

print("\n100 frames completed. Processing track history summary...", flush=True)

# Now let's check fusion identities
for tid in target_tracks:
    fused = alpr.fusion_engine.get_fused_identity(tid)
    frames = track_frame_history[tid]
    print(f"\n==================== TRACK: {tid} ====================")
    print(f"Total appearances: {len(frames)} frames")
    if frames:
        print(f"Frames present: {[f['frame_id'] for f in frames]}")
        print(f"Vehicle sizes: {[f['veh_size'] for f in frames]}")
        print(f"YOLO Plate detections per frame:")
        for f in frames:
            if f["detected_plates_in_veh"]:
                print(f"  Frame {f['frame_id']}: {f['detected_plates_in_veh']}")
    if fused:
        print(f"Fusion State: status={fused.status}, best_plate='{fused.best_plate_number}', pending='{fused.pending_plate_number}', conf={fused.overall_confidence:.3f}")
        print(f"Candidates history ({len(fused.candidates)} entries):")
        for c in fused.candidates:
            print(f"  Frame {c.frame_id}: text='{c.plate_text}', conf={c.confidence:.3f}, clean='{c.cleaned_text}', source={c.source}, w={c.crop_width}, h={c.crop_height}")
    else:
        print("No fused identity found in fusion engine.")
