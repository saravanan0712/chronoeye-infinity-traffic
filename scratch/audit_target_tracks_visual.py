import cv2
import numpy as np
import os
import sys

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

out_dir = r"E:\chronoeye\outputs\target_track_visual_audit"
os.makedirs(out_dir, exist_ok=True)

# Collect all frames for target tracks
track_data = {t: [] for t in target_tracks}

for frame_idx in range(1, 101):
    success, frame, meta = src.read_frame()
    if not success:
        break
    
    dets = detector.detect_frame(frame, "CAM_A_EAST", frame_idx, meta.timestamp, meta.width, meta.height)
    tracks = tracker.update(dets, meta.timestamp)
    
    for trk in tracks:
        tid = trk.track_id
        if tid in target_tracks:
            v_bbox = trk.current_bbox
            vx1, vy1, vx2, vy2 = int(v_bbox.x1), int(v_bbox.y1), int(v_bbox.x2), int(v_bbox.y2)
            vw, vh = vx2 - vx1, vy2 - vy1
            
            # Check localization path
            succ, vcrop, _ = alpr.plate_detector.extract_vehicle_crop(frame, v_bbox, meta.width, meta.height)
            candidates = []
            if succ and vcrop is not None:
                candidates = alpr.plate_detector.extract_plate_candidates(vcrop, v_bbox)
            
            cand_details = []
            for c_idx, (pcrop, p_bbox) in enumerate(candidates):
                cw, ch = int(p_bbox.width), int(p_bbox.height)
                # Test OCR
                variants = alpr.preprocessor.preprocess_plate_roi(pcrop)
                ocr_res = alpr.ocr_engine.recognize_text(variants)
                raw_t, norm_t, conf_t = ocr_res[:3]
                
                # Stacked test
                stacked_res = alpr.preprocessor.process_stacked_plate(pcrop, alpr.ocr_engine)
                
                cand_details.append({
                    "idx": c_idx,
                    "crop_size": (cw, ch),
                    "raw": raw_t,
                    "norm": norm_t,
                    "conf": conf_t,
                    "stacked": stacked_res
                })
            
            entry = {
                "frame_id": frame_idx,
                "bbox": (vx1, vy1, vx2, vy2),
                "size": (vw, vh),
                "candidates": cand_details
            }
            track_data[tid].append(entry)
            
            # Save frame crop and candidate crops for every frame
            t_dir = os.path.join(out_dir, tid)
            os.makedirs(t_dir, exist_ok=True)
            if vcrop is not None and vcrop.size > 0:
                cv2.imwrite(os.path.join(t_dir, f"f{frame_idx:03d}_veh_{vw}x{vh}.png"), vcrop)
            for c_idx, (pcrop, p_bbox) in enumerate(candidates):
                cw, ch = int(p_bbox.width), int(p_bbox.height)
                cv2.imwrite(os.path.join(t_dir, f"f{frame_idx:03d}_cand{c_idx}_{cw}x{ch}.png"), pcrop)
                
    # Run alpr
    for trk in tracks:
        alpr.process_track_frame(trk, frame, meta.timestamp, frame_idx, meta.width, meta.height)

src.release()

print("\n\n=================== COMPREHENSIVE TRACK AUDIT ===================")
for tid in target_tracks:
    fused = alpr.fusion_engine.get_fused_identity(tid)
    entries = track_data[tid]
    print(f"\n-----------------------------------------------------------")
    print(f"TRACK: {tid}")
    print(f"Frames Active: {len(entries)} (Range: {entries[0]['frame_id'] if entries else 'None'} -> {entries[-1]['frame_id'] if entries else 'None'})")
    if entries:
        print(f"Max Vehicle Size: {max([e['size'] for e in entries])}")
    print(f"Fusion Final: Best='{fused.best_plate_number if fused else None}', Pending='{fused.pending_plate_number if fused else None}', Status={fused.status if fused else None}")
    print(f"Frame-by-frame candidates:")
    for e in entries:
        if e["candidates"]:
            print(f"  Frame {e['frame_id']:03d} (Veh {e['size'][0]}x{e['size'][1]}):")
            for c in e["candidates"]:
                print(f"    Cand {c['idx']} ({c['crop_size'][0]}x{c['crop_size'][1]}): Single='{c['norm']}' (conf={c['conf']:.2f}) | Stacked={c['stacked']}")
