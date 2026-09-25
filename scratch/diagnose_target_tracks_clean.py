import sys
import os
import cv2
import json
import numpy as np

sys.path.insert(0, os.path.abspath("backend"))

from app.schemas.detection import DetectorConfig, BoundingBoxXYXY
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

# Also create subdirectories for each track
for tid in target_tracks:
    os.makedirs(os.path.join(out_dir, tid), exist_ok=True)

track_diagnostics = {tid: [] for tid in target_tracks}

print("Running 100-frame pipeline with deep tracking on target tracks...", flush=True)

for frame_idx in range(1, 101):
    success, frame, meta = src.read_frame()
    if not success:
        break
    
    dets = detector.detect_frame(frame, "CAM_A_EAST", frame_idx, meta.timestamp, meta.width, meta.height)
    tracks = tracker.update(dets, meta.timestamp)
    
    # Process target tracks
    for trk in tracks:
        tid = trk.track_id
        if tid in target_tracks:
            v_bbox = trk.current_bbox
            vx1, vy1, vx2, vy2 = int(v_bbox.x1), int(v_bbox.y1), int(v_bbox.x2), int(v_bbox.y2)
            vw = vx2 - vx1
            vh = vy2 - vy1
            
            # Extract vehicle crop
            succ_vcrop, veh_crop, _ = alpr.plate_detector.extract_vehicle_crop(frame, v_bbox, meta.width, meta.height)
            
            frame_diag = {
                "frame_id": frame_idx,
                "timestamp": meta.timestamp,
                "veh_bbox": (vx1, vy1, vx2, vy2),
                "veh_size": (vw, vh),
                "plate_candidates": [],
                "direct_plate_detections": [],
                "ocr_attempted_by_alpr": False,
                "alpr_skip_reason": None
            }
            
            # Save vehicle crop
            if veh_crop is not None and veh_crop.size > 0:
                cv2.imwrite(os.path.join(out_dir, tid, f"frame_{frame_idx:03d}_veh_{vw}x{vh}.png"), veh_crop)
            
            # Test direct plate model detection on the full frame
            if alpr.plate_detector.plate_model is not None:
                p_res = alpr.plate_detector.plate_model(frame, verbose=False)[0]
                if p_res.boxes is not None:
                    for pb in p_res.boxes:
                        pxy = pb.xyxy[0].cpu().numpy().astype(int)
                        # Check if within vehicle bounding box
                        if (pxy[0] >= vx1 - 30 and pxy[2] <= vx2 + 30 and
                            pxy[1] >= vy1 - 30 and pxy[3] <= vy2 + 30):
                            frame_diag["direct_plate_detections"].append({
                                "bbox": pxy.tolist(),
                                "conf": float(pb.conf[0]),
                                "size": [int(pxy[2]-pxy[0]), int(pxy[3]-pxy[1])]
                            })
            
            # Check what candidates extract_plate_candidates produces
            if succ_vcrop and veh_crop is not None:
                candidates = alpr.plate_detector.extract_plate_candidates(veh_crop, v_bbox)
                for c_idx, (plate_crop, p_bbox) in enumerate(candidates):
                    cw = int(p_bbox.width)
                    ch = int(p_bbox.height)
                    
                    # Run OCR on this candidate
                    ocr_res = None
                    is_stacked = alpr.preprocessor.is_likely_stacked_plate(plate_crop)
                    if is_stacked:
                        stacked_res = alpr.preprocessor.process_stacked_plate(plate_crop, alpr.ocr_engine)
                        if stacked_res is not None:
                            ocr_res = stacked_res
                    if ocr_res is None:
                        variants = alpr.preprocessor.preprocess_plate_roi(plate_crop)
                        res = alpr.ocr_engine.recognize_text(variants)
                        if len(res) == 4:
                            ocr_res = res
                        else:
                            ocr_res = (res[0], res[1], res[2], "ORIGINAL")
                    
                    raw_txt, norm_txt, ocr_c, variant_t = ocr_res
                    val_stat, val_conf = IndianPlateValidator.validate_format(norm_txt)
                    
                    cand_info = {
                        "cand_idx": c_idx,
                        "crop_size": (cw, ch),
                        "aspect": cw / float(max(1, ch)),
                        "is_stacked": is_stacked,
                        "raw_ocr": raw_txt,
                        "norm_ocr": norm_txt,
                        "ocr_conf": float(ocr_c),
                        "variant": variant_t,
                        "val_status": str(val_stat),
                        "val_conf": float(val_conf),
                    }
                    frame_diag["plate_candidates"].append(cand_info)
                    
                    # Save plate crop
                    cv2.imwrite(os.path.join(out_dir, tid, f"frame_{frame_idx:03d}_cand_{c_idx}_{cw}x{ch}_{norm_txt}.png"), plate_crop)

            track_diagnostics[tid].append(frame_diag)
            
        # Process track frame in ALPR pipeline
        alpr.process_track_frame(trk, frame, meta.timestamp, frame_idx, meta.width, meta.height)

src.release()

# Save detailed diagnostics to JSON
with open(os.path.join(out_dir, "diagnostic_summary.json"), "w") as f:
    json.dump(track_diagnostics, f, indent=2)

print("\nProcessing complete! Printing deep breakdown for each target track:")
for tid in target_tracks:
    fused = alpr.fusion_engine.get_fused_identity(tid)
    diag_list = track_diagnostics[tid]
    print(f"\n=======================================================")
    print(f"TRACK ID: {tid}")
    print(f"Total Frames Present: {len(diag_list)}")
    if diag_list:
        frame_ids = [d["frame_id"] for d in diag_list]
        print(f"Frame Range: {min(frame_ids)} to {max(frame_ids)} (Frames: {frame_ids})")
        print(f"Vehicle Dimensions (WxH) across track:")
        for d in diag_list:
            cand_str = ""
            if d["plate_candidates"]:
                cand_str = " | Candidates: " + ", ".join([f"size={c['crop_size']}, raw='{c['raw_ocr']}', norm='{c['norm_ocr']}', conf={c['ocr_conf']:.2f}, val={c['val_status']}" for c in d["plate_candidates"]])
            print(f"  Frame {d['frame_id']:03d}: Veh={d['veh_size'][0]}x{d['veh_size'][1]}, DirectPlates={d['direct_plate_detections']}{cand_str}")
    
    if fused:
        print(f"\nFused State: Status={fused.status}, Best='{fused.best_plate_number}', Pending='{fused.pending_plate_number}', OverallConf={fused.overall_confidence:.3f}")
        print(f"Accepted Candidates in Fusion ({len(fused.candidates)}):")
        for c in fused.candidates:
            print(f"  Frame {c.frame_id}: text='{c.plate_text}', norm='{c.cleaned_text}', conf={c.confidence:.3f}, source={c.source}, crop={c.crop_width}x{c.crop_height}")
    else:
        print("Fused State: None")
