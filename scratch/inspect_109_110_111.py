import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath("backend"))

from app.schemas.detection import DetectorConfig, BoundingBoxXYXY
from app.perception.detector import YOLOVehicleDetector
from app.perception.bytetrack import ByteTracker
from app.perception.plate_detector import PlateDetector
from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.ocr_engine import OCREngineFactory, IndianPlateValidator

video_path = r"data\videos\traffic_video_modified.mp4"
detector = YOLOVehicleDetector(DetectorConfig(model_path="yolov8n.pt", device="cpu", confidence_threshold=0.35))
tracker = ByteTracker(camera_id="CAM_A_EAST")
plate_detector = PlateDetector()
preprocessor = PlatePreprocessor()
ocr_engine = OCREngineFactory.create_engine(prefer_real=True)

target_tracks = ["TRK_104", "TRK_105", "TRK_107", "TRK_109", "TRK_110", "TRK_111"]

cap = cv2.VideoCapture(video_path)
track_records = {t: [] for t in target_tracks}

for frame_id in range(1, 101):
    ret, frame = cap.read()
    if not ret:
        break
    
    dets = detector.detect_frame(frame, "CAM_A_EAST", frame_id, float(frame_id)/30.0, frame.shape[1], frame.shape[0])
    tracks = tracker.update(dets, float(frame_id)/30.0)
    
    for trk in tracks:
        tid = trk.track_id
        if tid in target_tracks:
            vx1, vy1, vx2, vy2 = int(trk.bbox.x1), int(trk.bbox.y1), int(trk.bbox.x2), int(trk.bbox.y2)
            track_records[tid].append({
                "frame_id": frame_id,
                "bbox": (vx1, vy1, vx2, vy2),
                "size": (vx2 - vx1, vy2 - vy1),
                "frame": frame.copy()
            })

cap.release()

out_diag_dir = r"E:\chronoeye\outputs\target_track_deep_diagnosis"
os.makedirs(out_diag_dir, exist_ok=True)

for tid in ["TRK_109", "TRK_110", "TRK_111"]:
    recs = track_records[tid]
    print(f"\n==================================================================")
    print(f"TRACK: {tid}")
    if not recs:
        print("  Track never appeared!")
        continue
    
    frames_list = [r["frame_id"] for r in recs]
    sizes = [r["size"] for r in recs]
    valid_recs = [r for r in recs if r["size"][0] > 20 and r["size"][1] > 20]
    if not valid_recs:
        print(f"  All appearances have small size: {sizes}")
        continue
        
    max_size_idx = int(np.argmax([r["size"][0]*r["size"][1] for r in valid_recs]))
    best_rec = valid_recs[max_size_idx]
    
    print(f"Appearances: {len(recs)} frames (Range: Frame {frames_list[0]} to {frames_list[-1]})")
    print(f"Vehicle Size: Min={min(sizes)}, Max={max(sizes)} (Max at frame {best_rec['frame_id']})")
    
    t_out = os.path.join(out_diag_dir, tid)
    os.makedirs(t_out, exist_ok=True)
    
    for r in valid_recs:
        fid = r["frame_id"]
        vx1, vy1, vx2, vy2 = r["bbox"]
        vw, vh = r["size"]
        v_frame = r["frame"]
        vcrop = v_frame[max(0, vy1):min(v_frame.shape[0], vy2), max(0, vx1):min(v_frame.shape[1], vx2)]
        
        cv2.imwrite(os.path.join(t_out, f"f{fid:03d}_veh_{vw}x{vh}.png"), vcrop)
        
        v_bbox_obj = BoundingBoxXYXY(x1=vx1, y1=vy1, x2=vx2, y2=vy2)
        cands = plate_detector.extract_plate_candidates(vcrop, v_bbox_obj)
        
        p_res = plate_detector.plate_model(vcrop, verbose=False)[0] if plate_detector.plate_model else None
        
        print(f"\n  --- Frame {fid:03d} (Vehicle BBox: {vx1},{vy1},{vx2},{vy2} | Size: {vw}x{vh}) ---")
        if p_res and len(p_res.boxes) > 0:
            print(f"    Direct YOLO in vehicle crop found {len(p_res.boxes)} boxes:")
            for b in p_res.boxes:
                bxy = b.xyxy[0].cpu().numpy().astype(int)
                print(f"      Box: {bxy}, size=({bxy[2]-bxy[0]}x{bxy[3]-bxy[1]}), conf={float(b.conf[0]):.3f}")
        else:
            print("    Direct YOLO in vehicle crop: 0 boxes")
            
        print(f"    extract_plate_candidates produced {len(cands)} candidates:")
        for c_idx, (pcrop, pbox) in enumerate(cands):
            pw, ph = int(pbox.width), int(pbox.height)
            aspect = pw / float(max(1, ph))
            
            is_stacked = preprocessor.is_likely_stacked_plate(pcrop)
            stacked_res = None
            if is_stacked:
                stacked_res = preprocessor.process_stacked_plate(pcrop, ocr_engine)
                
            variants = preprocessor.preprocess_plate_roi(pcrop)
            res = ocr_engine.recognize_text(variants)
            raw_t, norm_t, conf_t = res[:3]
            
            val_stat, val_conf = IndianPlateValidator.validate_format(norm_t)
            
            cv2.imwrite(os.path.join(t_out, f"f{fid:03d}_cand_{c_idx}_{pw}x{ph}_{norm_t}.png"), pcrop)
            
            print(f"      Cand {c_idx}: size={pw}x{ph} (aspect={aspect:.2f}, stacked_flag={is_stacked})")
            print(f"        Single OCR: raw='{raw_t}', norm='{norm_t}', conf={conf_t:.3f}, val_status={val_stat.value if hasattr(val_stat, 'value') else val_stat}")
            if stacked_res:
                st_raw, st_norm, st_conf, st_var = stacked_res
                st_val, _ = IndianPlateValidator.validate_format(st_norm)
                print(f"        Stacked OCR: raw='{st_raw}', norm='{st_norm}', conf={st_conf:.3f}, val={st_val.value if hasattr(st_val, 'value') else st_val}")
