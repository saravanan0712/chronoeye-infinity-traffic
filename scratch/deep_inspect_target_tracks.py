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
from app.perception.plate_fusion import TemporalPlateFusionEngine

video_path = r"data\videos\traffic_video_modified.mp4"
cap = cv2.VideoCapture(video_path)

detector = YOLOVehicleDetector(DetectorConfig(model_path="yolov8n.pt", device="cpu", confidence_threshold=0.35))
tracker = ByteTracker(camera_id="CAM_A_EAST")
plate_detector = PlateDetector()
preprocessor = PlatePreprocessor()
ocr_engine = OCREngineFactory.create_engine(prefer_real=True)

target_tracks = ["TRK_104", "TRK_105", "TRK_107", "TRK_109", "TRK_110", "TRK_111"]

track_frames = {t: {} for t in target_tracks}

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    if frame_id > 100:
        break
    
    dets = detector.detect_frame(frame, "CAM_A_EAST", frame_id, float(frame_id)/30.0, frame.shape[1], frame.shape[0])
    tracks = tracker.update(dets, float(frame_id)/30.0)
    
    for trk in tracks:
        tid = trk.track_id
        if tid in target_tracks:
            vx1, vy1, vx2, vy2 = int(trk.bbox.x1), int(trk.bbox.y1), int(trk.bbox.x2), int(trk.bbox.y2)
            vw, vh = vx2 - vx1, vy2 - vy1
            vcrop = frame[max(0, vy1):min(frame.shape[0], vy2), max(0, vx1):min(frame.shape[1], vx2)].copy()
            
            # Extract candidate ROIs
            candidates = plate_detector.extract_plate_candidates(vcrop, trk.bbox)
            
            cand_info = []
            for pcrop, pbox in candidates:
                pw, ph = int(pbox.width), int(pbox.height)
                # Single line
                variants = preprocessor.preprocess_plate_roi(pcrop)
                ocr_single = ocr_engine.recognize_text(variants)
                
                # Stacked
                ocr_stacked = None
                if preprocessor.is_likely_stacked_plate(pcrop):
                    ocr_stacked = preprocessor.process_stacked_plate(pcrop, ocr_engine)
                    
                cand_info.append({
                    "crop_shape": pcrop.shape,
                    "box": (int(pbox.x1), int(pbox.y1), int(pbox.x2), int(pbox.y2)),
                    "single_ocr": ocr_single,
                    "stacked_ocr": ocr_stacked
                })
                
            track_frames[tid][frame_id] = {
                "veh_box": (vx1, vy1, vx2, vy2),
                "veh_size": (vw, vh),
                "candidates": cand_info,
                "vcrop": vcrop
            }

cap.release()

print("\n================ DETAILED AUDIT FOR TARGET TRACKS ================")
for tid in target_tracks:
    frames = sorted(track_frames[tid].keys())
    print(f"\n#################################################################")
    print(f"TRACK: {tid}")
    print(f"Active Frames ({len(frames)}): {frames}")
    if not frames:
        print("  No frames detected for this track.")
        continue
    
    # Check key frames
    print(f"Vehicle sizes: Start={track_frames[tid][frames[0]]['veh_size']}, Max={max([track_frames[tid][f]['veh_size'] for f in frames])}, End={track_frames[tid][frames[-1]]['veh_size']}")
    
    # Print candidate detections across frames
    detected_frames = [f for f in frames if track_frames[tid][f]["candidates"]]
    print(f"Frames with Plate Candidates ({len(detected_frames)}): {detected_frames}")
    for f in detected_frames:
        info = track_frames[tid][f]
        print(f"  Frame {f:03d} (Veh {info['veh_size'][0]}x{info['veh_size'][1]}):")
        for idx, c in enumerate(info["candidates"]):
            s_raw, s_norm, s_conf = c["single_ocr"][:3]
            st_str = f" | Stacked={c['stacked_ocr']}" if c['stacked_ocr'] else ""
            print(f"    Cand {idx}: Shape={c['crop_shape']} Single='{s_norm}' (raw='{s_raw}', conf={s_conf:.2f}){st_str}")
