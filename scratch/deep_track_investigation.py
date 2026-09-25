import cv2, json, os, sys
sys.path.insert(0, os.path.abspath("backend"))

from app.schemas.detection import DetectorConfig
from app.perception.detector import YOLOVehicleDetector
from app.perception.bytetrack import ByteTracker
from app.perception.plate_detector import PlateDetector
from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.ocr_engine import OCREngineFactory

video_path = r"E:\chronoeye\data\videos\traffic_video_modified.mp4"
cap = cv2.VideoCapture(video_path)

detector = YOLOVehicleDetector(DetectorConfig(model_path="yolov8n.pt", device="cpu", confidence_threshold=0.35))
tracker = ByteTracker(camera_id="CAM_A_EAST")
pdetector = PlateDetector(model_path="backend/models/license_plate_detector.pt")
preprocessor = PlatePreprocessor()
ocr = OCREngineFactory.create_engine(prefer_real=True)

out_dir = r"E:\chronoeye\outputs\point_e_deep_tracks"
os.makedirs(out_dir, exist_ok=True)

tracks_to_follow = ["TRK_104", "TRK_114", "TRK_116", "TRK_117", "TRK_125", "TRK_127", "TRK_111", "TRK_112", "TRK_109", "TRK_126", "TRK_128"]

track_data = {t: [] for t in tracks_to_follow}

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    timestamp = frame_idx / 25.0
    h, w = frame.shape[:2]
    
    dets = detector.detect_frame(frame, "CAM_A_EAST", frame_idx, timestamp, w, h)
    tracks = tracker.update(dets, timestamp)
    
    for trk in tracks:
        tid = trk.track_id
        if tid in track_data:
            vx1, vy1, vx2, vy2 = int(trk.bbox.x1), int(trk.bbox.y1), int(trk.bbox.x2), int(trk.bbox.y2)
            vw, vh = vx2 - vx1, vy2 - vy1
            
            ok, vcrop, v_bbox = pdetector.extract_vehicle_crop(frame, trk.bbox, w, h)
            if not ok or vcrop is None or vcrop.size == 0:
                continue
            
            # Find plate detections inside this vehicle
            plate_cands = pdetector.extract_plate_candidates(vcrop, v_bbox)
            
            cands_info = []
            for pcrop, p_bbox in plate_cands:
                if pcrop is None or (hasattr(pcrop, "size") and pcrop.size == 0):
                    continue
                
                ph, pw = pcrop.shape[:2] if hasattr(pcrop, "shape") else (0, 0)
                if pw == 0 or ph == 0:
                    continue
                
                is_stacked = preprocessor.is_likely_stacked_plate(pcrop)
                single_res = ocr.recognize_text(preprocessor.preprocess_plate_roi(pcrop))
                stacked_res = preprocessor.process_stacked_plate(pcrop, ocr) if is_stacked else None
                
                cands_info.append({
                    "bbox": (int(p_bbox.x1), int(p_bbox.y1), int(p_bbox.x2), int(p_bbox.y2)),
                    "size": (pw, ph),
                    "aspect": pw / float(max(1, ph)),
                    "is_stacked": is_stacked,
                    "single_ocr": single_res[:3],
                    "stacked_ocr": stacked_res,
                })
                
                # Save representative crop
                if frame_idx % 5 == 0 or len(track_data[tid]) < 5:
                    crop_fname = f"{tid}_f{frame_idx:03d}_{pw}x{ph}.png"
                    cv2.imwrite(os.path.join(out_dir, crop_fname), pcrop)
            
            track_data[tid].append({
                "frame": frame_idx,
                "v_bbox": (vx1, vy1, vx2, vy2),
                "v_size": (vw, vh),
                "plates": cands_info,
            })
            
            # Save largest vehicle crop
            if len(track_data[tid]) == 1 or (vw * vh) > max(x["v_size"][0]*x["v_size"][1] for x in track_data[tid][:-1]):
                cv2.imwrite(os.path.join(out_dir, f"{tid}_vehicle_best_f{frame_idx:03d}_{vw}x{vh}.png"), vcrop)

cap.release()

# Print detailed diagnostic report for each target track
for tid in tracks_to_follow:
    data = track_data[tid]
    print("\n" + "=" * 70)
    print(f"DEEP DIAGNOSIS: {tid} ({len(data)} tracked frames)")
    if not data:
        print("  No track appearances recorded.")
        continue
    
    start_f = data[0]["frame"]
    end_f = data[-1]["frame"]
    max_v = max(data, key=lambda x: x["v_size"][0] * x["v_size"][1])
    print(f"  Frame Range: f{start_f:03d} to f{end_f:03d}")
    print(f"  Max Vehicle Size: {max_v['v_size'][0]}x{max_v['v_size'][1]} at frame {max_v['frame']}")
    
    plates_seen = [d for d in data if d["plates"]]
    print(f"  Frames with Plate Detections: {len(plates_seen)}/{len(data)}")
    
    for d in plates_seen:
        f = d["frame"]
        for p in d["plates"]:
            print(f"    Frame {f:03d}: plate size={p['size'][0]}x{p['size'][1]} (aspect={p['aspect']:.2f}, stacked={p['is_stacked']})")
            print(f"      Single OCR:  {p['single_ocr']}")
            if p['stacked_ocr']:
                print(f"      Stacked OCR: {p['stacked_ocr']}")
