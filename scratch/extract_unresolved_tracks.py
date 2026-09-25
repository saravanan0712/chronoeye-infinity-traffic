import cv2
import os
import sys

sys.path.insert(0, r"E:\chronoeye\backend")

from ultralytics import YOLO
from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.ocr_engine import OCREngineFactory

video_path = r"E:\chronoeye\data\videos\traffic_video_modified.mp4"
out_dir = r"E:\chronoeye\outputs\unresolved_tracks_diagnosis"
os.makedirs(out_dir, exist_ok=True)

yolo_vehicle = YOLO(r"E:\chronoeye\yolov8n.pt")
yolo_plate = YOLO(r"E:\chronoeye\backend\models\license_plate_detector.pt")
preprocessor = PlatePreprocessor()
ocr_engine = OCREngineFactory.create_engine(prefer_real=True)

# Tracks to inspect:
# TRK_104 (Auto-rickshaw in middle lane: frames 1-48)
# TRK_105 (Truck crossing: frames 20-25)
# TRK_107 (Two-wheeler in distant lane: frames 23-48)
# TRK_109 (Vehicle entering at tail: frames 90-100)
# TRK_110 (Vehicle entering at tail: frames 93-100)
# TRK_111 (Distant vehicle: frame 98-100)

cap = cv2.VideoCapture(video_path)

# Let's inspect frame by frame for these specific tracks
# We can track them using ByteTrack or inspect vehicle boxes in their spatial regions
from app.perception.tracker import ByteTrackerManager
tracker = ByteTrackerManager()

frame_idx = 0
track_crops = {
    "TRK_104": [],
    "TRK_105": [],
    "TRK_107": [],
    "TRK_109": [],
    "TRK_110": [],
    "TRK_111": []
}

while cap.isOpened() and frame_idx < 100:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    # Run YOLO vehicle
    v_res = yolo_vehicle(frame, classes=[2, 3, 5, 7], verbose=False)[0]
    detections = []
    for b in v_res.boxes:
        xy = b.xyxy[0].cpu().numpy()
        conf = float(b.conf[0])
        cls = int(b.cls[0])
        detections.append((xy, conf, cls))
        
    # Update tracker
    tracks = tracker.update_tracks(detections, frame_id=frame_idx, timestamp=frame_idx/26.3)
    
    for t in tracks:
        tid = f"TRK_{t.track_id:03d}" if isinstance(t.track_id, int) else str(t.track_id)
        if tid in track_crops:
            bx = t.current_bbox
            x1, y1, x2, y2 = int(bx.x1), int(bx.y1), int(bx.x2), int(bx.y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            vcrop = frame[y1:y2, x1:x2]
            if vcrop.size > 0:
                track_crops[tid].append((frame_idx, (x1, y1, x2, y2), vcrop))

cap.release()

print("Summary of track frames collected:")
for tid, frames_list in track_crops.items():
    print(f"  {tid}: {len(frames_list)} frames (frames {frames_list[0][0] if frames_list else 'N/A'} to {frames_list[-1][0] if frames_list else 'N/A'})")
    # Save a few sample crops for inspection
    if frames_list:
        step = max(1, len(frames_list) // 4)
        for i in range(0, len(frames_list), step):
            f_num, bbox, crop = frames_list[i]
            fname = f"{tid}_frame_{f_num:04d}_{bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}.png"
            cv2.imwrite(os.path.join(out_dir, fname), crop)

