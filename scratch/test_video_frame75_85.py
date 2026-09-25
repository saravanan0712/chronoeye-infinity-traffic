import sys
import os
import cv2

sys.path.insert(0, os.path.abspath("backend"))

from app.schemas.detection import DetectorConfig
from app.perception.detector import YOLOVehicleDetector
from app.perception.bytetrack import ByteTracker
from app.perception.plate_association import PlateTrackerAssociationManager

cap = cv2.VideoCapture("data/videos/traffic_video_modified.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

detector = YOLOVehicleDetector(DetectorConfig(model_path="yolov8n.pt", device="cpu", confidence_threshold=0.35))
tracker = ByteTracker(camera_id="CAM_A_EAST")
alpr = PlateTrackerAssociationManager(ocr_frame_interval=5)

# Fast-forward to frame 75
cap.set(cv2.CAP_PROP_POS_FRAMES, 75)

print("Starting pipeline from frame 75 to 86...", flush=True)
for frame_id in range(75, 87):
    ret, frame = cap.read()
    if not ret:
        break
    timestamp = frame_id / float(fps)
    h, w = frame.shape[:2]
    
    dets = detector.detect_frame(frame, "CAM_A_EAST", frame_id, timestamp, w, h)
    tracks = tracker.update(dets, timestamp)
    
    for trk in tracks:
        ev = alpr.process_track_frame(trk, frame, timestamp, frame_id, w, h)
        if ev and ev.associated_plate:
            p = ev.associated_plate
            p_num = p.best_plate_number or p.pending_plate_number
            if p_num:
                print(f"[PLATE_RESULT] Frame {frame_id:04d} Track {trk.track_id}: Plate='{p_num}' Status={p.status} Conf={p.overall_confidence:.3f}", flush=True)

cap.release()
print("Pipeline run frames 75-86 finished.", flush=True)
