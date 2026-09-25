import sys
import os
import traceback
import cv2

sys.path.insert(0, os.path.abspath("backend"))

from app.schemas.detection import DetectorConfig
from app.perception.detector import YOLOVehicleDetector
from app.perception.bytetrack import ByteTracker
from app.perception.plate_association import PlateTrackerAssociationManager

cap = cv2.VideoCapture("data/videos/traffic_video_modified.mp4")
detector = YOLOVehicleDetector(DetectorConfig(model_path="yolov8n.pt", device="cpu", confidence_threshold=0.35))
tracker = ByteTracker(camera_id="CAM_A_EAST")
alpr = PlateTrackerAssociationManager(ocr_frame_interval=5)

for i in range(1, 15):
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    ts = i / 30.0
    dets = detector.detect_frame(frame, "CAM_A_EAST", i, ts, w, h)
    tracks = tracker.update(dets, ts)
    print(f"--- FRAME {i}: {len(tracks)} tracks ---")
    for trk in tracks:
        try:
            ev = alpr.process_track_frame(trk, frame, ts, i, w, h)
            p_str = ev.associated_plate.best_plate_number if ev and ev.associated_plate else "None"
            print(f"  Track {trk.track_id}: Plate={p_str}")
        except Exception as e:
            print(f"  ERROR on track {trk.track_id} in frame {i}: {e}")
            traceback.print_exc()
            sys.exit(1)

cap.release()
print("Frames 1-14 completed successfully!")
