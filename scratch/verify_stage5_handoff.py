import sys
import os
import cv2

sys.path.insert(0, os.path.abspath("backend"))

from app.schemas.detection import DetectorConfig
from app.perception.detector import YOLOVehicleDetector
from app.perception.bytetrack import ByteTracker
from app.perception.plate_association import PlateTrackerAssociationManager
from app.perception.journey import JourneyReconstructionEngine

cap = cv2.VideoCapture("data/videos/traffic_video_modified.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

detector = YOLOVehicleDetector(DetectorConfig(model_path="yolov8n.pt", device="cpu", confidence_threshold=0.35))
tracker = ByteTracker(camera_id="CAM_A_EAST")
alpr = PlateTrackerAssociationManager(ocr_frame_interval=5)
journey_engine = JourneyReconstructionEngine()

cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
print("Verifying Stage 4 -> Stage 5 Real-Video Handoff on frames 6-9...", flush=True)

for frame_id in range(6, 10):
    ret, frame = cap.read()
    if not ret:
        break
    timestamp = frame_id / float(fps)
    h, w = frame.shape[:2]
    
    dets = detector.detect_frame(frame, "CAM_A_EAST", frame_id, timestamp, w, h)
    tracks = tracker.update(dets, timestamp)
    
    for trk in tracks:
        ev = alpr.process_track_frame(trk, frame, timestamp, frame_id, w, h)
        if ev:
            try:
                journey = journey_engine.process_track_evidence(evidence=ev, track=trk, frame=frame)
                plate_str = ev.associated_plate.best_plate_number if (ev.associated_plate and ev.associated_plate.best_plate_number) else "None"
                status = ev.associated_plate.status if ev.associated_plate else "NO_PLATE"
                if status == "CONFIRMED":
                    print(f"[HANDOFF VERIFIED] Frame {frame_id:04d} | Track {trk.track_id} | Type: {trk.vehicle_type} | Confirmed Plate: {plate_str} | Journey ID: {journey.journey_id} | Global ID: {journey.global_vehicle_id}", flush=True)
            except Exception as e:
                import traceback
                print(f"[EXCEPTION in Journey Engine] Frame {frame_id} Track {trk.track_id}: {e}", flush=True)
                traceback.print_exc()

cap.release()
print("Stage 5 handoff test completed successfully!", flush=True)
