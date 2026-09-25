import sys
import os
import time

sys.path.insert(0, os.path.abspath("backend"))

from app.schemas.detection import DetectorConfig
from app.perception.detector import YOLOVehicleDetector
from app.perception.bytetrack import ByteTracker
from app.perception.plate_association import PlateTrackerAssociationManager
from app.perception.frame_source import VideoFrameSource

print("Initializing components...", flush=True)
src = VideoFrameSource("data/videos/traffic_video_modified.mp4")
detector = YOLOVehicleDetector(DetectorConfig(model_path="yolov8n.pt", device="cpu", confidence_threshold=0.35))
tracker = ByteTracker(camera_id="CAM_A_EAST")
alpr = PlateTrackerAssociationManager(ocr_frame_interval=5)

print("Starting 100-frame run...", flush=True)
for i in range(1, 101):
    t0 = time.time()
    success, frame, meta = src.read_frame()
    if not success:
        print(f"Frame {i}: read_frame returned False! Stream status: {src.status}", flush=True)
        break
    
    dets = detector.detect_frame(frame, "CAM_A_EAST", i, meta.timestamp, meta.width, meta.height)
    tracks = tracker.update(dets, meta.timestamp)
    
    for trk in tracks:
        alpr.process_track_frame(trk, frame, meta.timestamp, i, meta.width, meta.height)
    
    dt = time.time() - t0
    if i % 10 == 0 or i in (1, 6, 8, 80, 85):
        # Print summary
        plates = {t: alpr.fusion_engine.get_fused_identity(t).best_plate_number 
                  for t in alpr.fusion_engine.fused_identities 
                  if alpr.fusion_engine.get_fused_identity(t) and alpr.fusion_engine.get_fused_identity(t).best_plate_number}
        print(f"Frame {i:03d} ({dt:.2f}s) | Active tracks: {len(tracks)} | Confirmed/Best plates: {plates}", flush=True)

src.release()
print("\n=== FINAL 100-FRAME RESULTS ===", flush=True)
for tid, fused in alpr.fusion_engine.fused_identities.items():
    print(f"Track {tid}: Plate='{fused.best_plate_number or fused.pending_plate_number}' Status={fused.status} Conf={fused.overall_confidence:.3f}", flush=True)
