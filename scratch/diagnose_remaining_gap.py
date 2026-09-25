import cv2
import json
import os
import sys

sys.path.insert(0, r"E:\chronoeye\backend")

from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.ocr_engine import OCREngineFactory

# Let's inspect the video frames and tracks
video_path = r"E:\chronoeye\data\videos\traffic_video_modified.mp4"
log_path = r"C:\Users\Vijayakumari K\.gemini\antigravity-ide\brain\96c1d744-5e38-4d49-88ba-1a2f244a4166\.system_generated\tasks\task-458.log"

# Parse log file for all ANPR events
anpr_events = []
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "[ANPR_TRACE]" in line or "[ANPR_OCR]" in line:
                anpr_events.append(line.strip())

print(f"Loaded {len(anpr_events)} ANPR trace events from task-458.log")

# Track summary from log
tracks = {}
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        if "[ANPR_TRACE]" in line:
            # extract frame and track
            parts = line.split()
            frame_str = None
            track_str = None
            for p in parts:
                if p.startswith("frame="):
                    frame_str = p.split("=")[1]
                elif p.startswith("track="):
                    track_str = p.split("=")[1]
            if track_str:
                if track_str not in tracks:
                    tracks[track_str] = []
                tracks[track_str].append(line.strip())

print("\n================ TRACKS IN 100-FRAME VIDEO ================")
for t, events in sorted(tracks.items()):
    print(f"\n--- {t} ({len(events)} trace lines) ---")
    ocr_attempts = [e for e in events if "OCR_ATTEMPT" in e]
    validations = [e for e in events if "VALIDATION" in e]
    fusions = [e for e in events if "FUSION" in e]
    skips = [e for e in events if "SKIP" in e]
    print(f"  OCR Attempts: {len(ocr_attempts)}")
    for a in ocr_attempts[:5]:
        print(f"    {a}")
    print(f"  Validations: {len(validations)}")
    for v in validations[:5]:
        print(f"    {v}")
    print(f"  Fusions: {len(fusions)}")
    for fs in fusions:
        print(f"    {fs}")
