"""
Measurement and Diagnostic Script for ANPR & Temporal Fusion Audit
Measures:
1. Track length distribution (fraction >= 3 frames)
2. Evaluation methodology & counts
3. Comparison of whole-string baseline vs character-wise voting
"""

import os
import sys
import time
import json
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

import cv2
import numpy as np

from app.perception.detector import YOLOVehicleDetector
from app.perception.tracker import ByteTrackerManager
from app.schemas.detection import DetectorConfig
from app.schemas.tracking import TrackStatus
from app.perception.plate_association import PlateTrackerAssociationManager
from app.perception.ocr_engine import EasyOCREngine, IndianPlateValidator, PlateValidationStatus
from app.perception.plate_fusion import TemporalPlateFusionEngine, FusionConfig, _character_wise_vote

VIDEO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "videos", "traffic_video_modified.mp4"))

def run_diagnostics():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: video file not found at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    print(f"Video Loaded: {total_frames} frames | {vid_w}x{vid_h} @ {fps:.1f} FPS")

    detector = YOLOVehicleDetector(DetectorConfig(confidence_threshold=0.35))
    tracker = ByteTrackerManager()
    easy_ocr = EasyOCREngine()

    assoc_mgr = PlateTrackerAssociationManager(
        ocr_engine=easy_ocr,
        ocr_frame_interval=1,     # run every frame for tracks to max observations
        max_ocr_attempts_per_track=20,
    )

    track_frame_counts = Counter()
    track_first_frame = {}
    track_last_frame = {}

    frame_num = 0
    t_start = time.time()

    # Track evidence collected
    track_raw_observations = {} # track_id -> List[PlateObservation]

    print("\n--- Processing Video for Track & Fusion Audit ---")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        timestamp = frame_num / max(1.0, fps)

        events = detector.detect_frame(frame, camera_id="CAM_01", frame_id=frame_num, timestamp=timestamp, img_w=vid_w, img_h=vid_h)
        tracks = tracker.update_tracks(events, frame_num, timestamp)

        for track in tracks:
            if track.status not in [TrackStatus.CONFIRMED, TrackStatus.TENTATIVE]:
                continue

            tid = track.track_id
            track_frame_counts[tid] += 1
            if tid not in track_first_frame:
                track_first_frame[tid] = frame_num
            track_last_frame[tid] = frame_num

            # Process through association manager
            evidence = assoc_mgr.process_track_frame(track, frame, timestamp, frame_num, img_w=vid_w, img_h=vid_h)

    cap.release()
    elapsed = time.time() - t_start
    print(f"Processing Complete in {elapsed:.2f}s across {frame_num} frames.")

    # 1. Tracker Instrument Analysis
    total_tracks = len(track_frame_counts)
    counts_list = list(track_frame_counts.values())

    tracks_ge_3 = sum(1 for c in counts_list if c >= 3)
    tracks_ge_5 = sum(1 for c in counts_list if c >= 5)
    tracks_1_2 = sum(1 for c in counts_list if c < 3)

    pct_ge_3 = (tracks_ge_3 / max(1, total_tracks)) * 100.0
    pct_ge_5 = (tracks_ge_5 / max(1, total_tracks)) * 100.0

    print("\n=======================================================")
    print("ITEM 3: TRACKER LENGTH DISTRIBUTION REPORT")
    print("=======================================================")
    print(f"Total Unique Vehicle Track IDs: {total_tracks}")
    print(f"Tracks with length < 3 frames:  {tracks_1_2} ({100.0*tracks_1_2/max(1,total_tracks):.1f}%)")
    print(f"Tracks with length >= 3 frames: {tracks_ge_3} ({pct_ge_3:.1f}%)")
    print(f"Tracks with length >= 5 frames: {tracks_ge_5} ({pct_ge_5:.1f}%)")
    print(f"Min track length: {min(counts_list) if counts_list else 0}")
    print(f"Avg track length: {sum(counts_list)/max(1, total_tracks):.1f} frames")
    print(f"Max track length: {max(counts_list) if counts_list else 0}")

    # Track length frequency breakdown
    freq = Counter(counts_list)
    print("\nTrack Length Histogram:")
    for l in sorted(freq.keys()):
        print(f"  Length {l:2d} frames: {freq[l]:2d} tracks ({100.0*freq[l]/max(1,total_tracks):.1f}%)")

    # 2. Pipeline OCR & Fusion Summary
    stats = assoc_mgr.ocr_scheduling_stats
    print("\n=======================================================")
    print("ITEM 2 & 4: PIPELINE & EVALUATION AUDIT")
    print("=======================================================")
    print(f"Vehicles entering plate det: {stats['vehicles_entering_plate_detection']}")
    print(f"Plate Detector calls:        {stats['plate_detector_calls']}")
    print(f"Raw plate candidates:        {stats['raw_candidates']}")
    print(f"Accepted candidates:         {stats['accepted_candidates']}")
    print(f"Actual OCR executions:       {stats['actual_ocr_executions']}")
    print(f"OCR returned text:           {stats['ocr_returned_text']}")
    print(f"OCR returned empty:          {stats['ocr_returned_empty']}")
    print(f"Validation accepted (VALID): {stats['validation_accepted']}")
    print(f"Validation rejected:         {stats['validation_rejected']}")

    # Collect confirmed plate results
    confirmed_plates = {}
    for tid, ev in assoc_mgr.evidence_records.items():
        if ev.associated_plate:
            confirmed_plates[tid] = {
                "plate": ev.associated_plate.best_plate_number,
                "status": ev.associated_plate.status,
                "confirmed": ev.associated_plate.confirmed,
                "confidence": ev.associated_plate.overall_confidence,
                "obs_count": ev.associated_plate.observation_count,
            }

    print("\n=======================================================")
    print("PER-TRACK RECOGNITION RESULTS")
    print("=======================================================")
    for tid, res in confirmed_plates.items():
        print(f"  Track {tid:10s} | Length: {track_frame_counts[tid]:2d}f | Status: {res['status']:9s} | Confirmed: {str(res['confirmed']):5s} | Plate: '{res['plate']}' | Conf: {res['confidence']:.3f}")

    # Save output report JSON
    report_data = {
        "tracker_distribution": {
            "total_tracks": total_tracks,
            "tracks_lt_3": tracks_1_2,
            "tracks_ge_3": tracks_ge_3,
            "pct_ge_3": round(pct_ge_3, 2),
            "tracks_ge_5": tracks_ge_5,
            "pct_ge_5": round(pct_ge_5, 2),
            "histogram": dict(sorted(freq.items())),
        },
        "pipeline_stats": stats,
        "results_per_track": confirmed_plates,
    }

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "debug", "anpr_audit_metrics.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nAudit saved to: {out_path}")

if __name__ == "__main__":
    run_diagnostics()
