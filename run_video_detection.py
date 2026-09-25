"""
ChronoEye Infinity - Stage 1 to Stage 4 Standalone CLI Entry Point
Runs Real Video / Live Camera -> YOLO Vehicle Detection -> ByteTrack Tracking -> ANPR License Plate Recognition Pipeline.

Usage Examples:
  python run_video_detection.py --source data/videos/traffic.mp4 --anpr --show
  python run_video_detection.py --source 0 --anpr --show
  python run_video_detection.py --source "rtsp://admin:pass@192.168.1.100:554/stream1" --anpr
  python run_video_detection.py --source data/videos/traffic.mp4 --device cpu --conf 0.35 --save output_stage14.mp4
"""

import sys
import os
import time
import argparse
from typing import Optional, Any, Dict, Set

# Ensure backend directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from app.perception.frame_source import (
    VideoFrameSource,
    CameraFrameSource,
    SyntheticFrameSource,
    CameraStreamStatus,
    SourceType,
)
from app.schemas.detection import DetectorConfig
from app.perception.detector import YOLOVehicleDetector
from app.schemas.tracking import TrackerConfig
from app.perception.bytetrack import ByteTracker
from app.perception.annotator import FrameAnnotator
from app.perception.plate_association import PlateTrackerAssociationManager
from app.perception.journey import JourneyReconstructionEngine
from app.perception.reid_engine import ReIDMatchingEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="ChronoEye Infinity - Stage 1-5 Real Video / CCTV Detection, Tracking, ANPR & Re-ID Pipeline"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input video source: file path (.mp4/.avi/.mov/.mkv), webcam index ('0'), or RTSP URL ('rtsp://...')",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="CAM_A_EAST",
        help="Camera sensor identifier (default: CAM_A_EAST)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=5,
        help="Pipeline execution stage (1, 2, 3, 4, or 5) (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for YOLO vehicle detection (default: 0.35)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path or name of YOLO model weights (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Number of frames to skip between processed frames (0 = process every frame)",
    )
    parser.add_argument(
        "--ocr-interval",
        type=int,
        default=5,
        help="Frame interval for running ANPR/OCR on active vehicle tracks (default: 5)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process (0 = process entire video/stream)",
    )
    parser.add_argument(
        "--anpr",
        action="store_true",
        default=True,
        help="Enable Stage 4 License Plate Recognition / ANPR (default: enabled)",
    )
    parser.add_argument(
        "--no-anpr",
        action="store_false",
        dest="anpr",
        help="Disable Stage 4 ANPR plate recognition",
    )
    parser.add_argument(
        "--reid",
        action="store_true",
        default=True,
        help="Enable Stage 5 Cross-Camera Re-ID & Journey Reconstruction (default: enabled)",
    )
    parser.add_argument(
        "--no-reid",
        action="store_false",
        dest="reid",
        help="Disable Stage 5 Cross-Camera Re-ID",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live video window with bounding boxes, track IDs, plates, persistent vehicle IDs, FPS, and latency",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="outputs/chronoeye_annotated.mp4",
        help="Path to save annotated output video file (.mp4) (default: outputs/chronoeye_annotated.mp4)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving annotated output video",
    )
    parser.add_argument(
        "--target-plate",
        type=str,
        default="",
        help="Optional: evaluate whether a specific plate (e.g. SX8525) crossed this camera. "
             "Produces DETECTED / NOT_DETECTED / UNCERTAIN result after pipeline completes.",
    )
    return parser.parse_args()


def run_pipeline(args):
    print("=" * 70)
    print("  CHRONOEYE INFINITY - STAGE 1-4 PIPELINE RUNNER")
    print("  Real Video / CCTV -> YOLO Detection -> ByteTrack -> ANPR Plate Recognition")
    print("=" * 70)

    # 1. Determine Source Type & Initialize Input Stream
    if getattr(args, "no_save", False):
        args.save = ""

    src_input = args.source
    camera_id = args.camera_id
    frame_source: Optional[Any] = None

    if src_input.isdigit():
        device_idx = int(src_input)
        print(f"[Stage 1] Initializing Webcam (Device {device_idx}) ...")
        frame_source = CameraFrameSource(
            stream_url_or_device=device_idx,
            camera_id=camera_id,
            frame_skip=args.frame_skip,
        )
    elif src_input.startswith("rtsp://") or src_input.startswith("http://"):
        print(f"[Stage 1] Connecting to RTSP Stream ({src_input}) ...")
        frame_source = CameraFrameSource(
            stream_url_or_device=src_input,
            camera_id=camera_id,
            frame_skip=args.frame_skip,
        )
    else:
        print(f"[Stage 1] Loading Video File: {src_input} ...")
        if not os.path.exists(src_input):
            print(f"[ERROR] Video file not found: {src_input}")
            print("[INFO] Fallback: Searching in data/videos/ ...")
            alt_path = os.path.join("data", "videos", os.path.basename(src_input))
            if os.path.exists(alt_path):
                src_input = alt_path
                print(f"[INFO] Found video at: {src_input}")
            else:
                print(f"[ERROR] Could not locate video at {src_input} or {alt_path}")
                sys.exit(1)

        frame_source = VideoFrameSource(
            video_path=src_input,
            camera_id=camera_id,
            frame_skip=args.frame_skip,
        )

    if frame_source.status == CameraStreamStatus.ERROR:
        err_msg = getattr(frame_source, "error_message", "Unable to open video source")
        print(f"[ERROR] Stream initialization failed: {err_msg}")
        sys.exit(1)

    print(f"[Stage 1 Success] Stream active. Resolution: {frame_source.width}x{frame_source.height} @ {frame_source.original_fps:.1f} FPS")

    # 2. Initialize Stage 2 YOLO Vehicle Detector
    print(f"[Stage 2] Initializing YOLO Vehicle Detector (model={args.model}, device={args.device}, conf={args.conf}) ...")
    detector_config = DetectorConfig(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.conf,
        allowed_classes=["car", "bus", "truck", "motorcycle", "ambulance"],
    )
    detector = YOLOVehicleDetector(config=detector_config)

    # 3. Initialize Stage 3 ByteTracker & Visual Annotator
    print(f"[Stage 3] Initializing ByteTrack Multi-Object Tracker ...")
    tracker = ByteTracker(camera_id=camera_id, config=TrackerConfig())
    annotator = FrameAnnotator()

    # 4. Initialize Stage 4 ANPR / OCR Engine
    alpr_manager: Optional[PlateTrackerAssociationManager] = None
    if args.anpr:
        print(f"[Stage 4] Initializing License Plate Recognition (ANPR) Engine ...")
        alpr_manager = PlateTrackerAssociationManager(ocr_frame_interval=args.ocr_interval)
        print(f"[Stage 4 Success] ANPR engine ready (ocr_interval={args.ocr_interval}). OCR Engine: {alpr_manager.ocr_engine.__class__.__name__}")
    else:
        print(f"[Stage 4] ANPR disabled via --no-anpr flag.")

    # 5. Initialize Stage 5 Cross-Camera Re-ID & Journey Reconstruction Engine
    journey_engine: Optional[JourneyReconstructionEngine] = None
    if args.reid:
        print(f"[Stage 5] Initializing Cross-Camera Re-ID & Journey Reconstruction Engine ...")
        journey_engine = JourneyReconstructionEngine()
        print(f"[Stage 5 Success] Re-ID Journey Engine ready.")
    else:
        print(f"[Stage 5] Cross-Camera Re-ID disabled via --no-reid flag.")

    # Benchmark Tracker Initialization
    from app.core.benchmark import PipelineBenchmarkTracker
    benchmark = PipelineBenchmarkTracker(
        source_fps=frame_source.original_fps,
        frame_skip=args.frame_skip,
        device=detector.config.device,
        source=src_input,
        resolution=f"{frame_source.width}x{frame_source.height}",
    )
    benchmark.start_pipeline()

    # Video Writer setup if --save specified
    video_writer = None
    if args.save:
        try:
            import cv2
            out_dir = os.path.dirname(os.path.abspath(args.save))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = float(frame_source.original_fps) if getattr(frame_source, "original_fps", 0) > 0 else 25.0
            video_writer = cv2.VideoWriter(
                args.save, fourcc, fps, (int(frame_source.width), int(frame_source.height))
            )
            if video_writer.isOpened():
                print(f"[INFO] Video Writer initialized: Output will be saved to {args.save} ({frame_source.width}x{frame_source.height} @ {fps:.1f} FPS)")
            else:
                print(f"[WARNING] VideoWriter could not open file: {args.save}")
                video_writer = None
        except Exception as e:
            print(f"[WARNING] Could not initialize VideoWriter: {e}")
            video_writer = None

    # Processing Loop
    print("\n[Processing] Starting Stage 1-5 Pipeline Execution. Press 'q' or Ctrl+C to stop.\n")
    frames_read = 0
    frames_processed = 0
    total_detections_count = 0
    unique_track_ids = set()

    # Stage 4 ANPR statistics
    total_ocr_attempts = 0
    successful_plate_recognitions: Set[Tuple[str, str]] = set()  # (track_id, plate)
    pending_plate_recognitions: Set[Tuple[str, str]] = set()     # (track_id, plate)
    unknown_plate_count = 0
    total_ocr_time_seconds = 0.0

    # Stage 5 Re-ID statistics
    journey_map: Dict[str, Any] = {}
    total_reid_time_seconds = 0.0

    t_start_pipeline = time.time()

    try:
        import cv2
        cv2_available = True
    except ImportError:
        cv2_available = False

    try:
        while True:
            t0_frame = time.perf_counter()
            benchmark.start_timer("ingestion")
            success, frame, meta = frame_source.read_frame()

            if not success:
                benchmark.active_timers.pop("ingestion", None)
                if frame_source.status == CameraStreamStatus.OFFLINE:
                    print("\n[Stage 1] End of video stream reached.")
                    break
                if hasattr(frame_source, "frame_id") and frame_source.frame_id > frames_read:
                    skipped_delta = frame_source.frame_id - frames_read
                    frames_read += skipped_delta
                    for _ in range(skipped_delta):
                        benchmark.record_frame(read=True, processed=False)
                continue

            benchmark.stop_timer("ingestion")
            frames_read += 1
            frames_processed += 1
            benchmark.record_frame(read=True, processed=True)

            # Stage 2: YOLO Detection
            benchmark.start_timer("yolo")
            detections = detector.detect_frame(
                frame=frame,
                camera_id=camera_id,
                frame_id=meta.frame_id,
                timestamp=meta.timestamp,
                img_w=meta.width,
                img_h=meta.height,
            )
            det_latency_ms = benchmark.stop_timer("yolo")
            benchmark.record_detections(len(detections))
            total_detections_count += len(detections)

            # Stage 3: ByteTrack Vehicle Tracking
            benchmark.start_timer("bytetrack")
            active_tracks = tracker.update(detections=detections, timestamp=meta.timestamp)
            track_latency_ms = benchmark.stop_timer("bytetrack")

            for trk in active_tracks:
                unique_track_ids.add(trk.track_id)
                benchmark.record_track(trk.track_id)

            # Stage 4: ANPR / License Plate Recognition
            evidence_map: Dict[str, Any] = {}
            ocr_latency_ms = 0.0

            if args.anpr and alpr_manager is not None:
                for track in active_tracks:
                    prev_ocr_count = alpr_manager.track_ocr_stats.get(track.track_id, {}).get("attempts", 0)

                    benchmark.start_timer("plate_detection")
                    # Note: plate_detection timer is instrumented inside process_track_frame
                    benchmark.stop_timer("plate_detection")

                    benchmark.start_timer("ocr")
                    evidence = alpr_manager.process_track_frame(
                        track=track,
                        frame=frame,
                        timestamp=meta.timestamp,
                        frame_id=meta.frame_id,
                        img_w=meta.width,
                        img_h=meta.height,
                        all_tracks=active_tracks,
                    )
                    ocr_latency_ms += benchmark.stop_timer("ocr")

                    evidence_map[track.track_id] = evidence
                    new_ocr_count = alpr_manager.track_ocr_stats.get(track.track_id, {}).get("attempts", 0)

                    # If an OCR execution actually took place on this track
                    if new_ocr_count > prev_ocr_count:
                        total_ocr_attempts += 1
                        is_confirmed = False
                        is_pending = False
                        is_unk = True
                        if evidence.associated_plate:
                            plate_status = getattr(evidence.associated_plate, "status", "UNKNOWN")
                            if plate_status == "CONFIRMED" or evidence.associated_plate.confirmed:
                                is_confirmed = True
                                is_unk = False
                                if evidence.associated_plate.best_plate_number:
                                    successful_plate_recognitions.add(
                                        (track.track_id, evidence.associated_plate.best_plate_number)
                                    )
                            elif plate_status == "PENDING":
                                is_pending = True
                                is_unk = False
                                p_num = evidence.associated_plate.pending_plate_number or evidence.associated_plate.best_plate_number
                                if p_num:
                                    pending_plate_recognitions.add((track.track_id, p_num))

                        if is_unk:
                            unknown_plate_count += 1

                        benchmark.record_ocr_attempt(confirmed=is_confirmed, is_pending=is_pending, is_unknown=is_unk)

            # Stage 5: Cross-Camera Re-ID & Journey Reconstruction
            reid_latency_ms = 0.0
            if args.reid and journey_engine is not None:
                benchmark.start_timer("reid")
                for track in active_tracks:
                    ev = evidence_map.get(track.track_id)
                    if not ev:
                        from app.schemas.plate import VehicleIdentityEvidence
                        ev = VehicleIdentityEvidence(
                            track_id=track.track_id,
                            camera_id=camera_id,
                            vehicle_type=track.vehicle_type,
                            associated_plate=None,
                            last_updated_timestamp=meta.timestamp,
                        )
                    journey = journey_engine.process_track_evidence(evidence=ev, track=track, frame=frame)
                    journey_map[track.track_id] = journey
                    benchmark.record_global_identity(journey.global_vehicle_id)
                    benchmark.record_journey(journey.journey_id)

                reid_latency_ms = benchmark.stop_timer("reid")

            t_elapsed_frame = time.perf_counter() - t0_frame
            total_frame_latency_ms = t_elapsed_frame * 1000.0
            benchmark.record_latency("total", total_frame_latency_ms)

            current_fps = 1.0 / max(0.001, t_elapsed_frame)
            total_pipeline_latency_ms = det_latency_ms + track_latency_ms + ocr_latency_ms + reid_latency_ms

            # Render visual overlays
            annotated_frame = annotator.annotate_tracks(
                frame=frame,
                tracks=active_tracks,
                camera_id=camera_id,
                frame_id=meta.frame_id,
                evidence_map=evidence_map,
                journey_map=journey_map,
                fps=current_fps,
                vehicles_count=len(detections),
            )

            # Overlay FPS and Latency metrics onto frame header
            if cv2_available and isinstance(annotated_frame, cv2.Mat if hasattr(cv2, 'Mat') else type(frame)):
                stats_str = f"FPS: {current_fps:.1f} | Latency: {total_pipeline_latency_ms:.1f} ms | Tracks: {len(active_tracks)}"
                if args.anpr:
                    stats_str += f" | OCR: {len(successful_plate_recognitions)}"
                if args.reid and journey_engine:
                    stats_str += f" | Journeys: {len(journey_engine.journeys)}"
                cv2.putText(
                    annotated_frame,
                    stats_str,
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Display window if --show specified
            if args.show and cv2_available:
                cv2.imshow(f"ChronoEye Infinity - Stage 1-5 ({camera_id})", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or Esc
                    print("\n[User Request] User requested termination via keyboard ('q').")
                    break

            if video_writer is not None and cv2_available and annotated_frame is not None:
                video_writer.write(annotated_frame)

            # Progress log every 30 frames
            if frames_processed % 30 == 0 or frames_processed == 1:
                # Plates: count CONFIRMED + PENDING (both represent real OCR evidence)
                confirmed_count = len(successful_plate_recognitions)
                pending_count = len(pending_plate_recognitions)
                total_plate_evidence = confirmed_count + pending_count
                anpr_info = f"Plates: {confirmed_count:02d}(C) {pending_count:02d}(P)" if args.anpr else "ANPR: OFF"
                reid_info = f"Vehicles: {len(journey_engine.journeys):02d}" if (args.reid and journey_engine) else "Re-ID: OFF"
                print(
                    f"Frame {meta.frame_id:04d} | Detections: {len(detections):02d} | "
                    f"Tracks: {len(active_tracks):02d} | {anpr_info} | {reid_info} | FPS: {current_fps:04.1f} | "
                    f"Pipeline Latency: {total_pipeline_latency_ms:04.1f} ms"
                )

            if args.max_frames > 0 and frames_processed >= args.max_frames:
                print(f"\n[Info] Reached maximum frame count ({args.max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\n[User Request] Pipeline interrupted by user.")
    finally:
        benchmark.stop_pipeline()
        # Cleanup Resources
        frame_source.release()
        if video_writer is not None:
            video_writer.release()
        if cv2_available and args.show:
            import cv2
            cv2.destroyAllWindows()

    # Print Component 7 Formatted Benchmark Report
    benchmark.print_formatted_report()

    if args.anpr and alpr_manager is not None:
        stats = alpr_manager.ocr_scheduling_stats
        print("\n" + "=" * 70)
        print("  ANPR PIPELINE DIAGNOSTICS")
        print("=" * 70)
        print(f"  Vehicles entering plate detection: {stats.get('vehicles_entering_plate_detection', 0)}")
        print(f"  Plate detector calls:              {stats.get('plate_detector_calls', 0)}")
        print(f"  Raw plate candidates:              {stats.get('raw_candidates', 0)}")
        print(f"  Accepted candidates:               {stats.get('accepted_candidates', 0)}")
        print(f"  Rejected candidates:               {stats.get('rejected_candidates', 0)}")
        print(f"  Plate crops created:               {stats.get('plate_crops_created', 0)}")
        print(f"  OCR calls:                         {stats.get('ocr_calls', 0)}")
        print(f"  Valid OCR observations:            {stats.get('valid_ocr_observations', 0)}")
        print(f"  Validation rejections:             {stats.get('validation_rejections', 0)}")
        print(f"  Fusion rejections:                 {stats.get('fusion_rejections', 0)}")
        print(f"  Confirmed plates:                  {len(successful_plate_recognitions)}")
        print("=" * 70)
        print("\n" + "=" * 70)
        print("  OCR SCHEDULING STATISTICS")
        print("=" * 70)
        print(f"  Cache Skips:            {stats.get('cache_skips', 0)}")
        print(f"  Interval Skips:         {stats.get('interval_skips', 0)}")
        print(f"  ROI Quality Skips:      {stats.get('roi_quality_skips', 0)}")
        print(f"  Confirmed Plate Skips:  {stats.get('confirmed_plate_skips', 0)}")
        print(f"  Unchanged ROI Skips:    {stats.get('unchanged_roi_skips', 0)}")
        print(f"  Max Attempts Skips:     {stats.get('max_attempts_skips', 0)}")
        print(f"  ROI Too Small Skips:    {stats.get('roi_too_small_skips', 0)}")
        print(f"  Actual OCR Executions:  {stats.get('actual_ocr_executions', 0)}")
        print("=" * 70)
        print("[FP_STATS]", alpr_manager.plate_detector.fp_rejection_stats)

    # -------------------------------------------------------------------
    # Target Plate Evaluation (--target-plate flag)
    # -------------------------------------------------------------------
    if args.target_plate and args.anpr and alpr_manager is not None:
        from app.perception.target_plate_evaluator import TargetPlateEvaluator
        evaluator = TargetPlateEvaluator()
        result = evaluator.evaluate(args.target_plate, alpr_manager.fusion_engine)
        evaluator.print_report(result)
        # Machine-readable JSON summary
        import json
        print("[TARGET_PLATE_JSON]", json.dumps(result.to_dict()))

    if args.save and os.path.exists(args.save) and os.path.getsize(args.save) > 0:
        print(f"\n[OUTPUT VIDEO] Saved: {args.save}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)

