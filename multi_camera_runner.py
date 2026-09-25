"""
ChronoEye Infinity - Multi-Camera Video Runner & Orchestrator (Project B / Stage 5)
Orchestrates multiple camera streams (real video, webcam, RTSP) through independent
Stage 1-4 perception pipelines while feeding ONE shared JourneyReconstructionEngine
for global vehicle identity assignment and cross-camera journey reconstruction.

Usage Examples:
  # Sequential mode across 2 video files
  python multi_camera_runner.py --cameras CAM_A_EAST:data/videos/cam_a.mp4 CAM_B_WEST:data/videos/cam_b.mp4

  # With timestamp offsets (e.g. CAM_B starts 10 seconds after CAM_A)
  python multi_camera_runner.py --cameras CAM_A_EAST:cam_a.mp4:0.0 CAM_B_WEST:cam_b.mp4:10.0

  # Threaded mode with custom model and frame limit
  python multi_camera_runner.py --cameras CAM_A:video1.mp4 CAM_B:video2.mp4 --mode threaded --max-frames 300
"""

import os
import sys
import time
import argparse
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any

# Ensure backend directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from app.perception.frame_source import (
    VideoFrameSource,
    CameraFrameSource,
    CameraStreamStatus,
    FrameMetadata,
)
from app.schemas.detection import DetectorConfig, BoundingBoxXYXY
from app.perception.detector import YOLOVehicleDetector
from app.schemas.tracking import TrackerConfig, TrackState
from app.perception.bytetrack import ByteTracker
from app.perception.plate_association import PlateTrackerAssociationManager
from app.schemas.plate import VehicleIdentityEvidence
from app.schemas.reid import ReIDConfig, VehicleJourney, MatchDecision
from app.perception.camera_topology import CityCameraTopology
from app.perception.reid_engine import ReIDMatchingEngine
from app.perception.journey import JourneyReconstructionEngine


@dataclass
class CameraSourceConfig:
    """Configuration for a single camera source."""
    camera_id: str
    source: str
    frame_skip: int = 0
    max_frames: int = 0
    timestamp_offset_seconds: float = 0.0


@dataclass
class MultiCameraRunnerConfig:
    """Configuration for the multi-camera runner."""
    cameras: List[CameraSourceConfig] = field(default_factory=list)
    model_path: str = "yolov8n.pt"
    device: str = "cpu"
    conf_threshold: float = 0.35
    anpr_enabled: bool = True
    ocr_frame_interval: int = 5
    reid_enabled: bool = True
    processing_mode: str = "sequential"  # "sequential" | "threaded"
    journey_timeout_seconds: float = 60.0
    verbose: bool = True


@dataclass
class CameraStats:
    """Per-camera processing statistics."""
    camera_id: str
    frames_read: int = 0
    frames_processed: int = 0
    total_detections: int = 0
    unique_tracks: Set[str] = field(default_factory=set)
    ocr_attempts: int = 0
    confirmed_plates: Set[Tuple[str, str]] = field(default_factory=set)  # (track_id, plate)
    pending_plates: Set[Tuple[str, str]] = field(default_factory=set)    # (track_id, plate)
    total_processing_time: float = 0.0


@dataclass
class MultiCameraRunResult:
    """Aggregated result from a multi-camera pipeline run."""
    per_camera_stats: Dict[str, CameraStats] = field(default_factory=dict)
    journey_summary: List[Dict[str, Any]] = field(default_factory=list)
    global_vehicle_ids: List[str] = field(default_factory=list)
    total_runtime_seconds: float = 0.0
    journey_engine: Optional[JourneyReconstructionEngine] = None


class CameraWorker:
    """
    Worker instance managing Stage 1-4 perception for a single camera.
    Owns its own frame source, YOLO detector, ByteTracker, and ALPR manager.
    """

    def __init__(
        self,
        camera_config: CameraSourceConfig,
        runner_config: MultiCameraRunnerConfig,
    ):
        self.camera_config = camera_config
        self.runner_config = runner_config
        self.camera_id = camera_config.camera_id
        self.stats = CameraStats(camera_id=self.camera_id)
        self.frame_source = None
        self.detector = None
        self.tracker = None
        self.alpr_manager = None
        self._is_initialized = False

    def initialize(self):
        """Initializes Stage 1-4 components for this camera."""
        if self._is_initialized:
            return

        src_input = self.camera_config.source
        cam_id = self.camera_config.camera_id

        # 1. Initialize Frame Source
        if src_input.isdigit():
            device_idx = int(src_input)
            self.frame_source = CameraFrameSource(
                stream_url_or_device=device_idx,
                camera_id=cam_id,
                frame_skip=self.camera_config.frame_skip,
            )
        elif src_input.startswith("rtsp://") or src_input.startswith("http://"):
            self.frame_source = CameraFrameSource(
                stream_url_or_device=src_input,
                camera_id=cam_id,
                frame_skip=self.camera_config.frame_skip,
            )
        else:
            if not os.path.exists(src_input):
                alt_path = os.path.join("data", "videos", os.path.basename(src_input))
                if os.path.exists(alt_path):
                    src_input = alt_path
            self.frame_source = VideoFrameSource(
                video_path=src_input,
                camera_id=cam_id,
                frame_skip=self.camera_config.frame_skip,
            )

        # 2. Initialize YOLO Detector
        detector_config = DetectorConfig(
            model_path=self.runner_config.model_path,
            device=self.runner_config.device,
            confidence_threshold=self.runner_config.conf_threshold,
            allowed_classes=["car", "bus", "truck", "motorcycle", "ambulance"],
        )
        self.detector = YOLOVehicleDetector(config=detector_config)

        # 3. Initialize ByteTracker
        self.tracker = ByteTracker(camera_id=cam_id, config=TrackerConfig())

        # 4. Initialize ALPR Manager
        if self.runner_config.anpr_enabled:
            self.alpr_manager = PlateTrackerAssociationManager(
                ocr_frame_interval=self.runner_config.ocr_frame_interval
            )

        self._is_initialized = True

    def read_frame(self) -> Tuple[bool, Any, Optional[FrameMetadata]]:
        """Reads a frame and applies configured timestamp offset."""
        if self.frame_source is None:
            return False, None, None

        success, frame, meta = self.frame_source.read_frame()
        if success and meta is not None:
            self.stats.frames_read += 1
            # Adjust timestamp with camera offset
            if self.camera_config.timestamp_offset_seconds != 0.0:
                meta.timestamp += self.camera_config.timestamp_offset_seconds
        return success, frame, meta

    def process_frame(
        self, frame: Any, meta: FrameMetadata
    ) -> Tuple[List[TrackState], Dict[str, VehicleIdentityEvidence]]:
        """
        Executes Stage 2 (YOLO), Stage 3 (ByteTrack), and Stage 4 (ANPR) on a single frame.
        Returns active tracks and evidence map.
        """
        t0 = time.perf_counter()
        self.stats.frames_processed += 1

        # Stage 2: YOLO Detection
        detections = self.detector.detect_frame(
            frame=frame,
            camera_id=self.camera_id,
            frame_id=meta.frame_id,
            timestamp=meta.timestamp,
            img_w=meta.width,
            img_h=meta.height,
        )
        self.stats.total_detections += len(detections)

        # Stage 3: ByteTracker
        active_tracks = self.tracker.update(detections=detections, timestamp=meta.timestamp)
        for trk in active_tracks:
            self.stats.unique_tracks.add(trk.track_id)

        # Stage 4: ANPR
        evidence_map: Dict[str, VehicleIdentityEvidence] = {}
        if self.runner_config.anpr_enabled and self.alpr_manager is not None:
            for track in active_tracks:
                prev_ocr = self.alpr_manager.track_ocr_stats.get(track.track_id, {}).get("attempts", 0)
                evidence = self.alpr_manager.process_track_frame(
                    track=track,
                    frame=frame,
                    timestamp=meta.timestamp,
                    frame_id=meta.frame_id,
                    img_w=meta.width,
                    img_h=meta.height,
                    all_tracks=active_tracks,
                )
                evidence_map[track.track_id] = evidence
                new_ocr = self.alpr_manager.track_ocr_stats.get(track.track_id, {}).get("attempts", 0)
                if new_ocr > prev_ocr:
                    self.stats.ocr_attempts += 1
                    if evidence.associated_plate:
                        p_status = getattr(evidence.associated_plate, "status", "UNKNOWN")
                        if p_status == "CONFIRMED" or evidence.associated_plate.confirmed:
                            if evidence.associated_plate.best_plate_number:
                                self.stats.confirmed_plates.add(
                                    (track.track_id, evidence.associated_plate.best_plate_number)
                                )
                        elif p_status == "PENDING":
                            p_num = (
                                evidence.associated_plate.pending_plate_number
                                or evidence.associated_plate.best_plate_number
                            )
                            if p_num:
                                self.stats.pending_plates.add((track.track_id, p_num))

        self.stats.total_processing_time += time.perf_counter() - t0
        return active_tracks, evidence_map


class MultiCameraRunner:
    """
    Multi-camera orchestrator that coordinates multiple camera workers
    feeding ONE shared JourneyReconstructionEngine instance.
    """

    def __init__(
        self,
        config: MultiCameraRunnerConfig,
        journey_engine: Optional[JourneyReconstructionEngine] = None,
        matching_engine: Optional[ReIDMatchingEngine] = None,
        topology: Optional[CityCameraTopology] = None,
    ):
        self.config = config
        self.lock = threading.Lock()

        # Shared Journey Reconstruction Engine
        if journey_engine is not None:
            self.journey_engine = journey_engine
        else:
            topo = topology or CityCameraTopology()
            reid_cfg = ReIDConfig(journey_timeout_seconds=config.journey_timeout_seconds)
            matcher = matching_engine or ReIDMatchingEngine(topology=topo, config=reid_cfg)
            self.journey_engine = JourneyReconstructionEngine(
                matching_engine=matcher,
                config=reid_cfg,
                journey_timeout_seconds=config.journey_timeout_seconds,
            )

        self.workers: Dict[str, CameraWorker] = {}
        for cam_cfg in self.config.cameras:
            self.workers[cam_cfg.camera_id] = CameraWorker(
                camera_config=cam_cfg, runner_config=self.config
            )

    def feed_track_evidence(
        self,
        evidence: VehicleIdentityEvidence,
        track: TrackState,
        frame: Any = None,
    ) -> VehicleJourney:
        """
        Thread-safe method to feed evidence into the shared JourneyReconstructionEngine.
        """
        with self.lock:
            return self.journey_engine.process_track_evidence(
                evidence=evidence, track=track, frame=frame
            )

    def process_camera_frame(
        self, worker: CameraWorker, frame: Any, meta: FrameMetadata
    ) -> List[VehicleJourney]:
        """Processes one camera frame and updates the shared journey engine."""
        active_tracks, evidence_map = worker.process_frame(frame, meta)
        journeys = []

        if self.config.reid_enabled:
            for track in active_tracks:
                ev = evidence_map.get(track.track_id)
                if not ev:
                    ev = VehicleIdentityEvidence(
                        track_id=track.track_id,
                        camera_id=worker.camera_id,
                        vehicle_type=track.vehicle_type,
                        associated_plate=None,
                        last_updated_timestamp=meta.timestamp,
                    )
                journey = self.feed_track_evidence(evidence=ev, track=track, frame=frame)
                journeys.append(journey)

        return journeys

    def run_sequential(self) -> MultiCameraRunResult:
        """Runs cameras sequentially frame-by-frame (interleaved round-robin)."""
        if self.config.verbose:
            print("[MultiCameraRunner] Starting sequential round-robin execution...")

        for worker in self.workers.values():
            worker.initialize()

        active_workers = list(self.workers.values())
        frames_per_worker = {w.camera_id: 0 for w in active_workers}
        t0_total = time.time()

        while active_workers:
            for worker in list(active_workers):
                max_f = worker.camera_config.max_frames
                if max_f > 0 and frames_per_worker[worker.camera_id] >= max_f:
                    active_workers.remove(worker)
                    continue

                success, frame, meta = worker.read_frame()
                if not success:
                    if worker.frame_source and worker.frame_source.status == CameraStreamStatus.OFFLINE:
                        active_workers.remove(worker)
                    continue

                frames_per_worker[worker.camera_id] += 1
                self.process_camera_frame(worker, frame, meta)

        total_runtime = time.time() - t0_total
        return self._build_result(total_runtime)

    def _worker_thread_fn(self, worker: CameraWorker):
        """Thread target for processing a single camera source."""
        worker.initialize()
        max_f = worker.camera_config.max_frames
        frames_processed = 0

        while True:
            if max_f > 0 and frames_processed >= max_f:
                break

            success, frame, meta = worker.read_frame()
            if not success:
                if worker.frame_source and worker.frame_source.status == CameraStreamStatus.OFFLINE:
                    break
                time.sleep(0.005)
                continue

            frames_processed += 1
            self.process_camera_frame(worker, frame, meta)

    def run_threaded(self) -> MultiCameraRunResult:
        """Runs cameras concurrently in dedicated threads with thread-safe Re-ID locking."""
        if self.config.verbose:
            print("[MultiCameraRunner] Starting multi-threaded execution...")

        threads = []
        t0_total = time.time()

        for worker in self.workers.values():
            t = threading.Thread(
                target=self._worker_thread_fn,
                args=(worker,),
                name=f"Worker-{worker.camera_id}",
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        total_runtime = time.time() - t0_total
        return self._build_result(total_runtime)

    def run(self) -> MultiCameraRunResult:
        """Main execution entrypoint honoring the configured processing mode."""
        if self.config.processing_mode == "threaded":
            return self.run_threaded()
        return self.run_sequential()

    def _build_result(self, total_runtime: float) -> MultiCameraRunResult:
        """Builds and returns the MultiCameraRunResult."""
        summary = self.journey_engine.get_journey_summary_report()
        global_ids = sorted(list({j["global_vehicle_id"] for j in summary}))

        stats_map = {w.camera_id: w.stats for w in self.workers.values()}

        return MultiCameraRunResult(
            per_camera_stats=stats_map,
            journey_summary=summary,
            global_vehicle_ids=global_ids,
            total_runtime_seconds=total_runtime,
            journey_engine=self.journey_engine,
        )

    def search_plate(self, plate_number: str) -> List[Dict[str, Any]]:
        """Searches reconstructed journeys for a specific plate number."""
        clean_target = plate_number.strip().upper()
        results = []
        for journey in self.journey_engine.journeys.values():
            matched = False
            if journey.plate_number and clean_target in journey.plate_number.upper():
                matched = True
            else:
                for seg in journey.segments:
                    if seg.plate_number and clean_target in seg.plate_number.upper():
                        matched = True
                        break
            if matched:
                results.append({
                    "global_vehicle_id": journey.global_vehicle_id,
                    "journey_id": journey.journey_id,
                    "plate_number": journey.plate_number or "UNKNOWN",
                    "vehicle_type": journey.vehicle_type,
                    "status": journey.status,
                    "cameras": list(journey.cameras),
                    "segments_count": len(journey.segments),
                    "first_seen": journey.first_seen,
                    "last_seen": journey.last_seen,
                })
        return results

    def print_summary_report(self, result: MultiCameraRunResult):
        """Prints a rich formatted console report."""
        print("\n" + "=" * 80)
        print("           CHRONOEYE INFINITY — MULTI-CAMERA JOURNEY RECONSTRUCTION REPORT")
        print("=" * 80)
        print(f"Total Runtime: {result.total_runtime_seconds:.2f}s | Cameras Processed: {len(result.per_camera_stats)}")
        print(f"Total Global Vehicles Identified: {len(result.global_vehicle_ids)}")
        print("-" * 80)

        print("\n[CAMERA SENSOR INGESTION & PERCEPTION SUMMARY]")
        print(f"{'Camera ID':<15} | {'Frames':<8} | {'Detections':<11} | {'Tracks':<8} | {'OCR Attempts':<13} | {'Confirmed Plates'}")
        print("-" * 80)
        for cam_id, stats in result.per_camera_stats.items():
            confirmed_str = ", ".join(sorted({p[1] for p in stats.confirmed_plates})) or "None"
            print(f"{cam_id:<15} | {stats.frames_processed:<8} | {stats.total_detections:<11} | {len(stats.unique_tracks):<8} | {stats.ocr_attempts:<13} | {confirmed_str}")

        print("\n[STAGE 5 RECONSTRUCTED VEHICLE JOURNEYS]")
        if not result.journey_summary:
            print("  No multi-camera vehicle journeys reconstructed.")
        else:
            print(f"{'Global ID':<10} | {'Journey ID':<10} | {'Plate':<12} | {'Type':<8} | {'Status':<10} | {'Segments':<8} | {'Camera Route'}")
            print("-" * 80)
            for j in result.journey_summary:
                route = " -> ".join(j["camera_sequence"])
                print(f"{j['global_vehicle_id']:<10} | {j['journey_id']:<10} | {j['plate_number']:<12} | {j['vehicle_type']:<8} | {j['status']:<10} | {j['number_of_segments']:<8} | {route}")

        print("=" * 80 + "\n")


def parse_camera_arg(cam_str: str) -> CameraSourceConfig:
    """
    Parses a camera argument string into CameraSourceConfig.
    Formats supported:
      CAM_ID:source_path
      CAM_ID:source_path:offset_seconds
    """
    parts = cam_str.split(":")
    if len(parts) == 2:
        return CameraSourceConfig(camera_id=parts[0], source=parts[1])
    elif len(parts) == 3:
        try:
            offset = float(parts[2])
            return CameraSourceConfig(camera_id=parts[0], source=parts[1], timestamp_offset_seconds=offset)
        except ValueError:
            # Maybe Windows drive letter path, e.g. CAM_A:C:/video.mp4
            return CameraSourceConfig(camera_id=parts[0], source=f"{parts[1]}:{parts[2]}")
    elif len(parts) == 4:
        # Windows drive letter with offset: CAM_A:C:/video.mp4:10.5
        try:
            offset = float(parts[3])
            return CameraSourceConfig(camera_id=parts[0], source=f"{parts[1]}:{parts[2]}", timestamp_offset_seconds=offset)
        except ValueError:
            pass

    raise ValueError(f"Invalid camera format: '{cam_str}'. Expected CAM_ID:source or CAM_ID:source:offset")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ChronoEye Infinity — Shared Multi-Camera Journey Reconstruction Runner"
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        required=True,
        help="List of camera specifications in format CAM_ID:source or CAM_ID:source:offset (e.g. CAM_A:cam_a.mp4 CAM_B:cam_b.mp4:5.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path or name of YOLO model weights (default: yolov8n.pt)",
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
        "--mode",
        type=str,
        choices=["sequential", "threaded"],
        default="sequential",
        help="Multi-camera processing mode: 'sequential' (round-robin) or 'threaded' (default: sequential)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames per camera (0 = entire stream)",
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
        help="Frame interval for ANPR/OCR (default: 5)",
    )
    parser.add_argument(
        "--anpr",
        action="store_true",
        default=True,
        help="Enable Stage 4 ANPR License Plate Recognition (default: enabled)",
    )
    parser.add_argument(
        "--no-anpr",
        action="store_false",
        dest="anpr",
        help="Disable Stage 4 ANPR",
    )
    parser.add_argument(
        "--reid",
        action="store_true",
        default=True,
        help="Enable Stage 5 Re-ID Journey Reconstruction (default: enabled)",
    )
    parser.add_argument(
        "--no-reid",
        action="store_false",
        dest="reid",
        help="Disable Stage 5 Re-ID",
    )
    parser.add_argument(
        "--journey-timeout",
        type=float,
        default=60.0,
        help="Journey timeout in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--search-plate",
        type=str,
        default="",
        help="Search for a specific plate number in the reconstructed journeys after processing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cam_configs = []
    for c_str in args.cameras:
        cfg = parse_camera_arg(c_str)
        cfg.frame_skip = args.frame_skip
        cfg.max_frames = args.max_frames
        cam_configs.append(cfg)

    runner_config = MultiCameraRunnerConfig(
        cameras=cam_configs,
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        anpr_enabled=args.anpr,
        ocr_frame_interval=args.ocr_interval,
        reid_enabled=args.reid,
        processing_mode=args.mode,
        journey_timeout_seconds=args.journey_timeout,
        verbose=True,
    )

    runner = MultiCameraRunner(config=runner_config)
    result = runner.run()
    runner.print_summary_report(result)

    if args.search_plate:
        print(f"\n[SEARCH RESULTS FOR PLATE: {args.search_plate}]")
        search_res = runner.search_plate(args.search_plate)
        if search_res:
            for s in search_res:
                print(f"  Found in {s['global_vehicle_id']} ({s['journey_id']}) — Status: {s['status']} — Route: {' -> '.join(s['cameras'])}")
        else:
            print(f"  No journey found containing plate {args.search_plate}.")


if __name__ == "__main__":
    main()
