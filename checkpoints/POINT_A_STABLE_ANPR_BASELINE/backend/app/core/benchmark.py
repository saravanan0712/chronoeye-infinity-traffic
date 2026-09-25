"""
ChronoEye Infinity - Stage 1 to 5 Pipeline Benchmark Engine
Provides high-precision monotonic runtime latency tracking, throughput measurement,
and execution statistics for real-video pipeline verification.
"""

import time
from typing import Dict, List, Set, Optional, Any


class PipelineBenchmarkTracker:
    """
    High-precision real-time performance benchmark tracker for ChronoEye Infinity Stage 1-5.
    Measures 16 core pipeline indicators using monotonic performance counters.
    """

    def __init__(
        self,
        source_fps: float = 30.0,
        frame_skip: int = 0,
        device: str = "cpu",
        source: str = "unknown",
        resolution: str = "1920x1080",
    ):
        self.source_fps = source_fps
        self.frame_skip = frame_skip
        self.device = device
        self.source = source
        self.resolution = resolution

        # Monotonic time anchors
        self.t_start_pipeline: float = 0.0
        self.t_end_pipeline: float = 0.0
        self.active_timers: Dict[str, float] = {}

        # 16 Core Measured Indicators
        self.ingestion_latencies_ms: List[float] = []
        self.yolo_latencies_ms: List[float] = []
        self.bytetrack_latencies_ms: List[float] = []
        self.plate_detection_latencies_ms: List[float] = []
        self.ocr_latencies_ms: List[float] = []
        self.reid_latencies_ms: List[float] = []
        self.total_pipeline_latencies_ms: List[float] = []

        self.frames_read: int = 0
        self.frames_processed: int = 0
        self.total_vehicles_detected: int = 0
        self.unique_track_ids: Set[str] = set()

        self.total_ocr_attempts: int = 0
        self.confirmed_plate_observations: int = 0
        self.pending_plate_observations: int = 0
        self.unknown_plate_observations: int = 0

        self.global_vehicle_identities: Set[str] = set()
        self.reconstructed_journeys: Set[str] = set()

    def start_pipeline(self):
        """Starts total pipeline timer."""
        self.t_start_pipeline = time.perf_counter()

    def stop_pipeline(self):
        """Stops total pipeline timer."""
        self.t_end_pipeline = time.perf_counter()

    def start_timer(self, stage_name: str):
        """Starts high-precision timer for a specific stage."""
        self.active_timers[stage_name] = time.perf_counter()

    def stop_timer(self, stage_name: str) -> float:
        """Stops timer for a stage and returns elapsed latency in milliseconds."""
        t_start = self.active_timers.pop(stage_name, None)
        if t_start is None:
            return 0.0
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        self.record_latency(stage_name, latency_ms)
        return latency_ms

    def record_latency(self, stage_name: str, latency_ms: float):
        """Records a measured latency value for a stage."""
        if stage_name == "ingestion":
            self.ingestion_latencies_ms.append(latency_ms)
        elif stage_name in ("yolo", "detection"):
            self.yolo_latencies_ms.append(latency_ms)
        elif stage_name in ("bytetrack", "tracking"):
            self.bytetrack_latencies_ms.append(latency_ms)
        elif stage_name == "plate_detection":
            self.plate_detection_latencies_ms.append(latency_ms)
        elif stage_name == "ocr":
            self.ocr_latencies_ms.append(latency_ms)
        elif stage_name in ("reid", "journey"):
            self.reid_latencies_ms.append(latency_ms)
        elif stage_name in ("total", "pipeline"):
            self.total_pipeline_latencies_ms.append(latency_ms)

    def record_frame(self, read: bool = True, processed: bool = True):
        """Increments frame counts."""
        if read:
            self.frames_read += 1
        if processed:
            self.frames_processed += 1

    def record_detections(self, count: int):
        """Records vehicle detection count."""
        self.total_vehicles_detected += count

    def record_track(self, track_id: str):
        """Records a unique local track ID."""
        self.unique_track_ids.add(track_id)

    def record_ocr_attempt(self, confirmed: bool = False, is_pending: bool = False, is_unknown: bool = False):
        """Records an OCR attempt outcome."""
        self.total_ocr_attempts += 1
        if confirmed:
            self.confirmed_plate_observations += 1
        elif is_pending:
            self.pending_plate_observations += 1
        elif is_unknown:
            self.unknown_plate_observations += 1

    def record_global_identity(self, global_id: str):
        """Records a unique global vehicle identity."""
        self.global_vehicle_identities.add(global_id)

    def record_journey(self, journey_id: str):
        """Records a unique reconstructed journey ID."""
        self.reconstructed_journeys.add(journey_id)

    def _avg(self, values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def get_summary_report(self) -> Dict[str, Any]:
        """Calculates and returns complete benchmark metrics dictionary."""
        t_now = time.perf_counter()
        t_end = self.t_end_pipeline if self.t_end_pipeline > 0 else t_now
        total_runtime = max(0.001, t_end - self.t_start_pipeline) if self.t_start_pipeline > 0 else 0.0

        effective_fps = self.frames_processed / total_runtime if total_runtime > 0 else 0.0
        processing_fps = effective_fps

        avg_ingest = self._avg(self.ingestion_latencies_ms)
        avg_yolo = self._avg(self.yolo_latencies_ms)
        avg_bytetrack = self._avg(self.bytetrack_latencies_ms)
        avg_plate_det = self._avg(self.plate_detection_latencies_ms)
        avg_ocr = self._avg(self.ocr_latencies_ms)
        avg_reid = self._avg(self.reid_latencies_ms)
        avg_total = self._avg(self.total_pipeline_latencies_ms)

        max_total = max(self.total_pipeline_latencies_ms) if self.total_pipeline_latencies_ms else 0.0

        return {
            # Source & Config
            "source": self.source,
            "resolution": self.resolution,
            "source_fps": round(self.source_fps, 2),
            "frame_skip": self.frame_skip,
            "device": self.device,
            # Execution
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "total_runtime_seconds": round(total_runtime, 3),
            "effective_fps": round(effective_fps, 2),
            "processing_fps": round(processing_fps, 2),
            # Latencies (ms)
            "ingestion_latency_ms": round(avg_ingest, 2),
            "yolo_latency_ms": round(avg_yolo, 2),
            "bytetrack_latency_ms": round(avg_bytetrack, 2),
            "plate_detection_latency_ms": round(avg_plate_det, 2),
            "ocr_latency_ms": round(avg_ocr, 2),
            "reid_latency_ms": round(avg_reid, 2),
            "total_pipeline_latency_ms": round(avg_total, 2),
            "max_pipeline_latency_ms": round(max_total, 2),
            # Counts
            "total_vehicles_detected": self.total_vehicles_detected,
            "unique_local_tracks": len(self.unique_track_ids),
            "ocr_attempts": self.total_ocr_attempts,
            "confirmed_plates": self.confirmed_plate_observations,
            "pending_plates": self.pending_plate_observations,
            "unknown_plate_observations": self.unknown_plate_observations,
            "global_vehicle_identities": len(self.global_vehicle_identities),
            "reconstructed_journeys": len(self.reconstructed_journeys),
        }

    def print_formatted_report(self):
        """Prints standardized Component 7 benchmark output report."""
        rep = self.get_summary_report()

        print("\n" + "=" * 60)
        print("CHRONOEYE INFINITY — STAGE 1–5 REAL VIDEO BENCHMARK")
        print("=" * 60)
        print("\nInput:")
        print(f"Source:                     {rep['source']}")
        print(f"Resolution:                 {rep['resolution']}")
        print(f"Source FPS:                 {rep['source_fps']:.1f} FPS")
        print("\nExecution:")
        print(f"Device:                     {rep['device']}")
        print(f"Frames read:                {rep['frames_read']}")
        print(f"Frames processed:           {rep['frames_processed']}")
        print(f"Frame skip:                 {rep['frame_skip']}")
        print(f"Total runtime:              {rep['total_runtime_seconds']:.2f} seconds")
        print("\nPerformance:")
        print(f"Ingestion latency:          {rep['ingestion_latency_ms']:.2f} ms")
        print(f"YOLO latency:               {rep['yolo_latency_ms']:.2f} ms")
        print(f"ByteTrack latency:          {rep['bytetrack_latency_ms']:.2f} ms")
        print(f"Plate detection latency:    {rep['plate_detection_latency_ms']:.2f} ms")
        print(f"OCR latency:                {rep['ocr_latency_ms']:.2f} ms")
        print(f"Re-ID latency:              {rep['reid_latency_ms']:.2f} ms")
        print(f"Total pipeline latency:     {rep['total_pipeline_latency_ms']:.2f} ms")
        print(f"Effective FPS:              {rep['effective_fps']:.2f} FPS")
        print("\nDetection:")
        print(f"Total vehicles detected:    {rep['total_vehicles_detected']}")
        print(f"Unique local tracks:        {rep['unique_local_tracks']}")
        print("\nANPR:")
        print(f"OCR attempts:               {rep['ocr_attempts']}")
        print(f"Confirmed plates:           {rep['confirmed_plates']}")
        print(f"Pending plates:             {rep['pending_plates']}")
        print(f"UNKNOWN observations:       {rep['unknown_plate_observations']}")
        print("\nRE-ID:")
        print(f"Global vehicle identities:  {rep['global_vehicle_identities']}")
        print(f"Reconstructed journeys:     {rep['reconstructed_journeys']}")
        print("=" * 60 + "\n")

    def reset(self):
        """Resets all metrics counters and timers."""
        self.__init__(
            source_fps=self.source_fps,
            frame_skip=self.frame_skip,
            device=self.device,
            source=self.source,
            resolution=self.resolution,
        )
