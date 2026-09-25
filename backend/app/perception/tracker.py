"""
ChronoEye Infinity - Phase 3: Vehicle Tracker Manager
Factory and manager managing camera-isolated tracker instances (ByteTrack / DeepSORT)
across multiple CCTV camera streams.
"""

import time
from typing import Dict, List, Optional, Any
from app.schemas.detection import DetectionEvent
from app.schemas.tracking import (
    TrackState,
    TrackerConfig,
    TrackerBackend,
)
from app.perception.bytetrack import ByteTracker
from app.perception.deepsort import DeepSORTTracker


class VehicleTrackerManager:
    """
    Manages camera-specific tracking pipelines across urban CCTV sensors.
    Maintains camera-local track ID namespaces (e.g. CAM_A_EAST -> TRK_101).
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self.trackers: Dict[str, ByteTracker] = {}
        self.total_frames_tracked = 0
        self.total_detections_tracked = 0
        self.total_tracking_time_seconds = 0.0

    def get_tracker(self, camera_id: str) -> ByteTracker:
        """
        Retrieves or instantiates a camera-isolated tracker instance.
        """
        if camera_id not in self.trackers:
            if self.config.tracker_backend == TrackerBackend.DEEPSORT:
                self.trackers[camera_id] = DeepSORTTracker(camera_id=camera_id, config=self.config)
            else:
                self.trackers[camera_id] = ByteTracker(camera_id=camera_id, config=self.config)
        return self.trackers[camera_id]

    def update(
        self,
        camera_id: str,
        detections: List[DetectionEvent],
        timestamp: float,
    ) -> List[TrackState]:
        """
        Updates camera-specific tracker with new DetectionEvent list.
        """
        t_start = time.time()
        tracker = self.get_tracker(camera_id)
        active_tracks = tracker.update(detections, timestamp)

        t_elapsed = time.time() - t_start
        self.total_tracking_time_seconds += t_elapsed
        self.total_frames_tracked += 1
        self.total_detections_tracked += len(detections)

        return active_tracks

    def get_active_tracks(self, camera_id: str) -> List[TrackState]:
        """
        Returns active tracks for a specific camera.
        """
        tracker = self.get_tracker(camera_id)
        return [t for t in tracker.active_tracks.values() if t.active]

    def get_all_active_tracks(self) -> Dict[str, List[TrackState]]:
        """
        Returns all active tracks grouped by camera_id.
        """
        result = {}
        for cam_id, tracker in self.trackers.items():
            result[cam_id] = [t for t in tracker.active_tracks.values() if t.active]
        return result

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Returns tracking performance FPS, average latency, active track counts, and tracking quality metrics.
        """
        avg_latency_ms = (
            (self.total_tracking_time_seconds / self.total_frames_tracked) * 1000.0
            if self.total_frames_tracked > 0 else 0.0
        )
        fps = (
            self.total_frames_tracked / self.total_tracking_time_seconds
            if self.total_tracking_time_seconds > 0 else 0.0
        )
        total_active = sum(
            len([t for t in trk.active_tracks.values() if t.active])
            for trk in self.trackers.values()
        )
        total_reactivations = sum(
            getattr(trk, "total_reactivations", 0) for trk in self.trackers.values()
        )
        total_duplicates_suppressed = sum(
            getattr(trk, "total_duplicates_suppressed", 0) for trk in self.trackers.values()
        )
        total_lost_events = sum(
            getattr(trk, "total_lost_events", 0) for trk in self.trackers.values()
        )

        return {
            "tracker_backend": self.config.tracker_backend,
            "total_frames_tracked": self.total_frames_tracked,
            "total_detections_tracked": self.total_detections_tracked,
            "active_track_count": total_active,
            "total_reactivations": total_reactivations,
            "total_duplicates_suppressed": total_duplicates_suppressed,
            "total_lost_events": total_lost_events,
            "average_latency_ms": round(avg_latency_ms, 3),
            "average_fps": round(fps, 2),
            "total_tracking_time_seconds": round(self.total_tracking_time_seconds, 4),
        }

