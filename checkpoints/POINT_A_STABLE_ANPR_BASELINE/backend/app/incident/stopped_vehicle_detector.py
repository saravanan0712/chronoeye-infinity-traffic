"""
ChronoEye Infinity - Phase 12: Stopped Vehicle & Road Blockage Detector
Identifies stationary vehicles stranded on active road segments for extended durations (>30 seconds).
"""

from typing import List, Dict, Optional, Any
from app.perception.tracker import TrackState
from app.incident.incident_schema import IncidentEvent, IncidentType, IncidentSeverity, IncidentStatus


class StoppedVehicleDetector:
    """
    Stopped Vehicle Detector monitoring persistent vehicle track speeds and stationary durations.
    """

    def __init__(self, min_stopped_duration: float = 30.0, speed_threshold: float = 3.6):
        self.min_stopped_duration = min_stopped_duration  # seconds
        self.speed_threshold = speed_threshold          # km/h (1.0 m/s)

    def detect_stopped_vehicles(
        self,
        track_states: List[TrackState],
        segment_id: str = "ROAD_R_AB",
        timestamp: float = 0.0,
        incident_id_counter: int = 200,
    ) -> List[IncidentEvent]:
        """
        Detects stopped vehicle incidents from track states.
        """
        events: List[IncidentEvent] = []

        stopped_tracks = []
        for track in track_states:
            # Velocity in km/h
            speed_kmh = track.velocity * 3.6 if hasattr(track, "velocity") else 0.0
            if speed_kmh <= self.speed_threshold and len(track.history) >= 5:
                # Calculate stationary duration from history
                duration = timestamp - track.history[0][0] if track.history else 0.0
                if duration >= self.min_stopped_duration:
                    stopped_tracks.append((track.track_id, duration, speed_kmh))

        if stopped_tracks:
            count = len(stopped_tracks)
            inc_type = IncidentType.ROAD_BLOCKAGE if count >= 3 else IncidentType.STOPPED_VEHICLE
            sev = IncidentSeverity.CRITICAL if count >= 3 else IncidentSeverity.HIGH

            events.append(
                IncidentEvent(
                    incident_id=f"INC_{incident_id_counter}",
                    segment_id=segment_id,
                    incident_type=inc_type,
                    severity=sev,
                    confidence=min(1.0, 0.75 + (0.05 * count)),
                    detected_at=timestamp,
                    evidence={
                        "stopped_track_ids": [t[0] for t in stopped_tracks],
                        "stopped_vehicle_count": count,
                        "max_stopped_duration_seconds": max(t[1] for t in stopped_tracks),
                    },
                )
            )

        return events
