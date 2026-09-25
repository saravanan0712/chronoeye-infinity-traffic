"""
ChronoEye Infinity - Phase 3: Camera Calibration & Vehicle Speed Estimator
Calculates vehicle speed in km/h derived from pixel displacement across frames using
homography perspective transformation, reference-distance calibration, and road geometry.
Explicitly flags UNCALIBRATED state when perspective parameters are missing to prevent fake speeds.
"""

from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field


class CalibrationStatus(str, Enum):
    CALIBRATED = "CALIBRATED"
    HOMOGRAPHY_ESTIMATED = "HOMOGRAPHY_ESTIMATED"
    UNCALIBRATED = "UNCALIBRATED"
    UNAVAILABLE = "UNAVAILABLE"


class CalibrationConfig(BaseModel):
    """
    Camera calibration matrix and reference physical measurements.
    """
    camera_id: str
    calibration_status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
    # 3x3 perspective homography matrix mapping (x_px, y_px, 1) -> (x_m, y_m, 1)
    homography_matrix: Optional[List[List[float]]] = None
    pixels_per_meter: float = 20.0  # Simple linear scale fallback if homography matrix not set
    road_length_meters: float = 100.0
    lane_width_meters: float = 3.5
    speed_limit_kmh: float = 60.0


class SpeedEstimate(BaseModel):
    """
    Calculated speed result for a vehicle track.
    """
    track_id: str
    speed_kmh: float
    speed_mps: float
    speed_confidence: float
    calibration_status: CalibrationStatus
    distance_traveled_meters: float
    time_delta_seconds: float
    is_speed_available: bool = True


class CameraCalibrationEngine:
    """
    Engine executing geometric perspective mapping and calibrated speed calculations.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig(camera_id="CAM_UNKNOWN")

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """
        Maps a 2D pixel coordinate (px, py) to world coordinates (x_m, y_m) in meters.
        Uses 3x3 homography matrix if available, else linear pixels_per_meter scale.
        """
        if self.config.homography_matrix and len(self.config.homography_matrix) == 3:
            H = self.config.homography_matrix
            denom = H[2][0] * px + H[2][1] * py + H[2][2]
            if abs(denom) > 1e-6:
                wx = (H[0][0] * px + H[0][1] * py + H[0][2]) / denom
                wy = (H[1][0] * px + H[1][1] * py + H[1][2]) / denom
                return wx, wy

        # Linear scale fallback
        scale = max(1.0, self.config.pixels_per_meter)
        return px / scale, py / scale

    def calculate_speed(
        self,
        track_id: str,
        trajectory: List[Dict[str, Any]],
        window_frames: int = 5,
    ) -> SpeedEstimate:
        """
        Calculates instantaneous / windowed vehicle speed from trajectory time-series.
        Trajectory format: [{"timestamp": float, "x": float, "y": float}, ...]
        """
        if self.config.calibration_status == CalibrationStatus.UNCALIBRATED and not self.config.homography_matrix:
            return SpeedEstimate(
                track_id=track_id,
                speed_kmh=0.0,
                speed_mps=0.0,
                speed_confidence=0.0,
                calibration_status=CalibrationStatus.UNCALIBRATED,
                distance_traveled_meters=0.0,
                time_delta_seconds=0.0,
                is_speed_available=False,
            )

        if not trajectory or len(trajectory) < 2:
            return SpeedEstimate(
                track_id=track_id,
                speed_kmh=0.0,
                speed_mps=0.0,
                speed_confidence=0.0,
                calibration_status=self.config.calibration_status,
                distance_traveled_meters=0.0,
                time_delta_seconds=0.0,
                is_speed_available=False,
            )

        # Take window_frames latest points
        pts = trajectory[-window_frames:] if len(trajectory) >= window_frames else trajectory
        p1 = pts[0]
        p2 = pts[-1]

        t1 = p1.get("timestamp", 0.0)
        t2 = p2.get("timestamp", 0.0)
        dt = t2 - t1

        if dt <= 0.001:
            return SpeedEstimate(
                track_id=track_id,
                speed_kmh=0.0,
                speed_mps=0.0,
                speed_confidence=0.5,
                calibration_status=self.config.calibration_status,
                distance_traveled_meters=0.0,
                time_delta_seconds=dt,
                is_speed_available=True,
            )

        w1_x, w1_y = self.pixel_to_world(p1["x"], p1["y"])
        w2_x, w2_y = self.pixel_to_world(p2["x"], p2["y"])

        dist_m = math.sqrt((w2_x - w1_x) ** 2 + (w2_y - w1_y) ** 2)
        speed_mps = dist_m / dt
        speed_kmh = speed_mps * 3.6

        # Cap unrealistic vehicle speeds (> 200 km/h) due to pixel jitter
        speed_kmh = min(200.0, max(0.0, speed_kmh))
        speed_mps = speed_kmh / 3.6

        confidence = 0.90 if self.config.calibration_status == CalibrationStatus.CALIBRATED else 0.70

        return SpeedEstimate(
            track_id=track_id,
            speed_kmh=round(speed_kmh, 1),
            speed_mps=round(speed_mps, 2),
            speed_confidence=confidence,
            calibration_status=self.config.calibration_status,
            distance_traveled_meters=round(dist_m, 2),
            time_delta_seconds=round(dt, 3),
            is_speed_available=True,
        )
