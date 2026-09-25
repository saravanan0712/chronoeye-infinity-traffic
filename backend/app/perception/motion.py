"""
ChronoEye Infinity - Phase 3: Motion & Kinematics Module
Calculates velocity vectors (vx, vy), image-space speed estimates, and cardinal
movement directions (NORTH, SOUTH, EAST, WEST, etc.) from consecutive trajectory points.
"""

import math
from typing import Tuple
from app.schemas.detection import BoundingBoxXYXY
from app.schemas.tracking import Direction




class MotionEstimator:
    """
    Estimates velocity vectors, speed, and cardinal movement directions from spatial displacement over time.
    """

    @staticmethod
    def calculate_velocity_and_direction(
        prev_center: Tuple[float, float],
        curr_center: Tuple[float, float],
        dt_seconds: float,
        min_motion_threshold_pixels: float = 2.0,
    ) -> Tuple[float, float, float, Direction]:
        """
        Calculates (vx, vy, speed, direction) given previous center, current center, and dt.
        """
        if dt_seconds < 0.01:
            return 0.0, 0.0, 0.0, Direction.STATIONARY

        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        distance_pixels = math.hypot(dx, dy)

        velocity_x = dx / dt_seconds
        velocity_y = dy / dt_seconds
        speed_pixels_per_sec = distance_pixels / dt_seconds

        # Clamp max velocity to physical limits (1500 px/s) to prevent numerical explosion
        max_vel = 1500.0
        if speed_pixels_per_sec > max_vel:
            scale = max_vel / speed_pixels_per_sec
            velocity_x *= scale
            velocity_y *= scale
            speed_pixels_per_sec = max_vel

        if distance_pixels < min_motion_threshold_pixels:
            return round(velocity_x, 2), round(velocity_y, 2), round(speed_pixels_per_sec, 2), Direction.STATIONARY

        # Calculate angle in degrees (-180 to 180)
        # Note: image coordinates have Y pointing DOWNWARD (top-left origin)
        angle_rad = math.atan2(-dy, dx)  # Invert dy so positive Y is UP (North)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360.0

        if 67.5 <= angle_deg < 112.5:
            direction = Direction.NORTH
        elif 22.5 <= angle_deg < 67.5:
            direction = Direction.NORTHEAST
        elif angle_deg >= 337.5 or angle_deg < 22.5:
            direction = Direction.EAST
        elif 292.5 <= angle_deg < 337.5:
            direction = Direction.SOUTHEAST
        elif 247.5 <= angle_deg < 292.5:
            direction = Direction.SOUTH
        elif 202.5 <= angle_deg < 247.5:
            direction = Direction.SOUTHWEST
        elif 157.5 <= angle_deg < 202.5:
            direction = Direction.WEST
        else:
            direction = Direction.NORTHWEST

        return (
            round(velocity_x, 2),
            round(velocity_y, 2),
            round(speed_pixels_per_sec, 2),
            direction,
        )

    @staticmethod
    def predict_bbox(
        curr_bbox: BoundingBoxXYXY,
        velocity_x: float,
        velocity_y: float,
        dt_seconds: float,
        max_velocity_pixels_per_sec: float = 1500.0,
    ) -> BoundingBoxXYXY:
        """
        Projects a bounding box forward in time using constant velocity motion estimation.
        Displacement is capped to prevent unrealistic extrapolation.
        """
        if dt_seconds <= 0 or (velocity_x == 0.0 and velocity_y == 0.0):
            return curr_bbox

        speed = math.hypot(velocity_x, velocity_y)
        if speed > max_velocity_pixels_per_sec:
            scale = max_velocity_pixels_per_sec / speed
            vx = velocity_x * scale
            vy = velocity_y * scale
        else:
            vx = velocity_x
            vy = velocity_y

        dx = vx * dt_seconds
        dy = vy * dt_seconds

        x1 = max(0.0, curr_bbox.x1 + dx)
        y1 = max(0.0, curr_bbox.y1 + dy)
        x2 = max(x1 + 1.0, curr_bbox.x2 + dx)
        y2 = max(y1 + 1.0, curr_bbox.y2 + dy)

        return BoundingBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)


    @staticmethod
    def smooth_bbox(
        prev_bbox: BoundingBoxXYXY,
        det_bbox: BoundingBoxXYXY,
        alpha: float = 0.70,
    ) -> BoundingBoxXYXY:
        """
        Applies Exponential Moving Average (EMA) smoothing to bounding box coordinates:
        smoothed = alpha * det_bbox + (1 - alpha) * prev_bbox
        Alpha close to 1 gives low latency; alpha lower gives smoother boxes.
        """
        alpha = max(0.1, min(1.0, alpha))
        x1 = alpha * det_bbox.x1 + (1.0 - alpha) * prev_bbox.x1
        y1 = alpha * det_bbox.y1 + (1.0 - alpha) * prev_bbox.y1
        x2 = alpha * det_bbox.x2 + (1.0 - alpha) * prev_bbox.x2
        y2 = alpha * det_bbox.y2 + (1.0 - alpha) * prev_bbox.y2

        return BoundingBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)

