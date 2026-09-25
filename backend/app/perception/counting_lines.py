"""
ChronoEye Infinity - Phase 3: Virtual Line Counting & Direction Classifier
Detects when vehicle track trajectories cross virtual counting lines, calculates vehicle flow rates,
and determines cardinal direction vectors (NORTHBOUND, SOUTHBOUND, EASTBOUND, WESTBOUND).
"""

from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field


class Direction(str, Enum):
    NORTHBOUND = "NORTHBOUND"
    SOUTHBOUND = "SOUTHBOUND"
    EASTBOUND = "EASTBOUND"
    WESTBOUND = "WESTBOUND"
    UNKNOWN = "UNKNOWN"


class LineCrossingEvent(BaseModel):
    """
    Event generated when a vehicle track trajectory crosses a virtual counting line.
    """
    line_id: str
    track_id: str
    camera_id: str
    vehicle_class: str
    direction: Direction
    timestamp: float
    frame_id: int
    crossing_point: Tuple[float, float]


class VirtualLineCounter:
    """
    Detects line-segment intersection between vehicle trajectory segments and virtual counting lines.
    """

    def __init__(self, line_id: str, start_pt: Tuple[float, float], end_pt: Tuple[float, float], name: str = "ENTRY_LINE"):
        self.line_id = line_id
        self.name = name
        self.start_pt = start_pt  # (x1, y1)
        self.end_pt = end_pt      # (x2, y2)
        self.crossed_tracks: Dict[str, LineCrossingEvent] = {}
        self.total_count = 0
        self.counts_by_class: Dict[str, int] = {}

    @staticmethod
    def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(self, A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float], D: Tuple[float, float]) -> bool:
        """Determines if line segment AB intersects line segment CD."""
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def check_trajectory_crossing(
        self,
        track_id: str,
        vehicle_class: str,
        camera_id: str,
        trajectory: List[Dict[str, Any]],
        frame_id: int,
    ) -> Optional[LineCrossingEvent]:
        """
        Checks if the latest trajectory segment crossed this virtual counting line.
        """
        if track_id in self.crossed_tracks or len(trajectory) < 2:
            return None

        p1 = (trajectory[-2]["x"], trajectory[-2]["y"])
        p2 = (trajectory[-1]["x"], trajectory[-1]["y"])
        t2 = trajectory[-1]["timestamp"]

        if self.intersect(p1, p2, self.start_pt, self.end_pt):
            dir_enum = DirectionClassifier.classify_direction(p1, p2)
            event = LineCrossingEvent(
                line_id=self.line_id,
                track_id=track_id,
                camera_id=camera_id,
                vehicle_class=vehicle_class,
                direction=dir_enum,
                timestamp=t2,
                frame_id=frame_id,
                crossing_point=p2,
            )
            self.crossed_tracks[track_id] = event
            self.total_count += 1
            self.counts_by_class[vehicle_class] = self.counts_by_class.get(vehicle_class, 0) + 1
            return event

        return None


class DirectionClassifier:
    """
    Classifies movement direction vectors into cardinal directions based on trajectory delta.
    """

    @staticmethod
    def classify_direction(p1: Tuple[float, float], p2: Tuple[float, float]) -> Direction:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            return Direction.UNKNOWN

        # In image coordinates, Y increases downward
        # dx > 0 = EAST, dx < 0 = WEST
        # dy > 0 = SOUTH, dy < 0 = NORTH
        if abs(dy) >= abs(dx):
            return Direction.SOUTHBOUND if dy > 0 else Direction.NORTHBOUND
        else:
            return Direction.EASTBOUND if dx > 0 else Direction.WESTBOUND
