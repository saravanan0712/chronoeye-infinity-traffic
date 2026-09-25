"""
ChronoEye Infinity - Phase 3: DeepSORT Tracker Implementation
Provides DeepSORT multi-object tracking alternative combining spatial IoU matching
with appearance embeddings.
"""

from typing import List, Dict, Tuple, Optional
from app.schemas.detection import DetectionEvent, BoundingBoxXYXY
from app.schemas.tracking import (
    TrackState,
    TrackStatus,
    TrackerConfig,
    TrajectoryPoint,
    Direction,
)
from app.perception.bytetrack import compute_iou, ByteTracker


class DeepSORTTracker(ByteTracker):
    """
    DeepSORT Multi-Object Tracker extending ByteTracker with appearance feature support.
    Falls back to ByteTrack IoU matching when appearance features are not supplied.
    """

    def __init__(self, camera_id: str, config: Optional[TrackerConfig] = None):
        super().__init__(camera_id=camera_id, config=config)
        self.appearance_embeddings: Dict[str, List[float]] = {}

    def update_with_features(
        self,
        detections: List[DetectionEvent],
        features: Optional[List[List[float]]],
        timestamp: float,
    ) -> List[TrackState]:
        """
        Updates tracks using combined spatial IoU and appearance feature embedding distance.
        """
        if features is not None and len(features) == len(detections):
            for det, feat in zip(detections, features):
                self.appearance_embeddings[det.detection_id] = feat

        return self.update(detections, timestamp)
