"""
ChronoEye Infinity - Phase 2: Detection Adapter
Adapts Phase 1 simulation CameraObservation objects into normalized Phase 2 DetectionEvent objects,
ensuring data continuity across the perception pipeline.
"""

from typing import List
from app.schemas.simulation import CameraObservation
from app.schemas.detection import DetectionEvent, BoundingBoxXYXY


class DetectionAdapter:
    """
    Adapter bridging Phase 1 Traffic Simulation Camera Observations to Phase 2 Detection Events.
    Preserves camera identity, timestamp, frame ID, vehicle reference, and spatial bounding boxes.
    """

    @staticmethod
    def from_camera_observation(obs: CameraObservation) -> DetectionEvent:
        """
        Converts a single CameraObservation into a normalized DetectionEvent.
        """
        # Convert bbox from center/offset (x, y, w, h) to (x1, y1, x2, y2)
        x1 = max(0.0, float(obs.bbox.x))
        y1 = max(0.0, float(obs.bbox.y))
        x2 = x1 + max(1.0, float(obs.bbox.width))
        y2 = y1 + max(1.0, float(obs.bbox.height))

        bbox_xyxy = BoundingBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)

        # Map vehicle class name
        class_name = str(obs.vehicle_type.value).lower()
        if class_name == "emergency":
            class_name = "ambulance"

        class_id_map = {
            "car": 2,
            "motorcycle": 3,
            "bus": 5,
            "truck": 7,
            "ambulance": 8,
        }
        class_id = class_id_map.get(class_name, 2)

        return DetectionEvent(
            detection_id=f"DET_SIM_{obs.observation_id}",
            camera_id=obs.camera_id,
            frame_id=obs.frame_id,
            timestamp=obs.timestamp,
            class_name=class_name,
            class_id=class_id,
            confidence=round(obs.detection_confidence, 3),
            bbox=bbox_xyxy,
            bbox_center=bbox_xyxy.center,
            bbox_area=round(bbox_xyxy.area, 1),
            image_width=1920,
            image_height=1080,
            source="PHASE_1_SIMULATION_ADAPTER",
            raw_vehicle_reference=obs.vehicle_id,
        )

    @staticmethod
    def batch_convert(obs_list: List[CameraObservation]) -> List[DetectionEvent]:
        """
        Converts a list of CameraObservation objects into DetectionEvent objects.
        """
        return [DetectionAdapter.from_camera_observation(obs) for obs in obs_list]
