"""
ChronoEye Infinity - Phase 2 & 3: Visual Frame Annotator
Draws bounding boxes, detection events, persistent tracking IDs, trajectory trails,
velocity vectors, and direction labels on debug video frames.
"""

from typing import List, Any, Optional, Dict
from app.schemas.detection import DetectionEvent
from app.schemas.tracking import TrackState


class FrameAnnotator:
    """
    Annotates video frames with detection bounding boxes or persistent track state visual overlays.
    Uses OpenCV if available, with graceful fallback.
    """

    def __init__(
        self,
        box_color: tuple = (0, 255, 0),
        track_color: tuple = (255, 128, 0),
        trail_color: tuple = (0, 255, 255),
        text_color: tuple = (255, 255, 255),
    ):
        self.box_color = box_color
        self.track_color = track_color
        self.trail_color = trail_color
        self.text_color = text_color

    def annotate_frame(
        self,
        frame: Any,
        detections: List[DetectionEvent],
        camera_id: str = "CAM_01",
        frame_id: int = 0,
    ) -> Any:
        """
        Draws Phase 2 detection bounding boxes and labels onto the image frame.
        """
        try:
            import cv2
            annotated = frame.copy()

            header_str = f"Camera: {camera_id} | Frame: {frame_id} | Detections: {len(detections)}"
            cv2.putText(
                annotated,
                header_str,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for det in detections:
                x1, y1, x2, y2 = int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), self.box_color, 2)

                label = f"{det.class_name.upper()} {det.confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + w, y1), self.box_color, -1)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.text_color,
                    1,
                    cv2.LINE_AA,
                )

            return annotated
        except ImportError:
            return frame

    def annotate_tracks(
        self,
        frame: Any,
        tracks: List[TrackState],
        camera_id: str = "CAM_01",
        frame_id: int = 0,
        evidence_map: Optional[Dict[str, Any]] = None,
        journey_map: Optional[Dict[str, Any]] = None,
        fps: float = 0.0,
        vehicles_count: Optional[int] = None,
    ) -> Any:
        """
        Draws Phase 3 persistent tracking, Phase 4 ANPR, & Phase 5 Cross-Camera Re-ID overlays:
        global vehicle ID (GLOBAL ID: GID_xxx), local track ID (TRK_101), bounding boxes,
        trajectory trails, license plate readings (PLATE: xxx / PENDING / UNKNOWN), and status card.
        """
        try:
            import cv2
            annotated = frame.copy()

            v_count = vehicles_count if vehicles_count is not None else len(tracks)

            # Top-left ChronoEye status overlay card
            overlay_x, overlay_y = 15, 15
            overlay_w, overlay_h = 240, 105
            cv2.rectangle(
                annotated,
                (overlay_x, overlay_y),
                (overlay_x + overlay_w, overlay_y + overlay_h),
                (20, 20, 20),
                -1,
            )
            cv2.rectangle(
                annotated,
                (overlay_x, overlay_y),
                (overlay_x + overlay_w, overlay_y + overlay_h),
                (0, 200, 255),
                1,
            )

            cv2.putText(
                annotated,
                "CHRONOEYE INFINITY",
                (overlay_x + 10, overlay_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Frame: {frame_id:04d}",
                (overlay_x + 10, overlay_y + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Vehicles: {v_count:02d}",
                (overlay_x + 10, overlay_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"FPS: {fps:04.1f}",
                (overlay_x + 10, overlay_y + 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            for track in tracks:
                x1, y1, x2, y2 = (
                    int(track.current_bbox.x1),
                    int(track.current_bbox.y1),
                    int(track.current_bbox.x2),
                    int(track.current_bbox.y2),
                )

                # Draw Bounding Box
                color = self.track_color if track.confirmed else (128, 128, 128)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Draw Trajectory Trail
                if len(track.trajectory) > 1:
                    points = [(int(pt.center[0]), int(pt.center[1])) for pt in track.trajectory[-15:]]
                    for i in range(1, len(points)):
                        cv2.line(annotated, points[i - 1], points[i], self.trail_color, 2)

                # Stage 4 Plate Extraction — CONFIRMED / PENDING / UNKNOWN
                plate_str = "PLATE: UNKNOWN"
                conf_str = ""
                if evidence_map and track.track_id in evidence_map:
                    ev = evidence_map[track.track_id]
                    assoc_plate = getattr(ev, "associated_plate", None)
                    if assoc_plate:
                        status = getattr(assoc_plate, "status", "UNKNOWN")
                        is_conf = (status == "CONFIRMED") or getattr(assoc_plate, "confirmed", False)
                        plate_num = getattr(assoc_plate, "best_plate_number", None)
                        pending_num = getattr(assoc_plate, "pending_plate_number", None)
                        conf = getattr(assoc_plate, "overall_confidence", 0.0)

                        if is_conf and plate_num:
                            plate_str = f"PLATE: {plate_num}"
                            conf_str = f"CONF: {conf:.2f}"
                        elif (status == "PENDING" or pending_num) and not is_conf:
                            plate_str = "PLATE: PENDING"
                        else:
                            plate_str = "PLATE: UNKNOWN"

                # Stage 5 Global Vehicle Identity & Re-ID Score
                gid_str = ""
                if journey_map and track.track_id in journey_map:
                    journey = journey_map[track.track_id]
                    if journey and getattr(journey, "global_vehicle_id", None):
                        raw_gid = getattr(journey, "global_vehicle_id", "")
                        if raw_gid and "UNCONFIRMED" not in raw_gid.upper():
                            gid_str = f"GLOBAL ID: {raw_gid}"

                # Construct multi-line vehicle overlay header
                tid_str = track.track_id
                vtype_str = track.vehicle_type.upper()
                if gid_str:
                    line1 = f"{tid_str} | {gid_str}"
                else:
                    line1 = f"{tid_str} | {vtype_str}"

                if conf_str:
                    line2 = f"{plate_str} | {conf_str}"
                else:
                    line2 = plate_str

                # Calculate text dimensions for multi-line banner
                (w1, h1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                (w2, h2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                banner_w = max(w1, w2) + 10
                banner_h = h1 + h2 + 10

                banner_y1 = max(0, y1 - banner_h)
                cv2.rectangle(annotated, (x1, banner_y1), (x1 + banner_w, y1), color, -1)
                cv2.putText(
                    annotated,
                    line1,
                    (x1 + 5, max(12, banner_y1 + h1 + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    self.text_color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    line2,
                    (x1 + 5, max(24, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            return annotated
        except ImportError:
            return frame


