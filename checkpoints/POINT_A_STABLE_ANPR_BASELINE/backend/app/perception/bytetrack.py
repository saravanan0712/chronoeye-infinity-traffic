"""
ChronoEye Infinity - Phase 3: ByteTrack Tracker Implementation (Optimized Engine)
Implements an upgraded ByteTrack multi-object tracking algorithm featuring:
- Detection quality gating & frame boundary validation
- Short-term motion prediction (velocity-driven kinematic projection)
- Multi-factor data association (IoU + Motion + Size Ratio + Vehicle Class + Confidence)
- Explicit Track Lifecycle State Machine (NEW -> TENTATIVE -> CONFIRMED -> LOST -> REACTIVATED -> REMOVED)
- Occlusion handling & track reactivation
- Duplicate track suppression & vehicle class stability voting
- EMA bounding box smoothing & transparent track quality scoring
- Bounded trajectory memory management
"""

import math
from typing import List, Dict, Tuple, Optional, Set
from app.schemas.detection import DetectionEvent, BoundingBoxXYXY
from app.schemas.tracking import (
    TrackState,
    TrackStatus,
    TrackerConfig,
    TrajectoryPoint,
    Direction,
)
from app.perception.motion import MotionEstimator


def compute_iou(boxA: BoundingBoxXYXY, boxB: BoundingBoxXYXY) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes."""
    x1 = max(boxA.x1, boxB.x1)
    y1 = max(boxA.y1, boxB.y1)
    x2 = min(boxA.x2, boxB.x2)
    y2 = min(boxA.y2, boxB.y2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    areaA = boxA.area
    areaB = boxB.area
    union_area = areaA + areaB - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


class ByteTracker:
    """
    Upgraded ByteTrack Multi-Object Tracker.
    Maintains persistent track IDs per camera across video frames with robust occlusion recovery.
    """

    def __init__(self, camera_id: str, config: Optional[TrackerConfig] = None):
        self.camera_id = camera_id
        self.config = config or TrackerConfig()
        self.active_tracks: Dict[str, TrackState] = {}
        self.removed_tracks: Dict[str, TrackState] = {}
        self.next_track_number = 101
        self.frame_count = 0

        # Diagnostics & Proxy Metrics
        self.total_reactivations = 0
        self.total_duplicates_suppressed = 0
        self.total_lost_events = 0

    def update(self, detections: List[DetectionEvent], timestamp: float) -> List[TrackState]:
        """
        Updates tracker state with new frame detections.
        """
        self.frame_count += 1

        # 1. Detection Quality Gating & Pre-filtering
        gated_detections = self._gate_detections(detections)

        # Separate detections into high-confidence and low-confidence pools
        high_dets = [d for d in gated_detections if d.confidence >= self.config.high_conf_thresh]
        low_dets = [
            d for d in gated_detections
            if self.config.low_conf_thresh <= d.confidence < self.config.high_conf_thresh
        ]

        active_track_list = [t for t in self.active_tracks.values() if t.active]

        # 2. Stage 1 Association: High-confidence Detections vs Active / Lost Tracks
        matched_tracks_stage1, unmatched_tracks_stage1, unmatched_dets_high = self._associate(
            active_track_list, high_dets, timestamp, min_score_threshold=self.config.iou_threshold
        )

        # 3. Stage 2 Association: Remaining Unmatched Tracks vs Low-confidence Detections
        matched_tracks_stage2, unmatched_tracks_stage2, _ = self._associate(
            unmatched_tracks_stage1, low_dets, timestamp, min_score_threshold=self.config.iou_threshold
        )

        # Combine all matched pairs
        all_matched_pairs = matched_tracks_stage1 + matched_tracks_stage2

        # 4. Update Matched Tracks
        for track, det in all_matched_pairs:
            dt = max(0.001, timestamp - track.last_seen_timestamp)
            vx, vy, speed, direction = MotionEstimator.calculate_velocity_and_direction(
                track.current_center, det.bbox_center, dt
            )

            # Store raw detection bbox & EMA smoothed current bbox
            track.raw_bbox = det.bbox
            track.current_bbox = MotionEstimator.smooth_bbox(
                track.current_bbox, det.bbox, alpha=self.config.bbox_smoothing_alpha
            )
            track.current_center = track.current_bbox.center
            track.confidence = det.confidence
            track.last_seen_timestamp = timestamp
            track.age += 1
            track.hits += 1
            track.missed_frames = 0
            track.velocity_x = vx
            track.velocity_y = vy
            track.speed_estimate = speed
            track.direction = direction
            track.predicted_bbox = None  # Clear prediction on active match
            track.detection_history.append(det.detection_id)

            if det.raw_vehicle_reference:
                track.raw_vehicle_reference = det.raw_vehicle_reference

            # Class Voting & Dominant Vehicle Class Assignment
            track.class_votes[det.class_name] = track.class_votes.get(det.class_name, 0) + 1
            dominant_class = max(track.class_votes.items(), key=lambda x: x[1])[0]
            track.vehicle_type = dominant_class

            # State Transition & Track Confirmation / Reactivation
            if track.status == TrackStatus.LOST:
                track.status = TrackStatus.REACTIVATED
                track.reactivation_count += 1
                self.total_reactivations += 1
            elif track.hits >= self.config.min_confirmation_hits:
                track.status = TrackStatus.CONFIRMED
                track.confirmed = True
            elif track.status == TrackStatus.NEW:
                track.status = TrackStatus.TENTATIVE

            # Compute Track Quality Score
            hit_ratio = track.hits / max(1, track.age)
            conf_avg = track.confidence
            track.track_quality_score = round(min(1.0, 0.5 * hit_ratio + 0.5 * conf_avg), 3)

            # Append Trajectory Point (raw observed detection bbox and center)
            traj_pt = TrajectoryPoint(
                timestamp=timestamp,
                frame_id=self.frame_count,
                bbox=det.bbox,
                center=det.bbox_center,
                confidence=det.confidence,
                speed_estimate=speed,
                direction=direction,
                camera_id=self.camera_id,
            )
            track.trajectory.append(traj_pt)

            # Enforce Bounded Memory for Trajectory History
            if len(track.trajectory) > self.config.max_trajectory_length:
                track.trajectory = track.trajectory[-self.config.max_trajectory_length:]

        # 5. Initialize New Tracks for Unmatched High-Confidence Detections
        for det in unmatched_dets_high:
            track_id = f"{self.config.track_id_prefix}{self.next_track_number}"
            self.next_track_number += 1

            new_track = TrackState(
                track_id=track_id,
                camera_id=self.camera_id,
                vehicle_type=det.class_name,
                current_bbox=det.bbox,
                raw_bbox=det.bbox,
                current_center=det.bbox_center,
                confidence=det.confidence,
                first_seen_timestamp=timestamp,
                last_seen_timestamp=timestamp,
                age=1,
                hits=1,
                missed_frames=0,
                status=TrackStatus.TENTATIVE,
                confirmed=False,
                active=True,
                detection_history=[det.detection_id],
                raw_vehicle_reference=det.raw_vehicle_reference,
                class_votes={det.class_name: 1},
                track_quality_score=round(det.confidence, 3),
            )
            traj_pt = TrajectoryPoint(
                timestamp=timestamp,
                frame_id=self.frame_count,
                bbox=det.bbox,
                center=det.bbox_center,
                confidence=det.confidence,
                camera_id=self.camera_id,
            )
            new_track.trajectory.append(traj_pt)
            self.active_tracks[track_id] = new_track

        # 6. Process Unmatched Tracks (Missed Detections / Occlusion Handling)
        for track in unmatched_tracks_stage2:
            if track.status != TrackStatus.LOST:
                self.total_lost_events += 1
            track.missed_frames += 1
            track.age += 1
            track.status = TrackStatus.LOST

            # Motion Prediction during Occlusion / Missed Detection
            dt_missed = max(0.033, timestamp - track.last_seen_timestamp)
            track.predicted_bbox = MotionEstimator.predict_bbox(
                track.current_bbox,
                track.velocity_x,
                track.velocity_y,
                dt_missed,
                max_velocity_pixels_per_sec=self.config.max_velocity_pixels_per_sec,
            )
            # Update current_bbox to predicted_bbox during occlusion to maintain trajectory alignment
            track.current_bbox = track.predicted_bbox
            track.current_center = track.current_bbox.center

            # Track Expiration after configured max lost frames
            if track.missed_frames > self.config.max_lost_frames:
                track.status = TrackStatus.REMOVED
                track.active = False
                self.removed_tracks[track.track_id] = track

        # 7. Duplicate Track Suppression
        self._suppress_duplicate_tracks()

        # Return list of active tracks (tentative, confirmed, or lost/reactivated)
        return [t for t in self.active_tracks.values() if t.active]

    def _gate_detections(self, detections: List[DetectionEvent]) -> List[DetectionEvent]:
        """
        Validates detection bounding boxes and filters out noise / duplicates.
        """
        valid_dets: List[DetectionEvent] = []
        for det in detections:
            bbox = det.bbox
            if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
                continue

            width = bbox.x2 - bbox.x1
            height = bbox.y2 - bbox.y1
            area = width * height

            if area < self.config.min_bbox_area:
                continue

            aspect_ratio = max(width, height) / max(1.0, min(width, height))
            if aspect_ratio > self.config.max_aspect_ratio:
                continue

            valid_dets.append(det)

        # Intra-frame duplicate detection deduplication (NMS-style filter)
        if len(valid_dets) <= 1:
            return valid_dets

        keep_dets: List[DetectionEvent] = []
        valid_dets.sort(key=lambda d: d.confidence, reverse=True)

        for det in valid_dets:
            overlap = False
            for kept in keep_dets:
                if det.class_name == kept.class_name and compute_iou(det.bbox, kept.bbox) >= 0.85:
                    overlap = True
                    break
            if not overlap:
                keep_dets.append(det)

        return keep_dets

    def _compute_multi_factor_score(
        self, track: TrackState, det: DetectionEvent, timestamp: float
    ) -> float:
        """
        Calculates multi-factor association score between track and candidate detection.
        Score = w_iou * IoU + w_motion * MotionSim + w_size * SizeRatio + w_class * ClassMatch + w_conf * DetConf
        """
        # 1. Spatial IoU (using predicted bbox if track is lost/occluded)
        track_box = track.predicted_bbox if (track.status == TrackStatus.LOST and track.predicted_bbox) else track.current_bbox
        iou = compute_iou(track_box, det.bbox)

        # 2. Motion Consistency Score
        pred_center = track_box.center
        dist = math.hypot(det.bbox_center[0] - pred_center[0], det.bbox_center[1] - pred_center[1])
        expected_radius = max(50.0, math.hypot(track_box.width, track_box.height))
        motion_sim = max(0.0, 1.0 - (dist / expected_radius))

        # 3. Size Consistency Score
        area_track = max(1.0, track_box.area)
        area_det = max(1.0, det.bbox_area)
        size_ratio = min(area_track, area_det) / max(area_track, area_det)

        # 4. Class Consistency Score
        class_match = 1.0 if track.vehicle_type == det.class_name else 0.5

        # 5. Detection Confidence
        det_conf = det.confidence

        # Composite Weighted Score
        score = (
            self.config.weight_iou * iou
            + self.config.weight_motion * motion_sim
            + self.config.weight_size * size_ratio
            + self.config.weight_class * class_match
            + self.config.weight_confidence * det_conf
        )
        return score

    def _associate(
        self,
        tracks: List[TrackState],
        detections: List[DetectionEvent],
        timestamp: float,
        min_score_threshold: float,
    ) -> Tuple[List[Tuple[TrackState, DetectionEvent]], List[TrackState], List[DetectionEvent]]:
        """
        Multi-factor bipartite matching between tracks and detections.
        """
        if not tracks or not detections:
            return [], tracks, detections

        # Compute score matrix
        score_matrix: List[List[float]] = []
        for track in tracks:
            row = [self._compute_multi_factor_score(track, det, timestamp) for det in detections]
            score_matrix.append(row)

        matched_pairs: List[Tuple[TrackState, DetectionEvent]] = []
        matched_track_indices: Set[int] = set()
        matched_det_indices: Set[int] = set()

        candidates = []
        for t_idx in range(len(tracks)):
            for d_idx in range(len(detections)):
                score = score_matrix[t_idx][d_idx]
                if score >= min_score_threshold:
                    candidates.append((score, t_idx, d_idx))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for score, t_idx, d_idx in candidates:
            if t_idx not in matched_track_indices and d_idx not in matched_det_indices:
                matched_track_indices.add(t_idx)
                matched_det_indices.add(d_idx)
                matched_pairs.append((tracks[t_idx], detections[d_idx]))

        unmatched_tracks = [tracks[i] for i in range(len(tracks)) if i not in matched_track_indices]
        unmatched_dets = [detections[j] for j in range(len(detections)) if j not in matched_det_indices]

        return matched_pairs, unmatched_tracks, unmatched_dets

    def _suppress_duplicate_tracks(self):
        """
        Detects and deactivates duplicate active tracks representing the same physical vehicle.
        """
        active_list = [t for t in self.active_tracks.values() if t.active]
        if len(active_list) <= 1:
            return

        for i in range(len(active_list)):
            t1 = active_list[i]
            if not t1.active:
                continue

            for j in range(i + 1, len(active_list)):
                t2 = active_list[j]
                if not t2.active:
                    continue

                if t1.vehicle_type == t2.vehicle_type:
                    iou = compute_iou(t1.current_bbox, t2.current_bbox)
                    if iou >= self.config.duplicate_iou_thresh:
                        # Deactivate the weaker track (fewer hits / lower quality score)
                        if t1.hits >= t2.hits:
                            t2.active = False
                            t2.status = TrackStatus.REMOVED
                            self.removed_tracks[t2.track_id] = t2
                            self.total_duplicates_suppressed += 1
                        else:
                            t1.active = False
                            t1.status = TrackStatus.REMOVED
                            self.removed_tracks[t1.track_id] = t1
                            self.total_duplicates_suppressed += 1
                            break
