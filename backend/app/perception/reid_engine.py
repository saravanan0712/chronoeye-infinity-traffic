"""
ChronoEye Infinity - Phase 5: Re-ID Matching Engine
Computes multi-evidence cross-camera vehicle similarity scores combining license plate matching,
appearance embeddings, vehicle type, movement direction, temporal travel feasibility, and spatial topology.
"""

from typing import Optional, Tuple, Dict, Any, List
from app.schemas.detection import BoundingBoxXYXY
from app.schemas.tracking import TrackState
from app.schemas.plate import VehicleIdentityEvidence
from app.schemas.reid import (
    MatchDecision,
    ReIDScoreBreakdown,
    ReIDConfig,
)
from app.perception.camera_topology import CityCameraTopology
from app.perception.appearance import AppearanceEmbeddingExtractor


class ReIDMatchingEngine:
    """
    Multi-Evidence Cross-Camera Vehicle Re-Identification Engine.
    Fuses license plate matching, visual appearance cosine similarity, geometric visual features,
    vehicle type, movement direction, and physical spatio-temporal travel feasibility.
    """

    def __init__(
        self,
        topology: Optional[CityCameraTopology] = None,
        config: Optional[ReIDConfig] = None,
    ):
        self.topology = topology or CityCameraTopology()
        self.config = config or ReIDConfig()

    def calculate_plate_similarity(
        self, plate_a: Optional[str], plate_b: Optional[str]
    ) -> float:
        """
        Calculates license plate match similarity (0.0 to 1.0).
        Handles exact matches, normalized matches, partial Levenshtein similarity, and missing plates.
        """
        if not plate_a or not plate_b or plate_a.upper() == "UNKNOWN" or plate_b.upper() == "UNKNOWN":
            return 0.50  # Neutral score when plate is unreadable/missing

        str_a = plate_a.upper().replace(" ", "").replace("-", "")
        str_b = plate_b.upper().replace(" ", "").replace("-", "")

        if str_a == str_b:
            return 1.0  # Exact match

        # Compute character match ratio
        min_len = min(len(str_a), len(str_b))
        max_len = max(len(str_a), len(str_b))
        if max_len == 0:
            return 0.50

        matches = sum(1 for i in range(min_len) if str_a[i] == str_b[i])
        similarity = matches / float(max_len)

        # If state code differs (e.g. TN vs KA), penalize heavily
        if len(str_a) >= 2 and len(str_b) >= 2 and str_a[:2] != str_b[:2]:
            return min(0.05, round(similarity * 0.1, 3))

        return round(similarity, 3)

    def extract_visual_features(
        self, bbox: Optional[BoundingBoxXYXY], track: Optional[TrackState] = None
    ) -> Dict[str, Any]:
        """
        Extracts deterministic structural and geometric visual features from bounding box:
        - aspect_ratio: width / height
        - elongation: |width - height| / (width + height)
        - area: bbox pixel area
        - width: bbox width
        - height: bbox height
        - vehicle_type: vehicle category string
        """
        v_type = track.vehicle_type if track and hasattr(track, "vehicle_type") else "UNKNOWN"
        if bbox is None:
            return {"aspect_ratio": 1.0, "elongation": 0.0, "area": 0.0, "width": 0.0, "height": 0.0, "vehicle_type": v_type}

        w = max(1.0, float(bbox.width))
        h = max(1.0, float(bbox.height))
        aspect_ratio = w / h
        elongation = abs(w - h) / (w + h)
        area = float(bbox.area)

        return {
            "aspect_ratio": round(aspect_ratio, 4),
            "elongation": round(elongation, 4),
            "area": round(area, 2),
            "width": round(w, 2),
            "height": round(h, 2),
            "vehicle_type": v_type,
        }

    def calculate_visual_feature_similarity(
        self,
        feat_a: Optional[Dict[str, Any]],
        feat_b: Optional[Dict[str, Any]],
    ) -> float:
        """
        Calculates bounded similarity [0.0, 1.0] between geometric visual features.
        Compares normalized aspect ratio, elongation, and structural vehicle type compatibility.
        Falls back to neutral 0.50 if missing.
        """
        if not feat_a or not feat_b:
            return 0.50

        ar_a = max(0.01, float(feat_a.get("aspect_ratio", 1.0)))
        ar_b = max(0.01, float(feat_b.get("aspect_ratio", 1.0)))
        ar_sim = min(ar_a, ar_b) / max(ar_a, ar_b)

        el_a = float(feat_a.get("elongation", 0.0))
        el_b = float(feat_b.get("elongation", 0.0))
        el_sim = max(0.0, 1.0 - abs(el_a - el_b))

        type_a = str(feat_a.get("vehicle_type", "UNKNOWN")).lower()
        type_b = str(feat_b.get("vehicle_type", "UNKNOWN")).lower()
        type_scale = 1.0
        if type_a != "unknown" and type_b != "unknown" and type_a != type_b:
            type_scale = 0.35  # Structural mismatch penalty for different vehicle categories

        sim = (0.60 * ar_sim + 0.40 * el_sim) * type_scale
        return round(max(0.0, min(1.0, sim)), 3)

    def calculate_direction_similarity(self, dir_a: str, dir_b: str) -> float:
        """
        Calculates movement direction compatibility between cameras.
        """
        if dir_a == "UNKNOWN" or dir_b == "UNKNOWN":
            return 0.5

        if dir_a == dir_b:
            return 1.0

        # Opposite directions check
        opposites = {
            "NORTH": "SOUTH", "SOUTH": "NORTH",
            "EAST": "WEST", "WEST": "EAST",
            "NORTHEAST": "SOUTHWEST", "SOUTHWEST": "NORTHEAST",
            "NORTHWEST": "SOUTHEAST", "SOUTHEAST": "NORTHWEST",
        }
        if opposites.get(dir_a) == dir_b:
            return 0.1  # Low score for opposite travel direction

        return 0.6  # Partial match score for orthogonal direction

    def compute_match_score(
        self,
        evidence_a: VehicleIdentityEvidence,
        track_a: TrackState,
        evidence_b: VehicleIdentityEvidence,
        track_b: TrackState,
        embedding_a: Optional[list] = None,
        embedding_b: Optional[list] = None,
        visual_features_a: Optional[Dict[str, float]] = None,
        visual_features_b: Optional[Dict[str, float]] = None,
    ) -> ReIDScoreBreakdown:
        """
        Computes composite cross-camera match score and decision across 7 distinct signals.
        """
        cam_a = evidence_a.camera_id
        cam_b = evidence_b.camera_id
        t_a = evidence_a.last_updated_timestamp
        t_b = evidence_b.last_updated_timestamp

        # Enforce temporal ordering (t_a <= t_b)
        if t_a > t_b:
            evidence_a, evidence_b = evidence_b, evidence_a
            track_a, track_b = track_b, track_a
            cam_a, cam_b = cam_b, cam_a
            t_a, t_b = t_b, t_a
            embedding_a, embedding_b = embedding_b, embedding_a
            visual_features_a, visual_features_b = visual_features_b, visual_features_a

        delta_t = t_b - t_a

        # 1. Physical Spatio-Temporal Feasibility Check (Hard Filter)
        feasible, reason = self.topology.is_temporally_feasible(
            cam_a, cam_b, delta_t, self.config.max_speed_kmh, self.config.min_speed_kmh, self.config.max_time_gap_seconds
        )

        if not feasible:
            return ReIDScoreBreakdown(
                overall_score=0.0,
                decision=MatchDecision.MATCH_REJECTED,
                rejection_reason=reason,
            )

        # 2. Extract Sub-Evidence Similarity Scores
        plate_a = evidence_a.associated_plate.best_plate_number if evidence_a.associated_plate else None
        plate_b = evidence_b.associated_plate.best_plate_number if evidence_b.associated_plate else None
        s_plate = self.calculate_plate_similarity(plate_a, plate_b)

        s_app = AppearanceEmbeddingExtractor.cosine_similarity(embedding_a, embedding_b)

        # Visual features (geometric/structural characteristics)
        vf_a = visual_features_a if visual_features_a is not None else self.extract_visual_features(getattr(track_a, "current_bbox", getattr(track_a, "bbox", None)), track_a)
        vf_b = visual_features_b if visual_features_b is not None else self.extract_visual_features(getattr(track_b, "current_bbox", getattr(track_b, "bbox", None)), track_b)
        s_vf = self.calculate_visual_feature_similarity(vf_a, vf_b)

        s_type = 1.0 if evidence_a.vehicle_type.lower() == evidence_b.vehicle_type.lower() else 0.0

        dir_a = track_a.direction.value if track_a.direction else "UNKNOWN"
        dir_b = track_b.direction.value if track_b.direction else "UNKNOWN"
        s_dir = self.calculate_direction_similarity(dir_a, dir_b)

        # Temporal score based on speed reasonableness
        dist_m = self.topology.get_distance_meters(cam_a, cam_b)
        speed_m_s = dist_m / max(0.1, delta_t) if delta_t > 0 else 0.0
        speed_kmh = (speed_m_s * 3600.0) / 1000.0
        s_time = 1.0 if (15.0 <= speed_kmh <= 80.0 or delta_t == 0) else 0.7

        # Spatial topology connectivity score
        s_space = 1.0 if self.topology.get_distance_meters(cam_a, cam_b) < 1000.0 else 0.5

        # 3. Calculate Weighted Composite Score
        w_vf = getattr(self.config, "w_visual_features", 0.10)
        overall = (
            self.config.w_plate * s_plate
            + self.config.w_appearance * s_app
            + w_vf * s_vf
            + self.config.w_type * s_type
            + self.config.w_direction * s_dir
            + self.config.w_time * s_time
            + self.config.w_space * s_space
        )
        overall = round(overall, 3)

        # 4. Classify Match Decision
        if overall >= self.config.confirm_threshold:
            decision = MatchDecision.MATCH_CONFIRMED
        elif overall >= self.config.probable_threshold:
            decision = MatchDecision.MATCH_PROBABLE
        else:
            decision = MatchDecision.INSUFFICIENT_EVIDENCE

        return ReIDScoreBreakdown(
            plate_similarity=round(s_plate, 3),
            appearance_similarity=round(s_app, 3),
            visual_features_similarity=round(s_vf, 3),
            vehicle_type_similarity=round(s_type, 3),
            direction_similarity=round(s_dir, 3),
            temporal_compatibility=round(s_time, 3),
            spatial_compatibility=round(s_space, 3),
            overall_score=overall,
            decision=decision,
        )


# Backwards-compatible public alias (test_system_verification.py imports this name)
VehicleReIDEngine = ReIDMatchingEngine
