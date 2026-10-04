"""
ChronoEye Infinity - Phase 4: Track ↔ Plate Association Manager
Bridges Phase 3 persistent TrackState objects with Phase 4 license plate observations and
temporal fused plate identities, generating unified VehicleIdentityEvidence objects.
"""

from typing import Dict, List, Optional, Any, Tuple
import os as _os

_ANPR_DIAG = _os.environ.get("ANPR_DIAG", "0") == "1"

def _diag(*args):
    """Print diagnostic unconditionally for this run."""
    import sys
    print("[ANPR_TRACE]", *args, file=sys.stderr, flush=True)

from app.schemas.detection import BoundingBoxXYXY
from app.schemas.tracking import TrackState
from app.schemas.plate import (
    PlateObservation,
    FusedPlateIdentity,
    VehicleIdentityEvidence,
    PlateValidationStatus,
)
from app.perception.plate_detector import PlateDetector
from app.perception.plate_preprocessor import PlatePreprocessor
from app.perception.ocr_engine import (
    BaseOCREngine,
    IndianPlateValidator,
    TextNormalizer,
    OCREngineFactory,
)
from app.perception.plate_fusion import TemporalPlateFusionEngine


def compute_bbox_intersection_area(bbox1: BoundingBoxXYXY, bbox2: BoundingBoxXYXY) -> float:
    """Computes the intersection area between two bounding boxes."""
    ix1 = max(float(bbox1.x1), float(bbox2.x1))
    iy1 = max(float(bbox1.y1), float(bbox2.y1))
    ix2 = min(float(bbox1.x2), float(bbox2.x2))
    iy2 = min(float(bbox1.y2), float(bbox2.y2))
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    return inter_w * inter_h


def compute_plate_containment_fraction(
    plate_bbox: BoundingBoxXYXY, vehicle_bbox: BoundingBoxXYXY
) -> float:
    """
    Calculates the fraction of the plate bounding box area that lies inside the vehicle bounding box:
    plate_area_inside_vehicle / plate_area
    Returns float in [0.0, 1.0].
    """
    p_area = float(plate_bbox.area)
    if p_area <= 0.0:
        return 0.0
    inter_area = compute_bbox_intersection_area(plate_bbox, vehicle_bbox)
    return min(1.0, max(0.0, inter_area / p_area))


def compute_plate_vehicle_iou(
    plate_bbox: BoundingBoxXYXY, vehicle_bbox: BoundingBoxXYXY
) -> float:
    """
    Calculates the Intersection-over-Union (IoU) between plate bbox and vehicle bbox.
    """
    p_area = float(plate_bbox.area)
    v_area = float(vehicle_bbox.area)
    inter_area = compute_bbox_intersection_area(plate_bbox, vehicle_bbox)
    union_area = p_area + v_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return min(1.0, max(0.0, inter_area / union_area))


def resolve_plate_ownership(
    plate_bbox: BoundingBoxXYXY,
    candidate_tracks: List[TrackState],
    containment_threshold: float = 0.05,
    tie_tolerance: float = 0.05,
) -> Optional[TrackState]:
    """
    Resolves spatial ownership of a detected license plate bounding box among multiple candidate vehicle tracks.
    
    Rules:
    1. Determine all vehicle tracks whose bounding boxes contain/intersect the plate bbox (containment > containment_threshold).
    2. If only ONE vehicle contains the plate: preserve existing behavior exactly (return that track).
    3. If MULTIPLE vehicles contain/intersect the plate:
       - Calculate plate-to-vehicle containment: plate_area_inside_vehicle / plate_area
       - Assign the plate ONLY to the vehicle with the strongest spatial ownership.
       - Prefer the vehicle with the highest plate containment fraction.
       - If there is a meaningful tie (difference <= tie_tolerance), use the highest IoU / strongest overlap as the secondary criterion.
    4. If no vehicle contains/intersects the plate (all containment <= 0), return None.
    """
    if not candidate_tracks:
        return None

    # Step 1: Filter tracks with non-trivial intersection
    intersecting_tracks = []
    for track in candidate_tracks:
        v_bbox = track.current_bbox
        containment = compute_plate_containment_fraction(plate_bbox, v_bbox)
        if containment > containment_threshold:
            iou = compute_plate_vehicle_iou(plate_bbox, v_bbox)
            intersecting_tracks.append({
                "track": track,
                "containment": containment,
                "iou": iou,
            })

    if not intersecting_tracks:
        return None

    # Step 2: If only ONE vehicle contains the plate -> preserve existing behavior exactly
    if len(intersecting_tracks) == 1:
        return intersecting_tracks[0]["track"]

    # Step 3: Multiple vehicles contain/intersect the plate -> resolve by strongest spatial ownership
    max_containment = max(item["containment"] for item in intersecting_tracks)
    
    # Filter candidates within tie_tolerance of the maximum containment
    tied_candidates = [
        item for item in intersecting_tracks
        if (max_containment - item["containment"]) <= tie_tolerance
    ]
    
    if len(tied_candidates) == 1:
        return tied_candidates[0]["track"]
        
    # Meaningful tie on containment: resolve by highest IoU
    best_candidate = max(
        tied_candidates,
        key=lambda x: (
            x["iou"],
            -float(x["track"].current_bbox.area),  # Tertiary: prefer more compact vehicle bbox
            x["track"].track_id,                  # Deterministic tie-breaker
        )
    )
    return best_candidate["track"]


class PlateTrackerAssociationManager:
    """
    Coordinates end-to-end ALPR execution on active vehicle tracks:
    TrackState → Vehicle Crop → Plate ROI Candidate → Preprocessing → OCR → Format Validation → Temporal Fusion → VehicleIdentityEvidence
    Supports configurable OCR scheduling, ROI size filtering, track-level caching, and attempt limits.
    """

    def __init__(
        self,
        ocr_engine: Optional[BaseOCREngine] = None,
        ocr_frame_interval: int = 5,
        ocr_min_plate_size: Tuple[int, int] = (40, 15),
        ocr_confidence_threshold: float = 0.60,
        ocr_cache_duration: float = 5.0,
        max_ocr_attempts_per_track: int = 10,
        plate_model_path: str = "backend/models/license_plate_detector.pt",
        device: str = "cpu",
    ):
        import torch

        self.device = device
        self.plate_detector = PlateDetector(model_path=plate_model_path, device=device)
        self.preprocessor = PlatePreprocessor()

        gpu = str(device).lower().startswith("cuda") and torch.cuda.is_available()
        print(f"[ALPR] OCR GPU: {gpu}")

        self.ocr_engine = ocr_engine or OCREngineFactory.create_engine(
            prefer_real=True,
            gpu=gpu,
        )
        self.fusion_engine = TemporalPlateFusionEngine()
        self.evidence_records: Dict[str, VehicleIdentityEvidence] = {}

        try:
            from app.perception.anpr_verification import ANPRVerifier
            self.verifier = ANPRVerifier()
        except Exception:
            self.verifier = None

        # Configurable ALPR / OCR parameters
        self.ocr_frame_interval = ocr_frame_interval
        self.ocr_min_plate_size = ocr_min_plate_size  # (min_width, min_height)
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.ocr_cache_duration = ocr_cache_duration  # seconds
        self.max_ocr_attempts_per_track = max_ocr_attempts_per_track

        # Track OCR metadata: track_id -> dict with attempts, last_ocr_frame, last_ocr_timestamp, last_accepted_plate, last_crop_sig
        self.track_ocr_stats: Dict[str, Dict[str, Any]] = {}

        # Claimed plates per frame: (track_id, plate_bbox, vehicle_bbox) to support vehicle-aware dedup
        self.frame_claimed_plates: Dict[int, List[Tuple[str, BoundingBoxXYXY, BoundingBoxXYXY]]] = {}

        # Detailed OCR scheduling & pipeline diagnostic statistics
        self.ocr_scheduling_stats = {
            "cache_skips": 0,
            "interval_skips": 0,
            "roi_quality_skips": 0,
            "confirmed_plate_skips": 0,
            "unchanged_roi_skips": 0,
            "max_attempts_skips": 0,
            "roi_too_small_skips": 0,
            "spatial_association_rejections": 0,
            "duplicate_plate_skips": 0,
            "actual_ocr_executions": 0,
            "vehicles_entering_plate_detection": 0,
            "plate_detector_calls": 0,
            "raw_candidates": 0,
            "accepted_candidates": 0,
            "rejected_candidates": 0,
            "plate_crops_created": 0,
            "ocr_calls": 0,
            "ocr_returned_text": 0,
            "ocr_returned_empty": 0,
            "normalization_rejected": 0,
            "validation_rejected": 0,
            "validation_rejections": 0,
            "validation_accepted": 0,
            "valid_ocr_observations": 0,
            "plate_observations_created": 0,
            "fusion_accepted": 0,
            "fusion_rejected": 0,
            "confirmed_plates": 0,
            "saved_debug_crops": 0,
        }

    def should_run_ocr(
        self,
        track: TrackState,
        frame_id: int,
        timestamp: float,
        v_bbox: BoundingBoxXYXY,
    ) -> Tuple[bool, str]:
        """
        Determines whether OCR should be performed for a track on the current frame.
        ONLY a genuinely CONFIRMED fused plate skips OCR while cache is fresh.
        PENDING plates continue collecting temporal evidence subject to interval and attempt limits.
        """
        # 1. Check ROI dimensions before attempting extraction/OCR
        # v_bbox is the VEHICLE bbox — must be large enough to contain a plate sub-region
        min_w, min_h = self.ocr_min_plate_size
        if v_bbox.width < min_w or v_bbox.height < min_h:
            self.ocr_scheduling_stats["roi_too_small_skips"] += 1
            return False, "ROI_TOO_SMALL"


        # 2. Get track OCR history
        stats = self.track_ocr_stats.get(track.track_id)
        if stats is None:
            # First observation of a new track: always run OCR
            return True, "NEW_TRACK"

        attempts = stats.get("attempts", 0)
        last_frame = stats.get("last_ocr_frame", -1)
        last_time = stats.get("last_ocr_timestamp", 0.0)
        cached_fused = self.fusion_engine.get_fused_identity(track.track_id)

        # 3. CRITICAL: Only cache-skip if fused status is strictly CONFIRMED.
        #    PENDING plates MUST continue collecting temporal evidence.
        #    Do NOT use overall_confidence >= threshold as a proxy for confirmed.
        if cached_fused:
            is_truly_confirmed = (
                cached_fused.status == "CONFIRMED"
                and cached_fused.confirmed is True
            )
            time_since_last_ocr = timestamp - last_time
            if is_truly_confirmed and (time_since_last_ocr < self.ocr_cache_duration or self.ocr_cache_duration <= 0):
                self.ocr_scheduling_stats["confirmed_plate_skips"] += 1
                self.ocr_scheduling_stats["cache_skips"] += 1
                return False, "CONFIRMED_PLATE_CACHED"

        # 4. Check max attempts budget per track
        if attempts >= self.max_ocr_attempts_per_track:
            # If track is unconfirmed (PENDING/UNKNOWN) and ROI is larger/closer, allow retry
            is_unconfirmed = cached_fused is None or not cached_fused.confirmed
            if is_unconfirmed and v_bbox.width >= 40:
                pass  # Allow retry for PENDING tracks with sufficient ROI size
            else:
                self.ocr_scheduling_stats["max_attempts_skips"] += 1
                return False, "MAX_ATTEMPTS_REACHED"

        # 5. Check configured OCR frame interval
        if last_frame >= 0 and (frame_id - last_frame) < self.ocr_frame_interval:
            self.ocr_scheduling_stats["interval_skips"] += 1
            return False, "INTERVAL_NOT_MET"

        return True, "INTERVAL_MET"


    def _compute_crop_sig(self, crop: Any) -> float:
        """Computes a fast intensity-mean signature for ROI crop change comparison."""
        try:
            import numpy as np
            if isinstance(crop, np.ndarray) and crop.size > 0:
                return float(np.mean(crop))
        except Exception:
            pass
        return 0.0

    def process_track_frame(
        self,
        track: TrackState,
        frame: Any,
        timestamp: float,
        frame_id: int,
        img_w: int = 1920,
        img_h: int = 1080,
        all_tracks: Optional[List[TrackState]] = None,
    ) -> VehicleIdentityEvidence:
        """
        Executes ALPR pipeline for a single track on the current frame with scheduled caching and
        spatial ownership verification across overlapping vehicles.
        """
        v_bbox = track.current_bbox
        active_tracks = all_tracks if all_tracks is not None else [track]

        # Increment per-track frame counter BEFORE the OCR scheduling gate.
        # This counter is used by the single-observation fusion fast-path to enforce
        # the track_length >= 10 guard — it must count ALL frames, not just OCR frames.
        if track.track_id not in self.track_ocr_stats:
            self.track_ocr_stats[track.track_id] = {
                "attempts": 0,
                "last_ocr_frame": -1,
                "last_ocr_timestamp": 0.0,
                "last_accepted_plate": None,
                "last_crop_sig": None,
                "frame_count": 0,
            }
        self.track_ocr_stats[track.track_id]["frame_count"] = (
            self.track_ocr_stats[track.track_id].get("frame_count", 0) + 1
        )

        run_ocr, reason = self.should_run_ocr(track, frame_id, timestamp, v_bbox)
        if not run_ocr:
            _diag(f"SKIP  frame={frame_id} track={track.track_id} reason={reason}")
        self.ocr_scheduling_stats["vehicles_entering_plate_detection"] += 1

        observation: Optional[PlateObservation] = None

        if run_ocr:
            self.ocr_scheduling_stats["plate_detector_calls"] += 1
            stats = self.track_ocr_stats[track.track_id]

            # 1. Extract vehicle crop
            success, vehicle_crop, v_bbox = self.plate_detector.extract_vehicle_crop(
                frame, track.current_bbox, img_w, img_h
            )

            if success and vehicle_crop is not None:
                # 2. Extract plate candidate ROIs
                candidates = self.plate_detector.extract_plate_candidates(vehicle_crop, v_bbox)
                self.ocr_scheduling_stats["raw_candidates"] += len(candidates)

                # Check if simulation ground-truth reference is available
                sim_plate_hint = None
                if track.raw_vehicle_reference:
                    parts = track.raw_vehicle_reference.split("_")
                    if len(parts) >= 3:
                        sim_plate_hint = parts[2]

                for plate_crop, p_bbox in candidates:
                    # Spatial Association Protection (Point F)
                    # 1. Resolve vehicle ownership when multiple active tracks exist
                    if len(active_tracks) > 1:
                        owner_track = resolve_plate_ownership(p_bbox, active_tracks)
                        if owner_track is not None and owner_track.track_id != track.track_id:
                            self.ocr_scheduling_stats["spatial_association_rejections"] = (
                                self.ocr_scheduling_stats.get("spatial_association_rejections", 0) + 1
                            )
                            _diag(
                                f"SPATIAL_ASSOC_REJECTED frame={frame_id} cand_track={track.track_id} "
                                f"owner_track={owner_track.track_id} plate_bbox=({p_bbox.x1:.1f},{p_bbox.y1:.1f},{p_bbox.x2:.1f},{p_bbox.y2:.1f})"
                            )
                            continue

                    # 2. Vehicle-aware duplicate suppression (Point F fix).
                    #
                    # A plate claim from another track is a true duplicate of THIS plate ONLY when:
                    #   a) The two plate bboxes strongly overlap (plate IoU > 0.60), indicating the
                    #      detector found the same physical plate region, AND
                    #   b) The two vehicle bboxes also overlap meaningfully (vehicle IoU > 0.10),
                    #      confirming the plates originate from the same spatial region of the frame.
                    #
                    # This prevents TRK_125-style suppression of TRK_126: adjacent vehicles with
                    # similarly-sized plate bboxes are NOT duplicates — their vehicle bboxes are
                    # spatially separate even if their plate bboxes appear close in size/position.
                    claimed_list = self.frame_claimed_plates.get(frame_id, [])
                    already_claimed = False
                    for claimed_tid, claimed_pbox, claimed_v_bbox in claimed_list:
                        if claimed_tid == track.track_id:
                            continue
                        # (a) Strong plate-bbox overlap required
                        plate_iou = compute_plate_vehicle_iou(p_bbox, claimed_pbox)
                        if plate_iou <= 0.60:
                            continue
                        # (b) Vehicle bboxes must also overlap (same spatial region)
                        vehicle_iou = compute_plate_vehicle_iou(v_bbox, claimed_v_bbox)
                        if vehicle_iou > 0.10:
                            already_claimed = True
                            _diag(
                                f"DUPLICATE_PLATE_SKIPPED frame={frame_id} track={track.track_id} "
                                f"claiming_track={claimed_tid} plate_iou={plate_iou:.2f} "
                                f"vehicle_iou={vehicle_iou:.2f}"
                            )
                            break

                    if already_claimed:
                        self.ocr_scheduling_stats["duplicate_plate_skips"] = (
                            self.ocr_scheduling_stats.get("duplicate_plate_skips", 0) + 1
                        )
                        continue

                    # Calculate ROI quality score (Quality Gating)
                    quality_score = self.plate_detector.calculate_roi_quality(plate_crop, p_bbox, img_w, img_h)

                    # Reject obviously unusable crops (raised from 0.10 → 0.25 to eliminate dark/flat false positives)
                    if quality_score < 0.25 and not sim_plate_hint:
                        self.ocr_scheduling_stats["roi_quality_skips"] += 1
                        self.ocr_scheduling_stats["rejected_candidates"] += 1
                        continue

                    self.ocr_scheduling_stats["accepted_candidates"] += 1
                    self.ocr_scheduling_stats["plate_crops_created"] += 1

                    # Check crop similarity against last OCR attempt for this track
                    current_sig = self._compute_crop_sig(plate_crop)
                    last_sig = stats.get("last_crop_sig")
                    cached_fused = self.fusion_engine.get_fused_identity(track.track_id)

                    if (
                        last_sig is not None
                        and abs(current_sig - last_sig) < 3.0   # Raised from 1.0: dark/uniform frames all have mean≈0
                        and current_sig > 5.0                   # Guard: skip identical-signature check on near-black crops
                        and cached_fused is not None
                        and cached_fused.best_plate_number
                        and not sim_plate_hint
                    ):
                        self.ocr_scheduling_stats["unchanged_roi_skips"] += 1
                        continue

                    stats["attempts"] += 1
                    stats["last_ocr_frame"] = frame_id
                    stats["last_ocr_timestamp"] = timestamp
                    stats["last_crop_sig"] = current_sig
                    self.ocr_scheduling_stats["actual_ocr_executions"] += 1
                    self.ocr_scheduling_stats["ocr_calls"] += 1

                    # Safely apply perspective correction if reliable corners exist
                    warped_crop, _ = self.plate_detector.correct_perspective(plate_crop)

                    # Save diagnostic plate crops — skip during pytest to prevent production directory pollution.
                    # ANPRVerifier.log_verification_sample() handles all verification crops.
                    import sys as _sys
                    if not ("pytest" in _sys.modules):
                        if not hasattr(self, "_diag_crop_count"):
                            self._diag_crop_count = 0
                        if self._diag_crop_count < 20 and hasattr(plate_crop, "shape") and len(plate_crop.shape) >= 2:
                            try:
                                import os, cv2
                                diag_dir = r"E:\chronoeye\data\diagnostic_crops"
                                os.makedirs(diag_dir, exist_ok=True)
                                self._diag_crop_count += 1
                                cv2.imwrite(os.path.join(diag_dir, f"plate_crop_{self._diag_crop_count:02d}.png"), plate_crop)
                            except Exception:
                                pass

                    # 3. Image Preprocessing & OCR Text Recognition
                    ocr_res = None
                    if not sim_plate_hint and self.preprocessor.is_likely_stacked_plate(plate_crop):
                        stacked_res = self.preprocessor.process_stacked_plate(plate_crop, self.ocr_engine)
                        if stacked_res is not None:
                            raw_text, norm_text, ocr_conf, variant_type = stacked_res
                            ocr_res = (raw_text, norm_text, ocr_conf, variant_type)

                    # Existing single-line plate processing unchanged (and fallback for unstacked / incomplete split)
                    if ocr_res is None:
                        variants = self.preprocessor.preprocess_plate_roi(warped_crop)
                        if sim_plate_hint:
                            variants.append({"sim_plate": sim_plate_hint})

                        # 4. OCR Text Recognition
                        res = self.ocr_engine.recognize_text(variants)
                        if len(res) == 4:
                            raw_text, norm_text, ocr_conf, variant_type = res
                        else:
                            raw_text, norm_text, ocr_conf = res[:3]
                            variant_type = "ORIGINAL"

                    # Format validation for verification logging
                    val_status, val_conf = IndianPlateValidator.validate_format(norm_text)
                    val_status_str = str(val_status.name if hasattr(val_status, 'name') else val_status)

                    # Log verification sample (data/diagnostic_crops/anpr_verification.csv)
                    if getattr(self, "verifier", None) is not None:
                        self.verifier.log_verification_sample(
                            crop_image=plate_crop,
                            frame_number=frame_id,
                            track_id=track.track_id,
                            timestamp=timestamp,
                            raw_ocr=raw_text,
                            normalized_ocr=norm_text,
                            ocr_confidence=ocr_conf,
                            plate_detector_confidence=1.0,
                            indian_format=val_status_str,
                            indian_validation_reason=f"val_conf={val_conf:.2f}",
                            temporal_consistency=0.0,
                        )

                    # Calculate crop dimensions for diagnostic trace
                    c_h, c_w = (plate_crop.shape[:2]) if (hasattr(plate_crop, "shape") and len(plate_crop.shape) >= 2) else (0, 0)
                    aspect = c_w / float(max(1, c_h))
                    vw_int, vh_int = int(v_bbox.width), int(v_bbox.height)

                    # Diagnostic accounting & logging: OCR response tracking
                    ocr_trace_msg = (
                        f"OCR_ATTEMPT frame={frame_id:04d} track={track.track_id} "
                        f"vehicle={vw_int}x{vh_int} crop={c_w}x{c_h} aspect={aspect:.2f} "
                        f"raw='{raw_text}' norm='{norm_text}' conf={ocr_conf:.3f} variant={variant_type}"
                    )
                    _diag(ocr_trace_msg)
                    # Always print OCR attempt trace to stderr for CLI real-video verification
                    import sys
                    print(f"[ANPR_OCR] {ocr_trace_msg}", file=sys.stderr, flush=True)

                    if raw_text and raw_text.strip():
                        self.ocr_scheduling_stats["ocr_returned_text"] += 1
                    else:
                        self.ocr_scheduling_stats["ocr_returned_empty"] += 1

                    # Generate normalized candidates (original + prefix-stripped secondary candidate)
                    from app.perception.ocr_normalizer import generate_normalized_plate_candidates
                    candidates = generate_normalized_plate_candidates(raw_text, norm_text)
                    if not candidates and norm_text:
                        candidates = [norm_text]

                    if candidates:
                        # Save a limited number of representative debug crop images (10-15 max) to outputs/anpr_debug/
                        if self.ocr_scheduling_stats["saved_debug_crops"] < 15:
                            try:
                                import os, cv2, numpy as np
                                os.makedirs("outputs/anpr_debug", exist_ok=True)
                                out_path = f"outputs/anpr_debug/frame_{frame_id:04d}_track_{track.track_id}_{candidates[0]}.jpg"
                                if isinstance(plate_crop, np.ndarray) and plate_crop.size > 0:
                                    cv2.imwrite(out_path, plate_crop)
                                    self.ocr_scheduling_stats["saved_debug_crops"] += 1
                            except Exception:
                                pass

                        # Evaluate candidates: prefer valid secondary candidate over prefix-corrupted primary
                        selected_cand_text = None
                        selected_val_status = None
                        selected_val_conf = 0.0
                        selected_variant_type = variant_type

                        for c_idx, cand_text in enumerate(candidates):
                            val_status, val_conf = IndianPlateValidator.validate_format(cand_text)
                            
                            if val_status == PlateValidationStatus.VALID:
                                self.ocr_scheduling_stats["indian_format_valid_count"] = self.ocr_scheduling_stats.get("indian_format_valid_count", 0) + 1
                                self.ocr_scheduling_stats["validation_accepted"] += 1
                                self.ocr_scheduling_stats["valid_ocr_observations"] += 1
                                self.ocr_scheduling_stats["plate_observations_created"] += 1
                            elif val_status in (PlateValidationStatus.UNCERTAIN, PlateValidationStatus.FORMAT_MISMATCH):
                                if val_status == PlateValidationStatus.FORMAT_MISMATCH:
                                    self.ocr_scheduling_stats["format_mismatch_count"] = self.ocr_scheduling_stats.get("format_mismatch_count", 0) + 1
                                self.ocr_scheduling_stats["validation_accepted"] += 1
                                self.ocr_scheduling_stats["valid_ocr_observations"] += 1
                                self.ocr_scheduling_stats["plate_observations_created"] += 1
                            else:
                                self.ocr_scheduling_stats["validation_rejected"] += 1
                                self.ocr_scheduling_stats["validation_rejections"] += 1
                                _diag(
                                    f"VALIDATION_REJECTED frame={frame_id} track={track.track_id}"
                                    f" raw='{raw_text}' cand='{cand_text}' val_conf={val_conf:.3f}"
                                )
                                continue

                            _diag(
                                f"VALIDATION_ACCEPTED frame={frame_id} track={track.track_id}"
                                f" raw='{raw_text}' cand='{cand_text}' val_status={val_status.value if hasattr(val_status, 'value') else val_status} val_conf={val_conf:.3f}"
                            )

                            # If secondary stripped candidate is available, prefer it over prefix-corrupted candidate
                            if c_idx > 0:
                                selected_cand_text = cand_text
                                selected_val_status = val_status
                                selected_val_conf = val_conf
                                selected_variant_type = f"{variant_type}+PREFIX_NORM"
                                break
                            elif selected_cand_text is None:
                                selected_cand_text = cand_text
                                selected_val_status = val_status
                                selected_val_conf = val_conf
                                selected_variant_type = str(variant_type)

                        if selected_cand_text is not None:
                            q_val = quality_score if quality_score > 0 else 0.5
                            v_val = selected_val_conf if selected_val_conf > 0 else 0.5
                            overall_conf = round(0.4 * ocr_conf + 0.3 * q_val + 0.3 * v_val, 3)

                            observation = PlateObservation(
                                track_id=track.track_id,
                                camera_id=track.camera_id,
                                frame_id=frame_id,
                                timestamp=timestamp,
                                bbox=p_bbox,
                                raw_text=raw_text,
                                normalized_text=selected_cand_text,
                                ocr_confidence=ocr_conf,
                                validation_confidence=selected_val_conf,
                                overall_confidence=overall_conf,
                                quality_score=quality_score,
                                preprocessing_variant=selected_variant_type,
                                status=selected_val_status,
                                source="ALPR_PIPELINE",
                            )
                            if frame_id not in self.frame_claimed_plates:
                                self.frame_claimed_plates[frame_id] = []
                            # Store (track_id, plate_bbox, vehicle_bbox) for vehicle-aware dedup
                            self.frame_claimed_plates[frame_id].append((track.track_id, p_bbox, v_bbox))
                            _diag(
                                f"OBSERVATION_CREATED frame={frame_id} track={track.track_id}"
                                f" raw='{raw_text}' norm='{selected_cand_text}' overall={overall_conf:.3f}"
                            )
                            break
                    elif raw_text and raw_text.strip():
                        self.ocr_scheduling_stats["normalization_rejected"] += 1
                        _diag(
                            f"NORMALIZATION_REJECTED frame={frame_id} track={track.track_id}"
                            f" raw='{raw_text}'"
                        )

        # 6. Temporal Fusion (retains existing fused identity if observation is None)
        # Pass track_frame_count so single-obs fast-path can enforce the >=10 frame guard.
        _track_frame_count = self.track_ocr_stats.get(track.track_id, {}).get("frame_count", 0)
        fused_identity = self.fusion_engine.process_observation(
            track.track_id, track.camera_id, observation, track_frame_count=_track_frame_count
        )
        if observation is not None:
            if fused_identity:
                self.ocr_scheduling_stats["fusion_accepted"] += 1
                _diag(
                    f"FUSION_ACCEPTED frame={frame_id} track={track.track_id}"
                    f" best='{fused_identity.best_plate_number}' status={fused_identity.status}"
                )
            else:
                self.ocr_scheduling_stats["fusion_rejected"] += 1
                self.ocr_scheduling_stats["fusion_rejections"] += 1
                _diag(f"FUSION_REJECTED frame={frame_id} track={track.track_id}")

        try:
            import json, os
            diag_log_path = r"E:\chronoeye\data\diagnostic_crops\valid_observations.jsonl"
            os.makedirs(os.path.dirname(diag_log_path), exist_ok=True)
            if observation is not None:
                obs_data = {
                    "event": "VALID_OBSERVATION",
                    "frame_number": frame_id,
                    "track_id": track.track_id,
                    "raw_text": observation.raw_text,
                    "normalized_text": observation.normalized_text,
                    "ocr_confidence": observation.ocr_confidence,
                    "validator_status": str(observation.status.name if hasattr(observation.status, 'name') else observation.status),
                    "validation_confidence": observation.validation_confidence,
                    "overall_confidence": observation.overall_confidence,
                    "fusion_status": str(getattr(fused_identity, 'status', 'NONE')) if fused_identity else "NONE",
                    "crop_path": f"outputs/anpr_debug/frame_{frame_id:04d}_track_{track.track_id}_{observation.normalized_text}.jpg"
                }
                with open(diag_log_path, "a") as f:
                    f.write(json.dumps(obs_data) + "\n")
            if fused_identity and (getattr(fused_identity, "status", "") == "CONFIRMED" or getattr(fused_identity, "confirmed", False)):
                prev_best = self.track_ocr_stats.get(track.track_id, {}).get("last_accepted_plate")
                was_confirmed = (getattr(prev_best, "status", "") == "CONFIRMED" or getattr(prev_best, "confirmed", False)) if prev_best else False
                if not was_confirmed:
                    conf_data = {
                        "event": "PLATE_CONFIRMED",
                        "frame_number": frame_id,
                        "track_id": track.track_id,
                        "confirmed_plate_text": fused_identity.best_plate_number,
                        "confirmation_confidence": getattr(fused_identity, "overall_confidence", getattr(fused_identity, "confidence", 1.0)),
                        "crop_path": f"outputs/anpr_debug/frame_{frame_id:04d}_track_{track.track_id}_{fused_identity.best_plate_number}.jpg" if fused_identity.best_plate_number else ""
                    }
                    with open(diag_log_path, "a") as f:
                        f.write(json.dumps(conf_data) + "\n")
        except Exception:
            pass

        if track.track_id in self.track_ocr_stats and fused_identity:
            self.track_ocr_stats[track.track_id]["last_accepted_plate"] = fused_identity

        # 7. Update Vehicle Identity Evidence
        evidence = VehicleIdentityEvidence(
            track_id=track.track_id,
            camera_id=track.camera_id,
            vehicle_type=track.vehicle_type,
            associated_plate=fused_identity,
            raw_vehicle_reference=track.raw_vehicle_reference,
            last_updated_timestamp=timestamp,
        )
        self.evidence_records[track.track_id] = evidence
        return evidence

    def get_identity_evidence(self, track_id: str) -> Optional[VehicleIdentityEvidence]:
        """Returns VehicleIdentityEvidence for a track_id."""
        return self.evidence_records.get(track_id)

    def process_frame_tracks(
        self,
        tracks: List[TrackState],
        frame: Any,
        timestamp: float,
        frame_id: int,
        img_w: int = 1920,
        img_h: int = 1080,
    ) -> Dict[str, VehicleIdentityEvidence]:
        """
        Executes ALPR pipeline for all active tracks on the current frame with multi-vehicle spatial association protection.
        """
        evidence_map: Dict[str, VehicleIdentityEvidence] = {}
        for track in tracks:
            evidence = self.process_track_frame(
                track=track,
                frame=frame,
                timestamp=timestamp,
                frame_id=frame_id,
                img_w=img_w,
                img_h=img_h,
                all_tracks=tracks,
            )
            evidence_map[track.track_id] = evidence
        return evidence_map


