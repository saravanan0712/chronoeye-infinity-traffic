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
    ):
        self.plate_detector = PlateDetector(model_path=plate_model_path)
        self.preprocessor = PlatePreprocessor()
        self.ocr_engine = ocr_engine or OCREngineFactory.create_engine(prefer_real=True)
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

        # Detailed OCR scheduling & pipeline diagnostic statistics
        self.ocr_scheduling_stats = {
            "cache_skips": 0,
            "interval_skips": 0,
            "roi_quality_skips": 0,
            "confirmed_plate_skips": 0,
            "unchanged_roi_skips": 0,
            "max_attempts_skips": 0,
            "roi_too_small_skips": 0,
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
    ) -> VehicleIdentityEvidence:
        """
        Executes ALPR pipeline for a single track on the current frame with scheduled caching.
        """
        v_bbox = track.current_bbox
        run_ocr, reason = self.should_run_ocr(track, frame_id, timestamp, v_bbox)
        if not run_ocr:
            _diag(f"SKIP  frame={frame_id} track={track.track_id} reason={reason}")
        self.ocr_scheduling_stats["vehicles_entering_plate_detection"] += 1

        observation: Optional[PlateObservation] = None

        if run_ocr:
            self.ocr_scheduling_stats["plate_detector_calls"] += 1
            if track.track_id not in self.track_ocr_stats:
                self.track_ocr_stats[track.track_id] = {
                    "attempts": 0,
                    "last_ocr_frame": frame_id,
                    "last_ocr_timestamp": timestamp,
                    "last_accepted_plate": None,
                    "last_crop_sig": None,
                }

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

                    if norm_text:
                        # Save a limited number of representative debug crop images (10-15 max) to outputs/anpr_debug/
                        if self.ocr_scheduling_stats["saved_debug_crops"] < 15:
                            try:
                                import os, cv2, numpy as np
                                os.makedirs("outputs/anpr_debug", exist_ok=True)
                                out_path = f"outputs/anpr_debug/frame_{frame_id:04d}_track_{track.track_id}_{norm_text}.jpg"
                                if isinstance(plate_crop, np.ndarray) and plate_crop.size > 0:
                                    cv2.imwrite(out_path, plate_crop)
                                    self.ocr_scheduling_stats["saved_debug_crops"] += 1
                            except Exception:
                                pass

                        # 5. Format Validation
                        val_status, val_conf = IndianPlateValidator.validate_format(norm_text)
                        
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
                                f" raw='{raw_text}' norm='{norm_text}' val_conf={val_conf:.3f}"
                            )
                            continue  # Skip observation creation for rejected candidates
                            
                        # If we reach here, it is ACCEPTED (VALID, UNCERTAIN, or FORMAT_MISMATCH)
                        _diag(
                            f"VALIDATION_ACCEPTED frame={frame_id} track={track.track_id}"
                            f" raw='{raw_text}' norm='{norm_text}' val_status={val_status.value if hasattr(val_status, 'value') else val_status} val_conf={val_conf:.3f}"
                        )

                        q_val = quality_score if quality_score > 0 else 0.5
                        v_val = val_conf if val_conf > 0 else 0.5
                        overall_conf = round(0.4 * ocr_conf + 0.3 * q_val + 0.3 * v_val, 3)

                        observation = PlateObservation(
                            track_id=track.track_id,
                            camera_id=track.camera_id,
                            frame_id=frame_id,
                            timestamp=timestamp,
                            bbox=p_bbox,
                            raw_text=raw_text,
                            normalized_text=norm_text,
                            ocr_confidence=ocr_conf,
                            validation_confidence=val_conf,
                            overall_confidence=overall_conf,
                            quality_score=quality_score,
                            preprocessing_variant=str(variant_type),
                            status=val_status,
                            source="ALPR_PIPELINE",
                        )
                        _diag(
                            f"OBSERVATION_CREATED frame={frame_id} track={track.track_id}"
                            f" raw='{raw_text}' norm='{norm_text}' overall={overall_conf:.3f}"
                        )
                        break
                    elif raw_text and raw_text.strip():
                        self.ocr_scheduling_stats["normalization_rejected"] += 1
                        _diag(
                            f"NORMALIZATION_REJECTED frame={frame_id} track={track.track_id}"
                            f" raw='{raw_text}'"
                        )

        # 6. Temporal Fusion (retains existing fused identity if observation is None)
        fused_identity = self.fusion_engine.process_observation(
            track.track_id, track.camera_id, observation
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


