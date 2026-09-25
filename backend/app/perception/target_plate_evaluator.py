"""
ChronoEye Infinity - Target Plate Evaluator
Evaluates whether a specific target licence plate was reliably observed by a camera/video.

Returns one of three states:
  DETECTED     - target plate has an EXACT match against a CONFIRMED fused identity
  NOT_DETECTED - no relevant candidate found
  UNCERTAIN    - weak/conflicting/near candidate exists but cannot be confirmed

Matching rules (strict):
  DETECTED  : exact normalized match against a CONFIRMED fused identity.
  UNCERTAIN : exact normalized match against a PENDING fused identity, OR
              confusion-weighted Levenshtein distance <= 2 against any fused identity
              (covers OCR prefix errors like VN472 vs VN4712, WG219 vs WG2119).
  NOT_DETECTED : no match at all.

"Crossed camera" = the target vehicle was reliably observed by this camera/video.
No GPS or location data is invented.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from app.perception.plate_fusion import confusion_weighted_distance, TemporalPlateFusionEngine


def _normalize_plate(plate: str) -> str:
    """Normalize a plate string: uppercase, strip spaces/hyphens."""
    return plate.upper().replace(" ", "").replace("-", "").strip()


@dataclass
class TargetPlateResult:
    """Result of a target plate evaluation query."""
    target_plate: str           # Normalized target supplied by user
    status: str                 # "DETECTED", "NOT_DETECTED", "UNCERTAIN"

    # Populated when DETECTED or UNCERTAIN
    track_id: Optional[str] = None
    first_frame: Optional[int] = None
    best_frame: Optional[int] = None
    video_timestamp: Optional[float] = None
    recognized_plate: Optional[str] = None
    ocr_confidence: Optional[float] = None
    fusion_confidence: Optional[float] = None
    supporting_observations: int = 0
    fused_status: Optional[str] = None
    uncertain_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_plate": self.target_plate,
            "status": self.status,
            "track_id": self.track_id,
            "first_frame": self.first_frame,
            "best_frame": self.best_frame,
            "timestamp": self.video_timestamp,
            "recognized_plate": self.recognized_plate,
            "ocr_confidence": self.ocr_confidence,
            "fusion_confidence": self.fusion_confidence,
            "supporting_observations": self.supporting_observations,
        }


class TargetPlateEvaluator:
    """
    Evaluates whether a target plate was reliably observed by this camera.

    Usage:
        evaluator = TargetPlateEvaluator()
        result = evaluator.evaluate("SX8525", fusion_engine)
        evaluator.print_report(result)
    """

    # Maximum confusion-weighted edit distance for UNCERTAIN near-match classification
    UNCERTAIN_DIST_THRESHOLD: float = 2.0

    def evaluate(
        self,
        target_plate: str,
        fusion_engine: TemporalPlateFusionEngine,
    ) -> TargetPlateResult:
        """
        Evaluate target_plate against all fused identities in fusion_engine.

        Returns TargetPlateResult with status DETECTED / NOT_DETECTED / UNCERTAIN.
        """
        norm_target = _normalize_plate(target_plate)
        identities = list(fusion_engine.fused_identities.values())

        # ----------------------------------------------------------------
        # Pass 1: EXACT match against CONFIRMED identities -> DETECTED
        # ----------------------------------------------------------------
        for fused in identities:
            if not (fused.confirmed and fused.status == "CONFIRMED"):
                continue
            candidate = _normalize_plate(fused.best_plate_number or "")
            if candidate == norm_target:
                best_obs = _best_obs_from_audit(fused)
                return TargetPlateResult(
                    target_plate=norm_target,
                    status="DETECTED",
                    track_id=fused.track_id,
                    first_frame=_first_frame(fused),
                    best_frame=_best_frame_from_audit(fused),
                    video_timestamp=fused.first_seen_timestamp,
                    recognized_plate=fused.best_plate_number,
                    ocr_confidence=best_obs.get("ocr_confidence"),
                    fusion_confidence=fused.overall_confidence,
                    supporting_observations=fused.supporting_observations_count,
                    fused_status="CONFIRMED",
                )

        # ----------------------------------------------------------------
        # Pass 2: EXACT match against PENDING identities -> UNCERTAIN
        # ----------------------------------------------------------------
        for fused in identities:
            if fused.confirmed or fused.status == "CONFIRMED":
                continue
            candidate_text = _normalize_plate(
                fused.best_plate_number or fused.pending_plate_number or ""
            )
            if not candidate_text:
                continue
            if candidate_text == norm_target:
                best_obs = _best_obs_from_audit(fused)
                return TargetPlateResult(
                    target_plate=norm_target,
                    status="UNCERTAIN",
                    track_id=fused.track_id,
                    first_frame=_first_frame(fused),
                    best_frame=_best_frame_from_audit(fused),
                    video_timestamp=fused.first_seen_timestamp,
                    recognized_plate=candidate_text,
                    ocr_confidence=best_obs.get("ocr_confidence"),
                    fusion_confidence=fused.overall_confidence,
                    supporting_observations=fused.supporting_observations_count,
                    fused_status=fused.status,
                    uncertain_reason=(
                        f"Plate '{candidate_text}' was recognized by OCR "
                        f"(track {fused.track_id}, status={fused.status}) "
                        f"but did not reach CONFIRMED — only {fused.observation_count} "
                        f"observation(s), fusion conf={fused.overall_confidence:.3f}."
                    ),
                )

        # ----------------------------------------------------------------
        # Pass 3: Confusion-weighted near-match against ANY identity -> UNCERTAIN
        # Covers OCR prefix/suffix errors: VN472 vs VN4712, WG219 vs WG2119, etc.
        # ----------------------------------------------------------------
        best_near = None
        best_near_dist: float = float("inf")
        best_near_plate: str = ""

        for fused in identities:
            for plate_text in [fused.best_plate_number, fused.pending_plate_number]:
                if not plate_text:
                    continue
                candidate = _normalize_plate(plate_text)
                if not candidate:
                    continue
                dist = confusion_weighted_distance(norm_target, candidate)
                if dist <= self.UNCERTAIN_DIST_THRESHOLD and dist < best_near_dist:
                    best_near_dist = dist
                    best_near = fused
                    best_near_plate = candidate

        if best_near is not None:
            best_obs = _best_obs_from_audit(best_near)
            return TargetPlateResult(
                target_plate=norm_target,
                status="UNCERTAIN",
                track_id=best_near.track_id,
                first_frame=_first_frame(best_near),
                best_frame=_best_frame_from_audit(best_near),
                video_timestamp=best_near.first_seen_timestamp,
                recognized_plate=best_near_plate,
                ocr_confidence=best_obs.get("ocr_confidence"),
                fusion_confidence=best_near.overall_confidence,
                supporting_observations=best_near.supporting_observations_count,
                fused_status=best_near.status,
                uncertain_reason=(
                    f"Near-match: '{best_near_plate}' in track {best_near.track_id} "
                    f"(status={best_near.status}, confusion_dist={best_near_dist:.1f}) — "
                    f"OCR or prefix confusion may explain the difference. "
                    f"Cannot confirm this is exactly the target plate."
                ),
            )

        # ----------------------------------------------------------------
        # Pass 4: No match -> NOT_DETECTED
        # ----------------------------------------------------------------
        return TargetPlateResult(
            target_plate=norm_target,
            status="NOT_DETECTED",
        )

    def print_report(self, result: TargetPlateResult) -> None:
        """Print the formatted target plate evaluation report to stdout."""
        print()
        print("=" * 60)
        print("  TARGET PLATE EVALUATION")
        print("=" * 60)
        print(f"  TARGET PLATE : {result.target_plate}")
        print(f"  STATUS       : {result.status}")
        print()

        if result.status == "DETECTED":
            print(f"  Track ID               : {result.track_id}")
            print(f"  First detected frame   : {result.first_frame}")
            print(f"  Best frame             : {result.best_frame}")
            ts = f"{result.video_timestamp:.2f}s" if result.video_timestamp is not None else "N/A"
            print(f"  Video timestamp        : {ts}")
            print(f"  Recognized plate       : {result.recognized_plate}")
            oconf = f"{result.ocr_confidence:.3f}" if result.ocr_confidence is not None else "N/A"
            fconf = f"{result.fusion_confidence:.3f}" if result.fusion_confidence is not None else "N/A"
            print(f"  OCR confidence         : {oconf}")
            print(f"  Fusion confidence      : {fconf}")
            print(f"  Supporting observations: {result.supporting_observations}")
            print()
            print("  RESULT: TARGET VEHICLE CROSSED CAMERA = YES")

        elif result.status == "NOT_DETECTED":
            print(f"  No candidate matching '{result.target_plate}' was found in any track.")
            print()
            print("  RESULT: TARGET VEHICLE CROSSED CAMERA = NO")

        elif result.status == "UNCERTAIN":
            print(f"  Track ID               : {result.track_id}")
            print(f"  First detected frame   : {result.first_frame}")
            print(f"  Best frame             : {result.best_frame}")
            print(f"  Recognized plate       : {result.recognized_plate}")
            oconf = f"{result.ocr_confidence:.3f}" if result.ocr_confidence is not None else "N/A"
            fconf = f"{result.fusion_confidence:.3f}" if result.fusion_confidence is not None else "N/A"
            print(f"  OCR confidence         : {oconf}")
            print(f"  Fusion confidence      : {fconf}")
            print(f"  Supporting observations: {result.supporting_observations}")
            print(f"  Fusion status          : {result.fused_status}")
            print()
            print(f"  Reason: {result.uncertain_reason}")
            print()
            print("  RESULT: TARGET VEHICLE CROSSED CAMERA = UNCERTAIN")

        print("=" * 60)
        print()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_frame(fused) -> Optional[int]:
    """Return the first evidence frame from a fused identity."""
    if fused.evidence_frames:
        return min(fused.evidence_frames)
    if fused.raw_observations_audit:
        frames = [o.get("frame_id") for o in fused.raw_observations_audit
                  if o.get("frame_id") is not None]
        return min(frames) if frames else None
    return None


def _best_frame_from_audit(fused) -> Optional[int]:
    """Return the frame with the highest OCR confidence from the audit trail."""
    if not fused.raw_observations_audit:
        return _first_frame(fused)
    best = max(
        fused.raw_observations_audit,
        key=lambda o: o.get("ocr_confidence", 0.0),
        default=None,
    )
    return best.get("frame_id") if best else None


def _best_obs_from_audit(fused) -> Dict[str, Any]:
    """Return the observation dict with the highest OCR confidence."""
    if not fused.raw_observations_audit:
        return {}
    return max(
        fused.raw_observations_audit,
        key=lambda o: o.get("ocr_confidence", 0.0),
        default={},
    )
