"""
ChronoEye Infinity - Module 2 Checkpoint Query Service
Provides deterministic, evidence-backed evaluation for vehicle checkpoint crossings:
- "Did global vehicle VEH_101 cross checkpoint CAM_B?"
- "Did plate TN09AB1111 cross checkpoint CAM_C?"

Strict Semantic Guarantees:
- OBSERVED: Direct, reliable observation exists at the specified checkpoint/camera.
- NOT_OBSERVED: No reliable observation exists in the recorded evidence/time window.
  (CRITICAL: NOT_OBSERVED does not mean physical non-crossing; it denotes absence of recorded evidence).
- UNKNOWN / INSUFFICIENT_EVIDENCE: Weak, conflicting, pending, or unobserved-gap evidence.
"""

from typing import Dict, List, Optional, Tuple, Any
import math

from app.schemas.reid import (
    VehicleJourney,
    JourneySegment,
    CheckpointObservationStatus,
    CheckpointQueryResult,
    MatchDecision,
)
from app.perception.plate_fusion import confusion_weighted_distance


def normalize_plate(plate: Optional[str]) -> str:
    """Normalize license plate text: uppercase, stripped whitespace and hyphens."""
    if not plate:
        return ""
    return plate.upper().replace(" ", "").replace("-", "").strip()


def matches_camera_id(seg_camera: str, query_checkpoint: str) -> bool:
    """
    Checks if a segment's camera_id matches the queried checkpoint_id,
    handling exact matches or directional aliases (e.g., 'CAM_A' vs 'CAM_A_EAST').
    """
    if not seg_camera or not query_checkpoint:
        return False
    seg_clean = seg_camera.upper().strip()
    query_clean = query_checkpoint.upper().strip()
    if seg_clean == query_clean:
        return True
    if seg_clean.startswith(f"{query_clean}_") or query_clean.startswith(f"{seg_clean}_"):
        return True
    return False


class CheckpointQueryService:
    """
    Query evaluator for multi-camera vehicle checkpoint crossings.
    """

    UNCERTAIN_DIST_THRESHOLD: float = 2.0

    @classmethod
    def query_checkpoint(
        cls,
        journey_engine: Any,
        checkpoint_id: str,
        vehicle_id: Optional[str] = None,
        plate_number: Optional[str] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
    ) -> CheckpointQueryResult:
        """
        Evaluates whether a target global vehicle ID or license plate crossed a checkpoint.
        """
        norm_target_plate = normalize_plate(plate_number) if plate_number else None
        target_veh_id = vehicle_id.upper().strip() if vehicle_id else None

        # 1. Search across all journeys
        all_journeys: List[VehicleJourney] = list(journey_engine.journeys.values())

        matched_journey: Optional[VehicleJourney] = None
        plate_match_uncertain: bool = False
        uncertain_reason: Optional[str] = None

        # Try matching by vehicle_id / journey_id first
        if target_veh_id:
            for j in all_journeys:
                if j.global_vehicle_id.upper() == target_veh_id or j.journey_id.upper() == target_veh_id:
                    matched_journey = j
                    break

        # If not found or if querying primarily by plate_number
        if matched_journey is None and norm_target_plate:
            # Pass 1: Exact confirmed plate match
            for j in all_journeys:
                j_plate = normalize_plate(j.plate_number)
                if j_plate and j_plate == norm_target_plate:
                    matched_journey = j
                    break

            # Pass 2: Exact plate match on any segment
            if matched_journey is None:
                for j in all_journeys:
                    for seg in j.segments:
                        seg_plate = normalize_plate(seg.plate_number)
                        if seg_plate and seg_plate == norm_target_plate:
                            matched_journey = j
                            if seg.plate_status in ("PENDING", "UNKNOWN"):
                                plate_match_uncertain = True
                                uncertain_reason = (
                                    f"Plate '{plate_number}' matched segment in track {seg.track_id} "
                                    f"with status={seg.plate_status} (conf={seg.plate_confidence or 0.0:.2f})."
                                )
                            break
                    if matched_journey is not None:
                        break

            # Pass 3: Confusion-weighted edit distance near match
            if matched_journey is None:
                best_dist = float("inf")
                best_near_j = None
                best_near_plate = ""
                for j in all_journeys:
                    candidates = [normalize_plate(j.plate_number)] + [
                        normalize_plate(seg.plate_number) for seg in j.segments if seg.plate_number
                    ]
                    for cand in candidates:
                        if not cand:
                            continue
                        dist = confusion_weighted_distance(norm_target_plate, cand)
                        if dist <= cls.UNCERTAIN_DIST_THRESHOLD and dist < best_dist:
                            best_dist = dist
                            best_near_j = j
                            best_near_plate = cand

                if best_near_j is not None:
                    matched_journey = best_near_j
                    plate_match_uncertain = True
                    uncertain_reason = (
                        f"Near-match plate '{best_near_plate}' found (confusion_dist={best_dist:.1f}). "
                        f"Cannot confirm exact identity match for '{plate_number}'."
                    )

        # 2. Case: Target vehicle/plate not found anywhere in journey records
        if matched_journey is None:
            return CheckpointQueryResult(
                query_type="CHECKPOINT_CROSSING",
                target_vehicle_id=vehicle_id,
                target_plate_number=plate_number,
                checkpoint_id=checkpoint_id,
                status=CheckpointObservationStatus.NOT_OBSERVED,
                timestamp=None,
                timestamp_uncertainty_seconds=None,
                confidence=0.0,
                plate_number=None,
                plate_status=None,
                journey_id=None,
                supporting_segments_count=0,
                has_unobserved_gap=False,
                evidence={},
                uncertainty={
                    "status_reason": (
                        "No reliable observation of that vehicle at that checkpoint exists "
                        "in the available evidence/time window. Note: This indicates absence "
                        "of recorded evidence, not proof of non-crossing."
                    )
                },
                provenance={
                    "source_id": "JOURNEY_RECONSTRUCTION_ENGINE",
                    "process_name": "CheckpointQueryService",
                    "confidence": 0.0,
                },
            )

        # 3. Case: Target journey identified — evaluate segments for checkpoint observation
        matching_segments = [
            seg for seg in matched_journey.segments
            if matches_camera_id(seg.camera_id, checkpoint_id)
        ]

        # Filter by time window if specified
        if time_start is not None or time_end is not None:
            matching_segments = [
                seg for seg in matching_segments
                if (time_start is None or seg.timestamp >= time_start)
                and (time_end is None or seg.timestamp <= time_end)
            ]

        # 4. Checkpoint was OBSERVED (or UNCERTAIN due to plate/decision status)
        if matching_segments:
            best_seg = max(matching_segments, key=lambda s: (s.plate_confidence or 0.5, s.transition_score or 0.5))

            is_uncertain = (
                plate_match_uncertain
                or best_seg.plate_status in ("PENDING", "UNKNOWN")
                or best_seg.transition_decision == MatchDecision.INSUFFICIENT_EVIDENCE
            )

            if not uncertain_reason and best_seg.plate_status in ("PENDING", "UNKNOWN"):
                uncertain_reason = (
                    f"Observation at checkpoint {best_seg.camera_id} has plate status={best_seg.plate_status} "
                    f"(conf={best_seg.plate_confidence or 0.0:.2f})."
                )

            status = (
                CheckpointObservationStatus.UNKNOWN
                if is_uncertain
                else CheckpointObservationStatus.OBSERVED
            )

            breakdown_dict = {}
            if best_seg.transition_breakdown:
                tb = best_seg.transition_breakdown
                breakdown_dict = tb.model_dump() if hasattr(tb, "model_dump") else dict(tb)

            evidence_dict = {
                "track_id": best_seg.track_id,
                "camera_id": best_seg.camera_id,
                "direction": best_seg.direction,
                "speed_estimate_kmh": best_seg.speed_estimate,
                "visual_features": best_seg.visual_features or {},
                "transition_score": best_seg.transition_score,
                "transition_decision": (
                    best_seg.transition_decision.value
                    if hasattr(best_seg.transition_decision, "value")
                    else best_seg.transition_decision
                ),
                "transition_breakdown": breakdown_dict,
                "bounding_box": best_seg.bbox.model_dump() if best_seg.bbox else None,
            }

            reason = (
                uncertain_reason
                if uncertain_reason
                else f"Reliable observation confirmed at checkpoint {best_seg.camera_id}."
            )

            conf = best_seg.plate_confidence or best_seg.transition_score or matched_journey.overall_confidence

            return CheckpointQueryResult(
                query_type="CHECKPOINT_CROSSING",
                target_vehicle_id=matched_journey.global_vehicle_id,
                target_plate_number=plate_number or matched_journey.plate_number,
                checkpoint_id=checkpoint_id,
                status=status,
                timestamp=best_seg.timestamp,
                timestamp_uncertainty_seconds=best_seg.timestamp_uncertainty_seconds,
                confidence=round(conf, 3),
                plate_number=best_seg.plate_number or matched_journey.plate_number,
                plate_status=best_seg.plate_status or "CONFIRMED",
                journey_id=matched_journey.journey_id,
                supporting_segments_count=len(matching_segments),
                has_unobserved_gap=best_seg.has_unobserved_gap,
                evidence=evidence_dict,
                uncertainty={
                    "status_reason": reason,
                    "unobserved_gap": best_seg.has_unobserved_gap,
                    "timestamp_uncertainty_seconds": best_seg.timestamp_uncertainty_seconds,
                },
                provenance={
                    "source_id": "JOURNEY_RECONSTRUCTION_ENGINE",
                    "process_name": "CheckpointQueryService",
                    "confidence": round(conf, 3),
                },
            )

        # 5. Checkpoint was NOT directly observed in this journey
        # Check if there is an unobserved gap across visited cameras
        has_gap = any(seg.has_unobserved_gap for seg in matched_journey.segments)
        
        # Check if vehicle was observed before and after this camera in sequence
        cameras_visited = matched_journey.cameras

        status_reason = (
            f"No reliable observation of vehicle {matched_journey.global_vehicle_id} at checkpoint {checkpoint_id} "
            f"exists in recorded evidence. Visited cameras: {cameras_visited}. "
            "Note: This indicates absence of recorded evidence, not proof of non-crossing."
        )

        status = CheckpointObservationStatus.NOT_OBSERVED
        if has_gap:
            status_reason += " Vehicle trajectory contains unobserved intermediate gaps."

        return CheckpointQueryResult(
            query_type="CHECKPOINT_CROSSING",
            target_vehicle_id=matched_journey.global_vehicle_id,
            target_plate_number=plate_number or matched_journey.plate_number,
            checkpoint_id=checkpoint_id,
            status=status,
            timestamp=None,
            timestamp_uncertainty_seconds=None,
            confidence=0.0,
            plate_number=matched_journey.plate_number,
            plate_status=None,
            journey_id=matched_journey.journey_id,
            supporting_segments_count=0,
            has_unobserved_gap=has_gap,
            evidence={
                "cameras_visited": cameras_visited,
                "first_seen": matched_journey.first_seen,
                "last_seen": matched_journey.last_seen,
            },
            uncertainty={
                "status_reason": status_reason,
                "unobserved_gap": has_gap,
            },
            provenance={
                "source_id": "JOURNEY_RECONSTRUCTION_ENGINE",
                "process_name": "CheckpointQueryService",
                "confidence": 0.0,
            },
        )
