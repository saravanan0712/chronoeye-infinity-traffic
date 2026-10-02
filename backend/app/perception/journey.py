"""
ChronoEye Infinity - Phase 5: Spatio-Temporal Journey Reconstruction Engine
Performs optimal cross-camera track matching, assigns global vehicle identities (VEH_001),
and constructs continuous multi-camera vehicle journeys (JRN_001).
"""

from typing import Dict, List, Optional, Tuple, Any
from app.schemas.tracking import TrackState
from app.schemas.plate import VehicleIdentityEvidence
from app.schemas.reid import (
    VehicleJourney,
    JourneySegment,
    MatchDecision,
    ReIDScoreBreakdown,
    ReIDConfig,
)
from app.perception.reid_engine import ReIDMatchingEngine
from app.perception.camera_topology import CityCameraTopology
from app.perception.appearance import AppearanceEmbeddingExtractor


class JourneyReconstructionEngine:
    """
    Spatio-Temporal Vehicle Journey Reconstruction Engine.
    Reconciles camera-local track IDs into global vehicle identities and continuous journeys.
    Includes appearance embedding caching and direct track segment updating.
    """

    def __init__(
        self,
        matching_engine: Optional[ReIDMatchingEngine] = None,
        config: Optional[ReIDConfig] = None,
        embedding_refresh_interval: float = 10.0,
        journey_timeout_seconds: Optional[float] = None,
        temporal_graph_engine: Optional[Any] = None,
    ):
        self.config = config or ReIDConfig()
        self.matching_engine = matching_engine or ReIDMatchingEngine(config=self.config)
        self.journeys: Dict[str, VehicleJourney] = {}
        self.track_to_journey_map: Dict[str, str] = {}  # track_id -> journey_id
        self.track_embeddings: Dict[str, Tuple[float, List[float]]] = {}  # track_id -> (timestamp, embedding)
        self.embedding_refresh_interval = embedding_refresh_interval
        self.journey_timeout_seconds = (
            journey_timeout_seconds
            if journey_timeout_seconds is not None
            else getattr(self.config, "journey_timeout_seconds", 60.0)
        )
        self.temporal_graph_engine = temporal_graph_engine
        self.next_vehicle_number = 101

    def retire_completed_journeys(self, current_timestamp: float) -> List[str]:
        """
        Marks active journeys as COMPLETED if no observation was received
        within the configured journey_timeout_seconds.
        """
        completed = []
        for j_id, journey in self.journeys.items():
            if journey.status == "ACTIVE":
                if (current_timestamp - journey.last_seen) > self.journey_timeout_seconds:
                    journey.status = "COMPLETED"
                    completed.append(j_id)
        return completed

    def _get_or_extract_embedding(self, frame: Any, track: TrackState, timestamp: float) -> List[float]:
        """Retrieves cached appearance embedding or extracts a new one if expired/absent."""
        if track.track_id in self.track_embeddings:
            last_t, cached_emb = self.track_embeddings[track.track_id]
            if timestamp - last_t < self.embedding_refresh_interval:
                return cached_emb

        vehicle_crop = frame
        if frame is not None:
            try:
                import numpy as np
                if isinstance(frame, np.ndarray) and frame.size > 0:
                    x1 = max(0, int(track.current_bbox.x1))
                    y1 = max(0, int(track.current_bbox.y1))
                    x2 = min(frame.shape[1], int(track.current_bbox.x2))
                    y2 = min(frame.shape[0], int(track.current_bbox.y2))
                    if x2 > x1 and y2 > y1:
                        vehicle_crop = frame[y1:y2, x1:x2]
            except Exception:
                pass

        emb = AppearanceEmbeddingExtractor.extract_embedding(vehicle_crop, track)
        self.track_embeddings[track.track_id] = (timestamp, emb)
        return emb

    def process_track_evidence(
        self,
        evidence: VehicleIdentityEvidence,
        track: TrackState,
        frame: Any = None,
    ) -> VehicleJourney:
        """
        Processes new track evidence from a camera and matches/assigns it to an existing journey
        or initializes a new global vehicle journey.
        """
        # 0. Retire inactive journeys older than timeout
        self.retire_completed_journeys(evidence.last_updated_timestamp)

        # 1. If local track is already assigned to a journey, directly append segment using cached embedding
        if track.track_id in self.track_to_journey_map:
            j_id = self.track_to_journey_map[track.track_id]
            journey = self.journeys[j_id]
            if journey.status == "ACTIVE":
                embedding = self._get_or_extract_embedding(frame, track, evidence.last_updated_timestamp)
                self._append_segment_to_journey(
                    journey,
                    evidence,
                    track,
                    frame,
                    embedding,
                    transition_decision=MatchDecision.MATCH_CONFIRMED,
                    transition_score=1.0,
                    has_unobserved_gap=False,
                )
                return journey

        # 2. For unassigned/new tracks: Extract/cache appearance embedding
        embedding = self._get_or_extract_embedding(frame, track, evidence.last_updated_timestamp)

        # 3. Candidate Generation & Scoring against existing active journeys
        best_journey: Optional[VehicleJourney] = None
        best_score = 0.0
        best_decision = MatchDecision.INSUFFICIENT_EVIDENCE
        best_breakdown: Optional[ReIDScoreBreakdown] = None

        for journey in self.journeys.values():
            if journey.status != "ACTIVE":
                continue

            # Disallow matching to same camera if co-present in the same timestamp (simultaneous distinct tracks)
            if journey.segments and journey.segments[-1].camera_id == evidence.camera_id:
                if evidence.last_updated_timestamp <= journey.segments[-1].timestamp:
                    continue

            # Extract latest segment metadata for comparison
            last_seg = journey.segments[-1]
            dummy_evidence_b = VehicleIdentityEvidence(
                track_id=last_seg.track_id,
                camera_id=last_seg.camera_id,
                vehicle_type=journey.vehicle_type,
                associated_plate=None,
                last_updated_timestamp=last_seg.timestamp,
            )
            if journey.plate_number:
                from app.schemas.plate import FusedPlateIdentity
                dummy_evidence_b.associated_plate = FusedPlateIdentity(
                    track_id=last_seg.track_id,
                    camera_id=last_seg.camera_id,
                    best_plate_number=journey.plate_number,
                    overall_confidence=0.9,
                    first_seen_timestamp=last_seg.timestamp,
                    last_seen_timestamp=last_seg.timestamp,
                )

            dummy_track_b = TrackState(
                track_id=last_seg.track_id,
                camera_id=last_seg.camera_id,
                vehicle_type=journey.vehicle_type,
                current_bbox=last_seg.bbox,
                current_center=last_seg.bbox.center if last_seg.bbox else (0.0, 0.0),
                confidence=0.9,
                first_seen_timestamp=last_seg.timestamp,
                last_seen_timestamp=last_seg.timestamp,
            )

            breakdown = self.matching_engine.compute_match_score(
                evidence,
                track,
                dummy_evidence_b,
                dummy_track_b,
                embedding,
                last_seg.appearance_embedding,
                visual_features_b=last_seg.visual_features,
            )

            if breakdown.decision in [MatchDecision.MATCH_CONFIRMED, MatchDecision.MATCH_PROBABLE]:
                if breakdown.overall_score > best_score:
                    best_score = breakdown.overall_score
                    best_journey = journey
                    best_decision = breakdown.decision
                    best_breakdown = breakdown

        # 4. Assignment Decision
        if best_journey is not None and best_score >= self.config.probable_threshold:
            # Match confirmed/probable: merge track into existing journey
            self.track_to_journey_map[track.track_id] = best_journey.journey_id
            has_gap = False
            if best_journey.segments:
                prev_cam = best_journey.segments[-1].camera_id
                if prev_cam != evidence.camera_id:
                    has_gap = not self.matching_engine.topology.is_directly_connected(prev_cam, evidence.camera_id)

            self._append_segment_to_journey(
                best_journey,
                evidence,
                track,
                frame,
                embedding,
                breakdown=best_breakdown,
                transition_decision=best_decision,
                transition_score=best_score,
                has_unobserved_gap=has_gap,
            )
            if evidence.associated_plate and not best_journey.plate_number:
                best_journey.plate_number = evidence.associated_plate.best_plate_number
            return best_journey

        # 5. No match: Create new Global Vehicle Identity & Journey
        g_veh_id = f"VEH_{self.next_vehicle_number:03d}"
        j_id = f"JRN_{self.next_vehicle_number:03d}"
        self.next_vehicle_number += 1

        plate_str = evidence.associated_plate.best_plate_number if evidence.associated_plate else None

        new_journey = VehicleJourney(
            journey_id=j_id,
            global_vehicle_id=g_veh_id,
            plate_number=plate_str,
            vehicle_type=evidence.vehicle_type,
            first_seen=evidence.last_updated_timestamp,
            last_seen=evidence.last_updated_timestamp,
            cameras=[evidence.camera_id],
            overall_confidence=1.0,
            status="ACTIVE",
        )
        self.journeys[j_id] = new_journey
        self.track_to_journey_map[track.track_id] = j_id
        self._append_segment_to_journey(
            new_journey,
            evidence,
            track,
            frame,
            embedding,
            transition_decision=None,
            transition_score=None,
            breakdown=None,
            has_unobserved_gap=False,
        )

        return new_journey

    def _append_segment_to_journey(
        self,
        journey: VehicleJourney,
        evidence: VehicleIdentityEvidence,
        track: TrackState,
        frame: Any = None,
        embedding: Optional[List[float]] = None,
        visual_features: Optional[Dict[str, float]] = None,
        breakdown: Optional[ReIDScoreBreakdown] = None,
        transition_decision: Optional[MatchDecision] = None,
        transition_score: Optional[float] = None,
        has_unobserved_gap: bool = False,
    ):
        if embedding is None:
            embedding = self._get_or_extract_embedding(frame, track, evidence.last_updated_timestamp)

        if visual_features is None:
            bbox = getattr(track, "current_bbox", getattr(track, "bbox", None))
            visual_features = self.matching_engine.extract_visual_features(bbox, track)

        # Pool plate evidence under global vehicle identity
        plate_str = None
        plate_conf = None
        plate_stat = None
        if evidence.associated_plate:
            if evidence.associated_plate.best_plate_number:
                plate_str = evidence.associated_plate.best_plate_number
            elif evidence.associated_plate.pending_plate_number:
                plate_str = evidence.associated_plate.pending_plate_number

            plate_conf = evidence.associated_plate.overall_confidence
            plate_stat = evidence.associated_plate.status

        dir_str = track.direction.value if track.direction else "UNKNOWN"

        seg = JourneySegment(
            camera_id=evidence.camera_id,
            track_id=track.track_id,
            timestamp=evidence.last_updated_timestamp,
            timestamp_uncertainty_seconds=getattr(evidence, "timestamp_uncertainty_seconds", None),
            bbox=track.current_bbox,
            speed_estimate=track.speed_estimate,
            direction=dir_str,
            plate_number=plate_str,
            plate_confidence=plate_conf,
            plate_status=plate_stat,
            appearance_embedding=embedding,
            visual_features=visual_features,
            transition_decision=transition_decision,
            transition_score=transition_score,
            transition_breakdown=breakdown,
            has_unobserved_gap=has_unobserved_gap,
        )

        journey.segments.append(seg)
        journey.last_seen = evidence.last_updated_timestamp
        if evidence.camera_id not in journey.cameras:
            journey.cameras.append(evidence.camera_id)

        # Prioritize confirmed or best pooled plate string
        if plate_str and (not journey.plate_number or (evidence.associated_plate and evidence.associated_plate.confirmed)):
            journey.plate_number = plate_str

        if self.temporal_graph_engine is not None:
            self.temporal_graph_engine.update_journey(journey)

    def get_journey_for_track(self, track_id: str) -> Optional[VehicleJourney]:
        """Retrieves journey object associated with track_id."""
        j_id = self.track_to_journey_map.get(track_id)
        if j_id:
            return self.journeys.get(j_id)
        return None

    def add_segment(self, *args, **kwargs) -> Any:
        """
        Backward-compatible public wrapper for both process_track_evidence() and legacy test calls.
        Older callers that relied on `add_segment(evidence, track)` are routed through
        the canonical process_track_evidence() — no separate journey storage.
        """
        if (len(args) > 0 and hasattr(args[0], "vehicle_type")) or "evidence" in kwargs:
            return self.process_track_evidence(*args, **kwargs)

        # Legacy test signature: trk_id, cam_id, enter_ts, exit_ts, global_veh_id
        trk = args[0] if len(args) > 0 else kwargs.get("track_id")
        cam = args[1] if len(args) > 1 else kwargs.get("camera_id")
        start = args[2] if len(args) > 2 else kwargs.get("start")
        end = args[3] if len(args) > 3 else kwargs.get("end")
        g_id = args[4] if len(args) > 4 else kwargs.get("global_veh_id")

        j_id = self.track_to_journey_map.get(trk)
        if not j_id:
            j_id = f"JRN_{g_id}"
            if j_id not in self.journeys:
                self.journeys[j_id] = VehicleJourney(
                    journey_id=j_id, global_vehicle_id=g_id, vehicle_type="car", first_seen=start or 0.0, last_seen=end or 0.0
                )
            self.track_to_journey_map[trk] = j_id

        journey = self.journeys[j_id]
        journey.segments.append(
            JourneySegment(
                track_id=trk,
                camera_id=cam,
                timestamp=end or 0.0,
                timestamp_uncertainty_seconds=kwargs.get("timestamp_uncertainty_seconds"),
                bbox=kwargs.get("bbox"),
                plate_number=kwargs.get("plate_number"),
                plate_confidence=kwargs.get("plate_confidence"),
                plate_status=kwargs.get("plate_status"),
                transition_decision=kwargs.get("transition_decision"),
                transition_score=kwargs.get("transition_score"),
                transition_breakdown=kwargs.get("transition_breakdown"),
                has_unobserved_gap=kwargs.get("has_unobserved_gap", False),
            )
        )
        if end and end > journey.last_seen:
            journey.last_seen = end

        if self.temporal_graph_engine is not None:
            self.temporal_graph_engine.update_journey(journey)

        return journey

    def update_journeys(self, *args, **kwargs) -> Any:
        """
        Backward-compatible delegate method for update_journeys.
        If the first arg is a dictionary (fused_map), iterates and updates plate numbers on journeys.
        """
        if len(args) > 0 and isinstance(args[0], dict):
            fused_map = args[0]
            for track_id, fused_id in fused_map.items():
                j_id = self.track_to_journey_map.get(track_id)
                if j_id and j_id in self.journeys:
                    journey = self.journeys[j_id]
                    # Update plate number if currently none or if new identity is confirmed
                    if not journey.plate_number or getattr(fused_id, "confirmed", False):
                        if hasattr(fused_id, "best_plate_number"):
                            journey.plate_number = fused_id.best_plate_number
            return

        return self.add_segment(*args, **kwargs)

    def get_active_journeys(self) -> List[VehicleJourney]:
        """Returns a list of all active multi-camera journeys."""
        return [j for j in self.journeys.values() if j.status == "ACTIVE"]

    def get_completed_journeys(self) -> List[VehicleJourney]:
        """Returns a list of all completed multi-camera journeys."""
        return [j for j in self.journeys.values() if j.status in ("COMPLETED", "COMPLETE")]

    def get_journey_summary_report(self) -> List[Dict[str, Any]]:
        """
        Generates a concise, structured summary report for all reconstructed journeys.
        Exposes global_vehicle_id, journey_id, status, pooled plate, segment count,
        camera sequence, local track IDs, first timestamp, and last timestamp.
        """
        report = []
        for journey in self.journeys.values():
            local_track_ids = list(dict.fromkeys(seg.track_id for seg in journey.segments))
            report.append({
                "global_vehicle_id": journey.global_vehicle_id,
                "journey_id": journey.journey_id,
                "status": journey.status,
                "plate_number": journey.plate_number or "UNKNOWN",
                "vehicle_type": journey.vehicle_type,
                "number_of_segments": len(journey.segments),
                "camera_sequence": list(journey.cameras),
                "local_track_ids": local_track_ids,
                "first_timestamp": round(journey.first_seen, 3),
                "last_timestamp": round(journey.last_seen, 3),
            })
        return report

    def query_checkpoint(
        self,
        checkpoint_id: str,
        vehicle_id: Optional[str] = None,
        plate_number: Optional[str] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
    ) -> Any:
        """
        Queries whether a vehicle (by global_vehicle_id or license plate) crossed a checkpoint.
        Returns structured CheckpointQueryResult with OBSERVED / NOT_OBSERVED / UNKNOWN verdict.
        """
        from app.perception.checkpoint_query import CheckpointQueryService
        return CheckpointQueryService.query_checkpoint(
            journey_engine=self,
            checkpoint_id=checkpoint_id,
            vehicle_id=vehicle_id,
            plate_number=plate_number,
            time_start=time_start,
            time_end=time_end,
        )

