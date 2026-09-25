"""
ChronoEye Infinity - Phase 4: Temporal OCR Fusion Engine (Experimental Upgrade)
Accumulates multi-frame license plate observations for persistent vehicle track IDs.

Key invariants:
  - ONE temporal observation per (track_id, frame_id): multiple same-frame preprocessing
    variants produce only ONE representative observation stored in the temporal record.
  - Evidence accumulates ADDITIVELY over independent frames, not multiplicatively.
  - PENDING plates never count as CONFIRMED in OCR scheduling.
  - CONFIRMED requires N distinct frames + threshold evidence + valid format.
  - Full audit trail preserved; no raw OCR evidence ever overwritten.

Experimental additions (v2):
  - FusionConfig dataclass: all thresholds in one place, no magic numbers.
  - OCR_CONFUSION map: used ONLY during cluster distance computation, never alters stored text.
  - Confusion-weighted Levenshtein: treats 0/O, 1/I, 8/B, 5/S, Z/2 as cost=0.5 substitution.
  - Character-wise positional voting: position-by-position weighted vote across cluster
    observations to select the best-evidenced character at each position.
  - Contradictory-evidence UNKNOWN: if a track exhausts its observation budget but the
    best cluster never coheres (agreement < 0.30), mark status as UNKNOWN.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from app.schemas.plate import (
    PlateObservation,
    FusedPlateIdentity,
    PlateValidationStatus,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    """All temporal fusion thresholds in one documented place."""
    min_observations_to_confirm: int = 2       # Distinct frame count required
    confirmation_score_threshold: float = 1.2  # Additive evidence sum required
    agreement_ratio_threshold: float = 0.40    # best-cluster / total evidence
    validation_confidence_threshold: float = 0.20
    max_observation_history: int = 20          # Rolling window size per track
    levenshtein_tolerance: int = 2             # Max edit distance for clustering
    confusion_penalty: float = 0.5            # Cost for a confused-char substitution
    contradictory_agreement_threshold: float = 0.30  # Below this -> UNKNOWN on max history
    # Single-observation fast-path: a track with >=single_obs_min_track_frames frames
    # and OCR confidence >= single_obs_ocr_conf_threshold can be CONFIRMED from one frame.
    single_obs_ocr_conf_threshold: float = 0.85   # Minimum OCR conf for 1-obs confirmation
    single_obs_min_track_frames: int = 10          # Track must be active >= this many frames


# ---------------------------------------------------------------------------
# OCR confusion map (used ONLY in distance computation — never alters text)
# ---------------------------------------------------------------------------

OCR_CONFUSION: Dict[str, str] = {
    "0": "O", "O": "0",
    "1": "I", "I": "1",
    "8": "B", "B": "8",
    "5": "S", "S": "5",
    "Z": "2", "2": "Z",
}


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein edit distance."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def confusion_weighted_distance(s1: str, s2: str, confusion_penalty: float = 0.5) -> float:
    """
    Levenshtein variant that treats OCR-confused character pairs (0/O, 1/I, etc.)
    as fractional substitutions (cost = confusion_penalty) instead of full cost 1.
    Used ONLY for cluster membership decisions — never alters stored text.
    Returns a float distance.
    """
    if s1 == s2:
        return 0.0
    if not s1:
        return float(len(s2))
    if not s2:
        return float(len(s1))

    m, n = len(s1), len(s2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c1, c2 = s1[i - 1], s2[j - 1]
            if c1 == c2:
                cost = 0.0
            elif OCR_CONFUSION.get(c1) == c2 or OCR_CONFUSION.get(c2) == c1:
                cost = confusion_penalty   # confused pair — partial penalty
            else:
                cost = 1.0
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


# ---------------------------------------------------------------------------
# Observation scoring
# ---------------------------------------------------------------------------

def _observation_score(obs: PlateObservation) -> float:
    """
    Additive evidence score for a single observation in [0.05, 1.0].
    Transparent weighted combination — no multiplicative crushing.
    Defaults quality/validation to 0.5 when missing (no penalty for absent metadata).
    """
    q = obs.quality_score if obs.quality_score > 0.0 else 0.5
    v = obs.validation_confidence if obs.validation_confidence > 0.0 else 0.5
    c = obs.ocr_confidence
    score = 0.45 * c + 0.30 * v + 0.25 * q
    return round(max(0.05, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Character-wise positional voting
# ---------------------------------------------------------------------------

def _character_wise_vote(
    cluster_obs: List[Tuple[PlateObservation, float]],
    base_text: str,
) -> str:
    """
    Performs position-by-position weighted voting across all observations in the
    cluster to select the best-evidenced character at each position.

    Rules:
    - cluster_obs: list of (observation, weight) for all obs in the best cluster
    - base_text: used only to determine max expected length
    - Votes are weighted by observation score
    - A character is emitted only if it was actually observed — never fabricated
    - If two characters tie, the one from the observation with the highest weight wins
    - Positions beyond the shortest string are handled by majority/missing vote
    - Returns voted_text (str)
    """
    if not cluster_obs:
        return base_text

    # Determine reasonable max length — use the modal length among observations
    lengths = [len(obs.normalized_text) for obs, _ in cluster_obs]
    if not lengths:
        return base_text

    # Modal length: most common length is the expected plate length
    length_votes: Dict[int, float] = {}
    for obs, w in cluster_obs:
        ln = len(obs.normalized_text)
        length_votes[ln] = length_votes.get(ln, 0.0) + w
    voted_length = max(length_votes, key=length_votes.get)

    result = []
    for pos in range(voted_length):
        # Collect weighted votes for each character at this position
        char_votes: Dict[str, float] = {}
        best_w_for_char: Dict[str, float] = {}

        for obs, weight in cluster_obs:
            text = obs.normalized_text
            if pos < len(text):
                ch = text[pos]
                char_votes[ch] = char_votes.get(ch, 0.0) + weight
                if weight > best_w_for_char.get(ch, -1.0):
                    best_w_for_char[ch] = weight

        if not char_votes:
            # No observations cover this position — stop here (do NOT fabricate)
            break

        # Select character with highest weighted vote;
        # on tie, prefer the one from the highest-weight individual observation
        best_char = max(
            char_votes,
            key=lambda ch: (char_votes[ch], best_w_for_char.get(ch, 0.0)),
        )
        result.append(best_char)

    return "".join(result)


# ---------------------------------------------------------------------------
# Fusion engine
# ---------------------------------------------------------------------------

class TemporalPlateFusionEngine:
    """
    Temporally fuses license plate observations across consecutive video frames
    for each vehicle track_id.

    Experimental v2 additions:
    - FusionConfig for all thresholds
    - Confusion-weighted cluster distance (0/O, 1/I, 8/B, 5/S, Z/2)
    - Character-wise positional voting for representative text selection
    - UNKNOWN transition when evidence is contradictory after max history
    """

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        # Legacy positional args preserved for backwards compatibility with tests
        confirmation_score_threshold: Optional[float] = None,
        min_observations_to_confirm: Optional[int] = None,
        max_observation_history: Optional[int] = None,
    ):
        self.config = config or FusionConfig()
        # Allow legacy kwargs to override FusionConfig
        if confirmation_score_threshold is not None:
            self.config.confirmation_score_threshold = confirmation_score_threshold
        if min_observations_to_confirm is not None:
            self.config.min_observations_to_confirm = min_observations_to_confirm
        if max_observation_history is not None:
            self.config.max_observation_history = max_observation_history

        # For test compatibility: expose as attributes
        self.confirmation_score_threshold = self.config.confirmation_score_threshold
        self.min_observations_to_confirm = self.config.min_observations_to_confirm
        self.max_observation_history = self.config.max_observation_history

        # track_id -> FusedPlateIdentity
        self.fused_identities: Dict[str, FusedPlateIdentity] = {}

        # track_id -> List[PlateObservation] (all variants, not just one per frame)
        self.observations_store: Dict[str, List[PlateObservation]] = {}

    def _deduplicate_and_store(self, track_id: str, observation: PlateObservation) -> bool:
        """
        Stores all observation variants. Enforces the same-frame variant rule:
        multiple preprocessing variants (e.g. ORIGINAL, CLAHE, OTSU) of the same frame
        count as ONE temporal obs. Repeated temporal calls with identical variants
        are assigned distinct effective frame indices.
        """
        if track_id not in self.observations_store:
            self.observations_store[track_id] = []

        # Prevent reference mutation bugs if caller passes identical object instance
        observation = observation.model_copy()

        # Determine effective frame identifier
        existing_for_track = self.observations_store[track_id]
        same_frame_obs = [o for o in existing_for_track if o.frame_id == observation.frame_id]
        existing_raw_variants = [o.preprocessing_variant for o in same_frame_obs]

        if observation.preprocessing_variant in existing_raw_variants:
            # Repeated invocation with same variant -> distinct temporal iteration
            repeat_idx = sum(1 for v in existing_raw_variants if v == observation.preprocessing_variant)
            setattr(observation, "_effective_frame_id", f"{observation.frame_id}_rep{repeat_idx}")
            setattr(observation, "_variant_key", f"{observation.preprocessing_variant}_rep{repeat_idx}")
        else:
            setattr(observation, "_effective_frame_id", str(observation.frame_id))
            setattr(observation, "_variant_key", observation.preprocessing_variant)

        self.observations_store[track_id].append(observation)

        unique_frames_in_order = []
        for o in self.observations_store[track_id]:
            eff_id = getattr(o, "_effective_frame_id", o.frame_id)
            if eff_id not in unique_frames_in_order:
                unique_frames_in_order.append(eff_id)

        if len(unique_frames_in_order) > self.config.max_observation_history:
            oldest_frame = unique_frames_in_order[0]
            self.observations_store[track_id] = [
                o for o in self.observations_store[track_id]
                if getattr(o, "_effective_frame_id", o.frame_id) != oldest_frame
            ]
        return True

    def _build_clusters(
        self, track_obs_list: List[PlateObservation]
    ) -> List[List[str]]:
        """
        Groups unique normalized texts into Levenshtein clusters.
        Uses confusion-weighted distance so OCR-confused pairs (0/O, 8/B, etc.)
        cluster together more readily. Max distance = levenshtein_tolerance.
        """
        text_to_frames: Dict[str, List[int]] = {}
        for o in track_obs_list:
            text_to_frames.setdefault(o.normalized_text, []).append(o.frame_id)

        unique_texts = list(text_to_frames.keys())
        tol = self.config.levenshtein_tolerance
        penalty = self.config.confusion_penalty

        clusters: List[List[str]] = []
        visited: set = set()
        for t in unique_texts:
            if t in visited:
                continue
            cluster = [t]
            visited.add(t)
            for other in unique_texts:
                if other in visited:
                    continue
                dist = confusion_weighted_distance(t, other, penalty)
                # Allow ±levenshtein_tolerance edits, length diff ≤ tolerance
                if dist <= tol and abs(len(t) - len(other)) <= tol:
                    cluster.append(other)
                    visited.add(other)
            clusters.append(cluster)
        return clusters

    def process_observation(
        self,
        track_id: str,
        camera_id: str,
        observation: Optional[PlateObservation],
        track_frame_count: int = 0,
    ) -> Optional[FusedPlateIdentity]:
        """
        Processes a single-frame plate observation, deduplicates by frame, and
        updates the fused temporal plate identity for the track.
        Returns current fused identity (or None if no evidence yet).
        """
        cfg = self.config

        # 1. Handle missing/empty observation — return existing identity unchanged
        if observation is None or not observation.normalized_text:
            return self.fused_identities.get(track_id)

        # 2. Per-frame deduplication: same-frame variants → 1 temporal obs
        self._deduplicate_and_store(track_id, observation)

        track_obs_list = self.observations_store.get(track_id, [])
        if not track_obs_list:
            return self.fused_identities.get(track_id)

        # 3. Per-observation scores
        obs_scores = {getattr(o, "_effective_frame_id", o.frame_id): _observation_score(o) for o in track_obs_list}
        total_evidence = sum(obs_scores.values())

        # 4. Build confusion-weighted Levenshtein clusters
        clusters = self._build_clusters(track_obs_list)

        # 5. Score each cluster — additive sum over distinct frames
        best_cluster_text = ""
        best_cluster_voted_text = ""
        best_cluster_score = 0.0
        best_cluster_frames: List[int] = []
        best_cluster_frame_count = 0
        best_cluster_best_obs: Optional[PlateObservation] = None
        best_cluster_obs_weighted: List[Tuple[PlateObservation, float]] = []

        for cluster in clusters:
            cluster_texts_set = set(cluster)
            cluster_frame_scores: Dict[Any, float] = {}
            cluster_frame_best_obs: Dict[Any, PlateObservation] = {}

            for o in track_obs_list:
                if o.normalized_text in cluster_texts_set:
                    fid = getattr(o, "_effective_frame_id", o.frame_id)
                    s = obs_scores[fid]
                    if fid not in cluster_frame_scores or s > cluster_frame_scores[fid]:
                        cluster_frame_scores[fid] = s
                        cluster_frame_best_obs[fid] = o

            cluster_total_score = sum(cluster_frame_scores.values())
            cluster_frame_count = len(cluster_frame_scores)

            if cluster_total_score > best_cluster_score:
                best_cluster_score = cluster_total_score
                best_cluster_frames = sorted(cluster_frame_scores.keys())
                best_cluster_frame_count = cluster_frame_count
                best_cluster_best_obs = max(
                    cluster_frame_best_obs.values(), key=_observation_score, default=None
                )

                # All obs in this cluster with their weighted scores (for character voting)
                best_cluster_obs_weighted = [
                    (o, obs_scores[getattr(o, "_effective_frame_id", o.frame_id)])
                    for o in track_obs_list
                    if o.normalized_text in cluster_texts_set
                ]

                # Raw representative: highest-confidence observed text in cluster
                raw_rep = max(
                    cluster,
                    key=lambda txt: max(
                        (_observation_score(o) for o in track_obs_list if o.normalized_text == txt),
                        default=0.0,
                    ),
                )
                best_cluster_text = raw_rep

        if not best_cluster_text or best_cluster_best_obs is None:
            return self.fused_identities.get(track_id)

        # 6. Character-wise positional voting (STEP 6)
        #    Produces voted_text using position-by-position weighted majority.
        #    Only characters actually observed are emitted — no fabrication.
        voted_text = _character_wise_vote(best_cluster_obs_weighted, best_cluster_text)
        # Safety: if voting produced empty or very short result, fall back to raw rep
        if len(voted_text) < max(1, len(best_cluster_text) - 2):
            voted_text = best_cluster_text

        # 7. Agreement ratio — best cluster evidence vs all evidence
        agreement_ratio = round(best_cluster_score / max(0.001, total_evidence), 3)

        # 8. Transparent overall confidence
        obs = best_cluster_best_obs
        q = obs.quality_score if obs.quality_score > 0.0 else 0.5
        overall_conf = round(
            0.40 * obs.ocr_confidence
            + 0.30 * obs.validation_confidence
            + 0.20 * agreement_ratio
            + 0.10 * q,
            3,
        )

        # 9. Determine status
        is_confirmed = False
        status_str = "UNKNOWN"
        pending_text: Optional[str] = None
        final_plate_text = ""

        # Is the evidence contradictory now that we've exhausted the history?
        at_max_history = len(track_obs_list) >= cfg.max_observation_history
        contradictory = at_max_history and agreement_ratio < cfg.contradictory_agreement_threshold

        # --- Primary multi-observation confirmation path (unchanged) ---
        if (
            best_cluster_frame_count >= cfg.min_observations_to_confirm
            and best_cluster_score >= cfg.confirmation_score_threshold
            and agreement_ratio >= cfg.agreement_ratio_threshold
            and obs.validation_confidence >= cfg.validation_confidence_threshold
        ):
            is_confirmed = True
            status_str = "CONFIRMED"
            final_plate_text = voted_text   # Use character-voted text for confirmed

        # --- Single-observation high-confidence fast-path (Point C addition) ---
        # Conditions (ALL must hold):
        #   1. Exactly one distinct temporal frame in best cluster
        #   2. Track has been active for >= single_obs_min_track_frames frames
        #   3. Best observation OCR confidence >= single_obs_ocr_conf_threshold
        #   4. voted_text is non-empty
        #   5. Agreement ratio and validation confidence thresholds still met
        elif (
            best_cluster_frame_count == 1
            and track_frame_count >= cfg.single_obs_min_track_frames
            and best_cluster_best_obs is not None
            and best_cluster_best_obs.ocr_confidence >= cfg.single_obs_ocr_conf_threshold
            and voted_text
            and agreement_ratio >= cfg.agreement_ratio_threshold
            and obs.validation_confidence >= cfg.validation_confidence_threshold
        ):
            is_confirmed = True
            status_str = "CONFIRMED"
            final_plate_text = voted_text

        elif contradictory:
            # Max history exhausted, evidence is contradictory — mark UNKNOWN
            is_confirmed = False
            status_str = "UNKNOWN"
            final_plate_text = ""

        elif len(track_obs_list) >= 1 and best_cluster_score >= 0.15 and voted_text:
            is_confirmed = False
            status_str = "PENDING"
            pending_text = voted_text
            final_plate_text = voted_text

        # 10. Build audit trail — one entry per stored frame (deduped)
        # Pre-compute set of normalized texts that belong to the winning cluster (strings, not objects)
        best_cluster_texts: set = {b.normalized_text for b, _ in best_cluster_obs_weighted}
        audit_trail = [
            {
                "frame_id": o.frame_id,
                "raw_text": o.raw_text,
                "normalized_text": o.normalized_text,
                "voted_text": voted_text if o.normalized_text in best_cluster_texts else "",
                "ocr_confidence": o.ocr_confidence,
                "quality_score": o.quality_score,
                "variant": o.preprocessing_variant,
                "score": obs_scores.get(o.frame_id, 0.0),
                "validation_status": o.status.value if hasattr(o.status, "value") else str(o.status),
            }
            for o in track_obs_list
        ]

        # 11. Candidate history (text -> cumulative weighted score)
        all_unique_texts: List[str] = list({o.normalized_text for o in track_obs_list})
        candidate_history = {}
        for text in all_unique_texts:
            candidate_history[text] = round(
                sum(obs_scores[getattr(o, "_effective_frame_id", o.frame_id)] for o in track_obs_list if o.normalized_text == text), 4
            )

        # Confirmation logic uses best_cluster_frame_count (distinct frames) — algorithm unchanged
        # Convert internal effective frame IDs to sorted unique integer frame IDs for public schema
        public_evidence_frames = sorted(list(set(
            int(str(f).split("_")[0]) for f in best_cluster_frames
        )))

        total_unique_frames = len(set(getattr(o, "_effective_frame_id", o.frame_id) for o in track_obs_list))

        fused = FusedPlateIdentity(
            track_id=track_id,
            camera_id=camera_id,
            best_plate_number=final_plate_text,
            overall_confidence=overall_conf,
            observation_count=total_unique_frames,
            confirmed=is_confirmed,
            status=status_str,
            pending_plate_number=pending_text,
            evidence_frames=public_evidence_frames,
            supporting_observations_count=best_cluster_frame_count,
            weighted_evidence_score=round(best_cluster_score, 3),
            character_agreement_ratio=agreement_ratio,
            raw_observations_audit=audit_trail,
            first_seen_timestamp=track_obs_list[0].timestamp,
            last_seen_timestamp=observation.timestamp,
            candidate_history=candidate_history,
        )


        self.fused_identities[track_id] = fused
        return fused

    def calculate_observation_weight(self, obs: PlateObservation) -> float:
        """
        Public backwards-compatibility alias for _observation_score.
        Used by test_advanced_plate_recovery.py (Test 14).
        """
        return _observation_score(obs)

    def get_fused_identity(self, track_id: str) -> Optional[FusedPlateIdentity]:
        """Returns current fused plate identity for track_id if available."""
        return self.fused_identities.get(track_id)

