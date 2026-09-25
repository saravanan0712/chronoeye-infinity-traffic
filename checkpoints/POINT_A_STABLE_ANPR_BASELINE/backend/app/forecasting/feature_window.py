"""
ChronoEye Infinity - Phase 8: Feature Window Extractor
Extracts sliding lag windows from historical traffic state snapshots and flattens/imputes feature matrices.
"""

from typing import List, Dict, Optional, Any
from app.state.traffic_state_schema import NetworkTrafficSnapshot


class FeatureWindowExtractor:
    """
    Sliding Lag Window Extractor for Traffic Forecasting.
    """

    @staticmethod
    def extract_lag_windows(
        history: List[NetworkTrafficSnapshot],
        window_size: int = 12,
        segment_id: str = "ROAD_R_AB",
    ) -> List[List[float]]:
        """
        Extracts sliding window sequence of length W feature vectors for segment_id.
        Each feature vector: [norm_flow, norm_density, norm_speed, occupancy, norm_queue, congestion_score].
        Fills missing historical steps with neutral default features [0.0, 0.0, 0.5, 0.0, 0.0, 0.0].
        """
        raw_seq = []
        for snapshot in history[-window_size:]:
            seg_state = snapshot.segment_states.get(segment_id)
            if seg_state and seg_state.normalized_features:
                raw_seq.append(seg_state.normalized_features)
            else:
                raw_seq.append([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])

        # Impute missing steps if history has fewer than window_size elements
        missing_count = max(0, window_size - len(raw_seq))
        default_feat = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
        imputed_seq = [default_feat for _ in range(missing_count)] + raw_seq

        return imputed_seq

    @staticmethod
    def flatten_window(window: List[List[float]]) -> List[float]:
        """
        Flattens W x F feature matrix into 1D vector of length W * F.
        """
        flat = []
        for step in window:
            flat.extend(step)
        return flat
