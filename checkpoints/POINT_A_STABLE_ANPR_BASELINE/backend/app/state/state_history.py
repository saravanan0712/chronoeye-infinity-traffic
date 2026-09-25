"""
ChronoEye Infinity - Phase 7: Traffic State History & Sequence Manager
Maintains bounded sliding-window history of network traffic snapshots (t0 -> t1 -> ... -> tN)
enforcing strict temporal ordering, no future information leakage, and forecasting sequence generation.
"""

from typing import Dict, List, Optional, Any
from app.state.traffic_state_schema import NetworkTrafficSnapshot, RoadSegmentState


class StateHistoryManager:
    """
    Bounded Bounded Sliding-Window Traffic State History Manager.
    Prepares temporal sequences for Phase 8 ST-GNN forecasting without future information leakage.
    """

    def __init__(self, max_history_length: int = 60, sampling_interval_seconds: float = 5.0):
        self.max_history_length = max_history_length
        self.sampling_interval_seconds = sampling_interval_seconds
        self.history: List[NetworkTrafficSnapshot] = []

    def add_snapshot(self, snapshot: NetworkTrafficSnapshot):
        """
        Appends snapshot to temporal history enforcing strict temporal ordering (t_new >= t_last).
        """
        if self.history and snapshot.timestamp < self.history[-1].timestamp:
            raise ValueError(
                f"Future information leakage or out-of-order timestamp detected: {snapshot.timestamp} < {self.history[-1].timestamp}"
            )

        self.history.append(snapshot)
        # Enforce bounded history window
        if len(self.history) > self.max_history_length:
            self.history.pop(0)

    def get_latest_snapshot(self) -> Optional[NetworkTrafficSnapshot]:
        """Retrieves most recent network traffic snapshot."""
        return self.history[-1] if self.history else None

    def get_recent_sequence(
        self, window_size: int = 12, segment_id: Optional[str] = None
    ) -> List[Any]:
        """
        Returns recent K historical snapshots or normalized feature vectors for a segment.
        """
        snaps = self.history[-window_size:] if self.history else []
        if segment_id is None:
            return snaps

        # Extract normalized feature vector sequence for segment_id
        seq = []
        for s in snaps:
            seg_state = s.segment_states.get(segment_id)
            if seg_state:
                seq.append(seg_state.normalized_features)
            else:
                seq.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Default empty road normalized vector
        return seq

    def get_history(self, limit: Optional[int] = None) -> List[NetworkTrafficSnapshot]:
        """Backward-compatible accessor returning the canonical history list."""
        if limit is not None and limit > 0:
            return self.history[-limit:]
        return self.history

    def to_dict(self) -> Dict[str, Any]:
        """Serializes history to JSON-compatible dictionary."""
        return {
            "max_history_length": self.max_history_length,
            "sampling_interval_seconds": self.sampling_interval_seconds,
            "total_snapshots": len(self.history),
            "snapshots": [s.model_dump() for s in self.history],
        }

