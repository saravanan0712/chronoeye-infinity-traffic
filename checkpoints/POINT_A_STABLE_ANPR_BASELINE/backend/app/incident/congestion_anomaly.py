"""
ChronoEye Infinity - Phase 12: Congestion Anomaly Analyzer
Analyzes historical congestion baseline trends to identify unexpected non-recurrent congestion anomalies.
"""

from typing import List, Dict, Optional, Any
from app.state.traffic_state_schema import RoadSegmentState
from app.incident.incident_schema import IncidentEvent, IncidentType, IncidentSeverity, IncidentStatus


class CongestionAnomalyAnalyzer:
    """
    Congestion Anomaly Analyzer comparing real-time congestion against historical baseline expectations.
    """

    def __init__(self, congestion_surge_threshold: float = 0.45):
        self.congestion_surge_threshold = congestion_surge_threshold

    def analyze_congestion_anomaly(
        self,
        observed: RoadSegmentState,
        historical_baseline_congestion: float = 0.20,
        timestamp: float = 0.0,
        incident_id_counter: int = 300,
    ) -> Optional[IncidentEvent]:
        """
        Analyzes whether observed congestion score is an unexpected anomaly.
        """
        surge = observed.congestion_score - historical_baseline_congestion

        if surge >= self.congestion_surge_threshold:
            sev = IncidentSeverity.HIGH if surge >= 0.65 else IncidentSeverity.MEDIUM
            return IncidentEvent(
                incident_id=f"INC_{incident_id_counter}",
                segment_id=observed.segment_id,
                incident_type=IncidentType.UNEXPECTED_CONGESTION,
                severity=sev,
                confidence=min(1.0, round(surge, 2)),
                detected_at=timestamp or observed.timestamp,
                evidence={
                    "observed_congestion": observed.congestion_score,
                    "historical_baseline_congestion": historical_baseline_congestion,
                    "congestion_surge_delta": round(surge, 2),
                },
            )

        return None
