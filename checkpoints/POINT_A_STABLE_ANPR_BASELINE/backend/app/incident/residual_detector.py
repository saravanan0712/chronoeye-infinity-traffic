"""
ChronoEye Infinity - Phase 12: Residual Anomaly Detector
Detects traffic incidents from divergence residuals (observed_state - predicted_state).
"""

from typing import List, Dict, Optional, Tuple, Any
from app.state.traffic_state_schema import RoadSegmentState
from app.forecasting.forecasting_schema import SegmentForecast
from app.incident.incident_schema import IncidentEvent, IncidentType, IncidentSeverity, IncidentStatus


class ResidualAnomalyDetector:
    """
    Residual Anomaly Detector evaluating observed vs ST-GNN predicted traffic state divergence.
    """

    def __init__(
        self,
        speed_drop_threshold: float = 20.0,
        queue_surge_threshold: float = 8.0,
        flow_collapse_ratio: float = 0.25,
    ):
        self.speed_drop_threshold = speed_drop_threshold
        self.queue_surge_threshold = queue_surge_threshold
        self.flow_collapse_ratio = flow_collapse_ratio

    def detect_residuals(
        self,
        observed: RoadSegmentState,
        predicted: SegmentForecast,
        incident_id_counter: int = 100,
    ) -> List[IncidentEvent]:
        """
        Detects residual incidents comparing observed state against predicted forecast.
        """
        events: List[IncidentEvent] = []

        res_speed = predicted.predicted_speed - observed.average_speed
        res_queue = observed.queue_length - predicted.predicted_queue_length
        res_flow = predicted.predicted_flow - observed.flow_rate

        # 1. Abnormal Speed Drop Detection
        if res_speed >= self.speed_drop_threshold:
            sev = IncidentSeverity.HIGH if res_speed >= 35.0 else IncidentSeverity.MEDIUM
            events.append(
                IncidentEvent(
                    incident_id=f"INC_{incident_id_counter}",
                    segment_id=observed.segment_id,
                    incident_type=IncidentType.ABNORMAL_SPEED_DROP,
                    severity=sev,
                    confidence=min(1.0, round(res_speed / 40.0, 2)),
                    detected_at=observed.timestamp,
                    evidence={
                        "observed_speed": observed.average_speed,
                        "predicted_speed": predicted.predicted_speed,
                        "residual_speed_drop": round(res_speed, 2),
                    },
                )
            )

        # 2. Sudden Queue Growth Detection
        if res_queue >= self.queue_surge_threshold:
            sev = IncidentSeverity.CRITICAL if res_queue >= 20.0 else IncidentSeverity.HIGH
            events.append(
                IncidentEvent(
                    incident_id=f"INC_{incident_id_counter + 1}",
                    segment_id=observed.segment_id,
                    incident_type=IncidentType.SUDDEN_QUEUE_GROWTH,
                    severity=sev,
                    confidence=min(1.0, round(res_queue / 15.0, 2)),
                    detected_at=observed.timestamp,
                    evidence={
                        "observed_queue": observed.queue_length,
                        "predicted_queue": predicted.predicted_queue_length,
                        "residual_queue_growth": round(res_queue, 2),
                    },
                )
            )

        # 3. Flow Collapse Detection (High density, near-zero flow, near-zero speed)
        if (
            predicted.predicted_flow > 200.0
            and observed.flow_rate <= (predicted.predicted_flow * self.flow_collapse_ratio)
            and observed.density >= 35.0
        ):
            events.append(
                IncidentEvent(
                    incident_id=f"INC_{incident_id_counter + 2}",
                    segment_id=observed.segment_id,
                    incident_type=IncidentType.FLOW_COLLAPSE,
                    severity=IncidentSeverity.CRITICAL,
                    confidence=0.95,
                    detected_at=observed.timestamp,
                    evidence={
                        "observed_flow": observed.flow_rate,
                        "predicted_flow": predicted.predicted_flow,
                        "observed_density": observed.density,
                    },
                )
            )

        return events
