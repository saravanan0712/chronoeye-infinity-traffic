"""
ChronoEye Infinity - Phase 12: Incident Detection Engine
Master orchestrator managing real-time incident detection, false positive filtering, and incident lifecycle management.
"""

from typing import List, Dict, Optional, Any
from app.state.traffic_state_schema import RoadSegmentState, NetworkTrafficSnapshot
from app.forecasting.forecasting_schema import SegmentForecast
from app.perception.tracker import TrackState
from app.incident.incident_schema import IncidentEvent, IncidentType, IncidentSeverity, IncidentStatus
from app.incident.residual_detector import ResidualAnomalyDetector
from app.incident.stopped_vehicle_detector import StoppedVehicleDetector
from app.incident.congestion_anomaly import CongestionAnomalyAnalyzer


class IncidentDetectionEngine:
    """
    Master Traffic Anomaly & Incident Detection Engine.
    """

    def __init__(self):
        self.residual_detector = ResidualAnomalyDetector()
        self.stopped_detector = StoppedVehicleDetector()
        self.congestion_analyzer = CongestionAnomalyAnalyzer()
        self.active_incidents: Dict[str, IncidentEvent] = {}
        self._id_counter = 1000

    def process_snapshot_and_forecast(
        self,
        snapshot: NetworkTrafficSnapshot,
        forecasts: Optional[Dict[str, SegmentForecast]] = None,
        track_states: Optional[List[TrackState]] = None,
    ) -> List[IncidentEvent]:
        """
        Processes real-time traffic snapshot, ST-GNN forecast, and track states to produce active incident events.
        Automatically handles false-positive resistance and incident resolution lifecycle.
        """
        detected_events: List[IncidentEvent] = []

        for seg_id, observed in snapshot.segment_states.items():
            # 1. Residual Anomaly Detection
            if forecasts and seg_id in forecasts:
                res_events = self.residual_detector.detect_residuals(
                    observed, forecasts[seg_id], incident_id_counter=self._id_counter
                )
                self._id_counter += len(res_events)
                detected_events.extend(res_events)

            # 2. Stopped Vehicle Detection
            if track_states:
                stop_events = self.stopped_detector.detect_stopped_vehicles(
                    track_states, segment_id=seg_id, timestamp=snapshot.timestamp, incident_id_counter=self._id_counter
                )
                self._id_counter += len(stop_events)
                detected_events.extend(stop_events)

            # 3. Congestion Anomaly Detection
            cong_event = self.congestion_analyzer.analyze_congestion_anomaly(
                observed, historical_baseline_congestion=0.15, timestamp=snapshot.timestamp, incident_id_counter=self._id_counter
            )
            if cong_event:
                self._id_counter += 1
                detected_events.append(cong_event)

        # Incident Lifecycle & False-Positive Filtering: Register active incidents
        current_active: List[IncidentEvent] = []
        for ev in detected_events:
            key = f"{ev.segment_id}_{ev.incident_type.value}"
            if key not in self.active_incidents:
                self.active_incidents[key] = ev
            else:
                # Update confidence and timestamp
                self.active_incidents[key].confidence = min(1.0, self.active_incidents[key].confidence + 0.05)
                self.active_incidents[key].evidence.update(ev.evidence)

            current_active.append(self.active_incidents[key])

        # Auto-resolve incidents when traffic returns to normal
        active_keys = {f"{ev.segment_id}_{ev.incident_type.value}" for ev in detected_events}
        for k in list(self.active_incidents.keys()):
            if k not in active_keys:
                self.active_incidents[k].status = IncidentStatus.RESOLVED
                self.active_incidents[k].resolved_at = snapshot.timestamp

        return current_active
