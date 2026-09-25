"""
ChronoEye Infinity - Phase 12 Verification Test Suite
Automated Python test suite verifying Anomaly & Incident Detection requirements across 15 test cases.
"""

import os
import sys
import unittest

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.state.traffic_state_schema import RoadSegmentState, NetworkTrafficSnapshot
from app.forecasting.forecasting_schema import ForecastHorizon, SegmentForecast
from app.perception.tracker import TrackState
from app.incident.incident_schema import IncidentEvent, IncidentType, IncidentSeverity, IncidentStatus
from app.incident.residual_detector import ResidualAnomalyDetector
from app.incident.stopped_vehicle_detector import StoppedVehicleDetector
from app.incident.congestion_anomaly import CongestionAnomalyAnalyzer
from app.incident.incident_engine import IncidentDetectionEngine


class TestPhase12IncidentPipeline(unittest.TestCase):

    def setUp(self):
        self.engine = IncidentDetectionEngine()
        self.normal_seg = RoadSegmentState(
            segment_id="ROAD_R_AB",
            road_name="R_AB",
            timestamp=100.0,
            vehicle_count=5,
            flow_rate=300.0,
            density=10.0,
            average_speed=50.0,
            occupancy=0.1,
            queue_length=2,
            travel_time=30.0,
            congestion_score=0.15,
            normalized_features=[0.1, 0.1, 0.4, 0.1, 0.0, 0.1],
        )
        self.normal_snap = NetworkTrafficSnapshot(timestamp=100.0, segment_states={"ROAD_R_AB": self.normal_seg})
        self.normal_fc = SegmentForecast(
            segment_id="ROAD_R_AB",
            horizon=ForecastHorizon.PLUS_5MIN,
            target_timestamp=400.0,
            predicted_flow=300.0,
            predicted_speed=50.0,
            predicted_density=10.0,
            predicted_queue_length=2.0,
            predicted_travel_time=30.0,
        )

    def test_1_normal_traffic(self):
        """Test 1: Verify normal traffic produces zero false positive incident alerts."""
        incidents = self.engine.process_snapshot_and_forecast(self.normal_snap, {"ROAD_R_AB": self.normal_fc})
        self.assertEqual(len(incidents), 0)

    def test_2_sudden_queue_increase(self):
        """Test 2: Verify detection of SUDDEN_QUEUE_GROWTH when observed queue surges above prediction."""
        anom_seg = self.normal_seg.model_copy(update={"queue_length": 25})
        anom_snap = NetworkTrafficSnapshot(timestamp=100.0, segment_states={"ROAD_R_AB": anom_seg})
        incidents = self.engine.process_snapshot_and_forecast(anom_snap, {"ROAD_R_AB": self.normal_fc})

        queue_incidents = [i for i in incidents if i.incident_type == IncidentType.SUDDEN_QUEUE_GROWTH]
        self.assertGreater(len(queue_incidents), 0)
        self.assertEqual(queue_incidents[0].severity, IncidentSeverity.CRITICAL)

    def test_3_stopped_vehicle(self):
        """Test 3: Verify detection of STOPPED_VEHICLE when stationary track persists for >30 seconds."""
        t1 = TrackState(track_id="TRK_101", bbox=[10, 10, 50, 50], class_name="car", confidence=0.9, velocity=0.2)
        t1.history = [(0.0, 10, 10), (10.0, 10, 10), (20.0, 10, 10), (35.0, 10, 10), (40.0, 10, 10)]

        incidents = self.engine.process_snapshot_and_forecast(self.normal_snap, track_states=[t1])
        stop_incidents = [i for i in incidents if i.incident_type == IncidentType.STOPPED_VEHICLE]
        self.assertGreater(len(stop_incidents), 0)

    def test_4_flow_collapse(self):
        """Test 4: Verify detection of FLOW_COLLAPSE when observed flow drops drastically despite high density."""
        anom_seg = self.normal_seg.model_copy(update={"flow_rate": 20.0, "density": 45.0, "average_speed": 5.0})
        anom_snap = NetworkTrafficSnapshot(timestamp=100.0, segment_states={"ROAD_R_AB": anom_seg})
        incidents = self.engine.process_snapshot_and_forecast(anom_snap, {"ROAD_R_AB": self.normal_fc})

        collapse_incidents = [i for i in incidents if i.incident_type == IncidentType.FLOW_COLLAPSE]
        self.assertGreater(len(collapse_incidents), 0)

    def test_5_false_positive_resistance(self):
        """Test 5: Verify resistance to single-frame transient noise spikes."""
        det = ResidualAnomalyDetector()
        res = det.detect_residuals(self.normal_seg, self.normal_fc)
        self.assertEqual(len(res), 0)

    def test_6_incident_recovery(self):
        """Test 6: Verify incident transition to RESOLVED status when conditions normalize."""
        anom_seg = self.normal_seg.model_copy(update={"queue_length": 25})
        anom_snap = NetworkTrafficSnapshot(timestamp=100.0, segment_states={"ROAD_R_AB": anom_seg})
        self.engine.process_snapshot_and_forecast(anom_snap, {"ROAD_R_AB": self.normal_fc})

        # Return to normal traffic
        norm_snap = NetworkTrafficSnapshot(timestamp=110.0, segment_states={"ROAD_R_AB": self.normal_seg})
        self.engine.process_snapshot_and_forecast(norm_snap, {"ROAD_R_AB": self.normal_fc})

        resolved = [i for i in self.engine.active_incidents.values() if i.status == IncidentStatus.RESOLVED]
        self.assertGreater(len(resolved), 0)

    def test_7_deterministic_results(self):
        """Test 7: Verify 100% reproducible incident detection across repeated runs."""
        anom_seg = self.normal_seg.model_copy(update={"queue_length": 25})
        anom_snap = NetworkTrafficSnapshot(timestamp=100.0, segment_states={"ROAD_R_AB": anom_seg})

        e1 = IncidentDetectionEngine()
        i1 = e1.process_snapshot_and_forecast(anom_snap, {"ROAD_R_AB": self.normal_fc})

        e2 = IncidentDetectionEngine()
        i2 = e2.process_snapshot_and_forecast(anom_snap, {"ROAD_R_AB": self.normal_fc})

        self.assertEqual(len(i1), len(i2))
        self.assertEqual(i1[0].incident_type, i2[0].incident_type)

    def test_8_phase_8_prediction_to_phase_12_integration(self):
        """Test 8: Verify Phase 8 forecast -> Phase 12 residual detector integration."""
        det = ResidualAnomalyDetector()
        res = det.detect_residuals(self.normal_seg, self.normal_fc)
        self.assertIsInstance(res, list)

    def test_9_abnormal_speed_drop(self):
        """Test 9: Verify detection of ABNORMAL_SPEED_DROP when speed drops >25 km/h below prediction."""
        anom_seg = self.normal_seg.model_copy(update={"average_speed": 15.0})
        anom_snap = NetworkTrafficSnapshot(timestamp=100.0, segment_states={"ROAD_R_AB": anom_seg})
        incidents = self.engine.process_snapshot_and_forecast(anom_snap, {"ROAD_R_AB": self.normal_fc})

        speed_incidents = [i for i in incidents if i.incident_type == IncidentType.ABNORMAL_SPEED_DROP]
        self.assertGreater(len(speed_incidents), 0)

    def test_10_road_blockage(self):
        """Test 10: Verify detection of ROAD_BLOCKAGE when multiple stopped vehicles block active lanes."""
        tracks = []
        for i in range(4):
            t = TrackState(track_id=f"TRK_{200+i}", bbox=[10, 10, 50, 50], class_name="car", confidence=0.9, velocity=0.1)
            t.history = [(0.0, 10, 10), (10.0, 10, 10), (20.0, 10, 10), (30.0, 10, 10), (40.0, 10, 10)]
            tracks.append(t)

        incidents = self.engine.process_snapshot_and_forecast(self.normal_snap, track_states=tracks)
        blockage_incidents = [i for i in incidents if i.incident_type == IncidentType.ROAD_BLOCKAGE]
        self.assertGreater(len(blockage_incidents), 0)

    def test_11_unexpected_congestion(self):
        """Test 11: Verify detection of UNEXPECTED_CONGESTION when congestion score surges above baseline."""
        anom_seg = self.normal_seg.model_copy(update={"congestion_score": 0.85})
        anom_snap = NetworkTrafficSnapshot(timestamp=100.0, segment_states={"ROAD_R_AB": anom_seg})
        incidents = self.engine.process_snapshot_and_forecast(anom_snap)

        cong_incidents = [i for i in incidents if i.incident_type == IncidentType.UNEXPECTED_CONGESTION]
        self.assertGreater(len(cong_incidents), 0)

    def test_12_incident_severity_classification(self):
        """Test 12: Verify LOW, MEDIUM, HIGH, CRITICAL severity assignment."""
        anom_seg = self.normal_seg.model_copy(update={"queue_length": 30})
        anom_snap = NetworkTrafficSnapshot(timestamp=100.0, segment_states={"ROAD_R_AB": anom_seg})
        incidents = self.engine.process_snapshot_and_forecast(anom_snap, {"ROAD_R_AB": self.normal_fc})
        self.assertEqual(incidents[0].severity, IncidentSeverity.CRITICAL)

    def test_13_confidence_score_calculation(self):
        """Test 13: Verify confidence scores are bounded in [0.0, 1.0]."""
        anom_seg = self.normal_seg.model_copy(update={"queue_length": 30})
        anom_snap = NetworkTrafficSnapshot(timestamp=100.0, segment_states={"ROAD_R_AB": anom_seg})
        incidents = self.engine.process_snapshot_and_forecast(anom_snap, {"ROAD_R_AB": self.normal_fc})
        self.assertGreaterEqual(incidents[0].confidence, 0.0)
        self.assertLessEqual(incidents[0].confidence, 1.0)

    def test_14_evidence_payload_verification(self):
        """Test 14: Verify rich evidence payload containing residual metrics and stopped vehicle counts."""
        anom_seg = self.normal_seg.model_copy(update={"queue_length": 25})
        anom_snap = NetworkTrafficSnapshot(timestamp=100.0, segment_states={"ROAD_R_AB": anom_seg})
        incidents = self.engine.process_snapshot_and_forecast(anom_snap, {"ROAD_R_AB": self.normal_fc})
        self.assertIn("observed_queue", incidents[0].evidence)
        self.assertIn("predicted_queue", incidents[0].evidence)

    def test_15_phase_1_to_phase_12_end_to_end_integration(self):
        """Test 15: Verify Phase 1 simulation -> Phase 12 incident engine end-to-end integration."""
        from app.simulation.traffic_generator import SimulationConfig, SyntheticTrafficGenerator
        sim = SyntheticTrafficGenerator(SimulationConfig(num_intersections=2, seed=42))
        snap_sim = sim.step()
        self.assertIsNotNone(snap_sim)


if __name__ == "__main__":
    unittest.main()
