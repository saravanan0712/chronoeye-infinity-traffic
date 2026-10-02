"""
ChronoEye Infinity - Module 2 Checkpoint Query Test Suite
Tests deterministic, evidence-backed evaluation for:
- "Did global vehicle VEH_101 cross checkpoint CAM_B?"
- "Did plate TN09AB1111 cross checkpoint CAM_C?"
- Correct OBSERVED / NOT_OBSERVED / UNKNOWN semantics
- Time-window filtering
- Provenance and uncertainty verification
- FastAPI REST endpoints
"""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.perception.journey import JourneyReconstructionEngine
from app.perception.checkpoint_query import CheckpointQueryService, normalize_plate
from app.schemas.reid import (
    VehicleJourney,
    JourneySegment,
    CheckpointObservationStatus,
    CheckpointQueryResult,
    MatchDecision,
    ReIDScoreBreakdown,
)
from app.schemas.detection import BoundingBoxXYXY


@pytest.fixture
def populated_engine():
    """Builds a JourneyReconstructionEngine pre-populated with controlled test journeys."""
    engine = JourneyReconstructionEngine()

    # Journey 1: VEH_101 with plate TN-09-AB-1111 observed at CAM_A and CAM_B
    j1 = VehicleJourney(
        journey_id="JRN_101",
        global_vehicle_id="VEH_101",
        plate_number="TN09AB1111",
        vehicle_type="car",
        first_seen=10.0,
        last_seen=25.0,
        cameras=["CAM_A_EAST", "CAM_B_WEST"],
        overall_confidence=0.92,
        status="ACTIVE",
    )
    j1.segments.append(
        JourneySegment(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_01",
            timestamp=10.0,
            timestamp_uncertainty_seconds=0.2,
            speed_estimate=45.0,
            direction="EASTBOUND",
            plate_number="TN09AB1111",
            plate_confidence=0.95,
            plate_status="CONFIRMED",
            bbox=BoundingBoxXYXY(x1=100, y1=150, x2=250, y2=300),
        )
    )
    j1.segments.append(
        JourneySegment(
            camera_id="CAM_B_WEST",
            track_id="TRK_B_07",
            timestamp=25.0,
            timestamp_uncertainty_seconds=0.5,
            speed_estimate=48.5,
            direction="WESTBOUND",
            plate_number="TN09AB1111",
            plate_confidence=0.91,
            plate_status="CONFIRMED",
            transition_decision=MatchDecision.MATCH_CONFIRMED,
            transition_score=0.88,
            transition_breakdown=ReIDScoreBreakdown(
                plate_similarity=1.0,
                appearance_similarity=0.85,
                visual_features_similarity=0.80,
                vehicle_type_similarity=1.0,
                direction_similarity=0.9,
                temporal_compatibility=0.95,
                spatial_compatibility=1.0,
                overall_score=0.88,
                decision=MatchDecision.MATCH_CONFIRMED,
            ),
            bbox=BoundingBoxXYXY(x1=120, y1=160, x2=270, y2=310),
        )
    )
    engine.journeys[j1.journey_id] = j1

    # Journey 2: VEH_102 with pending plate MH-12-CD-2222 and unobserved gap (CAM_A -> CAM_C)
    j2 = VehicleJourney(
        journey_id="JRN_102",
        global_vehicle_id="VEH_102",
        plate_number="MH12CD2222",
        vehicle_type="bus",
        first_seen=30.0,
        last_seen=60.0,
        cameras=["CAM_A_EAST", "CAM_C_NORTH"],
        overall_confidence=0.78,
        status="ACTIVE",
    )
    j2.segments.append(
        JourneySegment(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_02",
            timestamp=30.0,
            speed_estimate=35.0,
            direction="EASTBOUND",
            plate_number="MH12CD2222",
            plate_confidence=0.65,
            plate_status="PENDING",
        )
    )
    j2.segments.append(
        JourneySegment(
            camera_id="CAM_C_NORTH",
            track_id="TRK_C_03",
            timestamp=60.0,
            speed_estimate=32.0,
            direction="NORTHBOUND",
            plate_number="MH12CD2222",
            plate_confidence=0.70,
            plate_status="PENDING",
            transition_decision=MatchDecision.MATCH_PROBABLE,
            transition_score=0.68,
            has_unobserved_gap=True,
        )
    )
    engine.journeys[j2.journey_id] = j2

    return engine


class TestCheckpointQuery:
    """Test suite for CheckpointQueryService and JourneyReconstructionEngine checkpoint querying."""

    def test_01_query_by_global_vehicle_id_observed(self, populated_engine):
        """Test 1: Global vehicle ID query for an observed checkpoint returns OBSERVED with evidence."""
        res: CheckpointQueryResult = populated_engine.query_checkpoint(
            checkpoint_id="CAM_B_WEST",
            vehicle_id="VEH_101",
        )
        assert res.status == CheckpointObservationStatus.OBSERVED
        assert res.target_vehicle_id == "VEH_101"
        assert res.checkpoint_id == "CAM_B_WEST"
        assert res.timestamp == 25.0
        assert res.timestamp_uncertainty_seconds == 0.5
        assert res.confidence > 0.8
        assert res.plate_number == "TN09AB1111"
        assert res.plate_status == "CONFIRMED"
        assert res.journey_id == "JRN_101"
        assert res.evidence["track_id"] == "TRK_B_07"
        assert res.evidence["direction"] == "WESTBOUND"
        assert res.evidence["speed_estimate_kmh"] == 48.5
        assert res.evidence["transition_score"] == 0.88
        assert res.uncertainty["unobserved_gap"] is False

    def test_02_query_by_plate_number_observed(self, populated_engine):
        """Test 2: Plate number query with spaces/hyphens returns OBSERVED across multi-camera records."""
        res = populated_engine.query_checkpoint(
            checkpoint_id="CAM_A",
            plate_number="tn-09 ab-1111",
        )
        assert res.status == CheckpointObservationStatus.OBSERVED
        assert res.target_vehicle_id == "VEH_101"
        assert res.timestamp == 10.0
        assert res.plate_number == "TN09AB1111"
        assert res.supporting_segments_count >= 1

    def test_03_query_not_observed_semantic_guarantee(self, populated_engine):
        """
        Test 3: Query for unvisited checkpoint returns NOT_OBSERVED and preserves semantic rule:
        NOT_OBSERVED denotes absence of recorded observation, not proof of non-crossing.
        """
        res = populated_engine.query_checkpoint(
            checkpoint_id="CAM_D_NORTH",
            vehicle_id="VEH_101",
        )
        assert res.status == CheckpointObservationStatus.NOT_OBSERVED
        assert res.timestamp is None
        assert res.confidence == 0.0
        assert res.journey_id == "JRN_101"
        assert "absence of recorded evidence" in res.uncertainty["status_reason"]
        assert "not proof of non-crossing" in res.uncertainty["status_reason"]

    def test_04_query_non_existent_vehicle(self, populated_engine):
        """Test 4: Querying a vehicle that does not exist in any journey returns NOT_OBSERVED."""
        res = populated_engine.query_checkpoint(
            checkpoint_id="CAM_A",
            vehicle_id="VEH_999",
        )
        assert res.status == CheckpointObservationStatus.NOT_OBSERVED
        assert res.confidence == 0.0
        assert res.journey_id is None

    def test_05_query_time_window_filtering(self, populated_engine):
        """Test 5: Time window boundaries filter matching segments accurately."""
        # Querying within time window [20.0, 30.0] matches CAM_B (t=25.0)
        res_in = populated_engine.query_checkpoint(
            checkpoint_id="CAM_B_WEST",
            vehicle_id="VEH_101",
            time_start=20.0,
            time_end=30.0,
        )
        assert res_in.status == CheckpointObservationStatus.OBSERVED
        assert res_in.timestamp == 25.0

        # Querying outside time window [0.0, 15.0] returns NOT_OBSERVED for CAM_B
        res_out = populated_engine.query_checkpoint(
            checkpoint_id="CAM_B_WEST",
            vehicle_id="VEH_101",
            time_start=0.0,
            time_end=15.0,
        )
        assert res_out.status == CheckpointObservationStatus.NOT_OBSERVED

    def test_06_query_unobserved_gap_and_pending_plate(self, populated_engine):
        """Test 6: Vehicle with unobserved gap or pending plate reflects uncertainty."""
        # Query VEH_102 at CAM_A_EAST (pending plate) -> UNKNOWN
        res_pending = populated_engine.query_checkpoint(
            checkpoint_id="CAM_A_EAST",
            vehicle_id="VEH_102",
        )
        assert res_pending.status == CheckpointObservationStatus.UNKNOWN
        assert "status=PENDING" in res_pending.uncertainty["status_reason"]

        # Query VEH_102 at CAM_B (bypassed with gap) -> NOT_OBSERVED with gap noted
        res_gap = populated_engine.query_checkpoint(
            checkpoint_id="CAM_B_WEST",
            vehicle_id="VEH_102",
        )
        assert res_gap.status == CheckpointObservationStatus.NOT_OBSERVED
        assert res_gap.has_unobserved_gap is True
        assert "unobserved intermediate gaps" in res_gap.uncertainty["status_reason"]

    def test_07_api_endpoint_checkpoint_query(self, populated_engine):
        """Test 7: FastAPI REST endpoint /api/v1/traffic/reid/checkpoint-query responds correctly."""
        # Inject populated engine into router singleton for test
        from app.api import router_traffic
        orig_engine = router_traffic._journey_engine
        router_traffic._journey_engine = populated_engine

        try:
            client = TestClient(app)

            # Valid query by vehicle_id
            response = client.get("/api/v1/traffic/reid/checkpoint-query?checkpoint_id=CAM_B_WEST&vehicle_id=VEH_101")
            assert response.status_code == 200
            data = response.json()
            assert data["query_type"] == "CHECKPOINT_CROSSING"
            assert data["status"] == "OBSERVED"
            assert data["timestamp"] == 25.0
            assert data["plate_number"] == "TN09AB1111"

            # Valid query by plate_number
            response_plate = client.get("/api/v1/traffic/reid/checkpoint-query?checkpoint_id=CAM_A_EAST&plate_number=TN09AB1111")
            assert response_plate.status_code == 200
            assert response_plate.json()["status"] == "OBSERVED"

            # Direct vehicle-checkpoint path route
            response_path = client.get("/api/v1/traffic/reid/vehicles/VEH_101/checkpoints/CAM_B_WEST")
            assert response_path.status_code == 200
            assert response_path.json()["status"] == "OBSERVED"

            # Missing required query parameters (neither vehicle_id nor plate_number)
            response_invalid = client.get("/api/v1/traffic/reid/checkpoint-query?checkpoint_id=CAM_A")
            assert response_invalid.status_code == 400

        finally:
            router_traffic._journey_engine = orig_engine
