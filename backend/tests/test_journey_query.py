"""
ChronoEye Infinity - Module 2 Global Vehicle Journey Query Test Suite
Tests dedicated global vehicle journey retrieval, plate lookup, time-window filtering,
evidence preservation, unobserved-gap preservation, and FastAPI REST endpoints:
- GET /api/v1/traffic/reid/vehicles/{vehicle_id}/journey
- GET /api/v1/traffic/reid/journey-query
"""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.perception.journey import JourneyReconstructionEngine
from app.schemas.reid import (
    VehicleJourney,
    JourneySegment,
    MatchDecision,
    ReIDScoreBreakdown,
)
from app.schemas.detection import BoundingBoxXYXY


@pytest.fixture
def multi_journey_engine():
    """Builds a JourneyReconstructionEngine with multiple controlled multi-camera journeys."""
    engine = JourneyReconstructionEngine()

    # Journey 1: VEH_101 with plate TN09AB1111 observed at CAM_A and CAM_B
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
            timestamp_uncertainty_seconds=0.4,
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
                direction_similarity=0.90,
                temporal_compatibility=0.95,
                spatial_compatibility=1.0,
                overall_score=0.88,
                decision=MatchDecision.MATCH_CONFIRMED,
            ),
            bbox=BoundingBoxXYXY(x1=120, y1=160, x2=270, y2=310),
        )
    )
    engine.journeys[j1.journey_id] = j1

    # Journey 2: VEH_102 with plate KA01XY9999 observed CAM_A -> CAM_C (Unobserved Gap)
    j2 = VehicleJourney(
        journey_id="JRN_102",
        global_vehicle_id="VEH_102",
        plate_number="KA01XY9999",
        vehicle_type="truck",
        first_seen=30.0,
        last_seen=60.0,
        cameras=["CAM_A_EAST", "CAM_C_NORTH"],
        overall_confidence=0.85,
        status="ACTIVE",
    )
    j2.segments.append(
        JourneySegment(
            camera_id="CAM_A_EAST",
            track_id="TRK_A_02",
            timestamp=30.0,
            timestamp_uncertainty_seconds=0.3,
            speed_estimate=35.0,
            direction="EASTBOUND",
            plate_number="KA01XY9999",
            plate_confidence=0.88,
            plate_status="CONFIRMED",
        )
    )
    j2.segments.append(
        JourneySegment(
            camera_id="CAM_C_NORTH",
            track_id="TRK_C_03",
            timestamp=60.0,
            timestamp_uncertainty_seconds=0.6,
            speed_estimate=32.0,
            direction="NORTHBOUND",
            plate_number="KA01XY9999",
            plate_confidence=0.85,
            plate_status="CONFIRMED",
            transition_decision=MatchDecision.MATCH_PROBABLE,
            transition_score=0.72,
            transition_breakdown=ReIDScoreBreakdown(
                plate_similarity=1.0,
                appearance_similarity=0.65,
                visual_features_similarity=0.70,
                vehicle_type_similarity=1.0,
                direction_similarity=0.80,
                temporal_compatibility=0.75,
                spatial_compatibility=0.70,
                overall_score=0.72,
                decision=MatchDecision.MATCH_PROBABLE,
            ),
            has_unobserved_gap=True,
        )
    )
    engine.journeys[j2.journey_id] = j2

    # Journey 3: Separate journey JRN_103 sharing plate TN09AB1111 at a much later time (Collision Test)
    j3 = VehicleJourney(
        journey_id="JRN_103",
        global_vehicle_id="VEH_103",
        plate_number="TN09AB1111",
        vehicle_type="car",
        first_seen=500.0,
        last_seen=530.0,
        cameras=["CAM_C_NORTH"],
        overall_confidence=0.90,
        status="COMPLETED",
    )
    j3.segments.append(
        JourneySegment(
            camera_id="CAM_C_NORTH",
            track_id="TRK_C_99",
            timestamp=500.0,
            timestamp_uncertainty_seconds=0.2,
            speed_estimate=50.0,
            direction="NORTHBOUND",
            plate_number="TN09AB1111",
            plate_confidence=0.96,
            plate_status="CONFIRMED",
        )
    )
    engine.journeys[j3.journey_id] = j3

    return engine


class TestGlobalVehicleJourneyQuery:
    """Test suite for global vehicle journey query engine methods and REST endpoints."""

    def test_01_get_vehicle_journey_by_global_id(self, multi_journey_engine):
        """Test 1: Engine and REST endpoint retrieve complete VehicleJourney by global_vehicle_id."""
        journey = multi_journey_engine.get_journey_by_vehicle_id("VEH_101")
        assert journey is not None
        assert journey.global_vehicle_id == "VEH_101"
        assert journey.journey_id == "JRN_101"
        assert journey.plate_number == "TN09AB1111"
        assert journey.vehicle_type == "car"
        assert len(journey.segments) == 2
        assert journey.cameras == ["CAM_A_EAST", "CAM_B_WEST"]
        assert journey.first_seen == 10.0
        assert journey.last_seen == 25.0

    def test_02_query_journey_by_plate_number(self, multi_journey_engine):
        """Test 2: Querying by license plate string retrieves the associated journeys."""
        journeys = multi_journey_engine.get_journeys_by_plate("KA01XY9999")
        assert len(journeys) == 1
        assert journeys[0].global_vehicle_id == "VEH_102"
        assert journeys[0].plate_number == "KA01XY9999"

    def test_03_query_journey_multiple_plate_matches(self, multi_journey_engine):
        """Test 3: Multiple distinct journeys sharing the same plate are returned as distinct objects without merging."""
        journeys = multi_journey_engine.get_journeys_by_plate("TN09AB1111")
        assert len(journeys) == 2
        veh_ids = {j.global_vehicle_id for j in journeys}
        assert veh_ids == {"VEH_101", "VEH_103"}
        j_ids = {j.journey_id for j in journeys}
        assert j_ids == {"JRN_101", "JRN_103"}

    def test_04_journey_unobserved_gap_preservation(self, multi_journey_engine):
        """Test 4: Unobserved gap (CAM_A -> CAM_C) is preserved on transition without inserting intermediate cameras."""
        journey = multi_journey_engine.get_journey_by_vehicle_id("VEH_102")
        assert journey is not None
        assert journey.cameras == ["CAM_A_EAST", "CAM_C_NORTH"]
        assert len(journey.segments) == 2
        assert "CAM_B" not in journey.cameras
        assert "CAM_B_WEST" not in journey.cameras
        assert journey.segments[1].has_unobserved_gap is True

    def test_05_journey_evidence_provenance(self, multi_journey_engine):
        """Test 5: Response preserves all seven ReID similarity scores and timestamp uncertainty."""
        journey = multi_journey_engine.get_journey_by_vehicle_id("VEH_101")
        assert journey is not None
        seg2 = journey.segments[1]
        assert seg2.timestamp_uncertainty_seconds == 0.4
        assert seg2.transition_score == 0.88
        assert seg2.transition_decision == MatchDecision.MATCH_CONFIRMED

        tb = seg2.transition_breakdown
        assert tb is not None
        assert tb.plate_similarity == 1.0
        assert tb.appearance_similarity == 0.85
        assert tb.visual_features_similarity == 0.80
        assert tb.vehicle_type_similarity == 1.0
        assert tb.direction_similarity == 0.90
        assert tb.temporal_compatibility == 0.95
        assert tb.spatial_compatibility == 1.0
        assert tb.overall_score == 0.88

    def test_06_journey_query_nonexistent_vehicle_404(self, multi_journey_engine):
        """Test 6: Non-existent vehicle ID returns None in engine and 404 on API."""
        journey = multi_journey_engine.get_journey_by_vehicle_id("VEH_999")
        assert journey is None

        from app.api import router_traffic
        orig_engine = router_traffic._journey_engine
        router_traffic._journey_engine = multi_journey_engine
        try:
            client = TestClient(app)
            response = client.get("/api/v1/traffic/reid/vehicles/VEH_999/journey")
            assert response.status_code == 404

            response_q = client.get("/api/v1/traffic/reid/journey-query?vehicle_id=VEH_999")
            assert response_q.status_code == 404
        finally:
            router_traffic._journey_engine = orig_engine

    def test_07_missing_vehicle_id_and_plate_400(self, multi_journey_engine):
        """Test 7: Calling /reid/journey-query without vehicle_id or plate_number returns HTTP 400."""
        from app.api import router_traffic
        orig_engine = router_traffic._journey_engine
        router_traffic._journey_engine = multi_journey_engine
        try:
            client = TestClient(app)
            response = client.get("/api/v1/traffic/reid/journey-query")
            assert response.status_code == 400
            assert "Either vehicle_id or plate_number must be specified" in response.json()["detail"]
        finally:
            router_traffic._journey_engine = orig_engine

    def test_08_time_window_filtering(self, multi_journey_engine):
        """Test 8: Time window boundaries accurately filter returned segments."""
        from app.api import router_traffic
        orig_engine = router_traffic._journey_engine
        router_traffic._journey_engine = multi_journey_engine
        try:
            client = TestClient(app)

            # VEH_101 has segments at t=10.0 and t=25.0
            # Filter time_start=15.0 -> only segment at t=25.0
            resp_start = client.get("/api/v1/traffic/reid/vehicles/VEH_101/journey?time_start=15.0")
            assert resp_start.status_code == 200
            data_start = resp_start.json()
            assert len(data_start["segments"]) == 1
            assert data_start["segments"][0]["timestamp"] == 25.0
            assert data_start["cameras"] == ["CAM_B_WEST"]

            # Filter time_end=15.0 -> only segment at t=10.0
            resp_end = client.get("/api/v1/traffic/reid/vehicles/VEH_101/journey?time_end=15.0")
            assert resp_end.status_code == 200
            data_end = resp_end.json()
            assert len(data_end["segments"]) == 1
            assert data_end["segments"][0]["timestamp"] == 10.0
            assert data_end["cameras"] == ["CAM_A_EAST"]

            # Filter [20.0, 30.0] on journey-query by plate
            resp_plate = client.get("/api/v1/traffic/reid/journey-query?plate_number=TN09AB1111&time_start=20.0&time_end=30.0")
            assert resp_plate.status_code == 200
            data_plate = resp_plate.json()
            # JRN_103 is at t=500.0 (excluded), JRN_101 is at t=10, 25 (t=25 segment included)
            assert data_plate["count"] == 1
            assert data_plate["journeys"][0]["journey_id"] == "JRN_101"
            assert len(data_plate["journeys"][0]["segments"]) == 1
            assert data_plate["journeys"][0]["segments"][0]["timestamp"] == 25.0
        finally:
            router_traffic._journey_engine = orig_engine

    def test_09_timestamp_boundary_inclusivity(self, multi_journey_engine):
        """Test 9: Exact boundary timestamps (timestamp == time_start and timestamp == time_end) are inclusively returned."""
        from app.api import router_traffic
        orig_engine = router_traffic._journey_engine
        router_traffic._journey_engine = multi_journey_engine
        try:
            client = TestClient(app)
            # Exactly t=10.0 to t=25.0
            resp = client.get("/api/v1/traffic/reid/vehicles/VEH_101/journey?time_start=10.0&time_end=25.0")
            assert resp.status_code == 200
            assert len(resp.json()["segments"]) == 2
        finally:
            router_traffic._journey_engine = orig_engine

    def test_10_plate_normalization(self, multi_journey_engine):
        """Test 10: Lookup with spaces, hyphens, and lowercase normalizes and resolves correctly."""
        journeys = multi_journey_engine.get_journeys_by_plate("  tn-09 ab 1111  ")
        assert len(journeys) == 2
        veh_ids = {j.global_vehicle_id for j in journeys}
        assert "VEH_101" in veh_ids
        assert "VEH_103" in veh_ids
