"""
ChronoEye Infinity - Phase 14: Graph & Vehicle Journey API Router
Provides REST endpoints for spatio-temporal traffic graph state and global vehicle journeys.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.perception.journey import VehicleJourney, JourneySegment

router = APIRouter(prefix="/api/v1", tags=["Graph & Vehicle Re-ID"])

_builder = SpatioTemporalGraphBuilder()
_builder.add_junction_node("JUNC_1")
_builder.add_junction_node("JUNC_2")
_builder.add_road_edge("ROAD_1_2", "JUNC_1", "JUNC_2", length_meters=500.0)


@router.get("/graph/state")
def get_graph_state() -> Dict[str, Any]:
    """Retrieves spatio-temporal traffic graph node & edge counts and topology summary."""
    g = _builder.graph
    return {
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "node_types": list({data.get("node_type") for _, data in g.nodes(data=True) if "node_type" in data}),
        "nodes": [n for n in g.nodes()],
    }


@router.get("/vehicles/journey/{vehicle_id}")
def get_vehicle_journey(vehicle_id: str) -> Dict[str, Any]:
    """Retrieves reconstructed continuous multi-camera journey for a global vehicle identity."""
    # Synthetic/Reconstructed vehicle journey response for demonstration
    journey = VehicleJourney(
        global_vehicle_id=vehicle_id,
        license_plate="TN-01-AB-1234",
        vehicle_type="car",
        segments=[
            JourneySegment(
                camera_id="CAM_A",
                track_id="TRK_101",
                enter_timestamp=100.0,
                exit_timestamp=130.0,
                average_speed=45.0,
            ),
            JourneySegment(
                camera_id="CAM_B",
                track_id="TRK_207",
                enter_timestamp=145.0,
                exit_timestamp=175.0,
                average_speed=48.0,
            ),
        ],
        total_distance_km=1.2,
        total_travel_time_seconds=75.0,
        reconstruction_confidence=0.92,
    )
    return journey.model_dump()
