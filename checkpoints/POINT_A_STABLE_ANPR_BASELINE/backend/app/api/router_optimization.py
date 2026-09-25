"""
ChronoEye Infinity - Phase 14: Optimization & Emergency API Router
Provides REST endpoints for predictive traffic signal optimization, route optimization, and emergency green corridor activation.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.optimization.signal_optimizer import TrafficSignalOptimizationEngine
from app.optimization.predictive_router import ChronoEyePredictiveAStarRouter
from app.incident.corridor_engine import EmergencyCorridorEngine
from app.incident.emergency_schema import EmergencyVehicle
from app.api.api_schema import RouteOptimizationRequest, EmergencyCorridorRequest

router = APIRouter(prefix="/api/v1", tags=["Optimization & Emergency"])

_builder = SpatioTemporalGraphBuilder()
_builder.add_junction_node("JUNC_1", latitude=13.0827, longitude=80.2707)
_builder.add_junction_node("JUNC_2", latitude=13.0850, longitude=80.2720)
_builder.add_junction_node("JUNC_4", latitude=13.0900, longitude=80.2800)
_builder.add_road_edge("ROAD_1_2", "JUNC_1", "JUNC_2", length_meters=500.0, travel_time=30.0)
_builder.add_road_edge("ROAD_2_4", "JUNC_2", "JUNC_4", length_meters=500.0, travel_time=30.0)

_sig_engine = TrafficSignalOptimizationEngine()
_pred_router = ChronoEyePredictiveAStarRouter()
_corridor_engine = EmergencyCorridorEngine()


@router.get("/signals/state")
def get_signals_state() -> Dict[str, Any]:
    """Retrieves real-time signal controller status and phase timing schedules."""
    res = _sig_engine.run_scenario_experiment(num_steps=1)
    return res if isinstance(res, dict) else res.model_dump()


@router.post("/routes/optimize")
def optimize_route(req: RouteOptimizationRequest) -> Dict[str, Any]:
    """Calculates optimal spatio-temporal route using ChronoEye Predictive A*."""
    route = _pred_router.find_route(
        _builder,
        origin=req.origin,
        destination=req.destination,
        departure_timestamp=req.departure_timestamp,
    )
    return route.model_dump()


@router.post("/emergency/corridor")
def create_emergency_corridor(req: EmergencyCorridorRequest) -> Dict[str, Any]:
    """Activates Emergency Green Corridor priority preemption plan."""
    veh = EmergencyVehicle(
        vehicle_id=req.vehicle_id,
        vehicle_type=req.vehicle_type,
        origin_junction=req.origin_junction,
        destination_junction=req.destination_junction,
        current_location_junction=req.origin_junction,
    )
    plan = _corridor_engine.create_corridor_plan(_builder, veh)
    return plan.model_dump()
