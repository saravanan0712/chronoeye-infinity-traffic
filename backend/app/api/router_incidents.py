"""
ChronoEye Infinity - Phase 14: Incidents API Router
Provides REST endpoints for active traffic incidents and residual anomalies.
"""

from fastapi import APIRouter
from typing import Dict, List, Any
from app.simulation.traffic_generator import SimulationConfig, SyntheticTrafficGenerator
from app.state.traffic_state_engine import DynamicTrafficStateEngine
from app.incident.incident_engine import IncidentDetectionEngine

router = APIRouter(prefix="/api/v1/incidents", tags=["Incident Detection"])

_sim = SyntheticTrafficGenerator(SimulationConfig(num_intersections=2, seed=42))
_state_engine = DynamicTrafficStateEngine()
_incident_engine = IncidentDetectionEngine()


@router.get("/active")
def get_active_incidents() -> Dict[str, Any]:
    """Retrieves active real-time traffic incidents."""
    sim_snap = _sim.step()
    net_snap = _state_engine.compute_network_snapshot(sim_snap)
    incidents = _incident_engine.process_snapshot_and_forecast(net_snap)
    return {
        "count": len(incidents),
        "incidents": [inc.model_dump() for inc in incidents],
    }
