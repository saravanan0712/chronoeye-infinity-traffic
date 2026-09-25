"""
ChronoEye Infinity - Phase 14: FastAPI Application & WebSocket Server
Main application entry point configuring REST API routes, CORS middleware, health check endpoints,
and non-blocking real-time WebSocket event streaming.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.api.api_schema import HealthResponse, WSMessage
from app.api.websocket_manager import WebSocketConnectionManager
from app.api.router_traffic import router as traffic_router
from app.api.router_graph import router as graph_router
from app.api.router_optimization import router as optimization_router
from app.api.router_incidents import router as incidents_router
from app.api.router_video import router as video_router
from app.api.router_forensics import router as forensics_router
from app.simulation.traffic_generator import SimulationConfig, SyntheticTrafficGenerator
from app.state.traffic_state_engine import DynamicTrafficStateEngine

logger = logging.getLogger("chronoeye.api")

ws_manager = WebSocketConnectionManager()
_sim = SyntheticTrafficGenerator(SimulationConfig(num_intersections=4, seed=42))
_state_engine = DynamicTrafficStateEngine()


async def background_broadcast_loop():
    """Non-blocking background simulation loop broadcasting traffic events to WebSocket subscribers."""
    while True:
        try:
            if ws_manager.active_connections:
                sim_snap = _sim.step()
                net_snap = _state_engine.compute_network_snapshot(sim_snap)
                msg = WSMessage(
                    event_type="TRAFFIC_STATE_UPDATE",
                    timestamp=net_snap.timestamp,
                    provenance={
                        "provenance_type": "SIMULATED" if getattr(_sim, "is_simulation", True) else "OBSERVED",
                        "source_id": "SYNTHETIC_TRAFFIC_GEN",
                        "process_name": "DynamicTrafficStateEngine",
                        "timestamp": net_snap.timestamp,
                        "confidence": 0.98,
                        "is_mock": False,
                    },
                    payload=net_snap.model_dump(),
                )
                await ws_manager.broadcast(msg.model_dump())
            await asyncio.sleep(1.0)
        except Exception as e:
            logger.warning(f"Error in background broadcast loop: {e}")
            await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager starting background WebSocket broadcaster on startup."""
    broadcast_task = asyncio.create_task(background_broadcast_loop())
    yield
    broadcast_task.cancel()


app = FastAPI(
    title="ChronoEye Infinity - Traffic Intelligence API",
    description="Unified REST API & WebSocket Real-time System for Phase 1-14 Traffic Intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(video_router)
app.include_router(forensics_router)
app.include_router(traffic_router)
app.include_router(graph_router)
app.include_router(optimization_router)
app.include_router(incidents_router)




@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health Check"])
@app.options("/api/v1/health", tags=["Health Check"])
def health_check():
    """Health check endpoint returning system status and active phase modules."""
    return HealthResponse(status="HEALTHY", version="1.0.0")


@app.websocket("/ws/traffic")
async def websocket_traffic_endpoint(websocket: WebSocket):
    """WebSocket streaming endpoint for real-time traffic state broadcasts."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep-alive receive loop
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket connection error: {e}")
        ws_manager.disconnect(websocket)
