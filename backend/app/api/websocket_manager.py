"""
ChronoEye Infinity - Phase 14: WebSocket Connection Manager
Non-blocking manager handling active WebSocket subscriptions and broadcasting real-time traffic events.
"""

import logging
from typing import List, Dict, Any
from fastapi import WebSocket

logger = logging.getLogger("chronoeye.websocket")


class WebSocketConnectionManager:
    """
    Manages active WebSocket client connections and provides thread-safe broadcasting.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcasts message to all active WebSocket clients. Automatically prunes stale connections.
        """
        stale: List[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Error broadcasting to WebSocket client: {e}")
                stale.append(connection)

        for s in stale:
            self.disconnect(s)

    async def broadcast_event(self, event_type: str, payload: Dict[str, Any]):
        """
        Broadcasts a typed real-time event.
        Types: FRAME_PROCESSED, VEHICLE_DETECTED, VEHICLE_TRACKED, VEHICLE_ENTERED,
               VEHICLE_EXITED, TRAFFIC_UPDATED, QUEUE_UPDATED, SPEED_UPDATED,
               CONGESTION_UPDATED, GRAPH_UPDATED, VEHICLE_IDENTITY_CREATED,
               CROSS_CAMERA_MATCH, CAMERA_TRANSITION, JOURNEY_UPDATED,
               IDENTITY_MATCH_REJECTED, JOURNEY_COMPLETED.
        """
        message = {
            "event": event_type,
            "timestamp": payload.get("timestamp", 0.0),
            "data": payload,
        }
        await self.broadcast(message)


