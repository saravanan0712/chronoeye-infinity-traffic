"""
ChronoEye Infinity - Phase 4: Simulation Fallback Adapter
Provides a seamless simulation fallback adapter implementing the identical Traffic Observation Interface
when physical CCTV camera streams or uploaded video files are unavailable during development/testing.
Clearly labels SIMULATION MODE across all metrics and schemas.
"""

import time
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from app.simulation.traffic_generator import SyntheticTrafficGenerator, SimulationConfig


class SimulationAdapter:
    """
    Simulation Adapter wrapping SyntheticTrafficGenerator into standard Traffic Observation outputs.
    """

    def __init__(self, num_intersections: int = 4, seed: int = 42):
        self.generator = SyntheticTrafficGenerator(SimulationConfig(num_intersections=num_intersections, seed=seed))
        self.is_simulation = True

    def get_latest_observation(self, camera_id: str = "CAM_SIMULATED_01", road_id: str = "ROAD_1_2") -> Dict[str, Any]:
        """
        Generates simulated traffic observation matching the exact Master Data Object schema (Section 58).
        """
        sim_snap = self.generator.step()
        now = time.time()

        if isinstance(sim_snap, dict):
            active_vehicles = sim_snap.get("active_vehicle_count", 0)
            traffic_metrics = sim_snap.get("traffic_metrics", [])
            
            # Aggregate metrics across the network
            if traffic_metrics:
                flow = sum(m.get("flow_rate", 0.0) for m in traffic_metrics)
                _speeds = [m.get("average_speed_kmh", 0.0) for m in traffic_metrics]
                speed = sum(_speeds) / len(_speeds) if _speeds else 60.0
                _densities = [m.get("density", 0.0) for m in traffic_metrics]
                density = sum(_densities) / len(_densities) if _densities else 0.0
                queue = sum(m.get("queue_length_meters", 0.0) for m in traffic_metrics)
                _congestions = [m.get("congestion_score", 0.0) for m in traffic_metrics]
                congestion = sum(_congestions) / len(_congestions) if _congestions else 0.0
            else:
                flow, speed, density, queue, congestion = 0.0, 60.0, 0.0, 0.0, 0.0
                
            state = "FREE_FLOW" if congestion < 0.2 else "MODERATE" if congestion < 0.5 else "HEAVY"
        else:
            # Fallback for mock/test objects
            active_vehicles = getattr(sim_snap, "active_vehicles", 0)
            flow = getattr(sim_snap, "overall_flow", 0.0)
            speed = getattr(sim_snap, "overall_speed", 0.0)
            density = getattr(sim_snap, "overall_density", 0.0)
            queue = getattr(sim_snap, "total_queue", 0.0)
            congestion = getattr(sim_snap, "congestion_score", 0.0)
            state = getattr(sim_snap, "traffic_state", "UNKNOWN")

        return {
            "observation_id": f"OBS_SIM_{int(now * 1000)}",
            "timestamp": round(now, 3),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(now)),
            "camera_id": camera_id,
            "road_id": road_id,
            "lane_id": "LANE_1",
            "source_type": "SIMULATION",
            "video_id": f"VID_SIM_{camera_id}",
            "video_trust_score": 1.0,
            "integrity_status": "SIMULATED_MODE",
            "vehicle_count": active_vehicles,
            "flow_rate": flow,
            "average_speed_kmh": round(speed, 1),
            "density": density,
            "queue_length": queue,
            "occupancy": round(density / 100.0, 2),
            "travel_time_seconds": round(500.0 / max(1.0, speed / 3.6), 1),
            "congestion_index": round(congestion, 2),
            "traffic_state": state,
            "measurement_confidence": 1.0,
            "data_quality": "GOOD",
            "provenance": {
                "provenance_type": "SIMULATED",
                "source_id": f"SIMULATOR_{camera_id}",
                "process_name": "SimulationAdapter",
                "timestamp": now,
                "confidence": 1.0,
                "is_mock": True,
            },
        }
