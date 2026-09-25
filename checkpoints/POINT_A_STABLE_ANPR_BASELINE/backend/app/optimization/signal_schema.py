"""
ChronoEye Infinity - Phase 10: Traffic Signal Optimization Schemas
Defines data models for signal phases, intersection state vectors, signal control constraints,
and optimization plans.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class SignalPhase(str, Enum):
    PHASE_NORTH_SOUTH = "PHASE_NORTH_SOUTH"
    PHASE_EAST_WEST = "PHASE_EAST_WEST"
    YELLOW_CLEARANCE = "YELLOW_CLEARANCE"
    ALL_RED_CLEARANCE = "ALL_RED_CLEARANCE"


class SignalState(BaseModel):
    """
    Traffic signal hardware & controller state vector.
    """
    signal_id: str
    current_phase: SignalPhase = SignalPhase.PHASE_NORTH_SOUTH
    elapsed_green: float = 0.0
    elapsed_yellow: float = 0.0
    min_green: float = 10.0        # Minimum green clearance seconds
    max_green: float = 60.0        # Maximum green duration seconds
    yellow_time: float = 3.0       # Yellow clearance duration seconds
    all_red_time: float = 2.0     # All-red safety clearance seconds
    cycle_length: float = 90.0     # Maximum total cycle time seconds


class IntersectionState(BaseModel):
    """
    Intersection traffic condition state vector.
    """
    intersection_id: str
    queue_ns: int = 0
    queue_ew: int = 0
    flow_ns: float = 300.0          # veh/hr
    flow_ew: float = 300.0          # veh/hr
    predicted_queue_ns: float = 0.0
    predicted_queue_ew: float = 0.0
    predicted_travel_time_ns: float = 30.0
    predicted_travel_time_ew: float = 30.0
    has_emergency_vehicle_ns: bool = False
    has_emergency_vehicle_ew: bool = False


class SignalOptimizationPlan(BaseModel):
    """
    Optimal signal timing decision produced by signal controller.
    """
    signal_id: str
    recommended_phase: SignalPhase
    recommended_green_duration: float
    expected_delay_reduction_pct: float = 0.0
    objective_cost: float = 0.0
    is_phase_switch: bool = False
