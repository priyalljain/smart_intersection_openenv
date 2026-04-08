"""
Pydantic models for traffic control environment.
Ensures OpenEnv spec compliance and type safety.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

# ============= ENUMS =============

class Lane(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

class Phase(str, Enum):
    """Traffic signal phase"""
    NS = "ns"  # North-South green
    EW = "ew"  # East-West green
    ALL_RED = "all_red"  # Emergency all-red

# ============= ACTION MODEL =============

class TrafficAction(BaseModel):
    """Action: which phase to set"""
    phase: Phase = Field(
        description="Target phase (NS or EW)"
    )

    class Config:
        json_schema_extra = {
            "example": {"phase": "ns"}
        }

# ============= OBSERVATION MODEL =============
class TrafficObservation(BaseModel):
    """Observation: current state of the intersection"""
    
    # Time
    time: float = Field(description="Current simulation time in seconds")
    phase: str = Field(description="Current traffic phase (ns, ew, all_red)")
    time_in_phase: float = Field(description="Seconds in current phase")
    
    # Vehicle queues
    queues: Dict[str, int] = Field(description="Queue length per lane {north, south, east, west}")
    total_queue_length: int = Field(description="Sum of all queue lengths")
    vehicles_cleared_this_step: int = Field(description="Vehicles cleared in last step")
    
    # Emergency vehicles
    active_emergencies: int = Field(description="Number of active emergency vehicles")
    
    # Pedestrians
    waiting_pedestrians: int = Field(description="Number of pedestrians waiting to cross")
    
    # Lane conditions
    flooded_lanes: List[str] = Field(description="Lanes that are flooded/impassable")
    blocked_lanes: List[str] = Field(description="Lanes blocked by crashes/construction")
    crashed_this_step: bool = Field(description="Whether a crash occurred this step")
    
    # Phase management
    phase_history: List[tuple] = Field(default_factory=list, description="Recent phase changes")
    
    # Placeholder reward for OpenEnv server compatibility (not used, but required)
    reward: float = Field(default=0.0, description="Placeholder reward (not used)")
    done: bool = Field(default=False, description="Placeholder done flag (not used)")
    
    # Scores (for debugging/monitoring)
    safety_score: float = Field(description="Safety sub-score (0-1000)")
    emergency_score: float = Field(description="Emergency handling score (0-1000)")
    efficiency_score: float = Field(description="Throughput efficiency score (0-1000)")
    equity_score: float = Field(description="Pedestrian fairness score (0-1000)")

    class Config:
        json_schema_extra = {
            "example": {
                "time": 10.5,
                "phase": "ns",
                "time_in_phase": 5.2,
                "queues": {"north": 3, "south": 2, "east": 5, "west": 1},
                "total_queue_length": 11,
                "vehicles_cleared_this_step": 2,
                "active_emergencies": 0,
                "waiting_pedestrians": 1,
                "flooded_lanes": [],
                "blocked_lanes": [],
                "crashed_this_step": False,
                "phase_history": [],
                "safety_score": 950.0,
                "emergency_score": 1000.0,
                "efficiency_score": 850.0,
                "equity_score": 900.0,
            }
        }

# ============= REWARD MODEL =============

class RewardInfo(BaseModel):
    """Detailed reward breakdown"""
    total: float = Field(description="Total reward (0-1)")
    safety: float = Field(description="Safety component (0-1)")
    emergency: float = Field(description="Emergency component (0-1)")
    efficiency: float = Field(description="Efficiency component (0-1)")
    equity: float = Field(description="Equity component (0-1)")

# ============= INFO MODEL =============

class StepInfo(BaseModel):
    """Additional info returned with each step"""
    episode_time: float = Field(description="Current episode time")
    is_done: bool = Field(description="Whether episode is finished")
    info_text: Optional[str] = Field(default=None, description="Human-readable info")