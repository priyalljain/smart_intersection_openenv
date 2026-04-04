from pydantic import BaseModel
from typing import List, Dict, Optional, Literal
from enum import Enum

class Lane(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

class Phase(str, Enum):
    NS = "ns"
    EW = "ew"

class TrafficAction(BaseModel):
    phase: Phase

class TrafficObservation(BaseModel):
    queues: Dict[Lane, int]
    current_phase: Phase
    time_in_phase: float
    emergency_detected: bool
    emergency_lane: Optional[Lane] = None
    pedestrian_waiting: bool
    blocked_lanes: List[Lane]
    flooded_lanes: List[Lane]
    time: float
    crash_detected: bool