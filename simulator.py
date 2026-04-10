"""
Advanced Traffic Simulator with Real-World Logic
- Priority-based event handling
- Dense reward signals
- Stochastic but deterministic (seeded) randomness
- Comprehensive edge case handling
- Crash resolution (30 sec clear time)
- Flood resolution (60–120 sec duration)
- Max green time (60 sec) to prevent starvation
- Fairness penalty (queue imbalance)
- Emergency preemption overrides all-red
"""

import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

STRICT_REWARD_EPSILON = 0.01

class Lane(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

class Phase(str, Enum):
    NS = "ns"
    EW = "ew"
    ALL_RED = "all_red"

class VehicleType(str, Enum):
    NORMAL = "normal"
    AMBULANCE = "ambulance"
    FIRE = "fire"
    POLICE = "police"
    PRISONER = "prisoner"
    SCHOOL_BUS = "school_bus"

class EventPriority(Enum):
    P0_CRASH = 0
    P1_AMBULANCE = 1
    P2_FIRE = 2
    P3_POLICE = 3
    P4_PRISONER = 4
    P5_ELDERLY_PEDESTRIAN = 5
    P6_SCHOOL_BUS = 6
    P7_FLOODED_LANE = 7
    P8_NORMAL = 8

@dataclass
class Vehicle:
    id: int
    lane: Lane
    type: VehicleType = VehicleType.NORMAL
    arrival_time: float = 0.0
    priority: int = 100
    def wait_time(self, current_time: float) -> float:
        return current_time - self.arrival_time

@dataclass
class Pedestrian:
    id: int
    arrival_time: float
    is_elderly: bool = False
    is_child: bool = False
    max_wait_time: float = 60.0
    def wait_time(self, current_time: float) -> float:
        return current_time - self.arrival_time

@dataclass
class Event:
    event_id: int
    event_type: str
    priority: EventPriority
    lane: Optional[Lane] = None
    data: Dict = field(default_factory=dict)
    created_at: float = 0.0
    duration: Optional[float] = None   # for flood events

class TrafficSimulator:
    def __init__(self, config: dict):
        self.cfg = config
        self.task = config.get("task", "easy")
        self.max_episode_time = config.get("max_episode_time", 300.0)
        self.min_green_time = config.get("min_green_time", 15.0)
        self.max_green_time = 60.0                 # enforce maximum green
        self.max_yellow_time = config.get("max_yellow_time", 5.0)
        self.all_red_duration = 2.0
        self.crash_clear_time = 30.0               # seconds before lane unblocks after crash
        self.flood_min_duration = 60.0
        self.flood_max_duration = 120.0

        # State
        self.time = 0.0
        self.phase = Phase.NS
        self.phase_timer = 0.0
        self.last_phase_change = 0.0
        self.phase_history = []

        # Vehicles
        self.vehicles: Dict[Lane, List[Vehicle]] = {lane: [] for lane in Lane}
        self.vehicle_id_counter = 0
        self.cleared_vehicles = []
        self.vehicles_cleared_this_step = 0
        self.total_vehicles_processed = 0
        self.total_wait_time = 0.0
        self.vehicles_spawned = 0
        self.emergency_vehicles_processed = 0

        # Pedestrians
        self.pedestrians: Dict[Lane, List[Pedestrian]] = {lane: [] for lane in Lane}
        self.pedestrian_id_counter = 0
        self.pedestrians_served = []

        # Events
        self.active_events: List[Event] = []
        self.event_id_counter = 0
        self.resolved_events = []

        # Lane conditions
        self.flooded_lanes: set = set()
        self.blocked_lanes: set = set()
        self.crashed_lanes: set = set()
        self.crashed_this_step = False

        # Scores
        self.safety_score = 1000.0
        self.emergency_score = 1000.0
        self.efficiency_score = 1000.0
        self.equity_score = 1000.0
        self.scores_history = []
        self.emergency_clear_time = {}

        self.reset()

    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is not None:
            random.seed(seed)
        self.time = 0.0
        self.phase = Phase.NS
        self.phase_timer = 0.0
        self.last_phase_change = 0.0
        self.phase_history = []
        self.vehicles = {lane: [] for lane in Lane}
        self.vehicle_id_counter = 0
        self.cleared_vehicles = []
        self.vehicles_cleared_this_step = 0
        self.total_vehicles_processed = 0
        self.total_wait_time = 0.0
        self.vehicles_spawned = 0
        self.emergency_vehicles_processed = 0
        self.pedestrians = {lane: [] for lane in Lane}
        self.pedestrian_id_counter = 0
        self.pedestrians_served = []
        self.active_events = []
        self.event_id_counter = 0
        self.resolved_events = []
        self.flooded_lanes = set()
        self.blocked_lanes = set()
        self.crashed_lanes = set()
        self.crashed_this_step = False
        self.safety_score = 1000.0
        self.emergency_score = 1000.0
        self.efficiency_score = 1000.0
        self.equity_score = 1000.0
        self.scores_history = []
        self.emergency_clear_time = {}
        self._spawn_initial_vehicles()
        return self._get_observation()

    def _spawn_initial_vehicles(self):
        for lane in Lane:
            for _ in range(random.randint(1, 3)):
                self._spawn_vehicle(lane, VehicleType.NORMAL)

    def step(self, action: Phase, dt: float = 1.0) -> Tuple[dict, float, bool]:
        self.time += dt
        self.vehicles_cleared_this_step = 0
        self.crashed_this_step = False

        self._handle_phase_transition(action)
        self._clear_vehicles(self.phase, dt)
        self._spawn_random_events(dt)
        self._update_pedestrians(dt)
        self._detect_crashes()
        self._resolve_events()
        reward = self._calculate_comprehensive_reward()
        # Clamp reward to be strictly between EPSILON and (1.0 - EPSILON)
        # Using 0.001 gives margin: reward in (0.001, 0.999)
        min_reward = STRICT_REWARD_EPSILON
        max_reward = 1.0 - STRICT_REWARD_EPSILON
        reward = max(min_reward, min(max_reward, reward))

        # Double-check we're not at the boundaries (floating point safety)
        if reward <= 0.0:
            reward = min_reward
        if reward >= 1.0:
            reward = max_reward
        done = self.time >= self.max_episode_time
        return self._get_observation(), reward, done

    # ---- Emergency Preemption ----
    def _emergency_preempt_active(self) -> bool:
        for event in self.active_events:
            if event.event_type == "emergency" and event.lane:
                return True
        return False

    def _apply_emergency_preemption(self):
        for event in self.active_events:
            if event.event_type == "emergency" and event.lane:
                if event.lane in [Lane.NORTH, Lane.SOUTH]:
                    self.phase = Phase.NS
                else:
                    self.phase = Phase.EW
                self.phase_timer = 0.0
                self.last_phase_change = self.time
                break

    # ---- Phase Management with Max Green Time ----
    def _handle_phase_transition(self, action: Phase):
        # Emergency preemption overrides everything
        if self._emergency_preempt_active():
            self._apply_emergency_preemption()
            return

        # ALL_RED handling
        if self.phase == Phase.ALL_RED:
            self.phase_timer += 1.0
            if self.phase_timer >= self.all_red_duration:
                new_phase = action if action != Phase.ALL_RED else Phase.NS
                self.phase = new_phase
                self.phase_timer = 0.0
                self.last_phase_change = self.time
                self.phase_history.append((self.time, self.phase.value))
            return

        # Force switch if maximum green time exceeded (prevents starvation)
        if self.phase_timer >= self.max_green_time:
            new_phase = Phase.EW if self.phase == Phase.NS else Phase.NS
            self.phase = new_phase
            self.phase_timer = 0.0
            self.last_phase_change = self.time
            self.phase_history.append((self.time, self.phase.value))
            return

        if action != self.phase:
            if self.phase_timer >= self.min_green_time:
                self.phase = action
                self.phase_timer = 0.0
                self.last_phase_change = self.time
                self.phase_history.append((self.time, self.phase.value))
            else:
                self.safety_score = max(0, self.safety_score - 10)
                self.safety_score = min(1000, self.safety_score)  # Cap at max
        self.phase_timer += 1.0

    # ---- Vehicle Movement ----
    def _clear_vehicles(self, phase: Phase, dt: float):
        if phase == Phase.ALL_RED:
            return
        allowed_lanes = [Lane.NORTH, Lane.SOUTH] if phase == Phase.NS else [Lane.EAST, Lane.WEST]
        for lane in allowed_lanes:
            if lane in self.flooded_lanes or lane in self.blocked_lanes:
                continue
            queue = self.vehicles[lane]
            while queue:
                vehicle = queue[0]
                if self._should_preempt_for_emergency(vehicle):
                    break
                queue.pop(0)
                self.cleared_vehicles.append(vehicle)
                self.vehicles_cleared_this_step += 1
                self.total_vehicles_processed += 1
                self.total_wait_time += vehicle.wait_time(self.time)
                if vehicle.type in [VehicleType.AMBULANCE, VehicleType.FIRE, VehicleType.POLICE]:
                    self.emergency_vehicles_processed += 1

    def _spawn_vehicle(self, lane: Lane, vehicle_type: VehicleType = VehicleType.NORMAL) -> Vehicle:
        v = Vehicle(self.vehicle_id_counter, lane, vehicle_type, self.time)
        self.vehicle_id_counter += 1
        self.vehicles[lane].append(v)
        self.vehicles_spawned += 1
        return v

    def _should_preempt_for_emergency(self, vehicle: Vehicle) -> bool:
        if vehicle.type in [VehicleType.AMBULANCE, VehicleType.FIRE, VehicleType.POLICE]:
            for event in self.active_events:
                if event.event_type == "emergency" and event.lane == vehicle.lane:
                    return True
        return False

    # ---- Event Spawning ----
    def _spawn_random_events(self, dt: float):
        # Normal vehicles
        if random.random() < 0.35 * dt:
            lane = random.choice(list(Lane))
            if lane not in self.flooded_lanes:
                self._spawn_vehicle(lane, VehicleType.NORMAL)

        # Emergencies (medium/hard)
        if self.task in ["medium", "hard"]:
            if random.random() < 0.003 * dt:
                lane = random.choice(list(Lane))
                self._spawn_vehicle(lane, VehicleType.AMBULANCE)
                self._create_event("emergency", lane, EventPriority.P1_AMBULANCE)
            if random.random() < 0.002 * dt:
                lane = random.choice(list(Lane))
                self._spawn_vehicle(lane, VehicleType.FIRE)
                self._create_event("emergency", lane, EventPriority.P2_FIRE)

        # Pedestrians
        if random.random() < 0.04 * dt:
            lane = random.choice(list(Lane))
            is_elderly = random.random() < 0.2
            is_child = random.random() < 0.15
            ped = Pedestrian(self.pedestrian_id_counter, self.time, is_elderly, is_child)
            self.pedestrian_id_counter += 1
            self.pedestrians[lane].append(ped)

        # Flooding (hard task) – with random duration
        if self.task == "hard" and random.random() < 0.001 * dt:
            lane = random.choice(list(Lane))
            if lane not in self.flooded_lanes:
                self.flooded_lanes.add(lane)
                duration = random.uniform(self.flood_min_duration, self.flood_max_duration)
                self._create_event("flood", lane, EventPriority.P7_FLOODED_LANE, duration=duration)

        # Crashes (rare)
        if random.random() < 0.0005 * dt:
            lane = random.choice(list(Lane))
            if lane not in self.crashed_lanes:
                self.crashed_lanes.add(lane)
                self.blocked_lanes.add(lane)
                self.crashed_this_step = True
                self._create_event("crash", lane, EventPriority.P0_CRASH)

    def _create_event(self, event_type: str, lane: Optional[Lane], priority: EventPriority, duration: Optional[float] = None):
        event = Event(self.event_id_counter, event_type, priority, lane, created_at=self.time, duration=duration)
        self.event_id_counter += 1
        self.active_events.append(event)

    def _update_pedestrians(self, dt: float):
        for lane in self.pedestrians:
            peds = self.pedestrians[lane]
            peds_to_remove = []
            for ped in peds:
                wait = ped.wait_time(self.time)
                if wait > ped.max_wait_time:
                    self.equity_score = max(0, self.equity_score - 50)
                    peds_to_remove.append(ped)
                    self.pedestrians_served.append(ped)
                # Crossing opportunity every 60 seconds (simplified)
                if int(self.time) % 60 == 0 and self.phase_timer < 1:
                    peds_to_remove.append(ped)
                    self.pedestrians_served.append(ped)
            for ped in peds_to_remove:
                peds.remove(ped)

    def _detect_crashes(self):
        if self.crashed_this_step:
            self.safety_score = max(0, self.safety_score - 500)
            self.phase = Phase.ALL_RED
            self.phase_timer = 0.0

    def _resolve_events(self):
        to_remove = []
        for event in self.active_events:
            if event.event_type == "emergency":
                if event.lane and self._is_lane_green(event.lane):
                    wait_time = self.time - event.created_at
                    self.emergency_clear_time[event.event_id] = wait_time
                    to_remove.append(event)
                    if wait_time < 10:
                        self.emergency_score = min(1000, self.emergency_score + 200)
                    elif wait_time < 30:
                        self.emergency_score = min(1000, self.emergency_score + 100)
                    elif wait_time < 60:
                        self.emergency_score = max(0, self.emergency_score - 100)
                    else:
                        self.emergency_score = max(0, self.emergency_score - 300)
            elif event.event_type == "flood":
                if event.duration and (self.time - event.created_at) >= event.duration:
                    if event.lane:
                        self.flooded_lanes.discard(event.lane)
                    to_remove.append(event)
                else:
                    if event.lane and self._is_lane_green(event.lane):
                        self.efficiency_score = max(0, self.efficiency_score - 10)
            elif event.event_type == "crash":
                if self.time - event.created_at >= self.crash_clear_time:
                    if event.lane:
                        self.crashed_lanes.discard(event.lane)
                        self.blocked_lanes.discard(event.lane)
                    to_remove.append(event)
        for event in to_remove:
            self.active_events.remove(event)
            self.resolved_events.append(event)

    def _is_lane_green(self, lane: Lane) -> bool:
        if self.phase == Phase.ALL_RED:
            return False
        if self.phase == Phase.NS:
            return lane in [Lane.NORTH, Lane.SOUTH]
        else:
            return lane in [Lane.EAST, Lane.WEST]

    # ---- Reward with Fairness Penalty ----
    def _calculate_comprehensive_reward(self) -> float:
        # Efficiency: queue penalty
        total_queue = sum(len(self.vehicles[l]) for l in Lane)
        self.efficiency_score = max(0, 1000 - total_queue * 15)
        if self.vehicles_cleared_this_step > 0:
            self.efficiency_score = min(1000, self.efficiency_score + min(50, self.vehicles_cleared_this_step * 10))
        for lane in Lane:
            if self._is_lane_green(lane) and len(self.vehicles[lane]) == 0:
                if lane not in self.flooded_lanes:
                    self.efficiency_score = max(0, self.efficiency_score - 5)

        # Equity: pedestrian waiting and fairness between directions
        for lane_peds in self.pedestrians.values():
            for ped in lane_peds:
                wait = ped.wait_time(self.time)
                if wait > ped.max_wait_time * 0.5:
                    self.equity_score = max(0, self.equity_score - 20)

        # Fairness penalty: if one direction's queue is more than double the other
        ns_queue = len(self.vehicles[Lane.NORTH]) + len(self.vehicles[Lane.SOUTH])
        ew_queue = len(self.vehicles[Lane.EAST]) + len(self.vehicles[Lane.WEST])
        if ns_queue > 0 or ew_queue > 0:
            queue_ratio = max(ns_queue, ew_queue) / (min(ns_queue, ew_queue) + 1)
            if queue_ratio > 2.0:
                fairness_penalty = min(200, (queue_ratio - 2) * 50)
                self.equity_score = max(0, self.equity_score - fairness_penalty)

        # Safety: thrashing penalty
        if len(self.phase_history) > 0:
            phase_changes = len(self.phase_history)
            expected_changes = int(self.time / self.min_green_time) + 1
            if phase_changes > expected_changes * 1.5:
                self.safety_score = max(0, self.safety_score - 20)

        # Combine
        combined = (self.safety_score * 0.30 +
                    self.emergency_score * 0.35 +
                    self.efficiency_score * 0.20 +
                    self.equity_score * 0.15)
        combined = max(0.0, min(1000.0, combined))
        reward = combined / 1000.0
        # FORCE margin: strictly between 0.01 and 0.99
        reward = max(STRICT_REWARD_EPSILON, min(1.0 - STRICT_REWARD_EPSILON, reward))

        self.scores_history.append({
            'time': self.time,
            'safety': self.safety_score,
            'emergency': self.emergency_score,
            'efficiency': self.efficiency_score,
            'equity': self.equity_score,
            'combined': combined
        })
        return round(float(reward), 4)

    def _get_observation(self) -> dict:
        return {
            "time": self.time,
            "phase": self.phase.value,
            "time_in_phase": self.phase_timer,
            "queues": {lane.value: len(self.vehicles[lane]) for lane in Lane},
            "total_queue_length": sum(len(self.vehicles[l]) for l in Lane),
            "active_emergencies": len([e for e in self.active_events if e.event_type == "emergency"]),
            "waiting_pedestrians": sum(len(p) for p in self.pedestrians.values()),
            "flooded_lanes": [l.value for l in self.flooded_lanes],
            "blocked_lanes": [l.value for l in self.blocked_lanes],
            "crashed_this_step": self.crashed_this_step,
            "vehicles_cleared_this_step": self.vehicles_cleared_this_step,
            "phase_history": self.phase_history[-10:],
            "safety_score": self.safety_score,
            "emergency_score": self.emergency_score,
            "efficiency_score": self.efficiency_score,
            "equity_score": self.equity_score,
        }

    def state(self) -> dict:
        return self._get_observation()
