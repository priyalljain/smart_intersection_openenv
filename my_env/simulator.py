import random
from typing import Dict, List, Tuple, Optional
from .models import Lane, Phase, TrafficObservation

class TrafficSimulator:
    def __init__(self, config: dict):
        self.cfg = config
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.time = 0.0
        self.phase = Phase.NS
        self.phase_timer = 0.0
        self.queues: Dict[Lane, int] = {Lane.NORTH: 0, Lane.SOUTH: 0, Lane.EAST: 0, Lane.WEST: 0}
        self.emergency_detected = False
        self.emergency_lane: Optional[Lane] = None
        self.emergency_arrival_time: Optional[float] = None
        self.pedestrian_waiting = False
        self.pedestrian_arrival_time: Optional[float] = None
        self.blocked_lanes: List[Lane] = []
        self.flooded_lanes: List[Lane] = []
        self.crash_detected = False
        self.last_phase_change: float = 0.0
        self._cleared_this_step = 0
        return self._get_observation()

    def step(self, action: Phase, dt: float = 1.0) -> Tuple[Dict, float, bool]:
        self.time += dt
        self._cleared_this_step = 0

        # Phase transition
        if action != self.phase:
            if self.phase_timer >= self.cfg.get("min_green_time", 15.0):
                self.phase = action
                self.phase_timer = 0.0
                self.last_phase_change = self.time
            # else penalty applied in reward
        self.phase_timer += dt

        # Clear vehicles based on phase
        for lane in self.queues:
            if lane in self.blocked_lanes or lane in self.flooded_lanes:
                continue
            if self.phase == Phase.NS and lane in [Lane.NORTH, Lane.SOUTH]:
                if self.queues[lane] > 0:
                    self.queues[lane] -= 1
                    self._cleared_this_step += 1
            elif self.phase == Phase.EW and lane in [Lane.EAST, Lane.WEST]:
                if self.queues[lane] > 0:
                    self.queues[lane] -= 1
                    self._cleared_this_step += 1

        # Spawn normal vehicles
        if random.random() < 0.3:
            lane = random.choice(list(Lane))
            self.queues[lane] += 1

        # Emergency vehicle (medium/hard tasks)
        if self.cfg.get("task") in ["medium", "hard"] and not self.emergency_detected:
            if random.random() < 0.005:
                self.emergency_detected = True
                self.emergency_lane = random.choice(list(Lane))
                self.emergency_arrival_time = self.time

        # Pedestrian
        if random.random() < 0.02 and not self.pedestrian_waiting:
            self.pedestrian_waiting = True
            self.pedestrian_arrival_time = self.time

        # Flood (hard task only)
        if self.cfg.get("task") == "hard" and random.random() < 0.001:
            lane = random.choice(list(Lane))
            if lane not in self.flooded_lanes:
                self.flooded_lanes.append(lane)

        # Crash detection
        if not self.crash_detected and random.random() < 0.001:
            self.crash_detected = True
            lane = random.choice(list(Lane))
            self.blocked_lanes.append(lane)

        # Reward
        reward = self._calculate_reward()
        done = self.time >= self.cfg.get("max_episode_time", 300.0)
        return self._get_observation(), reward, done

    def _calculate_reward(self) -> float:
        """
        Dense reward with heavy penalties for inefficiency, emergencies, and safety violations.
        Returns a value in [0.0, 1.0] after clamping.
        """
        # Start with a neutral baseline (0.5 after division)
        score = 500.0

        # 1. Queue length penalty (linear, heavy)
        total_queue = sum(self.queues.values())
        queue_penalty = min(300, total_queue * 15)
        score -= queue_penalty

        # 2. Starvation penalty: if any lane has >10 vehicles waiting
        for lane, q in self.queues.items():
            if q > 10:
                score -= 50
            if q > 20:
                score -= 100

        # 3. Emergency vehicle waiting penalty (progressive)
        if self.emergency_detected and self.emergency_lane and self.emergency_arrival_time:
            wait_time = self.time - self.emergency_arrival_time
            if wait_time > 30:
                score -= 400
            elif wait_time > 15:
                score -= 200
            elif wait_time > 5:
                score -= 50

        # 4. Pedestrian waiting penalty
        if self.pedestrian_waiting and self.pedestrian_arrival_time:
            ped_wait = self.time - self.pedestrian_arrival_time
            if ped_wait > 45:
                score -= 200
            elif ped_wait > 25:
                score -= 80
            elif ped_wait > 10:
                score -= 20

        # 5. Crash penalty (already -500 in detect_crashes, but add extra)
        if self.crash_detected:
            score -= 200

        # 6. Flooded lane penalty (per lane, per step)
        score -= len(self.flooded_lanes) * 10

        # 7. Phase thrashing penalty
        if self.last_phase_change and (self.time - self.last_phase_change) < 10:
            score -= 20

        # 8. Positive reward for clearing vehicles (small, capped)
        score += min(50, self._cleared_this_step * 5)

        # 9. Reward for giving green to emergency lane
        if self.emergency_detected and self.emergency_lane:
            if (self.phase == Phase.NS and self.emergency_lane in [Lane.NORTH, Lane.SOUTH]) or \
               (self.phase == Phase.EW and self.emergency_lane in [Lane.EAST, Lane.WEST]):
                score += 20

        # 10. Penalty for wasting green on empty lanes
        allowed_lanes = []
        if self.phase == Phase.NS:
            allowed_lanes = [Lane.NORTH, Lane.SOUTH]
        else:
            allowed_lanes = [Lane.EAST, Lane.WEST]
        total_allowed_queue = sum(self.queues[lane] for lane in allowed_lanes
                                  if lane not in self.blocked_lanes and lane not in self.flooded_lanes)
        if total_allowed_queue == 0 and self.phase_timer > 5:
            score -= 15

        # Clamp and convert to [0,1]
        final_score = max(0.0, min(1000.0, score))
        return final_score / 1000.0

    def _get_observation(self) -> dict:
        return {
            "queues": self.queues,
            "current_phase": self.phase,
            "time_in_phase": self.phase_timer,
            "emergency_detected": self.emergency_detected,
            "emergency_lane": self.emergency_lane,
            "pedestrian_waiting": self.pedestrian_waiting,
            "blocked_lanes": self.blocked_lanes,
            "flooded_lanes": self.flooded_lanes,
            "time": self.time,
            "crash_detected": self.crash_detected
        }