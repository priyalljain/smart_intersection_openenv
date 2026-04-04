from typing import Tuple, Dict, Any, Optional
from .models import TrafficAction, TrafficObservation, Phase
from .simulator import TrafficSimulator

class TrafficControlEnv:
    """
    OpenEnv-compliant traffic control environment.
    Implements reset(), step(), and state() methods.
    """
    def __init__(self, task: str = "easy"):
        config = {
            "task": task,
            "max_episode_time": 300.0,
            "min_green_time": 15.0,
        }
        self.sim = TrafficSimulator(config)

    def reset(self, seed: Optional[int] = None) -> TrafficObservation:
        """Reset the environment to initial state."""
        obs_dict = self.sim.reset(seed)
        return TrafficObservation(**obs_dict)

    def step(self, action: TrafficAction) -> Tuple[TrafficObservation, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        obs_dict, reward, done = self.sim.step(action.phase)
        return TrafficObservation(**obs_dict), reward, done, {}

    def state(self) -> Dict[str, Any]:
        """Return a JSON-serializable state snapshot."""
        return {
            "time": self.sim.time,
            "phase": self.sim.phase.value,
            "queues": {k.value: v for k, v in self.sim.queues.items()},
            "emergency_detected": self.sim.emergency_detected,
            "emergency_lane": self.sim.emergency_lane.value if self.sim.emergency_lane else None,
            "pedestrian_waiting": self.sim.pedestrian_waiting,
            "blocked_lanes": [l.value for l in self.sim.blocked_lanes],
            "flooded_lanes": [l.value for l in self.sim.flooded_lanes],
            "crash_detected": self.sim.crash_detected
        }