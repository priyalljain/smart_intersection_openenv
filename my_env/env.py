from typing import Tuple, Dict, Any, Optional
from my_env.models import TrafficAction, TrafficObservation, Phase
from my_env.simulator import TrafficSimulator

class TrafficControlEnv:
    """
    OpenEnv-compliant traffic control environment.
    
    KEY ISSUE FIX:
    - step() returns (obs, reward, done, info) for external use
    - step_async() returns ONLY obs for OpenEnv server
    - OpenEnv server calls step_async() and handles reward/done itself
    """
    
    def __init__(self, task: str = "easy"):
        if task not in ["easy", "medium", "hard"]:
            raise ValueError(f"Task must be easy/medium/hard, got {task}")
        self.task = task
        config = {
            "task": task,
            "max_episode_time": 300.0,
            "min_green_time": 15.0,
        }
        self.sim = TrafficSimulator(config)
        self.episode_count = 0
        self.step_count = 0
        self.current_reward = 0.0
        self.is_done = False

    def reset(self, seed: Optional[int] = None) -> TrafficObservation:
        """
        Reset environment.
        Returns ONLY observation (no tuple).
        """
        self.episode_count += 1
        self.step_count = 0
        self.current_reward = 0.0
        self.is_done = False
        obs_dict = self.sim.reset(seed)
        return self._dict_to_observation(obs_dict)

    async def reset_async(self, seed: Optional[int] = None) -> TrafficObservation:
        """
        Async reset for OpenEnv server.
        Returns ONLY observation.
        """
        return self.reset(seed)

    def step(self, action: TrafficAction) -> Tuple[TrafficObservation, float, bool, Dict[str, Any]]:
        """
        Synchronous step for manual agents/testing.
        Returns: (observation, reward, done, info)
        """
        self.step_count += 1
        obs_dict, reward, done = self.sim.step(action.phase, dt=1.0)
        obs = self._dict_to_observation(obs_dict)
        
        self.current_reward = reward
        self.is_done = done
        
        info = {
            "episode": self.episode_count,
            "step": self.step_count,
            "time": self.sim.time,
            "vehicles_processed": self.sim.total_vehicles_processed,
        }
        return obs, reward, done, info

    async def step_async(self, action: TrafficAction) -> TrafficObservation:
        """
        Async step for OpenEnv server.
        ⚠️ CRITICAL: Return ONLY the observation, NOT a tuple!
        OpenEnv will handle reward and done itself.
        """
        self.step_count += 1
        obs_dict, reward, done = self.sim.step(action.phase, dt=1.0)
        obs = self._dict_to_observation(obs_dict)
        
        self.current_reward = reward
        self.is_done = done
        
        return obs

    def state(self) -> Dict[str, Any]:
        """Return JSON-serializable state snapshot"""
        return self.sim.state()

    def close(self):
        """Close environment"""
        pass

    def _dict_to_observation(self, obs_dict: dict) -> TrafficObservation:
        """Convert simulator dict to Pydantic TrafficObservation"""
        return TrafficObservation(
            time=obs_dict["time"],
            phase=obs_dict["phase"],
            time_in_phase=obs_dict["time_in_phase"],
            queues={k: v for k, v in obs_dict["queues"].items()},
            total_queue_length=obs_dict.get("total_queue_length", 0),
            vehicles_cleared_this_step=obs_dict.get("vehicles_cleared_this_step", 0),
            active_emergencies=obs_dict.get("active_emergencies", 0),
            waiting_pedestrians=obs_dict.get("waiting_pedestrians", 0),
            flooded_lanes=obs_dict.get("flooded_lanes", []),
            blocked_lanes=obs_dict.get("blocked_lanes", []),
            crashed_this_step=obs_dict.get("crashed_this_step", False),
            phase_history=obs_dict.get("phase_history", []),
            safety_score=obs_dict.get("safety_score", 1000.0),
            emergency_score=obs_dict.get("emergency_score", 1000.0),
            efficiency_score=obs_dict.get("efficiency_score", 1000.0),
            equity_score=obs_dict.get("equity_score", 1000.0),
            reward=self.current_reward,
            done=self.is_done,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get episode metrics"""
        return {
            "episode": self.episode_count,
            "steps": self.step_count,
            "simulation_time": self.sim.time,
            "total_vehicles": self.sim.total_vehicles_processed,
            "vehicles_processed": self.sim.total_vehicles_processed,
            "avg_wait_time": (self.sim.total_wait_time / self.sim.total_vehicles_processed
                              if self.sim.total_vehicles_processed > 0 else 0.0),
            "vehicles_cleared": len(self.sim.cleared_vehicles),
            "emergency_events": len(self.sim.resolved_events),
            "scores": self.sim.scores_history[-1] if self.sim.scores_history else {},
        }