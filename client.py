from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import TrafficAction, TrafficObservation

class TrafficEnv(EnvClient[TrafficAction, TrafficObservation, State]):
    def _step_payload(self, action: TrafficAction) -> Dict:
        # Return the action as a plain dict (no "action" wrapper)
        return {"action": {"phase": action.phase.value}}

    def _parse_result(self, payload: Dict) -> StepResult[TrafficObservation]:
        obs_data = payload.get("observation", {})
        observation = TrafficObservation(
            time=obs_data.get("time", 0.0),
            phase=obs_data.get("phase", "ns"),
            time_in_phase=obs_data.get("time_in_phase", 0.0),
            queues=obs_data.get("queues", {}),
            total_queue_length=obs_data.get("total_queue_length", 0),
            vehicles_cleared_this_step=obs_data.get("vehicles_cleared_this_step", 0),
            active_emergencies=obs_data.get("active_emergencies", 0),
            waiting_pedestrians=obs_data.get("waiting_pedestrians", 0),
            flooded_lanes=obs_data.get("flooded_lanes", []),
            blocked_lanes=obs_data.get("blocked_lanes", []),
            crashed_this_step=obs_data.get("crashed_this_step", False),
            phase_history=obs_data.get("phase_history", []),
            safety_score=obs_data.get("safety_score", 1000.0),
            emergency_score=obs_data.get("emergency_score", 1000.0),
            efficiency_score=obs_data.get("efficiency_score", 1000.0),
            equity_score=obs_data.get("equity_score", 1000.0),
            reward=obs_data.get("reward", 0.0),
            done=obs_data.get("done", False),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0)
        )
