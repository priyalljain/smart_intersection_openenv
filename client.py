from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from my_env.models import TrafficAction, TrafficObservation

class TrafficEnv(EnvClient[TrafficAction, TrafficObservation, State]):
    def _step_payload(self, action: TrafficAction) -> Dict:
        return {"phase": action.phase.value}

    def _parse_result(self, payload: Dict) -> StepResult[TrafficObservation]:
        obs_data = payload.get("observation", {})
        # Convert queues back to Lane enum keys (simplified – use strings for safety)
        queues_raw = obs_data.get("queues", {})
        from my_env.models import Lane
        queues = {Lane(k): v for k, v in queues_raw.items()}
        observation = TrafficObservation(
            queues=queues,
            current_phase=obs_data["current_phase"],
            time_in_phase=obs_data["time_in_phase"],
            emergency_detected=obs_data["emergency_detected"],
            emergency_lane=obs_data.get("emergency_lane"),
            pedestrian_waiting=obs_data["pedestrian_waiting"],
            blocked_lanes=[Lane(l) for l in obs_data.get("blocked_lanes", [])],
            flooded_lanes=[Lane(l) for l in obs_data.get("flooded_lanes", [])],
            time=obs_data["time"],
            crash_detected=obs_data["crash_detected"]
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