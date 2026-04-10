"""
Inference script for OpenEnv baseline.
Uses the injected OpenAI-compatible API endpoint when available.
Otherwise uses the heuristic agent.
Outputs required [START]/[STEP]/[END] logs.
"""

import os
import time

from dotenv import load_dotenv

from agents import HeuristicExpertAgent
from env import TrafficControlEnv
from models import Phase, TrafficAction

load_dotenv()

# --- MODIFIED: Strict 1-decimal boundaries ---
MIN_REWARD = 0.1
MAX_REWARD = 0.9

def clamp_reward(reward: float) -> float:
    """Ensure reward is strictly between 0.1 and 0.9 at 1-decimal precision."""
    try:
        val = float(reward)
    except (TypeError, ValueError):
        return MIN_REWARD
    
    if val <= MIN_REWARD:
        return MIN_REWARD
    if val >= MAX_REWARD:
        return MAX_REWARD
    return round(val, 1)

# If the grader injects an API base URL and key, always use that proxy.
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
USE_LLM = (
    API_BASE_URL is not None
    and API_BASE_URL.strip() != ""
    and API_KEY is not None
    and API_KEY.strip() != ""
    and not API_KEY.startswith("#")
)

def _safe_log(message: str) -> None:
    """Avoid Windows console encoding crashes during smoke tests."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="replace").decode("ascii"), flush=True)

if USE_LLM:
    from openai import OpenAI
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    _safe_log(f"Using LLM: {MODEL_NAME}")
else:
    _safe_log("Using heuristic agent only.")
    MODEL_NAME = "heuristic"

def run_episode(task_name):
    env = TrafficControlEnv(task=task_name)
    obs = env.reset()
    done = False
    step_num = 0
    rewards = []
    print(f"[START] task={task_name} env=traffic_control_env model={MODEL_NAME}", flush=True)

    fallback_agent = HeuristicExpertAgent(env)

    while not done and step_num < 100:
        action_str = "ns"
        error_msg = "null"

        if USE_LLM:
            prompt = (
                "You control a traffic light. "
                f"Current phase: {obs.phase}. "
                f"Queues: {obs.queues}. "
                f"Emergency vehicles present: {obs.active_emergencies > 0}. "
                f"Pedestrians waiting: {obs.waiting_pedestrians}. "
                "Respond with only 'ns' or 'ew'."
            )
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=10,
                    timeout=15.0,
                )
                action_str = response.choices[0].message.content.strip().lower()
                if action_str not in ["ns", "ew"]:
                    action_str = "ns"
            except Exception as e:
                error_msg = str(e).replace("\n", " ")[:50]
                action_str = fallback_agent.get_action(obs).phase.value
        else:
            action_str = fallback_agent.get_action(obs).phase.value

        action = TrafficAction(phase=Phase(action_str))
        obs, reward, done, _info = env.step(action)

        step_num += 1  
        reward = clamp_reward(reward)
        rewards.append(reward)

        # --- MODIFIED: reward formatted to .1f ---
        print(
            f"[STEP] step={step_num} action={action_str} reward={reward:.1f} "
            f"done={str(done).lower()} error={error_msg}",
            flush=True
        )
        time.sleep(0.05)

    # Calculate average reward
    avg_reward = sum(rewards) / step_num if step_num > 0 else MIN_REWARD
    
    # --- MODIFIED: Ensure score and rewards list use .1f ---
    score = clamp_reward(avg_reward)
    success = "true" if score > 0.5 else "false"
    rewards_str = ",".join([f"{clamp_reward(r):.1f}" for r in rewards])
    
    print(f"[END] success={success} steps={step_num} score={score:.1f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        try:
            run_episode(task)
        except Exception:
            # Emergency fallback line for validator
            print(f"[END] success=false steps=0 score=0.1 rewards=0.1", flush=True)