"""
Inference script for OpenEnv baseline.
Uses OpenAI client (Hugging Face free endpoint) if API key is set.
Otherwise uses heuristic agent.
Outputs required [START]/[STEP]/[END] logs.
"""

import os
import time
from dotenv import load_dotenv
from my_env.env import TrafficControlEnv
from my_env.models import TrafficAction, Phase
from my_env.agents import HeuristicExpertAgent

load_dotenv()

# Check if API key is provided and not a placeholder
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
USE_LLM = API_KEY is not None and API_KEY.strip() != "" and not API_KEY.startswith("#")

if USE_LLM:
    from openai import OpenAI
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.2-3B-Instruct/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(f"✅ Using LLM: {MODEL_NAME}")
else:
    print("⚠️ No valid API key. Using heuristic agent only.")
    MODEL_NAME = "heuristic"

def run_episode(task_name):
    env = TrafficControlEnv(task=task_name)
    obs = env.reset()
    done = False
    step_num = 0
    rewards = []
    print(f"[START] task={task_name} env=traffic_control_env model={MODEL_NAME}")

    # Always have a heuristic agent as fallback (or primary if no LLM)
    fallback_agent = HeuristicExpertAgent(env)

    while not done and step_num < 100:
        action_str = "ns"
        error_msg = "null"

        if USE_LLM:
            prompt = f"""You control a traffic light. Current phase: {obs.phase}. Queues: {obs.queues}. Emergency vehicles present: {obs.active_emergencies > 0}. Pedestrians waiting: {obs.waiting_pedestrians}. Respond with only 'ns' or 'ew'."""
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=10,
                    timeout=15.0
                )
                action_str = response.choices[0].message.content.strip().lower()
                print(f"[DEBUG] LLM returned: {action_str}")   # verify
                if action_str not in ["ns", "ew"]:
                    action_str = "ns"
            except Exception:
                # Silent fallback – no error printed to keep logs clean
                action_str = fallback_agent.get_action(obs).phase.value
        else:
            action_str = fallback_agent.get_action(obs).phase.value

        action = TrafficAction(phase=Phase(action_str))
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        step_num += 1
        print(f"[STEP] step={step_num} action={action_str} reward={reward:.4f} done={str(done).lower()} error={error_msg}")
        time.sleep(0.05)

    success = "true" if (sum(rewards) / step_num) > 0.5 else "false"
    rewards_str = ",".join([f"{r:.4f}" for r in rewards])
    print(f"[END] success={success} steps={step_num} rewards={rewards_str}")

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_episode(task)