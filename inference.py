# import os
# from openai import OpenAI  # Required import – even if not used
# from dotenv import load_dotenv
# from my_env.env import TrafficControlEnv
# from my_env.models import TrafficAction, Phase

# load_dotenv()  # Optional – no keys needed

# def run_episode(task_name):
#     env = TrafficControlEnv(task=task_name)
#     obs = env.reset()
#     done = False
#     step_num = 0
#     rewards = []
#     print(f"[START] task={task_name} env=traffic_control_env model=rule-based")
#     while not done and step_num < 100:
#         # Simple rule‑based agent
#         if obs.emergency_detected:
#             # Give green to the lane with the emergency vehicle
#             if obs.emergency_lane in ["north", "south"]:
#                 action_str = "ns"
#             else:
#                 action_str = "ew"
#         else:
#             # Round‑robin: alternate phases every 15 steps
#             if step_num % 30 < 15:
#                 action_str = "ns"
#             else:
#                 action_str = "ew"
#         action = TrafficAction(phase=Phase(action_str))
#         obs, reward, done, info = env.step(action)
#         rewards.append(reward)
#         step_num += 1
#         print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")
#     success = "true" if sum(rewards) / step_num > 0.5 else "false"
#     rewards_str = ",".join([f"{r:.2f}" for r in rewards])
#     print(f"[END] success={success} steps={step_num} rewards={rewards_str}")

# if __name__ == "__main__":
#     for task in ["easy", "medium", "hard"]:
#         run_episode(task)
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from my_env.env import TrafficControlEnv
from my_env.models import TrafficAction, Phase

load_dotenv()

# Correct Hugging Face router endpoint for Meta Llama 3 (free, works)
HARDCODED_API_BASE = "https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.2-3B-Instruct/v1"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
if not API_KEY:
    raise ValueError("Missing API key. Set OPENAI_API_KEY or HF_TOKEN in .env or environment.")

client = OpenAI(base_url=HARDCODED_API_BASE, api_key=API_KEY)

def run_episode(task_name):
    env = TrafficControlEnv(task=task_name)
    obs = env.reset()
    done = False
    step_num = 0
    rewards = []
    print(f"[START] task={task_name} env=traffic_control_env model={MODEL_NAME}")
    while not done and step_num < 30:
        prompt = f"""You control a traffic light. Current phase: {obs.current_phase.value}. Queues: {obs.queues}. Emergency: {obs.emergency_detected}. Respond with only 'ns' or 'ew'."""
        action_str = "ns"
        error_msg = "null"
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=10,
                timeout=15.0
            )
            action_str = response.choices[0].message.content.strip().lower()
            if action_str not in ["ns", "ew"]:
                action_str = "ns"
        except Exception as e:
            error_msg = str(e).replace("\n", " ").replace('"', "'")
            # Fallback: simple rule
            if obs.emergency_detected:
                action_str = "ns" if obs.emergency_lane in ["north", "south"] else "ew"
            else:
                action_str = "ns" if step_num % 2 == 0 else "ew"
        action = TrafficAction(phase=Phase(action_str))
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        step_num += 1
        print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}")
        time.sleep(0.2)
    success = "true" if sum(rewards) / step_num > 0.5 else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={success} steps={step_num} rewards={rewards_str}")

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_episode(task)