import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_env.env import TrafficControlEnv
from my_env.models import TrafficAction, Phase

def grade_task(task_name):
    env = TrafficControlEnv(task=task_name)
    obs = env.reset()
    total_reward = 0.0
    done = False
    steps = 0
    while not done and steps < 500:
        action = TrafficAction(phase=Phase.NS)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
    return total_reward / steps if steps > 0 else 0.0

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        score = grade_task(task)
        print(f"{task}: {score:.4f}")
        assert 0.0 <= score <= 1.0