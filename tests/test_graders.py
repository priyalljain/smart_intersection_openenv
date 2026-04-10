"""
Three task graders for easy, medium, hard.
Each runs a heuristic agent and returns average reward per step (0-1).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import TrafficControlEnv
from agents import HeuristicExpertAgent

STRICT_SCORE_EPSILON = 1e-6


def _strict_unit_interval(value: float) -> float:
    """Clamp grader outputs to the validator's required open interval."""
    return min(1.0 - STRICT_SCORE_EPSILON, max(STRICT_SCORE_EPSILON, value))


def grade_task(task_name: str, episodes: int = 3) -> float:
    """Run multiple episodes and return average reward per step."""
    total_avg_reward = 0.0
    for _ in range(episodes):
        env = TrafficControlEnv(task=task_name)
        agent = HeuristicExpertAgent(env)
        total_reward, steps, _ = agent.run_episode(max_steps=200)
        # Average reward per step (normalized to 0-1)
        avg_reward = total_reward / steps if steps > 0 else STRICT_SCORE_EPSILON
        avg_reward = _strict_unit_interval(avg_reward)
        total_avg_reward += avg_reward
    return _strict_unit_interval(total_avg_reward / episodes)

if __name__ == "__main__":
    print(f"Easy:   {grade_task('easy'):.4f}")
    print(f"Medium: {grade_task('medium'):.4f}")
    print(f"Hard:   {grade_task('hard'):.4f}")
