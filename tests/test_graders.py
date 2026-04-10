"""
Three task graders for easy, medium, hard.
Each runs a heuristic agent and returns average reward per step (0-1).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import TrafficControlEnv
from agents import HeuristicExpertAgent

STRICT_SCORE_EPSILON = 0.001
MIN_SCORE = 0.001
MAX_SCORE = 0.999

def _strict_unit_interval(value: float) -> float:
    # Clamp to the safe range
    clamped = max(MIN_SCORE, min(MAX_SCORE, value))
    
    # Double-check we're not at the boundaries
    if clamped <= 0.0 or clamped >= 1.0:
        # This should never happen, but just in case
        return 0.5  # Return middle value as fallback
    
    return clamped


def grade_task(task_name: str, episodes: int = 3) -> float:
    """
    Run multiple episodes and return average reward per step.
    
    Ensures the returned score is strictly between 0 and 1.
    """
    episode_scores = []
    
    for ep_idx in range(episodes):
        env = TrafficControlEnv(task=task_name)
        agent = HeuristicExpertAgent(env)
        total_reward, steps, _ = agent.run_episode(max_steps=200)
        
        # Calculate average reward per step
        if steps > 0:
            avg_reward_per_step = total_reward / steps
        else:
            # Edge case: 0 steps (shouldn't happen)
            avg_reward_per_step = STRICT_SCORE_EPSILON
        
        # Clamp the episode score to ensure it's strictly in (0, 1)
        episode_score = _strict_unit_interval(avg_reward_per_step)
        episode_scores.append(episode_score)
    
    # Calculate average of episode scores
    average_score = sum(episode_scores) / len(episode_scores)
    
    # Final clamp to ensure output is strictly in (0, 1)
    # This is the score returned by the grader
    final_score = _strict_unit_interval(average_score)
    
    return final_score
 
 
if __name__ == "__main__":
    easy_score = grade_task('easy')
    medium_score = grade_task('medium')
    hard_score = grade_task('hard')
    
    print(f"Easy:   {easy_score:.4f}")
    print(f"Medium: {medium_score:.4f}")
    print(f"Hard:   {hard_score:.4f}")
    
    # Verify all scores are strictly in (0, 1)
    for task, score in [('easy', easy_score), ('medium', medium_score), ('hard', hard_score)]:
        if score <= 0.0 or score >= 1.0:
            print(f"ERROR: {task} score {score} is NOT strictly in (0, 1)!")
        else:
            print(f"✓ {task} score {score} is valid (strictly in (0, 1))")