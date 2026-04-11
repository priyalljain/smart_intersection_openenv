"""
Three task graders for easy, medium, hard.
PHASE 2 FIX: All scores clamped to 0.1-0.9 range (never 0.0 or 1.0)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import TrafficControlEnv
from agents import HeuristicExpertAgent

def clamp_score(value: float) -> float:
    """
    Clamp score to 0.1-0.9 range.
    CRITICAL: Never return 0.0 or 1.0 (validator rejects these).
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.1
    
    # Snap to safe zone
    if value <= 0.1:
        return 0.1
    if value >= 0.9:
        return 0.9
    
    return round(value, 1)


def grade_task(task_name: str, episodes: int = 3) -> float:
    """
    Grade a task: run episodes, return average reward.
    Output is strictly in 0.1-0.9 range.
    """
    episode_scores = []
    
    for ep_idx in range(episodes):
        env = TrafficControlEnv(task=task_name)
        agent = HeuristicExpertAgent(env)
        total_reward, steps, _ = agent.run_episode(max_steps=200)
        
        # Average reward per step
        if steps > 0:
            avg_reward = total_reward / steps
        else:
            avg_reward = 0.1
        
        # Clamp episode score to safe range
        episode_score = clamp_score(avg_reward)
        episode_scores.append(episode_score)
    
    # Average of episode scores
    average_score = sum(episode_scores) / len(episode_scores)
    
    # --- FINAL CLAMP ---
    # If the average is 0.0 or 0.99, move it to 0.1 or 0.9
    final_score = clamp_score(average_score)
    
    return final_score
 
 
if __name__ == "__main__":
    easy_score = grade_task('easy')
    medium_score = grade_task('medium')
    hard_score = grade_task('hard')
    
    print(f"\neasy_score: {easy_score:.2f}")
    print(f"medium_score: {medium_score:.2f}")
    print(f"hard_score: {hard_score:.2f}\n")
    
    # Validation: all scores must be in (0, 1) and NOT 0.0 or 1.0
    all_valid = True
    for task, score in [('easy', easy_score), ('medium', medium_score), ('hard', hard_score)]:
        # Check if strictly between 0 and 1
        if score <= 0.0 or score >= 1.0:
            print(f"✗ ERROR: {task} score {score} is out of bounds [0, 1]!")
            all_valid = False
        # Check if it's in our safe range
        elif score < 0.1 or score > 0.9:
            print(f"✗ WARNING: {task} score {score} is outside safe range [0.1, 0.9]!")
            all_valid = False
        else:
            print(f"✓ {task}: {score:.2f} (valid)")
    
    if all_valid:
        print("\n✓✓✓ ALL SCORES VALID - READY FOR PHASE 2 ✓✓✓")
        sys.exit(0)
    else:
        print("\n✗✗✗ SOME SCORES INVALID ✗✗✗")
        sys.exit(1)