"""
Multiple agent strategies for testing the environment.
No OpenAI API required - uses heuristics, simple RL, and local LLMs.
"""

import random
import numpy as np
from typing import Tuple
from my_env.env import TrafficControlEnv
from my_env.models import TrafficAction, Phase

# ============= BASE AGENT =============

class TrafficAgent:
    """Base class for traffic control agents"""
    
    def __init__(self, env: TrafficControlEnv):
        self.env = env
        self.episode_rewards = []
        self.episode_steps = 0

    def get_action(self, obs) -> TrafficAction:
        """Return action based on observation"""
        raise NotImplementedError

    def run_episode(self, max_steps: int = 300) -> Tuple[float, int, dict]:
        """Run one complete episode"""
        obs = self.env.reset()
        total_reward = 0.0
        steps = 0
        
        for step in range(max_steps):
            action = self.get_action(obs)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            steps = step + 1
            
            if done:
                break
        
        self.episode_rewards.append(total_reward)
        self.episode_steps = steps
        metrics = self.env.get_metrics()
        
        return total_reward, steps, metrics

# ============= AGENT 1: ROUND-ROBIN =============

class RoundRobinAgent(TrafficAgent):
    """Simple round-robin: alternate phases"""
    
    def __init__(self, env: TrafficControlEnv, switch_interval: int = 30):
        super().__init__(env)
        self.switch_interval = switch_interval
        self.step_counter = 0
    
    def get_action(self, obs) -> TrafficAction:
        self.step_counter += 1
        if (self.step_counter // self.switch_interval) % 2 == 0:
            return TrafficAction(phase=Phase.NS)
        else:
            return TrafficAction(phase=Phase.EW)

# ============= AGENT 2: QUEUE-BASED =============

class QueueBasedAgent(TrafficAgent):
    """
    Adaptive agent: prioritize the direction with longer queue.
    Also handles emergencies.
    """
    
    def __init__(self, env: TrafficControlEnv):
        super().__init__(env)
        self.last_phase = Phase.NS
        self.phase_timer = 0
        self.min_phase_time = 15
    
    def get_action(self, obs) -> TrafficAction:
        self.phase_timer += 1
        
        # Handle emergencies first
        if obs.active_emergencies > 0:
            # Heuristic: alternate to serve emergency
            if self.last_phase == Phase.NS:
                action = Phase.EW
            else:
                action = Phase.NS
            self.last_phase = action
            self.phase_timer = 0
            return TrafficAction(phase=action)
        
        # Respect minimum phase time
        if self.phase_timer < self.min_phase_time:
            return TrafficAction(phase=self.last_phase)
        
        # Choose based on queue lengths
        ns_queue = obs.queues.get("north", 0) + obs.queues.get("south", 0)
        ew_queue = obs.queues.get("east", 0) + obs.queues.get("west", 0)
        
        if ns_queue > ew_queue:
            action = Phase.NS
        else:
            action = Phase.EW
        
        if action != self.last_phase:
            self.phase_timer = 0
        
        self.last_phase = action
        return TrafficAction(phase=action)

# ============= AGENT 3: PREDICTIVE (SIMPLE RL) =============

class PredictiveAgent(TrafficAgent):
    """
    Simple Q-learning style agent that learns which phase is better.
    No neural network - uses lookup table.
    """

    def __init__(self, env: TrafficControlEnv, epsilon: float = 0.1):
        super().__init__(env)
        self.epsilon = epsilon
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        
        # Q-table: state -> action -> q-value
        # State: (ns_queue > ew_queue, has_emergency, has_pedestrian)
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
    
    def _get_state(self, obs) -> Tuple[int, int, int]:
        """Hash observation to state"""
        ns_queue = obs.queues.get("north", 0) + obs.queues.get("south", 0)
        ew_queue = obs.queues.get("east", 0) + obs.queues.get("west", 0)
        
        state = (
            1 if ns_queue > ew_queue else 0,
            1 if obs.active_emergencies > 0 else 0,
            1 if obs.waiting_pedestrians > 0 else 0,
        )
        return state
    
    def get_action(self, obs) -> TrafficAction:
        state = self._get_state(obs)
        
        # Initialize Q-table for new state
        if state not in self.q_table:
            self.q_table[state] = {
                Phase.NS: 0.0,
                Phase.EW: 0.0,
            }
        
        # Update previous action
        if self.last_state is not None:
            old_q = self.q_table[self.last_state][self.last_action]
            max_q = max(self.q_table[state].values())
            new_q = old_q + self.learning_rate * (
                self.last_reward + self.discount_factor * max_q - old_q
            )
            self.q_table[self.last_state][self.last_action] = new_q
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action = random.choice([Phase.NS, Phase.EW])
        else:
            # Best action
            action = max(self.q_table[state], key=self.q_table[state].get)
        
        self.last_state = state
        self.last_action = action
        
        return TrafficAction(phase=action)

# ============= AGENT 4: HEURISTIC EXPERT =============

class HeuristicExpertAgent(TrafficAgent):
    """
    Knowledge-based agent with hand-coded heuristics.
    Represents domain expert knowledge.
    """
    
    def __init__(self, env: TrafficControlEnv):
        super().__init__(env)
        self.phase_timer = 0
        self.current_phase = Phase.NS
        self.min_green = 15
        self.max_green = 60
    
    def get_action(self, obs) -> TrafficAction:
        self.phase_timer += 1
        
        # P0: Handle crashes -> all red
        if obs.crashed_this_step:
            return TrafficAction(phase=Phase.NS)  # Stay put
        
        # P1-P3: Handle emergencies
        if obs.active_emergencies > 0:
            # Aggressive: serve emergency direction
            return self._serve_emergencies(obs)
        
        # P5-P6: Prioritize pedestrians
        if obs.waiting_pedestrians > 2:
            return self._prioritize_pedestrians(obs)
        
        # P7: Avoid flooded lanes
        if obs.flooded_lanes:
            return self._avoid_flooded(obs)
        
        # P8: Normal operation
        return self._normal_operation(obs)
    
    def _serve_emergencies(self, obs) -> TrafficAction:
        """Route to serve emergencies"""
        ns_queue = obs.queues.get("north", 0) + obs.queues.get("south", 0)
        ew_queue = obs.queues.get("east", 0) + obs.queues.get("west", 0)
        
        # Simple heuristic: alternate
        if (self.phase_timer // 20) % 2 == 0:
            return TrafficAction(phase=Phase.NS)
        else:
            return TrafficAction(phase=Phase.EW)
    
    def _prioritize_pedestrians(self, obs) -> TrafficAction:
        """Give pedestrians longer cross time"""
        if self.phase_timer < 45:
            return TrafficAction(phase=self.current_phase)
        else:
            self.current_phase = Phase.EW if self.current_phase == Phase.NS else Phase.NS
            self.phase_timer = 0
            return TrafficAction(phase=self.current_phase)
    
    def _avoid_flooded(self, obs) -> TrafficAction:
        """Avoid phases that include flooded lanes"""
        flooded_ns = any(lane in obs.flooded_lanes for lane in ["north", "south"])
        flooded_ew = any(lane in obs.flooded_lanes for lane in ["east", "west"])
        
        if flooded_ns:
            return TrafficAction(phase=Phase.EW)
        elif flooded_ew:
            return TrafficAction(phase=Phase.NS)
        else:
            return self._normal_operation(obs)
    
    def _normal_operation(self, obs) -> TrafficAction:
        """Normal adaptive control"""
        # Minimum green time
        if self.phase_timer < self.min_green:
            return TrafficAction(phase=self.current_phase)
        
        # Maximum green time
        if self.phase_timer >= self.max_green:
            self.current_phase = Phase.EW if self.current_phase == Phase.NS else Phase.NS
            self.phase_timer = 0
            return TrafficAction(phase=self.current_phase)
        
        # Adaptive: queue-based
        ns_queue = obs.queues.get("north", 0) + obs.queues.get("south", 0)
        ew_queue = obs.queues.get("east", 0) + obs.queues.get("west", 0)
        
        if ns_queue > ew_queue + 3:
            return TrafficAction(phase=Phase.NS)
        elif ew_queue > ns_queue + 3:
            return TrafficAction(phase=Phase.EW)
        else:
            return TrafficAction(phase=self.current_phase)

# ============= AGENT 5: RANDOM (BASELINE) =============

class RandomAgent(TrafficAgent):
    """Random agent for baseline comparison"""
    
    def get_action(self, obs) -> TrafficAction:
        phase = random.choice([Phase.NS, Phase.EW])
        return TrafficAction(phase=phase)

# ============= BENCHMARK RUNNER =============

def run_benchmark(task: str = "medium", num_episodes: int = 5, verbose: bool = True):
    """
    Run benchmark with all agents.
    
    Args:
        task: "easy", "medium", or "hard"
        num_episodes: Episodes per agent
        verbose: Print progress
    
    Returns:
        Results dict
    """
    
    agents = [
        ("RoundRobin", RoundRobinAgent),
        ("QueueBased", QueueBasedAgent),
        ("Heuristic", HeuristicExpertAgent),
        ("Random", RandomAgent),
    ]
    
    results = {}
    
    for agent_name, AgentClass in agents:
        print(f"\n{'='*60}")
        print(f"Testing {agent_name} on task={task}")
        print(f"{'='*60}")
        
        env = TrafficControlEnv(task=task)
        agent = AgentClass(env)
        
        episode_rewards = []
        
        for ep in range(num_episodes):
            reward, steps, metrics = agent.run_episode()
            episode_rewards.append(reward)
            
            if verbose:
                print(f"  Episode {ep+1}: reward={reward:.4f}, steps={steps}, "
                      f"vehicles={metrics['vehicles_processed']}")
        
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        results[agent_name] = {
            "task": task,
            "episodes": num_episodes,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "rewards": episode_rewards,
        }
        
        print(f"\nResults: {avg_reward:.4f} ± {std_reward:.4f}")
    
    return results

if __name__ == "__main__":
    # Run benchmark
    print("\n" + "="*80)
    print("TRAFFIC CONTROL ENVIRONMENT BENCHMARK")
    print("="*80)
    
    for task in ["easy", "medium", "hard"]:
        results = run_benchmark(task=task, num_episodes=3, verbose=True)