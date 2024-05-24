"""Policy evaluation by using Monte Carlo method."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld


class RandomAgent:
    """Agent woking by random policy.

    Attributes:
        gamma (float): Discount rate.
        action_size (int): Number of actions (=4).
        pi (Dict[int, float]): Policy of the agent.
        V (Dict[Tuple[int, int], float]): State value function.
        cnts (Dict[Tuple[int, int], int]): Counts of each state.
        memory (List[Tuple[Tuple[int, int], int, float]]):
            Records of states, actions and rewards until reaching the goal.
    """

    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        # Policy: act randomly, by normal distribution
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        """Get an action of the agent.

        Args:
            state (Tuple[int, int]): Current state.

        Returns:
            (int): Action of the agent.
        """
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        """Add state, action and reward into a memory.

        Args:
            state (Tuple[int, int]): State before an action.
            action (int): Action of the agent.
            reward (float): Reward earned by the action.
        """
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        """Reset the memory."""
        self.memory.clear()

    def eval(self):
        """Evaluate value functions by Monte Carlo method."""
        G = 0  # Gain
        for data in reversed(self.memory):  # Backtrack from the goal
            state, action, reward = data
            G = self.gamma * G + reward  # Update the gain
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]  # Average


env = GridWorld()
agent = RandomAgent()

episodes = 1000  # Num of episodes
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:
            # Update V by Monte Carlo method
            agent.eval()
            break

        state = next_state

env.render_v(agent.V, to_file="mc_eval.png")
