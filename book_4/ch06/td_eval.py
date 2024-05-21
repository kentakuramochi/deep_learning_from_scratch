"""Policy evaluation by using TD method."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld


class TdAgent:
    """Agent which updates its value function by using TD (Time Difference) method.

    Attributes:
        gamma (float): Discount rate.
        alpha (float): Smoothing factor of V function.
        action_size (int): Number of actions (=4).
        pi: (Dict[int, float]): Policy of the agent (random).
        V (Dict[Tuple[int, int], float]): State value function.
    """

    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.1
        self.action_size = 4

        # Initialize policy in random by normal distribution
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)

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

    def eval(self, state, reward, next_state, done):
        """Evaluate value functions by 1-step TD method.

        Args:
            state (Tuple[int, int]): Current state.
            reward (float): Reward.
            next_state (Tuple[int, int]): Next state.
            done (bool): Flag, True if an episode finished.
        """
        next_V = 0 if done else self.V[next_state]  # Value of the goal is 0
        target = reward + self.gamma * next_V

        self.V[state] += (target - self.V[state]) * self.alpha  # EMA


env = GridWorld()
agent = TdAgent()

episodes = 1000  # Num of episodes
for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.eval(state, reward, next_state, done)
        if done:
            break
        state = next_state

env.render_v(agent.V, to_file="td_eval.png")
