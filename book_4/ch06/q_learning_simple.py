"""Policy iteration by Q-learning based on a sample model."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld


class QLearningAgent:
    """Agent which updates its policy by Q-learning based on a sample model.

    Attributes:
        gamma (float): Discount rate.
        alpha (float): Smoothing factor of Q function.
        epsilon (float): Probability of the exploration.
        action_size (int): Number of actions (=4).
        Q (Dict[Tuple[Tuple[int, int], int], float]): Action value function.
    """

    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        """Get an action of the agent.

        Args:
            state (Tuple[int, int]): Current state.

        Returns:
            (int): Action of the agent.
        """
        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = [self.Q[state, a] for a in range(self.action_size)]
            return np.argmax(qs)

    def update(self, state, action, reward, next_state, done):
        """Update the Q-function.

        Args:
            state (Tuple[int, int]): Current state.
            action (int): Action of the agent.
            reward (float): Reward.
            next_state (Tuple[int, int]): Next state.
            done (bool): Flag, True if an episode finished.
        """
        if done:
            next_q_max = 0
        else:
            # Select an optimal action based on the bellman optimality equation
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha


env = GridWorld()
agent = QLearningAgent()

episodes = 10000  # Num of episodes
for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state, done)
        if done:
            break
        state = next_state

env.render_q(agent.Q, to_file="q_learning_simple.png")
