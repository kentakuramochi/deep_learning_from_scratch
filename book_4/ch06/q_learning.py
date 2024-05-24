"""Policy iteration by Q-learning."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld
from common.utils import greedy_probs


class QLearningAgent:
    """Agent which updates its policy by Q-learning.

    Attributes:
        gamma (float): Discount rate.
        alpha (float): Smoothing factor of Q function.
        epsilon (float): Probability of the exploration.
        action_size (int): Number of actions (=4).
        pi (Dict[int, float]): Target policy of the agent.
        b (Dict[int, float]): Behavior policy of the agent.
        Q (Dict[Tuple[Tuple[int, int], int], float]): Action value function.
    """

    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        # Initialize policy in random by normal distribution
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        """Get an action of the agent.

        Args:
            state (Tuple[int, int]): Current state.

        Returns:
            (int): Action of the agent.
        """
        action_probs = self.b[state]  # Off-policy
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        """Update the policy.

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

        # Target policy: greedy
        self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
        # Behavior policy: epsilon-greedy
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


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

env.render_q(agent.Q, to_file="q_learning.png")
