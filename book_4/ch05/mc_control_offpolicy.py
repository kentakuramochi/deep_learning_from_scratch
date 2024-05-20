"""Policy iteration by using Monte Carlo method."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
import numpy as np
from common.utils import greedy_probs
from common.gridworld import GridWorld


class McOffPolicyAgent:
    """Agent which updates its policy by using off-policy Monte Carlo method.

    Attributes:
        gamma (float): Discount rate.
        epsilon (float): Probability of the exploration.
        alpha (float): Smoothing factor of the Q value.
        action_size (int): Number of actions (=4).
        pi: (Dict[int, float]): Target policy of the agent.
        b: (Dict[int, float]): Behavior policy of the agent.
        Q (Dict[Tuple[Tuple[int, int], int], float]): Action value function.
        memory (List[Tuple[Tuple[int, int], int, float]]):
            Records of states, actions and rewards until reaching the goal.
    """

    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.1
        self.action_size = 4

        # Initialize policy in random by normal distribution
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)  # Target policy
        self.b = defaultdict(lambda: random_actions)  # Behavior policy
        self.Q = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        """Get an action of the agent.

        Args:
            state (Tuple[int, int]): Current state.

        Returns:
            (int): Action of the agent.
        """
        action_probs = self.b[state]  # Action by the behavior policy
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        """Add state, action (trajectory) and reward into a memory.

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

    def update(self):
        """Update the policy."""
        G = 0  # Gain
        rho = 1  # Weight used for importance sampling
        for data in reversed(self.memory):
            state, action, reward = data
            key = (state, action)

            # Update the Q function by EMA and importance sampling
            G = self.gamma * rho * G + reward
            self.Q[key] += (G - self.Q[key]) * self.alpha
            rho *= self.pi[state][action] / self.b[state][action]

            # Target policy: greedy
            self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
            # Behavior policy: epsilon-greedy
            self.b[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = McOffPolicyAgent()

episodes = 1000  # Num of episodes
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:
            agent.update()
            break

        state = next_state

env.render_q(agent.Q, to_file="mc_control_offpolicy.png")
