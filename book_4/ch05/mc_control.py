"""Policy iteration by using Monte Carlo method."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld


def greedy_probs(Q, state, epsilon=0, action_size=4):
    """Get a greedy policy by using eplison-greedy method.

    Args:
        Q (Dict[Tuple[Tuple[int, int], int], float]): Action value function.
        state (Tuple[int, int]): Curren state.
        epsilon (float): Probability of the exploration.
        action_size (int): Number of actions (=4).

    Returns:
        (Dict[Tuple[int, int], float]): Greedy policy.
    """
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)  # Select an optimal action

    # Set the prob. of the exploration equally for non-optimal actions
    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += 1 - epsilon  # Greedy policy (=1)
    return action_probs


class McAgent:
    """Agent control its policy by using Monte Carlo method.

    Attributes:
        gamma (float): Discount rate.
        epsilon (float): Probability of the exploration.
        alpha (float): Smoothing factor of the Q value.
        action_size (int): Number of actions (=4).
        pi: (Dict[int, float]): Policy of the agent.
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
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
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

    def update(self):
        """Update the policy."""
        G = 0  # Gain
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)

            # Update the Q function by EMA (Exponential Moving Average)
            self.Q[key] += (G - self.Q[key]) * self.alpha

            self.pi[state] = greedy_probs(
                self.Q, state, self.epsilon
            )  # Set a greedy policy


env = GridWorld()
agent = McAgent()

episodes = 1000  # Num of episodes
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:
            # Update Q by Monte Carlo method
            agent.update()
            break

        state = next_state

env.render_q(agent.Q, to_file="mc_control.png")
