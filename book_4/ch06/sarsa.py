"""Policy iteration by SARSA."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict, deque
import numpy as np
from common.utils import greedy_probs
from common.gridworld import GridWorld


class SarsaAgent:
    """Agent which updates its policy by using SARSA method.

    Attributes:
        gamma (float): Discount rate.
        alpha (float): Smoothing factor of Q function.
        epsilon (float): Probability of the exploration.
        action_size (int): Number of actions (=4).
        pi: (Dict[int, float]): Policy of the agent (random).
        Q (Dict[Tuple[Tuple[int, int], int], float]): Action value function.
        memory (List[Tuple[Tuple[int, int], int, float]]):
            Records of states, actions and rewards until reaching the goal.
    """

    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        # Initialize policy in random by normal distribution
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)  # Deque, FIFO

    def get_action(self, state):
        """Get an action of the agent.

        Args:
            state (Tuple[int, int]): Current state.

        Returns:
            (int): Action of the agent.
        """
        action_probs = self.pi[state]  # On-policy
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        """Reset the memory."""
        self.memory.clear()

    def update(self, state, action, reward, done):
        """Update the policy.

        Args:
            state (Tuple[int, int]): Current state.
            action (int): Action of the agent.
            reward (float): Reward.
            done (bool): Flag, True if an episode finished.
        """
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        # SARSA, applying on-policy TD method for Q function
        state, action, reward, done = self.memory[0]  # (S_t, A_t, R_t)
        next_state, next_action, _, _ = self.memory[1]  # (S_(t+1), A_(t+1))
        next_q = 0 if done else self.Q[next_state, next_action]

        # Update by TD method
        target = reward + self.gamma * next_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)  # Epsilon-greedy


env = GridWorld()
agent = SarsaAgent()

episodes = 10000  # Num of episodes
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, done)

        if done:
            # Also update when arrived to the goal
            agent.update(next_state, None, None, None)
            break
        state = next_state

env.render_q(agent.Q, to_file="sarsa.png")
