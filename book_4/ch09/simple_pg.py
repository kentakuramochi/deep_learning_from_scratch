"""Play the CartPole with simple policy gradient method."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym

from dezero import Model
from dezero import optimizers

import dezero.functions as F
import dezero.layers as L


class Policy(Model):
    """Neural network model of a policy.

    Attributes:
        l1 (dezero.layers.Linear): 1st Linear layer.
        l2 (dezero.layers.Linear): 2nd Linear layer.
    """

    def __init__(self, action_size):
        """Initialize.

        Args:
            action_size (int): Size of an action space.
        """
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        """Forward propagation.

        Args:
            x (dezero.core.Variable): Current state.

        Returns:
            (dezero.core.Variable): Policy.
        """
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x


class Agent:
    """Agent which updates its policy by policy gradient method.

    Attributes:
        gamma (float): Discount rate.
        lr (float): Learning rate.
        action_size (int): Size of an action space.
        memory (List[Tuple[float, dezero.core.Variable]]):
            List of reward and action probabilities.
        pi (Policy): Neural network for the policy.
        optimizer (dezero.optimizer): Optimizer of the network.
    """

    def __init__(self):
        """Initialize."""
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)

    def get_action(self, state):
        """Get an action of the agent.

        Args:
            state (NDArray[float]): Current state.

        Returns:
            (Tuple[int, NDArray[float]]):
                (int): Action of the agent.
                (NDArray[float]): Policy of the agent.
        """
        state = state[np.newaxis, :]  # Add batch dim
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def add(self, reward, prob):
        """Add reward and action probabilities to the memory.

        Args:
            reward(float): Reward.
            prob(dezero.core.Variable): Action probabilities.
        """
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        """Update the policy."""
        self.pi.cleargrads()

        G, loss = 0, 0
        # Gain
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G

        # Loss
        for reward, prob in self.memory:
            loss += -F.log(prob) * G

        loss.backward()
        self.optimizer.update()
        self.memory = []


# Test
# env = gym.make("CartPole-v1", render_mode="human")
# agent = Agent()
# state = env.reset()[0]
# action, prob = agent.get_action(state)
# print("action:", action)
# print("prob:", prob)
#
# G = 100.0  # Dummy weight
# J = G * F.log(prob)  # Target function
# print("J:", J)
#
# J.backward()

# Train CartPole
env = gym.make("CartPole-v1")
episodes = 3000
agent = Agent()
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)

        agent.add(reward, prob)
        state = next_state
        total_reward += reward

    agent.update()

    reward_history.append(total_reward)
    if episode % 100 == 0:
        print(f"episode: {episode}, total reward: {total_reward}")

# Plot rewards
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.plot(range(len(reward_history)), reward_history)
plt.savefig("output/reward_history.png")

# Play CartPole
env = gym.make("CartPole-v1", render_mode="human")
state = env.reset()[0]
done = False
total_reward = 0

while not done:
    action, prob = agent.get_action(state)
    next_state, reward, done, truncated, info = env.step(action)
    state = next_state
    total_reward += reward
    env.render()

print("Total reward:", total_reward)
