"""Play the CartPole with actor-critic model."""

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


class PolicyNet(Model):
    """Neural network model of a policy.

    Attributes:
        l1 (dezero.layers.Linear): 1st Linear layer.
        l2 (dezero.layers.Linear): 2nd Linear layer.
    """

    def __init__(self, action_size=2):
        """Initialize.

        Args:
            action_size (int): Size of an action space (=2).
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
        x = self.l2(x)
        x = F.softmax(x)
        return x


class ValueNet(Model):
    """Neural network model of value function.

    Attributes:
        l1 (dezero.layers.Linear): 1st Linear layer.
        l2 (dezero.layers.Linear): 2nd Linear layer.
    """

    def __init__(self):
        """Initialize."""
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)

    def forward(self, x):
        """Forward propagation.

        Args:
            x (dezero.core.Variable): Current state.

        Returns:
            (dezero.core.Variable): Value function.
        """
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    """Agent which implements its policy and value function
       as actor-critic model.

    Attributes:
        gamma (float): Discount rate.
        lr_pi (float): Learning rate for the policy.
        lr_v (float): Learning rate for the value function.
        action_size (int): Size of an action space.
        pi (PolicyNet): Neural network for the policy.
        v (ValueNet): Neural network for the value function.
        optimizer_pi (dezero.optimizer): Optimizer for the PolicyNet.
        optimizer_v (dezero.optimizer): Optimizer for the ValueNet.
    """

    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet()
        self.v = ValueNet()
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

    def get_action(self, state):
        """Get an action of the agent.

        Args:
            state (NDArray[float]): Current state.

        Returns:
            (Tuple[int, NDArray[float]]):
                (int): Action of the agent.
                (NDArray[float]): Policy of the agent.
        """
        state = state[np.newaxis, :]  # Add batch dim.
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        """Update policy and value function.

        Args:
            state (NDArray[float]): Current state.
            action_prob (NDArray[float]): Action probabilities.
            reward (float): Reward.
            next_state (NDArray[float]): Next state.
            done (bool): Flag, True if the episode finished.
        """
        state = state[np.newaxis, :]  # Add batch dim.
        next_state = next_state[np.newaxis, :]

        # Update a ValueNet
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.unchain()
        v = self.v(state)
        loss_v = F.mean_squared_error(v, target)

        # Update a PolicyNet
        delta = target - v
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta

        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()


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

        agent.update(state, prob, reward, next_state, done)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 100 == 0:
        print(f"episode: {episode}, total reward: {total_reward}")

# Plot rewards
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.plot(range(len(reward_history)), reward_history)
plt.savefig("output/reward_history_actor_critic.png")

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
