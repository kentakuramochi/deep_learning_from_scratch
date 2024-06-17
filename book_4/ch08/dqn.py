"""Play the CartPole with using Deep Q Network."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import copy
import matplotlib.pyplot as plt
import random
from collections import deque

import numpy as np
import dezero.functions as F
import dezero.layers as L

from dezero import Model
from dezero import optimizers

import gymnasium as gym


class QNet(Model):
    """Neural network for Q function.

    Attributes:
        l1 (dezero.layers.Linear): 1st layer.
        l2 (dezero.layers.Linear): 2nd layer.
        l3 (dezero.layers.Linear): 3rd layer.
    """

    def __init__(self, action_size):
        """Initialize.

        Args:
            action_size (int): Size of an action space.
        """
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        """Forward propagation.

        Args:
            x (dezero.Variable): State in one-hot vector.

        Returns:
            (dezero.Variable): Value of the Q function.
        """
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class ReplayBuffer:
    """Buffer of experiences for the experience replay.

    Attributes:
        buffer (Deque[Tuple[NDArray[float], int, float, NDArray[float], bool]]):
            List of experiences composed of:
                - state
                - action
                - reward
                - next_state
                - done (bool)
        batch_size (int): Size of the mini-batch.
    """

    def __init__(self, buffer_size, batch_size):
        """Initialize.

        Args:
            buffer_size (int): Size of the buffer.
            batch_size (int): Size of the mini-batch.
        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer.

        Args:
            state (Tuple[NDArray[float]]): Current state.
            action (int): Agent's action.
            reward (float): Reward.
            next_state (NDArray[float]): Next state.
            done (bool): Flag, True when the episode finishes.
        """
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        """Length of the buffer.

        Returns:
            (int): Length of the buffer.
        """
        return len(self.buffer)

    def get_batch(self):
        """Get a random-sampled mini-batch experience from the buffer.

        Returns:
            (Tuple[NDArray[float], NDArray[int], NDArray[float], NDArray[float], NDArray[bool]]):
                Mini-batch experience.
        """
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)

        return state, action, reward, next_state, done


class DQNAgent:
    """Agent which updates its policy by Q-learning with a neural network.

    Attributes:
        gamma (float): Discount rate.
        lr (float): Learning rate.
        epsilon (float): Probability of the exploration.
        buffer_size (int): Size of the replay buffer.
        batch_size (int): Size of the mini-batch for the replay.
        action_size (int): Size of the action space.

        replay_buffer (ReplayBuffer): Buffer for experience replay.
        qnet (QNet): Neural network for the Q function.
        qnet_target (QNet): Target network.
        optimizer (dezero.optimizer): Optimizer of the network.
    """

    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)

    def sync_qnet(self):
        """Synchronize the network and the target network."""
        self.qnet_target = copy.deepcopy(self.qnet)  # Deep copy

    def get_action(self, state):
        """Get an action of the agent.

        Args:
            state (NDArray[float]): Current state.

        Returns:
            (int): Action of the agent.
        """
        # Epsilon-greedy method
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]  # Add batch dimension
            qs = self.qnet(state)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        """Update the network.

        Args:
            state (Tuple[NDArray[float]]): Current state.
            action (int): Agent's action.
            reward (float): Reward.
            next_state (NDArray[float]): Next state.
            done (bool): Flag, True when the episode finishes.
        """
        # Add en experience to the replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]  # Extract Q values

        next_qs = self.qnet_target(next_state)  # Next state from the target network
        next_q = next_qs.max(axis=1)
        next_q.unchain()
        target = reward + (1 - done) * self.gamma * next_q  # (1 - done): mask

        loss = F.mean_squared_error(q, target)

        # Backprop
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()


episodes = 300
sync_interval = 20
env = gym.make("CartPole-v1", render_mode="human")
agent = DQNAgent()
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)

        # Update
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    # Sync the target network
    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)
    if episode % 10 == 0:
        print(f"episode: {episode}, total reward: {total_reward}")


# Plot rewards
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.plot(range(len(reward_history)), reward_history)
plt.savefig("output/reward_history.png")

# Play CartPole
agent.epsilon = 0  # Greedy policy
state = env.reset()[0]
done = False
total_reward = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, done, truncated, info = env.step(action)
    state = next_state
    total_reward += reward
    env.render()

print("Total reward:", total_reward)
