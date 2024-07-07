"""Experience replay of the CartPole problem."""

from collections import deque
import random

import gymnasium as gym
import numpy as np


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
            state (NDArray[float]): Current state.
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


env = gym.make("CartPole-v1", render_mode="human")

# Allocate a buffer for 10000 experiences
replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

for episode in range(10):
    state = env.reset()[0]  # Get an observation from [observation, info]
    done = False

    while not done:
        # Cart Pole
        # https://gymnasium.farama.org/environments/classic_control/cart_pole/
        # Action space:
        #   - 0: Push cart to the left
        #   - 1: Push cart to the right
        # Observation space:
        #   - Dim 0: Cart position
        #   - Dim 1: Cart velocity
        #   - Dim 2: Pole angle
        #   - Dim 3: Pole angular velocity
        action = 0
        next_state, reward, done, truncated, info = env.step(action)
        # Save the experience
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

# Replay the experience as mini-batch
state, action, reward, next_state, done = replay_buffer.get_batch()
print(state.shape)
print(action.shape)
print(reward.shape)
print(next_state.shape)
print(done.shape)
