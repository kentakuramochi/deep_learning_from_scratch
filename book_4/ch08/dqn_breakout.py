"""Play the breakout with using Deep Q Network.

Reference:
    「DQNの進化史 ①DeepMindのDQN | どこから見てもメンダコ」
    (https://horomary.hatenablog.com/entry/2021/01/26/233351)
"""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import copy
import matplotlib.pyplot as plt
import pickle
import zlib
from collections import deque

import numpy as np
import dezero.functions as F
import dezero.layers as L

from PIL import Image
from dezero import Model
from dezero import optimizers

import gymnasium as gym


class QNet(Model):
    """Neural network for Q function.

    Attributes:
        c1 - c3 (dezero.layers.Linear): Conv2D layers.
        l1 - l2 (dezero.layers.Linear): Linear layers.
    """

    def __init__(self, action_size):
        """Initialize.

        Args:
            action_size (int): Size of an action space.
        """
        super().__init__()
        self.c1 = L.Conv2d(32, kernel_size=8, stride=4)
        self.c2 = L.Conv2d(64, kernel_size=4, stride=2)
        self.c3 = L.Conv2d(64, kernel_size=3, stride=1)
        self.l1 = L.Linear(512)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        """Forward propagation.

        Args:
            x (dezero.core.Variable): Current state.

        Returns:
            (dezero.core.Variable): Value of the Q function.
        """
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.reshape(x, (x.shape[0], -1))
        x = F.relu(self.l1(x))
        x = self.l2(x)
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
            state (NDArray[float]): Current state.
            action (int): Agent's action.
            reward (float): Reward.
            next_state (NDArray[float]): Next state.
            done (bool): Flag, True when the episode finishes.
        """
        data = (state, action, reward, next_state, done)
        # Compress the data to store a large amount of data
        data = zlib.compress(pickle.dumps(data))
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
        # Get random indices
        N = len(self.buffer)
        indices = np.random.choice(np.arange(N), replace=False, size=self.batch_size)
        # Decompress the data
        data = [pickle.loads(zlib.decompress(self.buffer[idx])) for idx in indices]

        state = np.concatenate([x[0] for x in data])  # Concat to a batch dimension
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.concatenate([x[3] for x in data])  # Concat to a batch dimension
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

    def __init__(self, weight_path=None):
        """Initialize.

        Args:
            weight_path (str): Path to the pretrained weight (.npz).
        """
        self.gamma = 0.99
        self.lr = 0.00025
        self.epsilon = 0.1
        self.buffer_size = 1000000
        self.batch_size = 32
        self.action_size = 4

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        # Load weights when it is specified
        if weight_path is not None:
            self.qnet.load_weights(weight_path)
            self.qnet_target.load_weights(weight_path)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)

    def sync_qnet(self):
        """Synchronize the network and the target network."""
        self.qnet_target = copy.deepcopy(self.qnet)  # Deep copy

    def get_action(self, state, epsilon=None):
        """Get an action of the agent.

        Args:
            state (NDArray[float]): Current state.

        Returns:
            (int): Action of the agent.
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Epsilon-greedy method
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        """Update the network.

        Args:
            state (NDArray[float]): Current state.
            action (int): Agent's action.
            reward (float): Reward.
            next_state (NDArray[float]): Next state.
            done (bool): Flag, True when the episode finishes.
        """
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


def preprocess(state):
    """Preprocess a state for the CNN operation.

    Args:
        state (NDArray[uint8]): 2-d state with the shape of (210, 160).

    Returns:
        (NDArray[float]): Normalized 2-d state with the shape of (84, 84).
    """
    # Crop game area: (164, 144)
    croped = state[32:-14, 8:-8]

    # Resize to (84, 84)
    resized = Image.fromarray(croped).resize((84, 84))
    resized = np.array(resized)

    # Normalize to [0.0, 1.0]
    return np.array(resized) / 255.0


# Atari Breakout
# https://gymnasium.farama.org/environments/atari/breakout/
# Action space:
#   - 0: No operation
#   - 1: Fire (throw the ball)
#   - 2: Move the paddle to right
#   - 3: Move the paddle to left
# Observation space:
#   - obs_type="rgb":
#       np.uint8 array with shape=(210,160,3)
#   - obs_type="ram":
#       np.uint8 array with shape=(128,)
#   - obs_type="grayscale":
#       np.uint8 array with shape=(210,160)
# Variant:
#   "BreakoutDeterministic-v4":
#       skip every 4 frames and doing deteministic actions
env_id = "BreakoutDeterministic-v4"

do_learning = True

if do_learning:
    env = gym.make(env_id, obs_type="grayscale")

    episodes = 10000
    agent = DQNAgent(weight_path="./output/qnet.npz")

    total_steps = 0
    reward_history = []

    for episode in range(episodes):
        state = env.reset()[0]
        lives = 5  # Game lives
        done = False
        total_reward = 0

        # Preprocess
        state = preprocess(state)

        # POMDP (Pertially Observable Markov Decision Process):
        # use recent 4 frames as a current state, use duplicated first frames firstly
        states = deque([state] * 4, maxlen=4)

        # Annealing of probability of the exploration:
        # reduce it at first 1,000,000 steps and then fix it to 0.1
        epsilon_scheduler = lambda steps: max(1.0 - 0.9 * steps / 1000000, 0.1)

        while not done:
            total_steps += 1
            epsilon = epsilon_scheduler(total_steps)

            # Get recent 4 frames
            state = np.stack(states)[np.newaxis, :]

            action = agent.get_action(state, epsilon=epsilon)
            next_state, reward, done, truncated, info = env.step(action)

            # Stack the current frame
            states.append(preprocess(next_state))
            next_state = np.stack(states)[np.newaxis, :]

            # If the lives decreased,
            # thought it as the end of the game as an experience
            if info["lives"] != lives:
                lives = info["lives"]
                done_experience = True
            else:
                done_experience = done

            # Add en experience to the replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done_experience)

            total_reward += reward

            # Start learning after enough experiences are stored in the buffer
            if len(agent.replay_buffer) > 50000:
                if total_steps % 4 == 0:
                    # Update the agent
                    agent.update(state, action, reward, next_state, done)
                if total_steps % 10000 == 0:
                    # Sync the target Q network
                    agent.sync_qnet()

        reward_history.append(total_reward)
        if episode % 100 == 0:
            print(f"episode: {episode}, total reward: {total_reward}")
            # Save the weight
            agent.qnet.save_weights("./output/qnet.npz")

    # Plot rewards
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.plot(range(len(reward_history)), reward_history)
    plt.savefig("output/breakout_reward_history.png")


# Play Breakout with using the trained model
env = gym.make(env_id, obs_type="grayscale", render_mode="human")
agent = DQNAgent(weight_path="./output/qnet.npz")

agent.epsilon = 0  # Greedy policy
done = False
total_reward = 0

state = env.reset()[0]

state = preprocess(state)
states = deque([state] * 4, maxlen=4)

while not done:
    state = np.stack(states)[np.newaxis, :]
    action = agent.get_action(state)
    next_state, reward, done, truncated, info = env.step(action)
    states.append(preprocess(next_state))
    next_state = np.stack(states)[np.newaxis, :]
    total_reward += reward
    env.render()

print("Total reward:", total_reward)
