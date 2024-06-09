"""Q-learning by using a neural network."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import matplotlib.pyplot as plt

import numpy as np

from common.gridworld import GridWorld

from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


def one_hot(state):
    """Convert a state into a one-hot vector.

    Args:
        state (Tuple[int, int]): State.

    Returns:
        (numpy.ndarray[float]): State in one-hot vector.
    """
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]


class QNet(Model):
    """Neural network for Q function.

    Attributes:
        l1 (dezero.layers.Linear): 1st layer.
        l2 (dezero.layers.Linear): 2nd layer.
    """

    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(100)  # Hidden size
        self.l2 = L.Linear(4)  # action_size

    def forward(self, x):
        """Forward propagation.

        Args:
            x (dezero.Variable): State in one-hot vector.

        Returns:
            (dezero.Variable): Value of the Q function.
        """
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class QLearningAgent:
    """Agent which updates its policy by Q-learning with a neural network.

    Attributes:
        gamma (float): Discount rate.
        lr (float): Learning rate.
        epsilon (float): Probability of the exploration.
        action_size (int): Number of actions (=4).
        qnet (QNet): Neural network for the Q function.
        optimizer (dezero.optimizer): Optimizer of the network.
    """

    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()
        self.optimizer = optimizers.SGD(self.lr)
        self.optimizer.setup(self.qnet)

    def get_action(self, state_vec):
        """Get an action of the agent.

        Args:
            state_vec (numpy.ndarray[float]): Current state in one-hot vector.

        Returns:
            (int): Action of the agent.
        """
        # Epsilon-greedy method
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state_vec)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        """Update the Q function.

        Args:
            state (numpy.ndarray[float]): Current state in one-hot vector.
            action (int): Action of the agent.
            reward (float): Reward.
            next_state (numpy.ndarray[float]): Next state in one-hot vector.
            done (bool): Flag, True if an episode finished.
        """
        if done:
            next_q = np.zeros(1)  # [0.]
        else:
            next_qs = self.qnet(next_state)
            next_q = next_qs.max(axis=1)
            next_q.unchain()

        target = self.gamma * next_q + reward
        qs = self.qnet(state)
        q = qs[:, action]
        loss = F.mean_squared_error(target, q)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        return loss.data


env = GridWorld()
agent = QLearningAgent()

episodes = 1000  # Num of episodes
loss_history = []

for episode in range(episodes):
    state = env.reset()
    state = one_hot(state)
    total_loss, cnt = 0, 0
    done = False

    # Learning
    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        next_state = one_hot(next_state)

        cnt += 1

        # Retry the episode when it takes over 100 steps
        if cnt >= 100 and not done:
            env.reset()
            cnt = 0

        state = next_state

    # Learn only from the finished episode
    loss = agent.update(state, action, reward, next_state, done)
    total_loss += loss

    # print(f"episode {episode}, count={cnt}, loss={loss}")

    average_loss = total_loss / cnt
    loss_history.append(average_loss)

# Plot a loss history
plt.plot(loss_history)
plt.xlabel("episode")
plt.ylabel("loss")
plt.savefig("output/loss_history.png")

# Visualize Q function
Q = {}
for state in env.states():
    for action in env.action_space:
        q = agent.qnet(one_hot(state))[:, action]
        Q[state, action] = float(q.data[0])
env.render_q(Q, to_file="output/q_learning_nn.png")
