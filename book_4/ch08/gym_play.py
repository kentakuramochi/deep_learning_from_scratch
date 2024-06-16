"""Play the CartPole with using a random agent."""

import gymnasium as gym
import numpy as np


# CartPole game
# env = gym.make("CartPole-v0")  # Out of date
env = gym.make("CartPole-v1", render_mode="human")  # Render the environment for human

state = env.reset()
done = False

while not done:
    env.render()
    action = np.random.choice([0, 1])  # Random agent
    next_state, reward, done, truncated, info = env.step(action)

env.close()
