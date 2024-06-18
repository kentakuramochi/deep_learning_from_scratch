"""Play the breakout with using Deep Q Network."""

import random

import gymnasium as gym


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
env = gym.make("ALE/Breakout-v5", render_mode="human")

episodes = 1
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action = random.randint(0, 3)  # Random
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 10 == 0:
        print(f"episode: {episode}, total reward: {total_reward}")
