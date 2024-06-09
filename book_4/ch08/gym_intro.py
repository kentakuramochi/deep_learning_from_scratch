"""Introduction to OpenAI Gym."""

import gym


# CartPole game
env = gym.make("CartPole-v0")

# State:
# [position, velocity, angle, angular velocity]
state = env.reset()
print(state)  # Initial state

# Action space: 2 (left/right)
action_space = env.action_space
print(action_space)

action = 0  # or 1
# next_state, reward, done, info = env.step(action)  # Obsoleted
# (observation, reward, terminated or not, truncated or not, diagnostic info)
next_state, reward, terminated, truncated, info = env.step(action)
print(next_state)
