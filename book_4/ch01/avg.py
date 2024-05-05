import numpy as np

np.random.seed(0)

# Estimate an action value
# rewards = []
# for n in range(1, 11):
#     reward = np.random.rand()  # Dummy rewards
#     rewards.append(reward)
#     Q = sum(rewards) / n  # Estimated action value from the current rewards
#     print(Q)


Q = 0
for n in range(1, 11):
    reward = np.random.rand()
    Q += (reward - Q) / n  # Update the action value by a recurrence relation
    print(Q)
