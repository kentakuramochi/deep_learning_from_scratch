"""Importance sampling, used for off-policy approeach."""

import numpy as np

# Distribution pi
x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])

# Expected value E_pi(x)
e = np.sum(x * pi)
print("E_pi[x]", e)

# Sampling
n = 100
samples = []
for _ in range(n):
    s = np.random.choice(x, p=pi)
    samples.append(s)

mean = np.mean(samples)  # E_pi(x)
var = np.var(samples)
print(f"MC: {mean:.2f} (var: {var:.2f})")


# Distribution b
# b = np.array([1 / 3, 1 / 3, 1 / 3])
b = np.array([0.2, 0.2, 0.6])

# Sampling
n = 100
samples = []
for _ in range(n):
    idx = np.arange(len(b))  # [0, 1, 2]
    i = np.random.choice(idx, p=b)
    s = x[i]
    rho = pi[i] / b[i]
    samples.append(rho * s)

mean = np.mean(samples)  # E_b(x * pi(x) / b(x)) (= E_pi(x))
var = np.var(samples)
print(f"MC: {mean:.2f} (var: {var:.2f})")
