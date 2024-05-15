"""Calculate an expected value of dice rolling by using Monte Carlo method."""

import numpy as np


def sample(dices=2):
    """Sample model of dice rolling.

    Args:
        dices (int): Number of dice.

    Returns:
        (int): Sum of the pips of rolled dice.
    """
    x = 0
    for _ in range(dices):
        x += np.random.choice([1, 2, 3, 4, 5, 6])
    return x


trial = 1000
V, n = 0, 0
# Get an average incrementally
for _ in range(trial):
    s = sample()
    n += 1
    V += (s - V) / n
    print(V)  # Approaching to 7
