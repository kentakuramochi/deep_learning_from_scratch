"""Utility functions."""

import numpy as np


def greedy_probs(Q, state, epsilon=0, action_size=4):
    """Get a greedy policy by using eplison-greedy method.

    Args:
        Q (Dict[Tuple[Tuple[int, int], int], float]): Action value function.
        state (Tuple[int, int]): Curren state.
        epsilon (float): Probability of the exploration.
        action_size (int): Number of actions (=4).

    Returns:
        (Dict[Tuple[int, int], float]): Greedy policy.
    """
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)  # Select an optimal action

    # Set the prob. of the exploration equally for non-optimal actions
    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += 1 - epsilon  # Greedy policy (=1)
    return action_probs
