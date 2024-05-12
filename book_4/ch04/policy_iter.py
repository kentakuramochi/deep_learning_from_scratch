"""Policy iteration."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from ch04.policy_eval import policy_eval


def argmax(d):
    """Argmax function.

    Args:
        d (Dict[str, float]): Key-value pair.

    Returns:
        (str): Key of the max value.
    """
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_poilcy(V, env, gamma):
    """Get a greedy policy.

    Args:
        V (Dict[Tuple[int, int], float]): State value function.
        env (GridWorld): Gridworld environment.
        gamma (float): Discount rate.

    Returns:
        (Dict[int, float]): Greedy policy.
    """
    pi = {}  # Policy

    for state in env.states():
        action_values = {}

        # Calculate Q functions
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value

        max_action = argmax(action_values)  # Get an optimal action

        # Set the greedy policy: prob. of the optimal action to 1.0
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi


def policy_iter(env, gamma, threshold=0.001, is_render=True, to_file=None):
    """Run the policy iteration.

    Args:
        env (GridWorld): Gridworld environment.
        gamma (float): Discount rate.
        threshold (float): Threshold to stop the policy evaluation.
        is_render (bool): If True, visualize a step of the policy iteration.
        to_file (str): If specified, output a visualized result as an image file to this path.

    Returns:
        (Dict[int, float]): Estimated optimal policy.
    """
    # Policy, initialize with the uniform distribution
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    # State value function
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)  # Evaluate
        new_pi = greedy_poilcy(V, env, gamma)  # Update

        if is_render:
            env.render_v(V, pi)

        # Finish if there's no more update
        if new_pi == pi:
            break
        pi = new_pi

    # Output the result as an image file
    if is_render:
        env.render_v(V, pi, to_file=to_file)

    return pi


def test():
    """Test of the policy iteration."""
    from common.gridworld import GridWorld

    env = GridWorld()
    gamma = 0.9

    pi = policy_iter(env, gamma, to_file="policy_iteration.png")


if __name__ == "__main__":
    test()
