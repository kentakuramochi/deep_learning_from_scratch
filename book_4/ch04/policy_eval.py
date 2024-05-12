"""Iterative policy evaluation."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def eval_onestep(pi, V, env, gamma=0.9):
    """Do a step of the iterative policy evaluation.

    Args:
        pi: (Dict[int, float]): Policy of the agent.
        V (Dict[Tuple[int, int], float]): State value function.
        env (GridWorld): Gridworld environment.
        gamma (float): Discount rate.

    Returns:
        (Dict[Tuple[int, int], float]): Updated state value function.
    """
    for state in env.states():
        if state == env.goal_state:
            # Value of the goal is 0, because there's no more actions
            V[state] = 0
            continue

        action_probs = pi[state]
        new_V = 0
        # Update state value function
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)  # Transit
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V


def policy_eval(pi, V, env, gamma, threshold=0.001):
    """Do a step of the iterative policy evaluation.

    Args:
        pi: (Dict[int, float]): Policy of the agent.
        V (Dict[Tuple[int, int], float]): State value function.
        env (GridWorld): Gridworld environment.
        gamma (float): Discount rate.
        threshold (float): Threshold to stop the policy evaluation.

    Returns:
        (Dict[Tuple[int, int], float]): Updated state value function.
    """
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        # Check the max amount of update
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        # Finish if there's no more enough update
        if delta < threshold:
            break
    return V


def test():
    """Test of the policy evaluation."""
    from collections import defaultdict
    from common.gridworld import GridWorld

    env = GridWorld()
    gamma = 0.9

    # Policy, initialized by the uniform dist.
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    # State value funciton
    V = defaultdict(lambda: 0)

    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi, to_file="V_in_random_policy.png")


if __name__ == "__main__":
    test()
