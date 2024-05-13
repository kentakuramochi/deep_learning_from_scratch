"""Policy iteration."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def value_iter_onestep(V, env, gamma):
    """Do a step of the value iteration.

    Args:
        V (Dict[Tuple[int, int], float]): State value function.
        env (GridWorld): Gridworld environment.
        gamma (float): Discount rate.

    Returns:
        (Dict[Tuple[int, int], float]): Updated state value function.
    """
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0  # Value of the goals is 0
            continue

        action_values = []
        # Update state value function
        for action in env.actions():
            next_state = env.next_state(state, action)  # Transit
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)

        V[state] = max(action_values)  # Greedy
    return V


def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    """Run the value iteration.

    Args:
        V (Dict[Tuple[int, int], float]): State value function.
        env (GridWorld): Gridworld environment.
        gamma (float): Discount rate.
        threshold (float): Threshold to stop the value itaration.
        is_render (bool): If True, visualize a step of the policy iteration.

    Returns:
        (Dict[Tuple[int, int], float]): Estimated state value function.
    """
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

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
