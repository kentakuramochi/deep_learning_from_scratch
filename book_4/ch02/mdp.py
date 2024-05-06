"""Example of MDP (Malcov Decision Process) on the grid world.

     L1    L2       State
   +-----+-----+
-1 | 0   | +1  | -1 Reward
   +-----+-----+
     <-L   R->      Action
"""

from enum import Enum


class State(Enum):
    """States on the grid world."""

    L1 = 0
    L2 = 1


class Action(Enum):
    """Action of the agent."""

    Right = 0
    Left = 1


# Policies of an agent
policies = [
    {State.L1: Action.Right, State.L2: Action.Right},  # Policy 1
    {State.L1: Action.Right, State.L2: Action.Left},  # Policy 2
    {State.L1: Action.Left, State.L2: Action.Right},  # Policy 3
    {State.L1: Action.Left, State.L2: Action.Left},  # Policy 4
]


def get_reward(state, action, next_state):
    """Get the reward.

    Args:
        state (State): Current state.
        action (Action): Action of the agent.
        next_state (State): State after the action.

    Returns:
        (int): Reward.

    Note:
        next_state is deterministic, therefore doesn't affect to the rewards.
    """

    if state == State.L1:
        if action == Action.Left:
            return -1
        else:  # Action.Right
            return 1
    else:  # State.L2
        if action == Action.Left:
            return 0
        else:  # Action.Right
            return -1


def transit(state, action):
    """Transit the state (deterministic).

    Args:
        state (State): Current state.
        action (Action): Action of the agent.

    Returns:
        (State): Next state.
    """
    if state == State.L1:
        if action == Action.Left:
            return State.L1
        else:  # Action.Right
            return State.L2
    else:  # State.L2
        if action == Action.Left:
            return State.L1
        else:  # Action.Right
            return State.L2


def approx_state_value(policy, initial_state, discount_rate):
    """Approximate a state value.

    Args:
        policy (List[Dict[State, Action]]): Policy of the agent.
        initial_state (State): Initial state.
        discount_rate (float): Discount rate of the rewards.

    Returns:
        (float): State value.
    """

    state = initial_state
    v = 0

    for i in range(100):  # Iteration
        action = policy[state]
        next_state = transit(state, action)
        v += get_reward(state, action, next_state) * (discount_rate**i)
        state = next_state

    return v


# Approximate all state values for each policy
vs = []
for policy in policies:
    v = {}
    for state in {State.L1, State.L2}:
        v[state] = approx_state_value(policy, state, discount_rate=0.9)
    vs.append(v)

# Print the state values
for i, v in enumerate(vs):
    print(f"Policy {i + 1}: v{i + 1}(L1)={v[State.L1]}, v{i + 1}(L2)={v[State.L2]}")
