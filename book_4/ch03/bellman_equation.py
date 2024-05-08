"""Example of MDP (Malcov Decision Process) on the grid world.

     L1    L2       State
   +-----+-----+
-1 | 0   | +1  | -1 Reward
   +-----+-----+
     <-L   R->      Action
"""

import numpy as np

from enum import Enum


class State(Enum):
    """States on the grid world."""

    L1 = 0
    L2 = 1


class Action(Enum):
    """Action of the agent."""

    Right = 0
    Left = 1


def policy(state):
    """Policy of the agent.

    Args:
        state (State): Current state.

    Returns:
        (Action): Action of the agent.

    Note:
        Random movement.
    """

    p = np.random.randn()

    if state == State.L1:
        return Action.Right if p < 0.5 else Action.Left
    else:  # State.L2
        return Action.Right if p < 0.5 else Action.Left


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


# Bellman equations
#
# v(L1) =
#   0.5 * (r(L1, Left, L1) + discount_rate * v(L1))
# + 0.5 * (r(L1, Right, L2) + discount_rate * v(L2))
#
# v(L2) =
#   0.5 * (r(L2, Left, L1) + discount_rate * v(L1))
# + 0.5 * (r(L2, Right, L2) + discount_rate * v(L2))
#
# -0.55 * v(L1) + 0.45 * v(L2) = 0
#  0.45 * v(L2) - 0.55 * v(L2) = 0.5
#
# v(L1) = -2.25
# v(L2) = -2.75
