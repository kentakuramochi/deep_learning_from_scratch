from enum import Enum


class State(Enum):
    L1 = 0
    L2 = 1


class Action(Enum):
    Right = 0
    Left = 1


# Policies of an agent
policies = [
    {State.L1: Action.Right, State.L2: Action.Left},  # 1
    {State.L1: Action.Right, State.L2: Action.Left},  # 2
    {State.L1: Action.Left, State.L2: Action.Right},  # 3
    {State.L1: Action.Left, State.L2: Action.Left},  # 4
]


discount_rate = 0.9


# State-value functions
v1 = {}
v1[State.L1] = 1 - discount_rate * (1 / (1 - discount_rate))
v1[State.L2] = -1 - discount_rate * (1 / (1 - discount_rate))
print("v1:", v1)
