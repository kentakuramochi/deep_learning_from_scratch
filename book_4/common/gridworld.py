"""Iterative policy evaluation by dynamic programming (DP).

An example of MDP on the 3x4 gridworld:

    WALL
   +----+----+----+----+
   |    |    |    |APPL| +1
   +----+----+----+----+
   |    |WALL|    |BOMB| -1
   +----+----+----+----+
   |AGNT|    |    |    |
   +----+----+----+----+

* Actions:
    up/down/left/right

* The agent cannot proceed to the walls.

* Rewards:
    Get an apple: +1
    Get a bomb: -1
    Otherwise: 0

* The transision of the state is definitive.

* The task finished when the agent gets the apple (episode task).
"""

import numpy as np
import common.gridworld_render as render_helper


class GridWorld:
    """Gridworld.

    Attributes:
        action_space (List[int]): Action space, candidates of the actions.
        action_meaning (Dict[int, str]): Meaning of each action.
        reward_map (numpy.ndarray(float)): Rewards of each grid.
        goal_state (Tuple[int, int]): Grid index of the goal.
        wall_state (Tuple[int, int]): Grid index of the wall.
        start_state (Tuple[int, int]): Grid index of the start.
        agent_state (Tuple[int, int]): Grid index of the agent.
    """

    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        self.reward_map = np.array([[0, 0, 0, 1.0], [0, None, 0, -1.0], [0, 0, 0, 0]])

        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property
    def height(self):
        """Height of the gridworld.

        Returns:
            (int): Height of the grid.
        """
        return len(self.reward_map)

    @property
    def width(self):
        """Width of the gridworld.

        Returns:
            (int): Width of the grid.
        """
        return len(self.reward_map[1])

    @property
    def shape(self):
        """Shape of the gridworld.

        Returns:
            (Tuple[int, int]): Shape of the grid (height, width).
        """
        return self.reward_map.shape

    def actions(self):
        """Actions of the agent.

        Returns:
            (List[int]): List of actions.
        """
        return self.action_space  # [0, 1, 2, 3]

    def states(self):
        """State of the gridworld.

        Returns:
            (Generator[int, int]): State (origin from the upper left).
        """
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state, action):
        """Get the next state which the agent transits to.

        Args:
            state (Tuple[int, int]): Current state.
            action (int): Action of the agent.

        Return:
            (Tuple[int, int]): Next state.
        """
        # Move the agent by the action
        action_move_map = [
            (-1, 0),  # Up
            (1, 0),  # Down
            (0, -1),  # Left
            (0, 1),  # Right
        ]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # Limit the state by walls
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        return next_state

    def reward(self, state, action, next_state):
        """Return a reward.

        Args:
            state (Tuple[int, int]): Current state.
            action (int): Action of the agent.
            next_state (Tuple[int, int]): Next state.

        Returns:
            (float): Reward.

        Note:
            Rewards are only decided from the next state in this example.
        """
        return self.reward_map[next_state]

    def reset(self):
        """Reset the gridworld into an initial state."""
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        """Get the agent into an action.

        Args:
            action (int): Action of the agent.

        Returns:
            (Tuple[Tuple[int, int], float, bool]):
                * Next state.
                * Reward.
                * Flag, True if an episode is done.
        """
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = next_state == self.goal_state

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True, to_file=None):
        """Visualize the gridworld with state values.

        Args:
            v (Tuple[int, int]): State values.
            policy (Dict[int, float]): Policy of the agent.
            print_value (bool): If True, print the state values.
            to_file (str): If specified, output a figure as an image file to this path.
        """
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_v(v, policy, print_value, to_file)
