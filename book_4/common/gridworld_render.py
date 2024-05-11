"""Utilities to visualize the gridworld."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Renderer:
    """Renderer of the gridworld.

    Attributes:
        reward_map (numpy.ndarray(float)): Rewards of each grid.
        goal_state (Tuple[int, int]): Grid index of the goal.
        wall_state (Tuple[int, int]): Grid index of the wall.
        ys (int): Height of the world.
        xs (int): Width of the world.
        ax (matplotlib.axes): Plot axes.
        fig (matplotlib.figure): Plot figure.
    """

    def __init__(self, reward_map, goal_state, wall_state):
        self.reward_map = reward_map
        self.goal_state = goal_state
        self.wall_state = wall_state
        # Grids
        self.ys = len(self.reward_map)
        self.xs = len(self.reward_map[0])

        # Axis
        self.ax = None
        self.fig = None
        # self.first_flg = True  # Unused?

    def set_figure(self, figsize=None):
        """Set configurations of an output figure.

        Args:
            figsize (Tuple[float, float]): Size of the output figure.
        """
        fig = plt.figure(figsize=figsize)
        self.ax = fig.add_subplot(111)  # 1 figure, 1x1
        ax = self.ax
        ax.clear()
        ax.tick_params(
            labelbottom=False, labelleft=False, labelright=False, labeltop=False
        )
        ax.set_xticks(range(self.xs))
        ax.set_yticks(range(self.ys))
        ax.set_xlim(0, self.xs)
        ax.set_ylim(0, self.ys)
        ax.grid(True)  # Draw grid

    def render_v(self, v=None, policy=None, print_value=True, to_file=None):
        """Visualize the gridworld with state values.

        Args:
            v (Tuple[int, int]): State values.
            policy (Dict[int, float]): Policy of the agent.
            print_value (bool): If True, print the state values.
            to_file (str): If specified, output a figure as an image file with this name.
        """
        self.set_figure()

        ys, xs = self.ys, self.xs
        ax = self.ax

        if v is not None:
            # Colormap: red to green in decending order by the state value
            color_list = ["red", "white", "green"]
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "colormap_name", color_list
            )

            # dict to ndarray
            v_dict = v
            v = np.zeros(self.reward_map.shape)
            for state, value in v_dict.items():
                v[state] = value

            vmax, vmin = v.max(), v.min()
            vmax = max(vmax, abs(vmin))
            vmin = -1 * vmax
            vmax = 1 if vmax < 1 else vmax
            vmin = -1 if vmin > -1 else vmin

            ax.pcolormesh(np.flipud(v), cmap=cmap, vmin=vmin, vmax=vmax)

        # Draw info into the grids
        for y in range(ys):
            for x in range(xs):
                state = (y, x)
                # Reward
                r = self.reward_map[y, x]
                if r != 0 and r is not None:
                    txt = "R " + str(r)
                    if state == self.goal_state:  # Goal
                        txt = txt + " (GOAL)"
                    ax.text(x + 0.1, ys - y - 0.9, txt)

                # State value
                if (v is not None) and state != self.wall_state:
                    if print_value:
                        # Upper right of the grid
                        offsets = [(0.4, -0.15), (-0.15, -0.3)]
                        key = 0
                        if v.shape[0] > 7:
                            key = 1
                        offset = offsets[key]
                        ax.text(
                            x + offset[0],
                            ys - y + offset[1],
                            "{:12.2f}".format(v[y, x]),
                        )

                # Action (by arrows)
                if policy is not None and state != self.wall_state:
                    actions = policy[state]
                    # Optimal action value
                    max_actions = [
                        kv[0]
                        for kv in actions.items()
                        if kv[1] == max(actions.values())
                    ]

                    arrows = ["↑", "↓", "←", "→"]
                    offsets = [(0, 0.1), (0, -0.1), (-0.1, 0), (0.1, 0)]
                    for action in max_actions:
                        arrow = arrows[action]
                        offset = offsets[action]
                        if state == self.goal_state:
                            continue
                        ax.text(x + 0.45 + offset[0], ys - y - 0.5 + offset[1], arrow)

                # Patch the wall with a gray square
                if state == self.wall_state:
                    ax.add_patch(
                        plt.Rectangle((x, ys - y - 1), 1, 1, fc=(0.4, 0.4, 0.4, 1.0))
                    )

        plt.show()
        if to_file is not None:  # Output an image file
            plt.savefig(to_file)
