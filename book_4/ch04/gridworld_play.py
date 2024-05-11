if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from common.gridworld import GridWorld


env = GridWorld()  # Gridworld
V = {}  # State value function

# Set random values
for state in env.states():
    V[state] = np.random.randn()

# Visualize
env.render_v(V, to_file=os.path.join(os.path.dirname(__file__), "gridworld.png"))
