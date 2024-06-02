"""Non-linear regression with DeZero."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F


# Toy dataset: y = sin(2πx) + ε
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000


class TwoLayerNet(Model):
    """Neural network with 2 layers.

    Attributes:
        l1 (dezero.layers.Linear): 1st linear layer.
        l2 (dezero.layers.Linear): 2nd linear layer.
    """

    def __init__(self, hidden_size, out_size):
        """Intialize.

        Args:
            hidden_size (int): Num of hidden parameters.
            out_size (int): Num of output elements.
        """
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        """Forward propagation.

        Args:
            x (dezero.Variable): Network input.

        Returns:
            (dezero.Variable): Network output.
        """
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


model = TwoLayerNet(10, 1)

# Stochastic gradient descent (SGD)
optimizer = optimizers.SGD(lr)
optimizer.setup(model)

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    if i % 1000 == 0:
        print(loss)

# Plot dataset and predictions
fig, ax = plt.subplots()
ax.scatter(x, y.flatten())

x_plot = np.linspace(0, 1, 100)
y_plot = model(np.expand_dims(x_plot, axis=1)).data.flatten()
ax.plot(x_plot, y_plot, c="r")

plt.savefig(os.path.join(os.path.dirname(__file__), "output", "predict_sin.png"))
