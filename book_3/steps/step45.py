if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable, Model
import dezero.layers as L
import dezero.functions as F


# Toy dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)  # y = sin(2πx) + ε

# Hyperparameters
lr = 0.2
iters = 10000
hidden_size = 10


# Model definition
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


model = TwoLayerNet(hidden_size, 1)

for i in range(iters):
    # Predict
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    # Backprop
    model.cleargrads()
    loss.backward()

    # Update
    for p in model.params():
        p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)


model.plot(
    x, to_file=os.path.join(os.path.dirname(__file__), "output", "twolayernet.png")
)


# Same with step 43
# fig, ax = plt.subplots()
# ax.scatter(x, y)
# x_plot = np.linspace(0, 1, 100)
# y_plot = model(np.expand_dims(x_plot, axis=1)).data
# ax.plot(x_plot, y_plot, c="r")
# plt.show()
# plt.savefig(os.path.join(os.path.dirname(__file__), "output", "sin_regression.png"))
