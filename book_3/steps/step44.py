if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
import dezero.layers as L


# Toy dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)  # y = sin(2πx) + ε

# Linear layers
l1 = L.Linear(10)
l2 = L.Linear(1)


def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    # Predict
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    # Backprop
    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    # Update
    for l in {l1, l2}:
        for p in l.params():
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)


# Same with step 43
# fig, ax = plt.subplots()
# ax.scatter(x, y)
# x_plot = np.linspace(0, 1, 100)
# y_plot = predict(np.expand_dims(x_plot, axis=1)).data
# ax.plot(x_plot, y_plot, c="r")
#
# plt.show()
# plt.savefig(os.path.join(os.path.dirname(__file__), "output", "sin_regression.png"))
