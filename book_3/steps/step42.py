if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F


# Toy dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)  # y = 5 + 2x + ε

# Linear regression: y = Wx + b
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    y = F.matmul(x, W) + b
    return y


# Naive MSE
def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff**2) / len(diff)


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()  # Backprop

    # Update
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)


# Plot samples and the prediction
fig, ax = plt.subplots()
ax.scatter(x, y)

x_plot = np.linspace(0, 1, 100)
ax.plot(x_plot, (np.dot(np.expand_dims(x_plot, axis=1), W.data) + b.data), c="r")

plt.show()
plt.savefig(os.path.join(os.path.dirname(__file__), "output", "linear_regression.png"))
