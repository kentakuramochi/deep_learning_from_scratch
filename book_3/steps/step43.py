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
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)  # y = sin(2πx) + ε

# Num. of input/hidden/output parameters
I, H, O = 1, 10, 1
# Parameters
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))


def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    # Predict
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    # Backprop
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    # Update
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    if i % 1000 == 0:  # Print every 1000 iterations
        print(loss)


# Plot samples and the prediction
fig, ax = plt.subplots()
ax.scatter(x, y)
x_plot = np.linspace(0, 1, 100)
y_plot = predict(np.expand_dims(x_plot, axis=1)).data
ax.plot(x_plot, y_plot, c="r")

plt.show()
plt.savefig(os.path.join(os.path.dirname(__file__), "output", "sin_regression.png"))
