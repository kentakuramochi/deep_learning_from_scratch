"""Linear regression with DeZero."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from dezero import Variable
import dezero.functions as F


# Toy dataset: y = 5 + 2x + ε
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros((1)))


def predict(x):
    """Predict y = Wx.
    
    Args:
        x (dezero.Variable): Input x.

    Returns:
        (dezero.Variable): Function y = Wx.
    """
    y = F.matmul(x, W) + b
    return y


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    # Print every 10 iteration
    if i % 10 == 0:
        print(loss.data)

print("====")
print("W =", W.data)  # 2
print("b =", b.data)  # 5
