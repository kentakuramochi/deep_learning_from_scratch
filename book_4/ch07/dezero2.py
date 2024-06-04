"""Get a minimum value of the Rosenbrock function with DeZero."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from dezero import Variable


def rosenbrock(x0, x1):
    """Rosenbrock function.
    
    Args:
        x0 (dezero.Variable): Variable x0.
        x1 (dezero.Variable): Variable x1.

    Returns:
        (dezero.Variable): Rosenbrock function at (x0, x1)
    """
    y = 100 * (x1 - x0**2) ** 2 + (x0 - 1) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

lr = 0.001  # Learning rate
iters = 10000  # Num of iterations

# Gradient descent
for i in range(iters):
    print(x0, x1)
    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad.data
    x1.data -= lr * x1.grad.data

print("====================")
print(x0, x1)
