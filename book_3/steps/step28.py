if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable


# Rosenbrock function
# y = b(x1 - x0^2)^2 + a(x0 - 1)^2
# (a = 1, b = 100)
def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2) ** 2 + (x0 - 1) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001  # Learning rate
iters = 10000  # Num. of iterations

# Gradient descent
for i in range(iters):
    # print(x0, x1)  # Minimum: (x0, x1) = (1, 1)

    y = rosenbrock(x0, x1)

    x0.cleargrad()  # Clear grad for every iteration
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

print(x0, x1)  # Minimum: (x0, x1) = (1, 1)
