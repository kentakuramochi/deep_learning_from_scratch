if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable


# f(x) = x^4 - 2x^2
def f(x):
    y = x**4 - 2 * x**2
    return y


# f''(x) = 12x^2 - 4
def gx2(x):
    return 12 * x**2 - 4


x = Variable(np.array(2.0))

iters = 10

# Newton's method
for i in range(iters):
    print(i, x)  # Minimum: x = 1 or x = -1

    y = f(x)
    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)  # x = x - f'(x) / f''(x), quadratic approx.
