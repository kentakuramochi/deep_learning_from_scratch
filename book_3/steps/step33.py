if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable


# f(x) = x^4 - 2x^2
def f(x):
    y = x**4 - 2 * x**2
    return y


x = Variable(np.array(2.0))
y = f(x)

# f'(x) = 4x^3 - 4x, f'(2) = 24
y.backward(create_graph=True)
print(x.grad)

# f''(x) = 12x^2 - 4, f''(2) = 44
gx = x.grad
x.cleargrad()
gx.backward()
print(x.grad)


x = Variable(np.array(2.0))
iters = 10

# Newton's method
for i in range(iters):
    print(i, x)  # Minimum: x = 1 or x = -1

    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data  # x = x - f'(x) / f''(x), quadratic approx.
