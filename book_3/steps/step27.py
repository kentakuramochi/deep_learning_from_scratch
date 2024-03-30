if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
from dezero import Variable, Function
from dezero.utils import plot_dot_graph


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)


# sin(x) by the Maclaurin series
# sin(x) = x/1! - x^3/3! + x^5/5! - ...
def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:  # Cut off the calculation
            break
    return y


x = Variable(np.array(np.pi / 4))
y = sin(x)
y.backward()

print(y.data)
print(x.grad)

x = Variable(np.array(np.pi / 4))
y = my_sin(x)
y.backward()

print(y.data)
print(x.grad)

plot_dot_graph(
    y, False, os.path.join(os.path.dirname(__file__), "output", "sin_th_1e-4.png")
)

x = Variable(np.array(np.pi / 4))
y = my_sin(x, threshold=1e-150)
y.backward()

print(y.data)
print(x.grad)

plot_dot_graph(
    y, False, os.path.join(os.path.dirname(__file__), "output", "sin_th_1e-150.png")
)
