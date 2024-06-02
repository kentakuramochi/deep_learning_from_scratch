"""Matmul with DeZero."""

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from dezero import Variable
import dezero.functions as F


# Multiply vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a, b = Variable(a), Variable(b)
c = F.matmul(a, b)
print(c)

# Multiply matrices
a = np.array([[1, 2], [3, 4]])
b = np.array([[3, 4], [5, 6]])
c = F.matmul(a, b)
print(c)
