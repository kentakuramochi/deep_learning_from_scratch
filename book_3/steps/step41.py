if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
import dezero.functions as F


x = Variable(np.array([[1, 2], [3, 4]]))
W = Variable(np.array([[5, 6, 7], [8, 9, 10]]))
y = F.matmul(x, W)
y.backward()

print(y)
print(x.grad)
print(W.grad)
