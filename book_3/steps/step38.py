if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
import dezero.functions as F


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(y.data)
print(y.grad)
print(x.grad)

print("##########")
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = x.reshape((2, 3))
print(y.data)

print("##########")
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = x.reshape(2, 3)
print(y.data)


print("##########")
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
y.backward()
print(y.data)
print(x.grad)

print("##########")
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = x.transpose()
print(y.data)

print("##########")
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = x.T
print(y.data)
