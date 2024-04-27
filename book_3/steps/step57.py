if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero.functions as F

from dezero.core import Variable
from dezero.functions_conv import conv2d_simple, pooling_simple


x1 = np.random.rand(1, 3, 7, 7)
col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(x1.shape, "->", col1.shape)

x2 = np.random.rand(10, 3, 7, 7)
kernel_size = (5, 5)
stride = (1, 1)
pad = (0, 0)
col2 = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
print(x2.shape, "->", col2.shape)

print("===== conv =====")
N, C, H, W = 1, 5, 15, 15
OC, (KH, KW) = 8, (3, 3)
x = Variable(np.random.randn(N, C, H, W))
W = np.random.randn(OC, C, KH, KW)
# y = conv2d_simple(x, W, b=None, stride=1, pad=1)
y = F.conv2d(x, W, b=None, stride=1, pad=1)
y.backward()

print(y.shape)
print(x.grad.shape)

print("===== pool =====")
N, C, H, W = 1, 3, 16, 16
x = Variable(np.random.randn(N, C, H, W))
# y = pooling_simple(x, 2, stride=2, pad=0)
y = F.pooling(x, 2, stride=2, pad=0)
y.backward()

print(y.shape)
print(x.grad.shape)
