# !usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()

network.load_params()
print("Loaded Network Parameters!")

test_size = 10000
x_test = x_test[:test_size]
t_test = t_test[:test_size]

acc = network.accuracy(x_test, t_test)
print("Accuracy(float64):{0}".format(acc))

x_test = x_test.astype(np.float16)
for param in network.params.values():
    param[...] = param.astype(np.float16)

acc = network.accuracy(x_test, t_test)
print("Accuracy(float64):{0}".format(acc))
