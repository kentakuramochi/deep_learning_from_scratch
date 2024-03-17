# !usr/bin/env python
# -*- coding: utf-8 -*-

"""
5章のTwoLayerNetを用いた比較
"""

import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from ch05.two_layer_net import TwoLayerNet
from common.optimizer import *

def smooth_curve(x):
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w / w.sum(), s, mode="valid")

    return y[5:len(y)-5]

(x_train, t_train), (x_test, t_test) =\
    load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000
iter_per_epoch = 100


optimizers = {}
optimizers["SGD"] = SGD()
optimizers["Momentum"] = Momentum()
optimizers["AdaGrad"] = AdaGrad()
optimizers["Adam"] = Adam()

networks = {}
train_loss = {}

for key in optimizers.keys():
    networks[key] = \
        TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    train_loss[key] = []

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % iter_per_epoch == 0:
        print("===== iteration:{0} =====".format(i))
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print("{0}:{1}".format(key, loss))

markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)

for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]))

plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()