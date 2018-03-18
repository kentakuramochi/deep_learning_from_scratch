# !usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from chap5.two_layer_net import TwoLayerNet
from common.optimizer import SGD

def smooth_curve(x):
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w / w.sum(), s, mode="valid")

    return y[5:len(y)-5]

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

#w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
#w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

weight_init_types = {"std=0.01": 0.01, "Xavier": "sigmoid", "He": "ReLU"}
optimizer = SGD(lr=0.01)

networks = {}
train_loss = {}

for key, weight_type in weight_init_types.items():
    networks[key] = \
        TwoLayerNet(input_size=784, hidden_size=100, output_size=10, \
        weight_init_std=weight_type)

    train_loss[key] = []

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in weight_init_types.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("===== iteration:{0} =====".format(i))
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print("{0}:{1}".format(key, loss))


markers = {"std=0.01": "o", "Xavier": "s", "He": "D"}
x = np.arange(max_iterations)

for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]))

plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()