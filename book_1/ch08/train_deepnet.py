# !usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
# from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()

# trainer = Trainer(network, x_train, t_train, x_test, t_test,
#     epochs=20, mini_batch_size=100,
#     optimizer="Adam", optimizer_param={"lr": 0.001},
#     evaluate_sample_num_per_epoch=1000)

# trainer.train()

# network.save_params("deep_convnet_params.pkl")
# print("Saved Network Parameters!")

network.load_params()
print("Loaded Network Parameters!")

test_size = 1000
x_test = x_test[:test_size]
t_test = t_test[:test_size]

acc = network.accuracy(x_test, t_test)

print("Accuracy:{0}".format(acc))
