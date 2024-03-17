"""
Deep Learning from Scrach
3. Neural network
"""
# !usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist

def sigmoid(x):
    """
    sigmoid function
    """
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    """
    identity function
    """
    return x

def softmax(a):
    """
    softmax
    """
    c = a.max()

    return np.exp(a - c) / np.sum(np.exp(a - c))

def init_network():
    """
    initialize 3 layer NN
    """
    with open("../dataset/sample_weight.pkl", "rb") as file:
        network = pickle.load(file)

    return network

def predict(network, x):
    """
    output prediction of 3 layer NN
    """
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    z1 = sigmoid(np.dot(x, W1) + b1)
    z2 = sigmoid(np.dot(z1, W2) + b2)

    return softmax(np.dot(z2, W3) + b3)

def main():
    """
    main
    """
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=False)

    network = init_network()

    batch_size = 100

    accuracy = 0

    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)

        accuracy += np.sum(p == t_test[i:i+batch_size])

    print("Accuracy: {0}".format(accuracy / len(t_test)))

if __name__ == "__main__":
    main()
