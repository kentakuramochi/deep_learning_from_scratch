"""
Deep Learning from Scrach
3. Neural network
"""
# !usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

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
    initialize 3-layer NN
    """
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["B1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["B2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["B3"] = np.array([0.1, 0.2])

    return network

def forward_propagation(network, x):
    """
    forward propagation of 3-layer NN
    """
    w1, w2, w3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["B1"], network["B2"], network["B3"]

    z1 = sigmoid(np.dot(x, w1) + b1)
    z2 = sigmoid(np.dot(z1, w2) + b2)

    return identity_function(np.dot(z2, w3) + b3)

def main():
    """
    main
    """
    mnist = fetch_mldata("MNIST original", data_home = "./mnist")

    print(type(mnist))
    print(mnist.keys())

if __name__ == "__main__":
    main()
