"""
Deep Learning from Scrach
3. Neural network
"""
# !usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

def change_one_hot_label(labels):
    """
    change label to one-hot array
    """
    t = np.zeros((labels.size, 10))

    for (index, row) in enumerate(t):
        row[labels[index]] = 1

    return t

def load_mnist(norm = True, flatten = True, one_hot_label = False):
    """
    load MNIST dataset
    """
    mnist = fetch_mldata("MNIST original", data_home = "./mnist")

    mnist_data = mnist.data.astype("float32")
    mnist_label = mnist.target.astype("int32")

    if norm:
        mnist_data /= 255

    if not flatten:
        mnist_data = mnist_data.reshape(-1, 1, 28, 28)

    if one_hot_label:
        mnist_label = change_one_hot_label(mnist_label)

    data_train, data_test, label_train, label_test =\
        train_test_split(mnist_data, mnist_label, train_size = 60000, test_size = 10000)

    return (data_train, label_train), (data_test, label_test)

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
    with open("./mnist/sample_weight.pkl", "rb") as file:
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

    probs = softmax(np.dot(z2, W3) + b3)

    return np.argmax(probs)

def main():
    """
    main
    """
    (vec_train, lab_train), (vec_test, lab_test) = \
        load_mnist(flatten = True, norm = True, one_hot_label = False)

    network = init_network()

    accuracy = 0

    for (vec, lab) in zip(vec_test, lab_test):
        out = predict(network, vec)

        if out == lab:
            accuracy += 1

    print("Accuracy: {0}".format(accuracy / len(lab_test)))

if __name__ == "__main__":
    main()
