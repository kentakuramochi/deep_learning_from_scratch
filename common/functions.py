# !usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

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

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    return -np.sum(t * np.log(y)) / batch_size
