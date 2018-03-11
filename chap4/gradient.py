# !usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pylab as plt

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i]

        x[i] = tmp + h
        fxh1 = f(x)
        x[i] = tmp - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)

        x -= lr * grad

    return x

def f1(x):
    return 0.01 * x ** 2 + 0.1 * x

def f2(x):
    return x[0] ** 2 + x[1] ** 2

# test
init_x = np.array([-3., 4.])
print(gradient_descent(f2, init_x=init_x, lr=1e-10, step_num=100))
