#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep Learning from scrach
2. Perceptron
"""

import numpy as np

def step(x):
    """
    step function
    """
    if x > 0:
        return 1
    else:
        return 0

def AND(x1, x2):
    """
    AND
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    return step(np.dot(w, x) + b)

def NAND(x1, x2):
    """
    NAND
    """
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    return step(np.dot(w, x) + b)

def OR(x1, x2):
    """
    OR
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    return step(np.dot(w, x) + b)

def XOR(x1, x2):
    """
    XOR
    """
    return AND(NAND(x1, x2), OR(x1, x2))

def main():
    """
    main
    """
    print("***AND***")
    print("0 0:", AND(0, 0))
    print("0 1:", AND(0, 1))
    print("1 0:", AND(1, 0))
    print("1 1:", AND(1, 1))
    print("***NAND***")
    print("0 0:", NAND(0, 0))
    print("0 1:", NAND(0, 1))
    print("1 0:", NAND(1, 0))
    print("1 1:", NAND(1, 1))
    print("***OR***")
    print("0 0:", OR(0, 0))
    print("0 1:", OR(0, 1))
    print("1 0:", OR(1, 0))
    print("1 1:", OR(1, 1))
    print("***XOR***")
    print("0 0:", XOR(0, 0))
    print("0 1:", XOR(0, 1))
    print("1 0:", XOR(1, 0))
    print("1 1:", XOR(1, 1))

if __name__ == "__main__":
    main()
