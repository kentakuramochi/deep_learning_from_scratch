import sys
sys.path.append("..")

import numpy as np

from common.layers import MatMul


def main():
    # CBOW (Continuous Bag-Of-Words) model: predict a target from contexts
    # Context data
    c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
    c1 = np.array([[1, 0, 0, 0, 0, 0, 0]])

    # Initialize weights
    W_in = np.random.randn(7, 3)
    W_out = np.random.randn(3, 7)

    # Layers
    in_layer0 = MatMul(W_in)
    in_layer1 = MatMul(W_in)
    out_layer = MatMul(W_out)

    # Forward propagation
    h0 = in_layer0.forward(c0)
    h1 = in_layer1.forward(c1)
    h = 0.5 * (h0 + h1)  # Average
    s = out_layer.forward(h)

    print(s)


if __name__ == "__main__":
    main()
