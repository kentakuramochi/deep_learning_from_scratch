from common.np import *


def sigmoid(x):
    """ Sigmoid function (for TimeSigmoidWithLoss)
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """ Softmax function (for TimeSoftmaxWithLoss)
    """
    if x.ndim == 2:  # For 2-D tensor
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    """ Cross entropy error
    """
    if y.ndim == 1:
        # Expand a batch dimension for a 1-D tensor
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # When the training data is one-hot-vector, get its index
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    # Access the y element corresponding with the index by fancy indexing
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
