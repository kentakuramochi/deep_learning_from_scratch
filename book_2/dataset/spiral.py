import sys
sys.path.append("..")

from common.np import *


def load_data(seed=1984):
    """ Load the spiral dataset
    """
    np.random.seed(seed)
    N = 100  # Num of samples for each class
    DIM = 2  # Num of elements in one sample
    CLS_NUM = 3  # Num of classes

    # Data and labels
    x = np.zeros((N*CLS_NUM, DIM))
    t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=np.int32)

    for j in range(CLS_NUM):
        for i in range(N):  # [N*j, N*(j+1))
            # Parameters for the distribution
            rate = i / N
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

            ix = N * j + i

            x[ix] = np.array(
                [radius*np.sin(theta), radius*np.cos(theta)]
            ).flatten()

            t[ix, j] = 1

    return x, t
