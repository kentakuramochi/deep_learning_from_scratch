import numpy as np
import matplotlib.pyplot as plt


def main():
    N = 2  # Mini-batch size
    H = 3  # Num of dimensions of a hidden state vector
    T = 20  # Length of time series data

    dh = np.ones((N, H))

    np.random.seed(3)  # Fix a seed
    # Wh = np.random.randn(H, H)  # Before
    Wh = np.random.randn(H, H) * 0.5  # After

    norm_list = []
    for t in range(T):
        dh = np.dot(dh, Wh.T)
        norm = np.sqrt(np.sum(dh**2)) / N  # L2 norm
        norm_list.append(norm)

    print(norm_list)

    plt.plot(np.arange(len(norm_list)), norm_list)
    plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
    plt.xlabel("time step")
    plt.ylabel("norm")
    plt.show()
    plt.savefig("dh.png")


if __name__ == "__main__":
    main()
