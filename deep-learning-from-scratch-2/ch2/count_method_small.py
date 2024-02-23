import sys
sys.path.append("..")

import numpy
import matplotlib.pyplot as plt

from common.util import preprocess, create_co_matrix, ppmi


def main():
    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)
    W = ppmi(C)

    # SVD (Singular Value Decomposition)
    U, S, V = numpy.linalg.svd(W)

    print(C[0])  # Co-occurrance matrix
    print(W[0])  # PPMI matrix
    print(U[0])  # SVD

    # Plot the words by 2-d vectors
    for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

    plt.scatter(U[:,0], U[:,1], alpha=0.5)
    plt.show()
    plt.savefig("svd.png")


if __name__ == "__main__":
    main()
