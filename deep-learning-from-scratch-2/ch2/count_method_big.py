import sys
sys.path.append("..")

import numpy

from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb


def main():
    window_size = 2
    wordvec_size = 100  # Number of singular values and vectors

    corpus, word_to_id, id_to_word = ptb.load_data("train")
    vocab_size = len(word_to_id)
    print("counting co-occurrence ...")
    C = create_co_matrix(corpus, vocab_size, window_size)
    print("calculating PPMI ...")
    W = ppmi(C, verbose=True)

    print("calculating SVD ...")
    try:
        # Truncated SVD (fast!)
        from sklearn.utils.extmath import randomized_svd
        U, S, V = randomized_svd(
            W, n_components=wordvec_size, n_iter=5, random_state=None  # No seed
        )
    except ImportError:  # If an import of scikit-learn failed
        # SVD (slow)
        U, S, V = numpy.linalg.svd(W)

    # Extract the vectors
    word_vecs = U[:, :wordvec_size]

    # Print top 5 words similar with each query
    queries = ["you", "year", "car", "toyota"]
    for query in queries:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)


if __name__ == "__main__":
    main()
