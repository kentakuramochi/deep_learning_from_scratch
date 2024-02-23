import sys
sys.path.append("..")

from common.np import *


def preprocess(text):
    """ Create a simple corpus and word-id dictionaries from the text
    """
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def cos_similarity(x, y, eps=1e-8):
    """ Calculate cosine similarity between 2 word vectors
    """
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    """ Find top N words which are similar to the query, from a word matrix
    """
    if query not in word_to_id:
        print(f"{query} is not found")
        return

    print(f"\n[query] {query}")
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # Calculate cosine simiralities between all of the other word vectors
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # Print top N words and the simiralities
    count = 0
    for i in (-1 * similarity).argsort():  # (* -1) to sort with descending order
        if id_to_word[i] == query:
            continue
        print(f" {id_to_word[i]}: {similarity[i]}")

        count += 1
        if count >= top:
            return


def create_co_matrix(corpus, vocab_size, window_size=1):
    """ Create a co-occurence matrix from the corpus
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def clip_grads(grads, max_norm):
    """ Clip gradients with the specified maximum value
    """
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
