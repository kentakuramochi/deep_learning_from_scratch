# Penn Treebank (PTB) dataset
import sys
import os
sys.path.append("..")

try:
    import urllib.request
except ImportError:
    raise ImportError("Use Python3!")

import pickle
import numpy as np


URL_BASE = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/"

# Data type and correponding .txt file
key_file = {
    "train": "ptb.train.txt",
    "test": "ptb.test.txt",
    "valid": "ptb.valid.txt"
}
# Data type and correponding .npy file
save_file = {
    "train": "ptb.train.npy",
    "test": "ptb.test.npy",
    "valid": "ptb.valid.npy"
}
vocab_file = "ptb.vocab.pkl"

# Put dataset files in the same directory with this script
dataset_dir = os.path.dirname(os.path.abspath(__file__))


def _download(file_name):
    """ Download the PTB dataset
    """
    # If the file already exists, skip
    file_path = dataset_dir + "/" + file_name
    if os.path.exists(file_path):
        return

    print(f"Downloading {file_name} ...")

    # Download the specified file
    try:
        urllib.request.urlretrieve(URL_BASE + file_name, file_path)
    except urllib.error.URLError:
        # If the download failed, retry without SSL verification
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(URL_BASE + file_name, file_path)

    print("Done")


def load_vocab():
    """ Load a vocabulary file
    """
    vocab_path = dataset_dir + "/" + vocab_file

    # If the file already exists, load it
    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word

    word_to_id = {}
    id_to_word = {}
    data_type = "train"
    file_name = key_file[data_type]
    file_path = dataset_dir + "/" + file_name

    _download(file_name)

    words = open(file_path).read().replace("\n", "<eos>").strip().split()

    # Register words into the word-id dictionaries
    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    # And save them as a pickle file
    with open(vocab_path, "wb") as f:
        pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word


def load_data(data_type="train"):
    if data_type == "val":
        data_type = "valid"

    save_path = dataset_dir + "/" + save_file[data_type]

    word_to_id, id_to_word = load_vocab()

    # If an .npy file already exists, load it
    if os.path.exists(save_path):
        corpus = np.load(save_path)
        return corpus, word_to_id, id_to_word

    file_name = key_file[data_type]
    file_path = dataset_dir + "/" + file_name
    _download(file_name)

    # Create a corpus and save it as the .npy file
    words = open(file_path).read().replace("\n", "<eos>").strip().split()
    corpus = np.array([word_to_id[w] for w in words])

    np.save(save_path, corpus)

    return corpus, word_to_id, id_to_word


if __name__ == "__main__":
    for data_type in ("train", "val", "test"):
        load_data(data_type)
