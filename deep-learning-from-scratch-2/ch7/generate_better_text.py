import sys
sys.path.append("..")

from common.np import *
from rnnlm_gen import BetterRnnlmGen
from dataset import ptb


def main():
    corpus, word_to_id, id_to_word = ptb.load_data("train")
    vocab_size = len(word_to_id)
    corpus_size = len(corpus)

    model = BetterRnnlmGen(wordvec_size=650, hidden_size=650)
    model.load_params("../ch6/BetterRnnlm.pkl")  # Load pre-trained parameters

    # Set start/skip words
    start_word = "you"
    start_id = word_to_id[start_word]
    skip_words = ["N", "<unk>", "$"]
    skip_ids = [word_to_id[w] for w in skip_words]

    # Generate a sentence
    word_ids = model.generate(start_id, skip_ids)
    txt = " ".join([id_to_word[i] for i in word_ids])
    txt = txt.replace(" <eos>", ".\n")
    print(txt)

    # Generate following sentences to "the meaning for life is" ...
    model.reset_state()
    start_words = "the meaning of life is"
    start_ids = [word_to_id[w] for w in start_words.split(" ")]
    for i in start_ids[:-1]:
        x = np.array(i).reshape(1, 1)
        model.predict(x)

    word_ids = model.generate(start_ids[-1], skip_ids)
    word_ids = start_ids[:-1] + word_ids
    txt = " ".join([id_to_word[i] for i in word_ids])
    txt = txt.replace(" <eos>", ".\n")
    print("-" * 50)
    print(txt)


if __name__ == "__main__":
    main()
