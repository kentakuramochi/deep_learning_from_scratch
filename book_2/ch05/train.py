import sys
sys.path.append("..")

from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm


def main():
    # Hyperparameters
    batch_size = 10
    wordvec_size = 100
    hidden_size = 100  # Num. of hidden state vectors in RNN
    time_size = 5  # Size of a time series of the Truncated BPTT
    lr = 0.1
    max_epoch = 100

    # Load dataset (first 1000 data)
    corpus, word_to_id, id_to_word = ptb.load_data("train")
    corpus_size = 1000
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)

    xs = corpus[:-1]  # Input
    ts = corpus[1:]  # Output (label)

    # Create a model
    model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)

    # Train the model
    trainer.fit(xs, ts, max_epoch, batch_size, time_size)
    trainer.plot(saveas="train.png")


if __name__ == "__main__":
    main()
