import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np

from common.optimizer import SGD
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
    data_size = len(xs)
    print(f"corpus size: {corpus_size}, vocabulary size: {vocab_size}")

    max_iters = data_size // (batch_size * time_size)
    time_idx = 0
    total_loss = 0
    loss_count = 0
    ppl_list = []  # List of perplexities

    # Create a model
    model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)

    # Calculate a start offset for loading each sample in mini batch
    jump = (corpus_size - 1) // batch_size
    offsets = [i*jump for i in range(batch_size)]

    for epoch in range(max_epoch):
        for iter in range(max_iters):
            # Create mini batches for sequential data
            batch_x = np.empty((batch_size, time_size), dtype="i")
            batch_t = np.empty((batch_size, time_size), dtype="i")
            for t in range(time_size):
                for i, offset in enumerate(offsets):
                    # Add offset
                    batch_x[i, t] = xs[(offset+time_idx) % data_size]
                    batch_t[i, t] = ts[(offset+time_idx) % data_size]
                time_idx += 1

            # Calculate a loss and update parameters
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1

        # Evaluate a perplexity for each epoch
        ppl = np.exp(total_loss / loss_count)
        print(f"| epoch {epoch+1} | perplexity {ppl:.2f}")
        ppl_list.append(float(ppl))
        total_loss, loss_count = 0, 0

    x = np.arange(len(ppl_list))
    plt.plot(x, ppl_list, label="train")
    plt.xlabel("epochs")
    plt.ylabel("perplexity")
    plt.show()
    plt.savefig("perplexity.png")


if __name__ == "__main__":
    main()
