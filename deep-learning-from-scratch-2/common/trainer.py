import sys
sys.path.append("..")

import time
import numpy  # Use NumPy function explicitly
import matplotlib.pyplot as plt

from common.np import *
from common.util import clip_grads

class Trainer:
    """ Trainer of the neural network
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(
        self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=10
    ):
        """ Train the network
        """
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # Shuffle the dataset
            idx = numpy.random.permutation(numpy.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                # Create a mini batch
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # Calculate a loss and update parameters
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1

                # Output a status periodically
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print(
                        "| epoch %d |  iter %d / %d | time %d[s] | loss %.2f"
                        % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss)
                    )
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, saveas=None, ylim=None):
        """ Plot a figure of the progress of the training
        """
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label="train")
        plt.xlabel(f"iterations (x{self.eval_interval})")
        plt.ylabel("loss")
        plt.show()
        if saveas is not None:
            plt.savefig(saveas)


def remove_duplicate(params, grads):
    """ Aggregate duplicated weights in the parameter list,
    and accumurate gradients corresponding with them
    """
    params, grads = params[:], grads[:]  # Copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                if params[i] is params[j]:
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                    params[i].T.shape == params[j].shape and \
                    np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg:
                    break
            if find_flg:
                break

        if not find_flg:
            break

    return params, grads


class RnnlmTrainer:
    """ Trainer of the RNN Language Model network
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size):
        """ Create mini batch for sequential data
        """
        batch_x = np.empty((batch_size, time_size), dtype="i")
        batch_t = np.empty((batch_size, time_size), dtype="i")

        # Calculate a start offset for loading each sample in mini batch
        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                # Add offset
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1

        return batch_x, batch_t


    def fit(
        self, xs, ts, max_epoch=10, batch_size=20, time_size=35, max_grad=None, eval_interval=20
    ):
        """ Train the network
        """
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # Calculate a loss and update parameters
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1

                # Output a status periodically
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print(
                        "| epoch %d |  iter %d / %d | time %d[s] | perplexity %.2f"
                        % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl)
                    )
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, saveas=None, ylim=None):
        """ Plot a figure of the progress of the training
        """
        x = numpy.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label="train")
        plt.xlabel(f"iterations (x{self.eval_interval})")
        plt.ylabel("perplexity")
        plt.show()
        if saveas is not None:
            plt.savefig(saveas)


def remove_duplicate(params, grads):
    """ Aggregate duplicated weights in the parameter list,
    and accumurate gradients corresponding with them
    """
    params, grads = params[:], grads[:]  # Copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # When the weights are shared
                if params[i] is params[j]:
                    grads[i] += grads[j]  # Accumulate grads
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # When the weights are shared with a transposed matrix (weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                    params[i].T.shape == params[j].shape and \
                    np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg:
                    break
            if find_flg:
                break

        if not find_flg:
            break

    return params, grads
