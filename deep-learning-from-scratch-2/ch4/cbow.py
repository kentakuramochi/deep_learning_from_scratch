import sys
sys.path.append("..")

from common.np import *
from common.layers import Embedding
from ch4.negative_sampling_layer import NegativeSamplingLoss


class CBOW:
    """ CBOW (Continuous Bag-Of-Words) model
    with an embedding layer and a negative sampling
    """
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # Initialize weights
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(V, H).astype("f")

        # Create layers
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(
            W_out, corpus, power=0.75, sample_size=5
        )

        # Aggregate all weights and gradients to a list
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # Distributed representation
        self.word_vecs = W_in

    def forward(self, contexts, target):
        """ Forward propagation (mini-batch)
        """
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:,i])
        h += 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        """ Backward propagation (mini-batch)
        """
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
