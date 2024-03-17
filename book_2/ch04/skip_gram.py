import sys
sys.path.append("..")

from common.layers import *
from ch04.negative_sampling_layer import NegativeSamplingLoss


class SkipGram:
    """ Skip-gram model with a negative sampling
    """
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # Initialize weights
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(V, H).astype("f")

        # Create layers
        self.in_layer = Embedding(W_in)
        self.loss_layers = []
        for i in range(2 * window_size):
            layer = NegativeSamplingLoss(
                W_out, corpus, power=0.75, sample_size=5
            )
            self.loss_layers.append(layer)

        # Aggregate all weights and gradients to a list
        layers = [self.in_layer] + self.loss_layers
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # Distributed representation
        self.word_vecs = W_in

    def forward(self, contexts, target):
        """ Forward propagation (mini-batch)
        """
        h = self.in_layer.forward(target)

        loss = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:,i])
        return loss

    def backward(self, dout=1):
        """ Backward propagation (mini-batch)
        """
        dh = 0
        for i, layer in enumerate(self.loss_layers):
            dh += layer.backward(dout)
        self.in_layer.backward(dh)
        return None
