import numpy as np


class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        # Integrate parameters into a list
        params = [p for p in self.target.params() if p.grad is not None]

        # Preprocess
        for f in self.hooks:
            f(params)

        # Update parameters
        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    # Add a hook for the preprocess
    def add_hook(self, f):
        self.hooks.append(f)


# Stochastic gradient descent (SGD)
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        # Create an individual momentum v for each parameter
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        # Update with the momentum
        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v
