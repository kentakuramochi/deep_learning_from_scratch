if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Model
import dezero.functions as F
import dezero.layers as L


# Simple RNN
class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y


seq_data = [np.random.randn(1, 1) for _ in range(1000)]  # Dummy sequential data
xs = seq_data[0:-1]  # Training data: t = [0, N-1]
ts = seq_data[1:]  # Training label: t = N

model = SimpleRNN(10, 1)

loss, cnt = 0, 0
for x, t in zip(xs, ts):
    y = model(x)
    loss += F.mean_squared_error(y, t)

    cnt += 1
    if cnt == 2:  # Stop at the second step
        model.cleargrads()
        loss.backward()
        break

model.plot(
    x, to_file=os.path.join(os.path.dirname(__file__), "output", "simple_rnn.png")
)
