if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import dezero
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


# seq_data = [np.random.randn(1, 1) for _ in range(1000)]  # Dummy sequential data
# xs = seq_data[0:-1]  # Training data: t = [0, N-1]
# ts = seq_data[1:]  # Training label: t = N
#
# model = SimpleRNN(10, 1)
#
# loss, cnt = 0, 0
# for x, t in zip(xs, ts):
#     y = model(x)
#     loss += F.mean_squared_error(y, t)
#
#     cnt += 1
#     if cnt == 2:  # Stop at the second step
#         model.cleargrads()
#         loss.backward()
#         break
#
# model.plot(
#     x, to_file=os.path.join(os.path.dirname(__file__), "output", "simple_rnn.png")
# )

# Hyper parameters
max_epoch = 100
hidden_size = 100
bptt_length = 30  # Length of BPTT (Backpropagation-Through-Time)

train_set = dezero.datasets.SinCurve(train=True)
seqlen = len(train_set)

model = SimpleRNN(hidden_size, 1)
optimizer = dezero.optimizers.Adam().setup(model)

# Training
for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in train_set:
        x = x.reshape(1, 1)  # Reshape an input
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1

        # Backward every 30 steps, or reached to the end of the dataset
        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

    avg_loss = float(loss.data) / count
    print("| epoch %d | loss %f" % (epoch + 1, avg_loss))


# Plot the dataset and the prediction
xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(y.data.flatten())

plt.plot(np.arange(len(xs)), xs, label="y=cos(x)")
plt.plot(np.arange(len(xs)), pred_list, label="predict")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), "output", "sin_rnn.png"))
