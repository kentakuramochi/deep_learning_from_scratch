if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import math
import numpy as np
import matplotlib.pyplot as plt
import dezero
import dezero.functions as F
from dezero.models import MLP
from dezero import optimizers


def plot_decision_boundary(x, y, t, model, output_path):
    _, ax = plt.subplots()

    # Create a data mesh
    plot_range = np.arange(-1, 1, 0.01)
    x0_plot, x1_plot = np.meshgrid(plot_range, plot_range)
    y_mesh = model(np.array([x0_plot.ravel(), x1_plot.ravel()]).T)

    # Plot the decision boundary by the data mesh
    y_plot = np.array([np.argmax(yp) for yp in y_mesh.data])
    y_plot = y_plot.reshape(x0_plot.shape)
    plt.contourf(x0_plot, x1_plot, y_plot)

    # Plot samples in the dataset
    MARKERS = [("orange", "o"), ("blue", "x"), ("green", "^")]
    for i, label in enumerate(t):
        ax.scatter(x[i], y[i], c=MARKERS[label][0], marker=MARKERS[label][1])

    plt.savefig(output_path)


# Hyperparameters
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# Load dataset and create optimizer
x, t = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    # Shuffle indices for the dataset
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # Create a minibatch
        batch_index = index[i * batch_size : (i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        # Caculate gradients and update parameters
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print("epoch %d, loss %.2f" % (epoch + 1, avg_loss))


plot_decision_boundary(
    x[:, 0],
    x[:, 1],
    t,
    model,
    os.path.join(os.path.dirname(__file__), "output", "classify_spiral_dataset.png"),
)
