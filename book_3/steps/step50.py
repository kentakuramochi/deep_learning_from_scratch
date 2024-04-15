if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import math
import numpy as np
import matplotlib.pyplot as plt
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP


max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# Mini-batch for training
train_set = dezero.datasets.Spiral(train=True)
train_loader = DataLoader(train_set, batch_size)

# Mini-batch for test
test_set = dezero.datasets.Spiral(train=False)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)

train_acc = []
train_loss = []
test_acc = []
test_loss = []

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    # Training
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print("epoch: {}".format(epoch + 1))
    print(
        "train loss: {:.4f}, accuracy: {:.4f}".format(
            sum_loss / len(train_set), sum_acc / len(train_set)
        )
    )

    train_loss.append(sum_loss / len(train_set))
    train_acc.append(sum_acc / len(train_set))

    # Test
    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():  # Without grads
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print(
        "test loss: {:.4f}, accuracy: {:.4f}".format(
            sum_loss / len(train_set), sum_acc / len(train_set)
        )
    )

    test_loss.append(sum_loss / len(train_set))
    test_acc.append(sum_acc / len(train_set))

epoch = np.arange(max_epoch)

# Plot loss
plt.plot(epoch, np.array(train_loss), label="train")
plt.plot(epoch, np.array(test_loss), label="test")
plt.legend(loc="upper right")
plt.savefig(os.path.join(os.path.dirname(__file__), "output", "minibatch_loss.png"))

plt.cla()
# Plot accuracy
plt.plot(epoch, np.array(train_acc), label="train")
plt.plot(epoch, np.array(test_acc), label="test")
plt.legend(loc="upper right")
plt.savefig(os.path.join(os.path.dirname(__file__), "output", "minibatch_acc.png"))
