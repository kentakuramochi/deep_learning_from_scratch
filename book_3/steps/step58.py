if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

import dezero
from dezero.models import VGG16


url = (
    "https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg"
)
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
x = VGG16.preprocess(img)
x = x[np.newaxis]  # CHW to NCHW (N=1)

model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

# Visualize a computational graph
model.plot(x, to_file=os.path.join(os.path.dirname(__file__), "output", "vgg.pdf"))
labels = dezero.datasets.ImageNet.labels()  # ImageNet labels
print(labels[predict_id])
