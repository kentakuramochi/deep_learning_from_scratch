if "__file__" in globals():  # __file__ is an absolute path of this script
    import os, sys

    # Add a path of dezero to the system path, by a relative path from this script
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable


def main():
    x = Variable(np.array(1.0))
    y = (x + 3) ** 2
    y.backward()

    print(y)

    print(x.grad)


if __name__ == "__main__":
    main()
