import numpy as np
from dezero import cuda
from dezero.core import Function
from dezero.utils import pair, get_conv_outsize


# =============================================================================
# im2col/col2im
# =============================================================================
class Im2Col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        # gx = col2im(gy, self.input_shape, self.kernel_size, self.stride,
        #             self.pad, self.to_matrix)
        # return gx
        raise NotImplementedError


def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
    """Extract patches from an image based on the filter."""
    y = Im2Col(kernel_size, stride, pad, to_matrix)(x)
    return y


# =============================================================================
# NumPy im2col
# =============================================================================
def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    # Batch, channel, height, width
    N, C, H, W = img.shape
    # Kernel parameters
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    # Output height/width
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = cuda.get_array_module(img)
    if xp != np:
        # col = _im2col_gpu(img, kernel_size, stride, pad)
        raise NotImplementedError
    else:
        # Fill zero paddings
        img = np.pad(
            img,
            ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
            mode="constant",
            constant_values=(0,),
        )
        col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    # Reshape to a matrix
    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col
