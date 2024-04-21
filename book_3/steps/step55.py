def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


H, W = 4, 4  # Input height/width
KH, KW = 3, 3  # Kernel height/width
SH, SW = 1, 1  # Stride Y/X
PH, PW = 1, 1  # Padding Y/X

# Output size
OH = get_conv_outsize(H, KH, SH, PH)
OW = get_conv_outsize(W, KW, SW, PW)
print(OH, OW)
