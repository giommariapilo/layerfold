from .fold import fold
from .layers import calculate_equivalent_fc_layer as fuse_fc
from .layers import calculate_equivalent_convolutional_layer as fuse_conv
from .layers import fuse_conv_and_bn as fuse_convbn
