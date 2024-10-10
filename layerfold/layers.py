import torch
import torch.nn.functional as F
from torch import nn

def calculate_equivalent_fc_layer(fc1, fc2, device):
    fc_eq = nn.Linear(fc1.in_features, fc2.out_features).to(device)
    fc_eq.weight.data = torch.mm(fc2.weight, fc1.weight)
    fc_eq.bias.data = torch.add(torch.matmul(fc2.weight, fc1.bias), fc2.bias)
    return fc_eq

def calculate_equivalent_convolutional_layer(conv1, conv2, device):
    if conv1.bias is None:
        conv1_bias = torch.zeros(conv1.out_channels).to(device)
    else:
        conv1_bias = conv1.bias

    if conv2.bias is None:
        conv2_bias = torch.zeros(conv2.out_channels).to(device)
    else:
        conv2_bias = conv2.bias

    conv_eq_weight_data = torch.conv2d(conv1.weight.data.permute(1, 0, 2, 3),
                                    conv2.weight.data.flip(-1, -2),
                                    padding=[conv2.kernel_size[0] - 1, conv2.kernel_size[0] - 1]).permute(1, 0, 2, 3).to(device)
    conv_eq = nn.Conv2d(in_channels=conv1.in_channels,
                        out_channels=conv2.out_channels,
                        kernel_size=3,
                        padding=conv1.padding[0],
                        stride=conv1.stride[0],
                        # bias=False
                        ).to(device)
    # print(conv_eq_weight_data[:,:,1:4,1:4].shape)
    conv_eq.weight = nn.Parameter(conv_eq_weight_data[:,:,1:4,1:4])
    bias_eq = conv2(torch.ones(1, conv1.out_channels, * conv2.kernel_size).to(device) * conv1_bias[None, :, None, None]).sum(dim=(0,2,3)).to(device) + conv2_bias
    # bias_eq = conv2(torch.ones(1, conv1.out_channels, * conv2.kernel_size).to('cyda') * conv1_bias[None, :, None, None]).flatten().to(device)    
    conv_eq.bias = nn.Parameter(bias_eq)

    return conv_eq   

# function from https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
def fuse_conv_and_bn(conv, bn, device):
	#
	# init
	fusedconv = torch.nn.Conv2d(
		conv.in_channels,
		conv.out_channels,
		kernel_size=conv.kernel_size,
		stride=conv.stride,
		padding=conv.padding,
		bias=True
	).to(device)
	#
	# prepare filters
	w_conv = conv.weight.clone().view(conv.out_channels, -1).to(device)
	w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var))).to(device)
	fusedconv.weight = nn.Parameter(torch.mm(w_bn, w_conv).view(fusedconv.weight.size())) 
	#
	# prepare spatial bias
	if conv.bias is not None:
		b_conv = conv.bias
	else:
		b_conv = torch.zeros( conv.weight.size(0) ).to(device)
	b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
	fusedconv.bias = nn.Parameter( torch.matmul(w_bn, b_conv) + b_bn )
	#
	# we're done
	return fusedconv



