import pytest
import torch
import sys 
sys.path.append('..')
from torch import nn
from layerfold.layers import calculate_equivalent_fc_layer, calculate_equivalent_convolutional_layer, fuse_conv_and_bn

@pytest.fixture
def conv_layers():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1).to(device)
    conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1).to(device)
    bn = nn.BatchNorm2d(16).to(device)
    return conv1, conv2, bn

@pytest.fixture
def fc_layers():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fc1 = nn.Linear(128, 64).to(device)
    fc2 = nn.Linear(64, 32).to(device)
    return fc1, fc2

def test_fuse_conv_and_bn(conv_layers):
    conv1, _, bn = conv_layers
    fused_conv = fuse_conv_and_bn(conv1, bn, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    assert isinstance(fused_conv, nn.Conv2d), "Fused layer is not Conv2d!"
    assert fused_conv.bias is not None, "Fused layer doesn't have a bias!"

def test_calculate_equivalent_fc_layer(fc_layers):
    fc1, fc2 = fc_layers
    fused_fc = calculate_equivalent_fc_layer(fc1, fc2, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    assert fused_fc.in_features == fc1.in_features, "Input features mismatch!"
    assert fused_fc.out_features == fc2.out_features, "Output features mismatch!"

def test_calculate_equivalent_convolutional_layer(conv_layers):
    conv1, conv2, _ = conv_layers
    fused_conv = calculate_equivalent_convolutional_layer(conv1, conv2, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    assert fused_conv.in_channels == conv1.in_channels, "Input channels mismatch!"
    assert fused_conv.out_channels == conv2.out_channels, "Output channels mismatch!"
    assert isinstance(fused_conv, nn.Conv2d), "Fused layer is not Conv2d!"
