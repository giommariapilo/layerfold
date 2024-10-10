import pytest
import torch
import sys
sys.path.append('./')
from torch import nn
from layerfold.fold import fold, inside_block_fusion, inside_MLP_fusion

class ResnetBlock(nn.Module):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(),
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu2(out)
        return out
    
class SwinTBlock(nn.Module):
    def __init__(self):
        super(SwinTBlock, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64))
    def forward(self, x):
        out = self.MLP(x)
        return out
    

@pytest.fixture
def resnet_block():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    block = ResnetBlock().to(device)
    block.relu1 = nn.Identity()
    return block

@pytest.fixture
def swint_block():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    block = SwinTBlock().to(device)
    block.MLP[1] = nn.Identity()
    return block

def test_fold_resnet(resnet_block):
    folded_model = fold(resnet_block, 'resnet', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(folded_model)
    assert isinstance(folded_model.conv1, nn.Identity), "Folding of ResNet failed!"

def test_fold_swint(swint_block):
    folded_model = fold(swint_block, 'swint', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    assert isinstance(folded_model.MLP[2], nn.Identity), "Folding of Swin Transformer failed!"

def test_inside_block_fusion(resnet_block):
    inside_block_fusion(resnet_block, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), "")
    assert isinstance(resnet_block.bn1, nn.Identity), "BatchNorm not fused correctly in ResNet block!"

def test_fold_invalid_type(resnet_block):
    with pytest.raises(NotImplementedError):
        fold(resnet_block, 'unsupported_model', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
