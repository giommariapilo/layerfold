import os
import torch
import pickle
import random
import numpy as np
import re 
from .layers import calculate_equivalent_fc_layer, calculate_equivalent_convolutional_layer, fuse_conv_and_bn
from torch import nn

def set_seed(seed):
  ## SEEDING
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  torch.use_deterministic_algorithms(True, warn_only=True)
  torch.backends.cudnn.benchmark = False
  os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
  return

def inside_block_fusion(model, device, name):
  namespace = {'model': model, 'device': device, 'torch':torch, 'fold': fold, 'nn': nn, 'test': None}
  # print(name)
  exec(f'test = isinstance(model.{name+"downsample"}, nn.Sequential)', namespace)
  exec(f'''
conv = model.{name+"conv1"}
if isinstance(model.{name+"bn1"}, nn.BatchNorm2d):
  bn = model.{name+"bn1"}
  fused_conv = fold.fuse_conv_and_bn(conv, bn, device)
  model.{name + "conv1"} = fused_conv
  model.{name + "bn1"} = nn.Identity()
''', namespace)

  exec(f'''
conv = model.{name+"conv2"}
if isinstance(model.{name+"bn2"}, nn.BatchNorm2d):
  bn = model.{name+"bn2"}
  fused_conv = fold.fuse_conv_and_bn(conv, bn, device)
  model.{name + "conv2"} = fused_conv
  model.{name + "bn2"} = nn.Identity()
''', namespace)


  exec(f'''
conv1 = model.{name+"conv1"}
conv2 = model.{name+"conv2"}
# print(conv1.bias.shape)
# print(conv2.bias.shape)
fused_conv = fold.calculate_equivalent_convolutional_layer(conv1, conv2, device)
model.{name+"conv1"} = nn.Identity()
model.{name+"conv2"}= fused_conv
# print(fused_conv.bias.shape)
''',namespace)

  return


def fold_resnet(model, device):
  def numrepl(matchobj):
    return f'[{matchobj.group(0)[1]}]'
  # something to analize the network
  relu_list = []
  for name, layer in model.named_modules():
    if isinstance(layer, (nn.Identity)):
      #  is this enough?
      #probably the name needs to be changed
      pattern = r'\.[0-9]'  
      name = re.sub(pattern=pattern, repl=numrepl, string=name)
      relu_list.append(name)
  # something to fuse the right layers
  # I will need exec to adapt this code

  count = 0

  for i in range(len(relu_list)):
  # for relu in relu_list:
    if relu_list[i][-1] =='1':
      inside_block_fusion(model, device, relu_list[i][:-5])
      with open(f'/raid/gpilo/fusing_layers/fused_models/CIFAR-10_ResNet-18/fused_layers{count+1}', 'wb') as f:
        pickle.dump(model, f)
      count += 1


  return

def fold_swint():
    # TODO implement this
    pass

def fold(model, type):
  return

if __name__ == '__main__':
  pass