import os
import torch
import random
import numpy as np
import re 
from .layers import calculate_equivalent_fc_layer, calculate_equivalent_convolutional_layer, fuse_conv_and_bn
from torch import nn
from copy import deepcopy

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
  namespace = {'model': model, 
               'device': device, 
               'torch':torch, 
               'fuse_conv_and_bn': fuse_conv_and_bn, 
               'calculate_equivalent_convolutional_layer': calculate_equivalent_convolutional_layer,
               'nn': nn}

  exec(f'''
conv = model.{name+"conv1"}
if isinstance(model.{name+"bn1"}, nn.BatchNorm2d):
  bn = model.{name+"bn1"}
  fused_conv = fuse_conv_and_bn(conv, bn, device)
  model.{name + "conv1"} = fused_conv
  model.{name + "bn1"} = nn.Identity()
''', namespace)

  exec(f'''
conv = model.{name+"conv2"}
if isinstance(model.{name+"bn2"}, nn.BatchNorm2d):
  bn = model.{name+"bn2"}
  fused_conv = fuse_conv_and_bn(conv, bn, device)
  model.{name + "conv2"} = fused_conv
  model.{name + "bn2"} = nn.Identity()
''', namespace)


  exec(f'''
conv1 = model.{name+"conv1"}
conv2 = model.{name+"conv2"}
fused_conv = calculate_equivalent_convolutional_layer(conv1, conv2, device)
model.{name+"conv1"} = nn.Identity()
model.{name+"conv2"}= fused_conv
''',namespace)
  return


def fold_resnet(model, device):
  def numrepl(matchobj):
    return f'[{matchobj.group(0)[1]}]'
  fused_model = deepcopy(model)
  # something to analize the network
  relu_list = []
  for name, layer in fused_model.named_modules():
    if isinstance(layer, (nn.Identity)):
      pattern = r'\.[0-9]'  
      name = re.sub(pattern=pattern, repl=numrepl, string=name)
      relu_list.append(name)

  for i in range(len(relu_list)):
  # for relu in relu_list:
    if relu_list[i][-1] =='1':
      inside_block_fusion(fused_model, device, relu_list[i][:-5])

  return fused_model

def inside_MLP_fusion(model, device, name):
    namespace = {'model': model, 
                 'device': device, 
                 'torch':torch, 
                 'calculate_equivalent_fc_layer': calculate_equivalent_fc_layer,
                 'nn': nn}
    exec(f'''
fc1 = model.{name}[0]
fc2 = model.{name}[2]
''', namespace)
    exec(f'''
fc1 = model.{name}[0]
fc2 = model.{name}[2]
fused_fc = calculate_equivalent_fc_layer(fc1,fc2,device)
model.{name}[0] = fused_fc
model.{name}[2] = nn.Identity()
''', namespace)
    return

def fold_swint(model, device):
    def numrepl(matchobj):
        return f'[{matchobj.group(0)[1]}]'
    fused_model = deepcopy(model)
    relu_list = []
    for name, layer in fused_model.named_modules():
      if isinstance(layer, (nn.Identity)):
        pattern = r'\.[0-9]'  
        name = re.sub(pattern=pattern, repl=numrepl, string=name)
        relu_list.append(name)
    for relu in relu_list:
      inside_MLP_fusion(fused_model, device, relu[:-3])

    return fused_model

def fold(model, type, device):
  if type == 'resnet':
    return fold_resnet(model, device)
  elif type == 'swint':
    return fold_swint(model, device)
  else:
    raise NotImplementedError(f'{type} not supported')


if __name__ == '__main__':
  pass