# LayerFold

Layer collapse of DNNs for accelerated inference

## Installation

Layerfold last version can be installed downloading the last version from github, running the `setup.py` to create the installation file, and installing it using pip:

```bash
git clone https://github.com/giommariapilo/layerfold
cd layerfold
python setup.py sdist
pip instal ./dist/layerfold-1.0.tar.gz
```

## Usage

### Main function

The main function, `fold` can be used to fold layers in Resnet and Swin-T architectures where some of the non-linearities have been replaced by Identities. 

#### Example
```python
import torch
from torchvision import models
from layerfold import fold

model = models.resnet18()

# Apply some layer collapse strategy or load a collapsed model

architecture = 'resnet' # or swint in case of a Swin Transformer
folded_model = fold(model, architecture, device='cuda')
```

### Subfunctions
If the operation needs to be applied to another architecture, the single functions `fusefc`, `fuse_conv`, and `fuse_convbn` can be used to fuse layers two at a time.

#### Example
```python
import torch
from torchvision import models
from layerfold import *

model = models.vgg16()

# Apply some layer collapse strategy or load a collapsed model

# Since VGG!6 is not supported we can fold it manually
conv1 = model.features[10]
conv2 = model.features[12]
eq_conv = fuse_conv(conv1, conv2, device='cuda')
# now put the fused layer in the model
model.features[12] = torch.nn.Identity()
model.features[10] = eq_conv

```