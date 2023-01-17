import numpy as np
import os
import torch
import random
import sys

from IPython.display import clear_output
!nvidia-smi
!git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user && cd .. && rm -rf apex
clear_output(wait=True)
print('Done!')
# install dependencies: (use cu101 because colab has CUDA 10.1)
!pip install -U torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install mmcv-full thus we could use CUDA operators
!pip install mmcv-full

clear_output(wait=True)
print('Done!')
# Install mmdetection
!rm -rf mmdetection
!git clone https://github.com/open-mmlab/mmdetection.git  # !git clone https://github.com/t0efL/mmdetection.git
%cd mmdetection

clear_output(wait=True)
print('Done!')
!pip install -e .

# install Pillow 7.0.0 back in order to avoid bug in colab
!pip install Pillow==7.0.0

clear_output(wait=True)
print('Done!')
!pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
clear_output(wait=True)
print('Done!')
# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())
def set_seed(seed=27):
    """Sets the random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
 
set_seed()
!python /content/mmdetection/tools/train.py /content/mmdetection/configs/faster_rcnn_r101_fpn_1x.py