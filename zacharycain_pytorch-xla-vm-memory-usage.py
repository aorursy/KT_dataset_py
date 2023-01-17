#!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
#!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
!pip install torch==1.6.0 torchvision==0.7.0 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.6-cp37-cp37m-linux_x86_64.whl
!pip install psutil
import os
import shutil
import sys
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.bn1 = nn.BatchNorm2d(10)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(20)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = self.bn1(x)
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = self.bn2(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)
def train_mnist():
  torch.manual_seed(1)

  train_loader = xu.SampleGenerator(
      data=(torch.zeros(128, 1, 28,
                        28), torch.zeros(128,
                                           dtype=torch.int64)),
      sample_count=60000 // 128 // xm.xrt_world_size())
  test_loader = xu.SampleGenerator(
      data=(torch.zeros(128, 1, 28,
                        28), torch.zeros(128,
                                           dtype=torch.int64)),
      sample_count=10000 // 128 // xm.xrt_world_size())

  # Scale learning rate to num cores
  lr = 0.01 * xm.xrt_world_size()

  device = xm.xla_device()
  model = MNIST().to(device)
  writer = None
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
  loss_fn = nn.NLLLoss()

  def train_loop_fn(loader):
    tracker = xm.RateTracker()

    model.train()
    for step, (data, target) in enumerate(loader):
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      xm.optimizer_step(optimizer)

  def test_loop_fn(loader):
    total_samples = 0
    correct = 0
    model.eval()
    for data, target in loader:
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum()
      total_samples += data.size()[0]

    accuracy = 100.0 * correct.item() / total_samples
    accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
    return accuracy

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  accuracy, max_accuracy = 0.0, 0.0
  for epoch in range(1, 2):
    xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    train_loop_fn(train_device_loader)
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))

    accuracy = test_loop_fn(test_device_loader)
    max_accuracy = max(accuracy, max_accuracy)

  xm.master_print('Done training/eval for all epochs.')
  return max_accuracy
def mp_fn(index, flags):
  torch.set_default_tensor_type('torch.FloatTensor')
  accuracy = train_mnist()
FLAGS={}
import gc
def print_mem():
    print(psutil.virtual_memory())

print('Memory stats before training:')
print_mem()
gc.collect()
print('Memory stats before training and after collect:')
print_mem()

xmp.spawn(mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
print('Memory stats after 1 round training:')
print_mem()
gc.collect()
print('Memory stats after 1 round training and after another collect:')
print_mem()

xmp.spawn(mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
print('Memory stats after 2 round training:')
print_mem()
gc.collect()
print('Memory stats after 2 round training and after another collect:')
print_mem()