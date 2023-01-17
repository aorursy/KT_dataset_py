import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
# import visdom
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import os
from graphviz import Digraph
import tarfile
os.listdir('../input/')
t = tarfile.open('../input/cifar-10-python.tar.gz')
t.extractall(path = 'input') 
os.listdir('input')


#viz = visdom.Visdom()
class mycnn(nn.Module):
    def __init__(self):
        super(mycnn, self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),  # (32,32)
            nn.ReLU(True),
            nn.MaxPool2d(2),  # (32,32) >> (16,16)
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True))  # (16,16) >> (8,8)）

        self.con2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            #nn.MaxPool2d(2, 2),  # 32*8*8
            nn.ReLU())
        self.con3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),  # 6*16*16---16*16*16
            nn.Conv2d(256, 256, kernel_size=3),
            #nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            #nn.MaxPool2d(2, 2),  # 32*8*8
            nn.ReLU())
        self.con4 = nn.Sequential(
            nn.Conv2d(256,512, kernel_size=3),  # 6*16*16---16*16*16
            nn.Conv2d(512, 512, kernel_size=3),  # 6*16*16---16*16*16
            #nn.Conv2d(512, 512, kernel_size=3),  # 6*16*16---16*16*16
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),  # 32*8*8

            nn.ReLU())
        self.fcon = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        x = self.con3(x)
        x = self.con4(x)
        x = x.view(x.size(0), -1)
        x = self.fcon(x)
        return x


transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
trainset = torchvision.datasets.CIFAR10(root='../working/input', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
    # 测试集
testset = torchvision.datasets.CIFAR10(root='../working/input', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                 shape='plaintext',
                 align='left',
                 fontsize='12',
                 ranksep='0.1',
                 height='0.2')

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"),format='png')
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'
    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot


cnn = mycnn()
if torch.cuda.is_available():
    device = 'cuda'
    cnn = torch.nn.DataParallel(cnn)
for inputs, targets in trainloader:
    inputs, targets = inputs.to(device), targets.to(device)
    break
y = cnn(inputs)
g = make_dot(y)
g 
try:
    g.view()
except:
    print('error')
