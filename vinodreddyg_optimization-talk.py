from IPython.display import Image
Image("../input/optimization-talk-images/gradient_descent.png")
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
path = './data/'
!tar -zxvf ../input/cifar10-python/cifar-10-python.tar.gz
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='.', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='.', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv_layers = nn.Sequential(*[
            *self.conv_block(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=1),
        ])
        self.classifier = nn.Linear(2304, 10)
        
    def conv_block(self, in_channels, out_channels):
        return [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
def train_model(mod, optimizer, trn_loader, nepochs=1, verbose=True):
    crit = nn.CrossEntropyLoss()
    loss_arr = []
    for epoch in range(nepochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        for i, data in enumerate(trn_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = Variable(inputs, requires_grad=False).cuda(), Variable(labels, requires_grad=False).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = mod(inputs)
            loss = crit(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 99:    # print every 2000 mini-batches
                if verbose:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                loss_arr.append((epoch*len(trn_loader) + i+1, running_loss / 100))
                running_loss = 0.0
    print('Finished Training')
    return loss_arr
def ploterr(arr):
    xs = [x[0] for x in arr]
    ys = [x[1] for x in arr]
    plt.figure(figsize=(15,10))
    plt.subplot(1, 1, 1)
    plt.plot(xs, ys, )
net = CifarNet().cuda(0)
optimizer = opt.SGD(net.parameters(), lr=0.001)
sgd_res = train_model(net, optimizer, trainloader, verbose=False)
ploterr(sgd_res)
Image("../input/optimization-talk-images/momentum.png")
Image("../input/optimization-talk-images/ravines.gif")
Image("../input/optimization-talk-images/with_momentum.gif")
net = CifarNet().cuda(0)
optimizer = opt.SGD(net.parameters(), lr=0.001, momentum=0.9)
mom_res = train_model(net, optimizer, trainloader, verbose=False)
ploterr(mom_res)
net = CifarNet().cuda(0)
optimizer = opt.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)
nest_res = train_model(net, optimizer, trainloader, verbose=False)
ploterr(nest_res)
Image("../input/optimization-talk-images/Nestorov.jpeg")
net = CifarNet().cuda(0)
optimizer = opt.Adagrad(net.parameters(), lr=0.001)
adag_res = train_model(net, optimizer, trainloader, verbose=False)
ploterr(adag_res)
net = CifarNet().cuda(0)
optimizer = opt.RMSprop(net.parameters(), lr=0.001, alpha=0.9, momentum=0.9)
rms_res = train_model(net, optimizer, trainloader, verbose=False)
ploterr(rms_res)
net = CifarNet().cuda(0)
optimizer = opt.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999))
adam_res = train_model(net, optimizer, trainloader, verbose=False)
ploterr(adam_res)
Image("../input/optimization-talk-images/contours_evaluation_optimizers.gif")