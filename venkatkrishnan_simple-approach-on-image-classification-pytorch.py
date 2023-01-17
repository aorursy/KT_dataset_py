import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as fn

import torch.optim as optim # Optim module for loss functions

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# The output of torchvision datasets are PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
trainset = torchvision.datasets.CIFAR10(root='/kaggle/working/', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# Loading testset
testset = torchvision.datasets.CIFAR10(root='/kaggle/working/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# Class labels
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def show_img(img):
    img = img / 2 + 0.5
    img_np = img.numpy()
    
    plt.imshow(np.transpose(img_np, (1,2,0)))
    plt.show()
    

# get some random training images
detaiter = iter(trainloader)
images,labels = detaiter.next()
show_img(torchvision.utils.make_grid(images))

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Class CNN inherits nn.Module
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(fn.relu(self.conv1(x)))
        x = self.pool(fn.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = fn.relu(self.fc1(x))
        x = fn.relu(self.fc2(x))
        x = self.fc3(x)
        return x

cnn_model = CNN()
criterian = nn.CrossEntropyLoss()

optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        # get inputs and labels
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = cnn_model(inputs)
        loss = criterian(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i+1, running_loss/2000))
            
            running_loss = 0.0
            
print('Finished Training')
        
PATH = '/kaggle/working/cifar_cnn_net.pth'

torch.save(cnn_model.state_dict(), PATH)
