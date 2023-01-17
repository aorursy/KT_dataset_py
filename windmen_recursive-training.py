%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import numpy as np

import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models

from torch.optim.lr_scheduler import StepLR

import psutil

import os
data_dir = '../input/age-predict/train'

def load_split_train_test(datadir, valid_size = .1):

    train_transforms = transforms.Compose([transforms.Resize((224,224)),

                                           transforms.RandomCrop([196, 196]),

                                           transforms.RandomHorizontalFlip(),

                                           transforms.ToTensor(),

                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),

                                          ])

    test_transforms = transforms.Compose([transforms.Resize((224,224)),

                                          transforms.ToTensor(),

                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),

                                         ])

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)

    test_data = datasets.ImageFolder(datadir, transform=test_transforms)    

    num_train = len(train_data)

    indices = list(range(num_train))

    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    from torch.utils.data.sampler import SubsetRandomSampler

    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)

    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data,

                   sampler=train_sampler, batch_size=64)

    testloader = torch.utils.data.DataLoader(test_data,

                   sampler=test_sampler, batch_size=64)

    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_dir, .2)

print(trainloader.dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() 

                                  else "cpu")

model = torch.load('../input/recursive-training/aerialmodel.pth')

#model = models.resnet18(pretrained=True)

print(device)
for param in model.parameters():

    param.requires_grad = True

    

#model.fc = nn.Sequential(nn.Linear(512, 128),

#                         nn.ReLU(),

#                         nn.Dropout(0.2),

#                         nn.Linear(128, 1))

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0000005)

scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

model.to(device);
def forward_model_train(inputs, labels):

    inputs, labels = inputs.to(device), labels.to(device).type(torch.float32)

    #sceduler

    optimizer.zero_grad()

    #scheduler.step()

    logps = torch.flatten(model.forward(inputs))

    loss = criterion(logps, labels)

    loss.backward()

    optimizer.step()

    return loss.item()

def forward_model_val(inputs, labels):

    inputs, labels = inputs.to(device), labels.to(device).type(torch.float32)

    logps = torch.flatten(model.forward(inputs))

    batch_loss = criterion(logps, labels)

    return batch_loss.item()
epochs = 1

steps = 0

running_loss = 0

print_every = 50

train_losses, test_losses = [], []

for epoch in range(epochs):

    print(scheduler.get_lr())

    scheduler.step()

    print(scheduler.get_lr())

    for inputs, labels in trainloader:

        steps += 1

        running_loss += forward_model_train(inputs, labels)

        #torch.cuda.ipc_collect()

        if steps % print_every == 0:

            test_loss = 0

            accuracy = 0

            model.eval()

            with torch.no_grad():

                for inputs, labels in testloader:

                    test_loss += forward_model_val(inputs, labels)

                    #torch.cuda.ipc_collect()

                    

            print(f"Epoch {epoch + 1}/{epochs}.. "

                  f"Train loss: {running_loss / print_every:.3f}.. "

                  f"Test loss: {test_loss / len(testloader):.3f}.. ")

            running_loss = 0

            model.train()

torch.save(model, 'aerialmodel.pth')