# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch

import torch.nn.functional as F

from torch import nn, optim

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms, models

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import warnings

warnings.filterwarnings('ignore')

dataset = pd.read_csv('../input/train.csv')

dataset.head()

#The row of csv file represent a picture
data = dataset.iloc[0].values



image = data[1:].reshape(28, 28).astype(np.uint8)

label = data[0]

plt.imshow(image, cmap='gray')

print(label)
class Mnist(torch.utils.data.Dataset):

    def __init__(self, data, transforms=None):

        self.data = data

        self.transforms = transforms

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        #Each time we get a row of data

        item = self.data.iloc[index]

        image = item[1:].values.astype(np.uint8).reshape((28, 28))

        label = item[0]

        

        if self.transforms is not None:

            image = self.transforms(image)

            

        return image, label

    
batch_size = 32

transforms = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5, ), std=(0.5,))

])
mydataset = Mnist(dataset, transforms=transforms)

len(mydataset)
num_train = len(mydataset)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(0.2 * num_train))

# print(split)

train_idx, test_idx = indices[split:], indices[:split]

# print(train_idx)

train_sampler = SubsetRandomSampler(train_idx)

test_sampler = SubsetRandomSampler(test_idx)



batch_size = 100

train_loader = DataLoader(mydataset, batch_size=batch_size, sampler=train_sampler)

test_loader = DataLoader(mydataset, batch_size=batch_size, sampler=test_sampler)
#可视化

fig, axis = plt.subplots(3, 10, figsize=(15, 10))

images, labels = next(iter(train_loader))

for i, ax in enumerate(axis.flatten()):

    with torch.no_grad():

        image, label = images[i], labels[i]

#         print(image.size())

#         break

        ax.imshow(image.view(28, 28), cmap='gray')

        ax.set(title = f"{label}")

        
import torch.nn as nn

class Lenet5(nn.Module):

    def __init__(self):

            super(Lenet5, self).__init__()

            self.conv = nn.Sequential(

                nn.Conv2d(1, 6, 5),   # out = (in - kernal + 2*padding) / stride + 1  (1, 28, 28) -> (6, 24, 24)

                nn.ReLU(),

                nn.AvgPool2d(2, stride=2), # (N, 6, 24, 24) - > (N, 6, 12, 12)

                nn.Conv2d(6, 16, 5), #  (N, 6, 12, 12) -> (N. 6, 8, 8)

                nn.ReLU(),

                nn.AvgPool2d(2, stride=2)# (N, 16, 8, 8) -> (N, 16, 4, 4)

            )

            self.fc = nn.Sequential(

                nn.Linear(256, 120), #(N, 256) - > (N, 120)

                nn.ReLU(),

                nn.Linear(120, 84), #(N, 120) - > (N, 84)

                nn.ReLU(),

                nn.Linear(84, 10)

            )

    def forward(self, x):

        x = self.conv(x)

        #flatten

        x = x.view(x.size(0), -1)

#         print(x.size())

        x = self.fc(x)

        return x

        
model = Lenet5()

# out = net(image)
device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')

learning_rate = 0.001

criterion = nn.CrossEntropyLoss()

model = model.to(device)

# optim = optim.Adam(net.parameters(), lr=learning_rate)

optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
def evaluation(dataloader):

    total, correct = 0, 0

    for data in dataloader:

        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)

        _, pred = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (pred == labels).sum().item()

    return 100 * correct / total
num_epochs = 50

best_acc = 0

checkpoint = {}

for epoch in range(num_epochs):

    

    model.train()

    train_loss = 0

    for i, data in enumerate(train_loader, 0):

#         data = data.to(device)

        images, labels = data

        images = images.to(device)

        labels = labels.to(device)

#         print(images.device)

        optim.zero_grad()

        out = model(images)

        loss = criterion(out, labels)

        loss.backward()

        optim.step()

        train_loss += loss.item()

        

    test_loss = test_acc = correct = 0

    model.eval()

    with torch.no_grad():

        for i, data in enumerate(test_loader):

            images, labels = data

            images, labels = images.to(device), labels.to(device)

            out = model(images)

            test_loss += criterion(out, labels)

            _, pred = torch.max(out, 1)

            correct += pred.eq(labels).sum().item()

    cur_acc = correct/len(test_loader)

    

    print(f"Epoch: {epoch+1}/{num_epochs}",

          f"Train loss:{train_loss/len(train_loader):.4f}",

          f"Test loss:{test_loss/len(test_loader):.4f}",

          f"Test acc:{correct/len(test_loader):.4f}")

    #Save model state

    if cur_acc > best_acc:

        best_acc = cur_acc

        torch.save(model, 'model.pt')

# 



print(f"Best acc:{best_acc:.4f}")

#torch.save(checkpoint, 'ckp.s')        
class DatasetSubmissionMNIST(torch.utils.data.Dataset):

    def __init__(self, file_path, transforms=None):

        self.data = pd.read_csv(file_path)

        self.transforms = transforms

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        image = self.data.iloc[index].values.astype(np.uint8).reshape((28, 28, 1))



        

        if self.transforms is not None:

            image = self.transforms(image)

            

        return image
model = torch.load('model.pt')

# model.load_state_dict(model)

submissonset = DatasetSubmissionMNIST('../input/test.csv', transforms=transforms)

sub_loader = DataLoader(submissonset, batch_size=batch_size, shuffle=False)
submission = [['ImageId', 'Label']]

model.to(device)

tt = None

with torch.no_grad():

    model.eval()

    image_id = 1

    for _, images in enumerate(sub_loader, 1):

        images = images.to(device)

        out = model(images)

        _, pred = torch.max(out, 1)

        for p in pred:

            submission.append([image_id, p.item()])

            image_id += 1

print(len(submission) - 1)
import csv

with open('submission.csv', 'w') as file:

    writer = csv.writer(file)

    writer.writerows(submission)
