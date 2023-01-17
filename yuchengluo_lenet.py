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

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
class LeNet(nn.Module):

    def __init__(self):

        super(LeNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.softmax = nn.Softmax()

        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)

        self.conv2 = nn.Conv2d(6, 16, 5, 1, 2)

        self.fc1 = nn.Linear(7*7*16 ,120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)

        self.pooling = nn.MaxPool2d(2, 2)

        

    def forward(self, x):

        x = self.pooling(self.relu(self.conv1(x)))

        x = self.pooling(self.relu(self.conv2(x)))

        x = x.view(-1, 7*7*16)

        x = self.relu(self.fc1(x))

        x = self.relu(self.fc2(x))

        out = self.fc3(x)

        return out
lenet = LeNet()

lenet = lenet.to("cuda")

print(lenet)
class mnist_dataset(Dataset):

    def __init__(self, dfdata, is_train=True):

        if is_train:

            self.dfdata = dfdata.iloc[:40000]

        else:

            self.dfdata = dfdata.iloc[40000:]

        self.is_train = is_train

    def __len__(self):

        return len(self.dfdata)



    def __getitem__(self, idx):

        data = self.dfdata.iloc[idx:idx+1]

        data = data.values

        label = data[0, 0]

        img = data[0, 1:].reshape(1, 28, 28)



        sample = {

            'label': label,

            'img': img,

        }

        return sample



class Test_Dataset(Dataset):

    def __init__(self, dfdata):

        self.dfdata = dfdata

        

    def __len__(self):

        return len(self.dfdata)



    def __getitem__(self, idx):

        data = self.dfdata.iloc[idx:idx+1]

        data = data.values

        img = data[0, :].reshape(1, 28, 28)



        sample = {

            'img': img,

        }

        return sample
train_dataset = mnist_dataset(train_data)

print(len(train_dataset))

dataload = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

val_dataset = mnist_dataset(train_data, is_train=False)

print(len(val_dataset))

val_dataload = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

test_dataset = Test_Dataset(test_data)

print(len(test_dataset))

test_dataload = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(lenet.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)



for e in range(30):

    total = 0

    correct = 0

    for i, batch_data in enumerate(dataload):

        img = batch_data['img']

        label = batch_data['label']

        img = img.type(torch.FloatTensor)

        label = label.type(torch.LongTensor)

        img = img.to("cuda")

        label = label.to("cuda")

        

        optimizer.zero_grad()

        predict = lenet(img)

        loss = criterion(predict, label)

        loss.backward()

        optimizer.step() 



        _, predicted = torch.max(predict.data, 1)

        total += label.size(0)

        correct += (predicted == label).sum().item()





    print('Epoch %d Accuracy of the network on the train images: %.2f %%' % (e,

        100 * correct / total))



    with torch.no_grad():

        total = 0

        correct = 0

        for i, batch_data in enumerate(val_dataload):

            img = batch_data['img']

            label = batch_data['label']

            img = img.type(torch.FloatTensor)

            label = label.type(torch.LongTensor)

            img = img.to("cuda")

            label = label.to("cuda")

            predict = lenet(img)

            _, predicted = torch.max(predict.data, 1)

            total += label.size(0)

            correct += (predicted == label).sum().item()

        print('Epoch %d Accuracy of the network on the 2000 val images: %.2f %%' % (e,

            100 * correct / total))



    

with torch.no_grad():

    total = 0

    correct = 0

    for i, batch_data in enumerate(test_dataload):

        img = batch_data['img']

        img = img.type(torch.FloatTensor)



        img = img.to("cuda")

        predict = lenet(img)

        _, predicted = torch.max(predict.data, 1)

        

        predicted = predicted.cpu().numpy()

        sub['Label'][i*100:(i+1)*100] = predicted
sub.to_csv("CNN.csv",index=False)