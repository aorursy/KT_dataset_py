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
import pandas as pd

import numpy as np

import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms, models

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

%matplotlib inline
train_csv_path = '../input/train.csv'

test_csv_path = '../input/test.csv'



train_df = pd.read_csv(train_csv_path)

test_df = pd.read_csv(test_csv_path)



# have a glimpse of train dataframe structure

n_train = len(train_df)

n_pixels = len(train_df.columns) - 1

n_class = len(set(train_df['label']))

print('Number of training samples: {0}'.format(n_train))

print('Number of training pixels: {0}'.format(n_pixels))

print('Number of classes: {0}'.format(n_class))

print(train_df.head())



# have a glimpse of test dataframe structure

n_test = len(test_df)

n_pixels = len(test_df.columns)

print('Number of test samples: {0}'.format(n_test))

print('Number of test pixels: {0}'.format(n_pixels))

print(test_df.head())
class MNISTDataset(Dataset):

    """MNIST data set"""

    

    def __init__(self, dataframe, 

                 transform = transforms.Compose([transforms.ToPILImage(),

                                                 transforms.ToTensor(),

                                                 transforms.Normalize(mean=(0.5,), std=(0.5,))])

                ):

        df = dataframe

        # for MNIST dataset n_pixels should be 784

        self.n_pixels = 784

        

        if len(df.columns) == self.n_pixels:

            # test data

            self.X = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

            self.y = None

        else:

            # training data

            self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

            self.y = torch.from_numpy(df.iloc[:,0].values)

            

        self.transform = transform

    

    def __len__(self):

        return len(self.X)



    def __getitem__(self, idx):

        if self.y is not None:

            return self.transform(self.X[idx]), self.y[idx]

        else:

            return self.transform(self.X[idx])
RandAffine = transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2))
batch_size = 64



train_transforms = transforms.Compose(

    [transforms.ToPILImage(),

     RandAffine,

     transforms.ToTensor(),

     transforms.Normalize(mean=(0.5,), std=(0.5,))])



val_test_transforms = transforms.Compose(

    [transforms.ToPILImage(),

     transforms.ToTensor(),

     transforms.Normalize(mean=(0.5,), std=(0.5,))])



def get_dataset(dataframe, dataset=MNISTDataset,

                transform=transforms.Compose([transforms.ToPILImage(),

                                              transforms.ToTensor(),

                                              transforms.Normalize(mean=(0.5,), std=(0.5,))])):

    return dataset(dataframe, transform=transform)
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck



class MNISTResNet(ResNet):

    def __init__(self):

        super(MNISTResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10) # Based on ResNet50

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)



model = MNISTResNet()

print(model)
def train(train_loader, model, criterion, optimizer, epoch):

    model.train()



    for batch_idx, (data, target) in enumerate(train_loader):

        # if GPU available, move data and target to GPU

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()

                

        # compute output and loss

        output = model(data)

        loss = criterion(output, target)

               

        # backward and update model

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if (batch_idx + 1)% 100 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),

                100. * (batch_idx + 1) / len(train_loader), loss.data.item()))
def validate(val_loader, model, criterion):

    model.eval()

    loss = 0

    correct = 0

    

    for _, (data, target) in enumerate(val_loader):

        # if GPU available, move data and target to GPU

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()

        

        output = model(data)

        

        loss += criterion(output, target).data.item()



        pred = output.data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        

    loss /= len(val_loader.dataset)

        

    print('\nOn Val set Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(

        loss, correct, len(val_loader.dataset),

        100.0 * float(correct) / len(val_loader.dataset)))
# example config, use the comments to get higher accuracy

total_epoches = 50

step_size = 10

base_lr = 0.01 



optimizer = optim.Adam(model.parameters(), lr=base_lr)

criterion = nn.CrossEntropyLoss()

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)



# move data and target to GPU    

if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()
def split_dataframe(dataframe=None, fraction=0.9, rand_seed=1):

    df_1 = dataframe.sample(frac=fraction, random_state=rand_seed)

    df_2 = dataframe.drop(df_1.index)

    return df_1, df_2



for epoch in range(total_epoches):

    print("\nTrain Epoch {}: lr = {}".format(epoch, exp_lr_scheduler.get_lr()[0]))



    train_df_new, val_df = split_dataframe(dataframe=train_df, fraction=0.9, rand_seed=epoch)

    

    train_dataset = get_dataset(train_df_new, transform=train_transforms)

    val_dataset = get_dataset(val_df, transform=val_test_transforms)



    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                               batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,

                                             batch_size=batch_size, shuffle=False)



    train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch)

    validate(val_loader=val_loader, model=model, criterion=criterion)

    exp_lr_scheduler.step()