import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

import torch

import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn

import torch.nn.functional as F

import gc

import memory_profiler
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



#The following line of code deals with loading the data from two csv files and preparing it in a format which can be trained using MLP



# load csv data into dataframes

train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# create train dataset class which is a subclass of torch.utils.data.Dataset

class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.df = df

        self.data = df.values      #convert dataframe to numpy array

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        label = self.data[idx,0]

        image = self.data[idx,1:]

        

        #normalize image 

        image = image / 255

    

        #sample = {'label':label, 'image':image} #return a dict containing label and corresponding image

        

        return label, image



# create test dataset class which is a subclass of torch.utils.data.Dataset

class TestDataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.df = df

        self.data = df.values      #convert dataframe to numpy array

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        image = self.data[idx]

        

        #normalize image using output = (input - mean) / std

        image = image / 255

        

        #sample = {'image':image} #return a dict containing image present at index idx

        

        return image
# helper function to show image from numpy array

def img_show(image):

    image = np.reshape(image, (28,-1))  # as MNIST dataset has image size of 28 * 28



    plt.imshow(image)

    plt.show()



# basic sanity check function to verify correctness of our dataset and dataloader 

def sanity_check(train=True):

    if train == True:

        dataiter = iter(train_loader)

    else:

        dataiter = iter(test_loader)

        

    for i in range(5):

        if train == True:

            labels, images = next(dataiter)

            print(labels)

        else:

            images = next(dataiter)

            

        print(images.shape)

    

        img_show(images[0])
valid_split = 0.2



# create train and test dataset class from dataframe

train_dataset = TrainDataset(train_df)

test_dataset = TestDataset(test_df)



# split indices of train dateset to train and validation indices

num_train = len(train_dataset)

idx_arr = list(range(num_train))

np.random.shuffle(idx_arr)

split = int(np.floor(valid_split*num_train))

valid_idx, train_idx = idx_arr[:split], idx_arr[split:]



train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



# create dataloader classes to load dataset in batch_size of 64

train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=64, num_workers=4, pin_memory=True)

valid_loader = torch.utils.data.DataLoader(train_dataset, sampler=valid_sampler, batch_size=64, num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=4, pin_memory=True)



sanity_check()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.linear1 = nn.Linear(784, 256)

        self.linear2 = nn.Linear(256, 64)

        self.linear3 = nn.Linear(64, 10)

        self.logsoft = nn.LogSoftmax(dim=1)

        self.drop = nn.Dropout(p=0.2)



    def forward(self, x):

        x = F.relu(self.linear1(x))

        x = self.drop(x)

        

        x = F.relu(self.linear2(x))

        x = self.drop(x)

        

        x = self.logsoft(self.linear3(x))

        

        return x



network = Model() # create network object

network.to(device)
criterion = nn.NLLLoss() # define loss. Using negative log likelyhood loss here.

optimizer = torch.optim.Adam(network.parameters(), lr=0.0001) # define optimization algorithm. Here we'll use Adam



network.double() # as our input image is of type tensor.double



n_epoch = 75 # number of epochs



train_losses = [] # set of training loss for each epoch

valid_losses = [] # set of validation loss for each epoch



valid_loss_min = np.inf



for i in range(n_epoch):

    train_loss = 0

    valid_loss = 0 



    network.train() # training loop

    for labels, images in train_loader:

        images, labels = images.to(device), labels.to(device)



        optimizer.zero_grad()



        logits = network(images)



        loss = criterion(logits, labels)



        train_loss += loss



        loss.backward()



        optimizer.step()

    

    #print(loss.device)

    train_loss = train_loss/ len(train_loader.dataset)

    train_losses.append(train_loss)



    with torch.no_grad():

        network.eval() # validation loop

        for labels, images in valid_loader:

            images, labels = images.to(device), labels.to(device)



            logits = network(images)



            loss = criterion(logits, labels)



            valid_loss += loss



        valid_loss = valid_loss/ len(valid_loader.dataset)

        valid_losses.append(valid_loss)



        # save our model when validation loss decreases

    if valid_loss < valid_loss_min:

        print("valid loss min decreased epoch: ",i, "--> ",valid_loss,", saving model state")

        valid_loss_min = valid_loss

        torch.save(network.state_dict(), 'net.pt')

    

    gc.collect()
# plot valid and train loss

plt.plot(train_losses, label='train')

plt.plot(valid_losses, label='valid')

plt.ylabel('losses')

plt.xlabel('epochs')

plt.legend()

plt.show()

#gc.get_objects()

print(gc.garbage)
network.load_state_dict(torch.load('net.pt'))

network.cpu()

network.eval()



solutions = []

with torch.no_grad():

    for images in test_loader:

        logits = network(images)

        labels = np.exp(logits)

    

        labels = labels.max(dim=1).indices

        

        solutions.extend(labels.numpy())



final = pd.DataFrame()

final['ImageId'] = [i+1 for i in range(len(test_dataset))]

final['Label'] = solutions

final.to_csv('submission.csv', index=False)