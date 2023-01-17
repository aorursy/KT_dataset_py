import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import torch

import glob

import cv2

from torch import nn,optim

import torch.nn.functional as F

from torch.utils.data import DataLoader

from torchvision import transforms

from torch.utils.data import SubsetRandomSampler

from tqdm import tqdm_notebook
all_faces=glob.glob("../input/fei-face-data/originalimages_part1/*.jpg")

all_faces.extend(glob.glob("../input/fei-face-data/originalimages_part2/*.jpg"))

all_faces.extend(glob.glob("../input/fei-face-data/originalimages_part3/*.jpg"))

all_faces.extend(glob.glob("../input/fei-face-data/originalimages_part4/*.jpg"))

data_size=len(all_faces)
class FaceDataset(torch.utils.data.Dataset):

    def __init__(self,data):

        self.data=data

        

    def __getitem__(self,idx):

        

        image=self.data[idx]

        

        #image=directory/subjectno_imageno.jpg

        label=int(image.split("/")[-1].split(".")[0].split("-")[0]) #taking the subject number

        #one_hot=np.zeros(shape=(200))

        #one_hot[label-1]=1 #label

        

        # extracting image

        img=cv2.imread(self.data[idx])# 480*640*3

        img=torch.tensor(img,dtype=torch.float32)

        img=img.permute(2,0,1)

        

        return img,label-1 # class should be between (0-c-1d)

        

    def __len__(self):

        return len(self.data)
valid_split=0.1

batch_size=64

is_cuda=torch.cuda.is_available()



dataset=FaceDataset(all_faces)



index= list(range(data_size))

np.random.shuffle(index)



split=int(data_size*valid_split)

valid_index,train_index=index[:split],index[split:]



trainsampler=SubsetRandomSampler(train_index)

validsampler=SubsetRandomSampler(valid_index)



trainloader=DataLoader(dataset,sampler=trainsampler,batch_size=batch_size)

validloader=DataLoader(dataset,sampler=validsampler,batch_size=batch_size)
class Net(nn.Module):

    

    def __init__(self):

        super().__init__()

        self.conv1=nn.Sequential(

            nn.Conv2d(3,5,3,padding=1),

            nn.ReLU(),

            nn.MaxPool2d(4)

        )

        self.conv2=nn.Sequential(

            nn.Conv2d(5,10,3,padding=1),

            nn.ReLU(),

            nn.MaxPool2d(4)

        )

        self.fc=nn.Sequential(

            nn.Linear(30*40*10,50),

            nn.ReLU(),

            nn.Linear(50,200)

        )

        

    def forward(self,x):

        

        x=self.conv1(x)

        x=self.conv2(x)

        x=x.view(x.shape[0],-1)

        x=self.fc(x)

        return x
net=Net()

if(is_cuda):

    net=net.cuda()

    

criterion=nn.CrossEntropyLoss()

optimiser=optim.SGD(net.parameters(),lr=0.001)
train_loss=[]

valid_loss=[]

for epoch in tqdm_notebook(range(1)):



    net.train()    

    print(f"Epoch : {epoch} :-")

    for x,y in trainloader:

        if(is_cuda):

            x,y=x.cuda(),y.cuda()

        out=net(x)

        optimiser.zero_grad()

        loss=criterion(out,y)

        loss.backward()

        optimiser.step()

    train_loss.append(loss)

    print(f"\tTrain Loss : {loss}")



    net.eval()

    for x,y in validloader:

        if(is_cuda):

            x,y=x.cuda(),y.cuda()

        out=net(x)

        loss=criterion(out,y)

    valid_loss.append(loss)

    print(f"\tValid Loss : {loss}")