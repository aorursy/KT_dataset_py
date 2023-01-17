import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import torch

from torch import nn,optim

import torch.nn.functional as F

from torch.utils.data import DataLoader

from torchvision import transforms

from torch.utils.data import SubsetRandomSampler

from tqdm import tqdm_notebook
dftrain=pd.read_csv("../input/digit-recognizer/train.csv")
class MnistDataset(torch.utils.data.Dataset):

    def __init__(self,data):

        self.data=data

        

    def __getitem__(self,idx):

        

        label=self.data.iloc[idx,0]

        img=self.data.iloc[idx,1:].values.astype(np.float32)

        return img,label

        

    def __len__(self):

        return len(self.data)
valid_split=0.1

is_cuda=torch.cuda.is_available()

batch_size=64



trainset=MnistDataset(dftrain)

validset=MnistDataset(dftrain)



index= list(range(len(dftrain)))

np.random.shuffle(index)



split=int(len(dftrain)*valid_split)

valid_index,train_index=index[:split],index[split:]



trainsampler=SubsetRandomSampler(train_index)

validsampler=SubsetRandomSampler(valid_index)



trainloader=DataLoader(trainset,sampler=trainsampler,batch_size=batch_size)

validloader=DataLoader(validset,sampler=validsampler,batch_size=batch_size)

def plot_images(x):

    plt.figure(figsize=(5,5))

    for i in range(9):

        plt.subplot(3,3,i+1)

        plt.imshow(x[i].reshape((28,28)))
class Net(nn.Module):   

    def __init__(self):

        super().__init__()

        self.conv1=nn.Sequential(

            nn.Conv2d(1,5,3,padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        self.conv2=nn.Sequential(

            nn.Conv2d(5,10,3,padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        self.fc=nn.Sequential(

            nn.Linear(7*7*10,50),

            nn.ReLU(),

            nn.Linear(50,10),

        )

        

    def forward(self,x):

        x=x.reshape((-1,1,28,28))

        

        x=self.conv1(x)

        x=self.conv2(x)

        

        x=x.view(x.shape[0],-1)

        x=self.fc(x)

        return x

    

class Net_flatten(nn.Module):

    def __init__(self):

        super().__ini__()

        self.l1=nn.Sequential(

            nn.Linear(784,500),

            nn.ReLU()

        )

        self.l2=nn.Sequential(

            nn.Linear(784,100),

            nn.ReLU()

        )

        self.l3=nn.Sequential(

            nn.Linear(784,10),

        )

        

    def forward(self,x):

        x=x.reshape(-1,784)

        x=self.l3(self.l2(self.l1(x)))

        return x
def train(model_type="cnn",optim_type="adam"):

    

    if(model_type=="cnn"):

        model=Net()

        

    if(model_type=="flat"):

        model=Net_flatten()

    

    if(is_cuda):

        model=model.cuda()

        

    criterion=nn.CrossEntropyLoss()

    if(optim_type=="adam"):

        optim=torch.optim.Adam(model.parameters(),lr=0.001)

        

    if(optim_type=="adagrad"):

        optim=torch.optim.Adagrad(model.parameters(),lr=0.001)



    if(optim_type=="batch_gradient" or optim_type=="mini_batch" or optim_type=="new_mini_batch"):

        optim=torch.optim.SGD(model.parameters(),lr=0.001)

    

    epoch_loss=[]

    model.train()

    for epoch in range(20):

        iter_loss=0

        for x,y in trainloader:

            if(is_cuda):

                x,y=x.cuda(),y.cuda()



            if(optim_type=="new_mini_batch"):

                x=x/x.shape[0]

                y=y/y.shape[0]

                

            out=model(x)

            loss=criterion(out,y)

            iter_loss+=loss.item()

            model.zero_grad()

            loss.backward()

            optim.step()



        print(f"loss : {iter_loss}")

        epoch_loss.append(iter_loss)

        

        

    model.eval()

    pred=[]

    for x,y in validloader:

        if(is_cuda):

            x,y=x.cuda(),y.cuda()

        out=model(x)

        pred.extend((torch.argmax(out,dim=1)==y).to(float).detach().cpu().tolist())

    pred=np.array(pred)

    accuracy=(sum(pred)/len(pred))*100

        

    return epoch_loss,accuracy
model_types=["cnn","flattent"]

optim_types=["adam","adagrad","batch_gradient","mini_batch","new_mini_batch"]



plt.figure(figsize=(10,10))

for model in model_types:

    i=0

    losses=[]

    for optim in optim_types:

        

        loss,accuracy=train(model,optim)

        losses.append(loss)

        print(f"\nModel : {model}, Optim : {optim} : ")

        print(f"Accuracy : {accuracy}")

        

        if(optim_type=="new_mini_batch"):

            plt.plot(list(range(len(loss))),loss)

    

    for i in range(len(losses)):

        plt.plot(list(range(len(losses[i]))),losses[i],label=optim_types[i])

    

    plt.legend(loc="upper right")

    plt.show()    