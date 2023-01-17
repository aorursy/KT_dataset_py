import pandas as pd

import numpy as np

import torch

import torch.utils.data

from torch.utils.data import DataLoader

import torch.nn as nn

import torch.nn.functional as F

import matplotlib.pyplot as plt
train=pd.read_csv("../input/digit-recognizer/train.csv").values

test=pd.read_csv("../input/digit-recognizer/test.csv").values
class dataset(torch.utils.data.Dataset):

    

    def __init__(self,data):

        self.data=data

        

    def __getitem__(self,idx):

        image=self.data[idx,1:].astype(np.float32)

        label=self.data[idx,0:1].astype(np.float32)

        

        return image,label

    

    def __len__(self):

        return len(self.data)
trainset=dataset(train)

trainloader=DataLoader(trainset,batch_size=64,shuffle=True)
inp=784

out=784

hid=100



datasize=train.shape[0]

batchsize=64

is_cuda=torch.cuda.is_available()
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        

        self.encoder=nn.Linear(inp,hid)

        self.decoder=nn.Linear(hid,out)

        

    def forward(self,x):

        latent=self.encoder(x)

        output=self.decoder(latent)

        

        return latent,output
net=Net()

if(is_cuda):

    net=net.cuda()

optim=torch.optim.Adam(net.parameters(),lr=0.001)



def loss_function(out,real):

    mse=F.mse_loss(out,real,reduction="mean")  

    return mse
total_loss=[]

for epoch in range(10):

    loss_count=0

    for x,y in trainloader:

        if(is_cuda):

            x,y=x.cuda(),y.cuda()



        _,output=net(x)

        loss=loss_function(output,x)

        loss_count+=loss.item()



        net.zero_grad()

        loss.backward()

        optim.step()

        

    print(f"epoch : {epoch} - loss : {loss_count}")

    total_loss.append(loss_count)
plt.plot(list(range(len(total_loss))),total_loss)

plt.show()
x=torch.ones(size=(2,4))
net=Net()

if(is_cuda):

    net=net.cuda()

optim=torch.optim.Adam(net.parameters(),lr=0.001)



def loss_function(out,real,latent):

    mse=F.mse_loss(out,real,reduction="mean")  

    latent=latent.mean(axis=0)

    kld=F.kl_div(latent,torch.tensor(0.3).cuda(),reduction="batchmean")

    return mse
total_loss=[]

for epoch in range(10):

    loss_count=0

    for x,y in trainloader:

        if(is_cuda):

            x,y=x.cuda(),y.cuda()



        latent,output=net(x)

        loss=loss_function(output,x,latent)

        loss_count+=loss.item()



        net.zero_grad()

        loss.backward()

        optim.step()

        

    print(f"epoch : {epoch} - loss : {loss_count}")

    total_loss.append(loss_count)
plt.plot(list(range(len(total_loss))),total_loss)

plt.show()
net=Net()

if(is_cuda):

    net=net.cuda()

optim=torch.optim.Adam(net.parameters(),lr=0.001)



def loss_function(out,real,latent):

    mse=F.mse_loss(out,real,reduction="mean")  

    w=net.state_dict()["encoder.weight"]

    latent=latent*(1-latent)

    contractive=torch.mean(0.0004*torch.sum(latent**2 * torch.sum(w**2,axis=1),axis=1))

    return mse+contractive
total_loss=[]

for epoch in range(10):

    loss_count=0

    for x,y in trainloader:

        if(is_cuda):

            x,y=x.cuda(),y.cuda()



        latent,output=net(x)

        loss=loss_function(output,x,latent)

        loss_count+=loss.item()



        net.zero_grad()

        loss.backward()

        optim.step()

        

    print(f"epoch : {epoch} - loss : {loss_count}")

    total_loss.append(loss_count)
plt.plot(list(range(len(total_loss))),total_loss)

plt.show()