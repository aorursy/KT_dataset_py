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
def load_dataset(path):

    df=pd.read_csv(path)

    print("Dataset Contains : {} rows and {} columns".format(df.shape[0],df.shape[1]))

    return df
print("Loading train dataset :")

dftrain=load_dataset('../input/train.csv')

print("Loading test dataset :")

dftest=load_dataset('../input/test.csv')
print("Label datatype : {} \nImage datatype : {}".format(dftrain.label.dtype,dftrain.pixel0.dtype))

print("Test Image datatype : {}".format(dftest.pixel0.dtype))
maxp=dftrain.iloc[:,1:].max().max()

minp=dftrain.iloc[:,1:].min().min()

print("Pixel range : {}-{}".format(minp,maxp))
class MnistDataset(torch.utils.data.Dataset):

    def __init__(self,data,transform=None):

        self.data=data

        self.transform=transform

        

    def __getitem__(self,idx):

        

        label=self.data.iloc[idx,0]

        img=self.data.iloc[idx,1:].values.astype(np.uint8).reshape(28,28)

        if(self.transform):

            img=self.transform(img)

        return img,label

        

    def __len__(self):

        return len(self.data)
batch_size=16

valid_split=0.1

is_cuda=torch.cuda.is_available()



# preprocessing on image

# as of now I'm only mean normalizing the image

# todo - convert the image in range (0,1)



transform_train = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



transform_valid = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])





trainset=MnistDataset(dftrain,transform_train)

validset=MnistDataset(dftrain,transform_valid)



index= list(range(len(dftrain)))

np.random.shuffle(index)



split=int(len(dftrain)*valid_split)

valid_index,train_index=index[:split],index[split:]



trainsampler=SubsetRandomSampler(train_index)

validsampler=SubsetRandomSampler(valid_index)



trainloader=DataLoader(trainset,sampler=trainsampler,batch_size=batch_size)

validloader=DataLoader(validset,sampler=validsampler,batch_size=batch_size)



print(f"Train Length {len(train_index)}")

print(f"Valid Lenght {len(valid_index)}")

print(f"Total {len(train_index)+len(valid_index)}")
for x,y in trainloader:

    plt.figure(figsize=(5,5))

    for i in range(3):

        plt.subplot(1,3,i+1)

        plt.imshow(x[i].view(28,28))

        plt.title(str(y[i]))

    break
for x,y in validloader:

    plt.figure(figsize=(5,5))

    for i in range(3):

        plt.subplot(1,3,i+1)

        plt.imshow(x[i].view(28,28))

        plt.title(str(y[i]))

    break
class TestDataset(torch.utils.data.Dataset):

    def __init__(self,data,transform=None):

        self.data=data

        self.transform=transform

        

    def __getitem__(self,idx):

        

        img=self.data.iloc[idx,:].values.astype(np.uint8).reshape(28,28)

        if(self.transform):

            img=self.transform(img)

        return img

        

    def __len__(self):

        return len(self.data)
transform_test = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



testset=TestDataset(dftest,transform_test)

testloader=DataLoader(testset,batch_size=1)
i=0

plt.figure(figsize=(5,5))

for x in testloader:

    plt.subplot(1,3,i+1)

    plt.imshow(x[0].view(28,28))

    i+=1

    if(i==3):

        break
def train(net,criterion,optimiser,num_epoch=10):

    train_loss=[]

    valid_loss=[]

    for epoch in tqdm_notebook(range(num_epoch)):



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



        torch.cuda.empty_cache()

        

    return net,train_loss,valid_loss
def plot_losses(train_loss,valid_loss):

    plt.figure(figsize=(15,5))

    plt.subplot(1,2,1)

    plt.plot(range(len(train_loss)),train_loss)

    plt.title("Training Loss")

    plt.subplot(1,2,2)

    plt.plot(range(len(valid_loss)),valid_loss)

    plt.title("Validation Loss")
def predict_output(net):

    pred=[]

    net.eval()

    for x in tqdm_notebook(testloader):

        if(is_cuda):

            x=x.cuda()

        p=net(x)

        pred.append(torch.argmax(p).data.cpu().numpy())

    return pred
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

            nn.Linear(50,10)

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
net,train_loss,valid_loss=train(net,criterion,optimiser,num_epoch=50)
plot_losses(train_loss,valid_loss)
pred=predict_output(net)

dfpred=pd.DataFrame({"ImageId":list(range(1,len(dftest)+1)),"Label":pred})

dfpred.to_csv("pred3.csv",index=False)
class Net_bn(nn.Module):

    

    def __init__(self):

        super().__init__()

        self.conv1=nn.Sequential(

            nn.Conv2d(1,5,3,padding=1),

            nn.BatchNorm2d(5),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        self.conv2=nn.Sequential(

            nn.Conv2d(5,10,3,padding=1),

            nn.BatchNorm2d(10),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        self.fc=nn.Sequential(

            nn.Linear(7*7*10,50),

            nn.ReLU(),

            nn.Linear(50,10)

        )

        

    def forward(self,x):

        

        x=self.conv1(x)

        x=self.conv2(x)

        x=x.view(x.shape[0],-1)

        x=self.fc(x)

        return x
net_bn=Net_bn()

if(is_cuda):

    net_bn=net_bn.cuda()

    

criterion=nn.CrossEntropyLoss()

optimiser=optim.SGD(net_bn.parameters(),lr=0.001)
net_bn,train_loss,valid_loss=train(net_bn,criterion,optimiser,num_epoch=50)
plot_losses(train_loss,valid_loss)
pred=predict_output(net_bn)

dfpred=pd.DataFrame({"ImageId":list(range(1,len(dftest)+1)),"Label":pred})

dfpred.to_csv("pred4.csv",index=False)
class Net_deep(nn.Module):

    

    def __init__(self):

        super().__init__()

        self.conv1=nn.Sequential(

            nn.Conv2d(1,5,3,padding=1),

            nn.ReLU(),

            nn.Conv2d(5,5,3,padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        self.conv2=nn.Sequential(

            nn.Conv2d(5,10,3,padding=1),

            nn.ReLU(),

            nn.Conv2d(10,10,3,padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        self.fc=nn.Sequential(

            nn.Linear(7*7*10,100),

            nn.ReLU(),

            nn.Linear(100,10)

        )

        

    def forward(self,x):

        

        x=self.conv1(x)

        x=self.conv2(x)

        x=x.view(x.shape[0],-1)

        x=self.fc(x)

        return x
net_deep=Net_deep()

if(is_cuda):

    net_deep=net_deep.cuda()

    

criterion=nn.CrossEntropyLoss()

optimiser=optim.SGD(net_deep.parameters(),lr=0.001)
net_deep,train_loss,valid_loss=train(net_deep,criterion,optimiser,num_epoch=50)
plot_losses(train_loss,valid_loss)
pred=predict_output(net_deep)

dfpred=pd.DataFrame({"ImageId":list(range(1,len(dftest)+1)),"Label":pred})

dfpred.to_csv("pred5.csv",index=False)