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

import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt

torch.__version__
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

x_train = train.iloc[:,1:].values.reshape(-1, 1, 28,28)/255

y_train = train.iloc[:,0].values

x_test = test.values.reshape(-1, 1, 28,28)/255

print(x_train.shape, y_train.shape)

plt.imshow(x_train[6][0])
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1=nn.Conv2d(1,512,5)

        self.conv2=nn.Conv2d(512,256,5)

        self.conv3 = nn.Conv2d(256,128,5)

        self.conv4 = nn.Conv2d(128,64,5)

        self.fc1 = nn.Linear(64*2*2,512)

        self.fc2 = nn.Linear(512,256)

        self.fc3 = nn.Linear(256,128)

        self.fc4 = nn.Linear(128,10)

    def forward(self,x):

        out = F.relu(self.conv1(x))#512x24x24

        out = F.dropout(out,p=0.6, training=self.training)

        out = F.relu(self.conv2(out))#256x20x20

        out = F.dropout(out,p=0.5, training=self.training)

        out = F.max_pool2d(F.relu(self.conv3(out)),2)#128x8x8

        out = F.dropout(out,p=0.4, training=self.training)

        out = F.max_pool2d(F.relu(self.conv4(out)),2)#64x2x2

        out = out.view(out.size(0),-1)

        out = F.relu(self.fc1(out))

        out = F.relu(self.fc2(out))

        out = F.relu(self.fc3(out))

        out = self.fc4(out)

        out = F.log_softmax(out,dim=1)

        return out

net = Net()

print(net)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(DEVICE)

optimizer = optim.Adam(model.parameters(),lr=0.0013)
A=100

B=int(42000/A)

C=10

model.train()

while C>0:

    for i in range(B):

        x_t=torch.from_numpy(x_train[i*A:(i+1)*A]).to(DEVICE).float()

        y_t=torch.from_numpy(y_train[i*A:(i+1)*A]).to(DEVICE)

        optimizer.zero_grad()

        y_hat = model(x_t)

        loss = F.nll_loss(y_hat,y_t)

        loss.backward()

        optimizer.step()

    C-=1

    print('C='+str(C)+':',loss.item())
model.eval()

test_pred = torch.LongTensor()

with torch.no_grad():

    for i in range(28):

        y_test_hat=model(torch.from_numpy(x_test[i*1000:(i+1)*1000]).to(DEVICE).float())

        pred = y_test_hat.max(1, keepdim=True)[1]

        test_pred = torch.cat((test_pred, pred.cpu()), dim=0)



print(test_pred.shape)
def look(u):

    print(test_pred[u])

    plt.imshow(x_test[u][0])
look(20000)
r=pd.DataFrame({'ImageId':range(1,28001),'Label':test_pred.numpy().reshape(-1)})

r.to_csv("submission7.csv", index=False)

r.head()