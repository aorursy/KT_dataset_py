# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import warnings

warnings.simplefilter("ignore", UserWarning)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import matplotlib.pyplot as plt

import PIL

from matplotlib import image

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from tqdm import tqdm

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = image.imread('/kaggle/input/sign-language-mnist/american_sign_language.PNG')

plt.imshow(data)

plt.show()
data_train =pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')

data_train.shape
def showImage(row):

    d = data_train.drop('label',axis = 1).loc[row,:]

    print(data_train['label'][row])

    plt.imshow(d.values.reshape(28,28))

    plt.show()
showImage(15)
class Neural_Network(nn.Module):

    def __init__(self):

        super(Neural_Network, self).__init__()

        self.conv1 = nn.Conv2d(1,6,kernel_size = 3,stride = 1,padding = 1) #input: (m,28,28,1) output: (m,28,28,6)

        self.max1 = nn.MaxPool2d(kernel_size = (2,2),stride = 2) #input: (m,28,28,6) output: (m,14,14,6)

        self.conv2 = nn.Conv2d(6,16,kernel_size = 5,stride = 1,padding = 0) #input: (m,14,14,6) output: (m,10,10,16)

        self.max2 = nn.MaxPool2d(kernel_size = (2,2),stride = 2) #input: (m,10,10,16) output: (m,5,5,16)

        self.fc1 = nn.Linear(400,120) #input: (m,400) output: (m,120)

        self.fc2 = nn.Linear(120,84) #input: (m,120) output: (m,84)

        self.fc3 = nn.Linear(84,25) #input: (m,84) output: (m,25)

    def forward(self,x):

        x = F.relu(self.conv1(x))

        x = self.max1(x)

        x = F.relu(self.conv2(x))

        x = self.max2(x)

        x = torch.flatten(x,start_dim = 1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x

            
net = Neural_Network()

net

print(torch.cuda.is_available())

device = torch.device("cuda:0")

net.to(device)
#STORE THE LOSS AT EACH EPOCH

loss_list = []



#FUNCTION

def RUN_NETWORK(train,EPOCHS,test,batch_size):

    # WE USE THE ADAM OPTIMIZER

    optimizer = optim.Adam(net.parameters(),lr = 0.001)

    # WE USE SCHEDULER SO THAT AT EVERY 10 EPOCHS LEARNING REDUCES BY A FACTOR OF 10

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 10,gamma = 0.1)

    # WE USE CROSS ENTROPY LOSS

    loss = nn.CrossEntropyLoss()

    

    for epoch in (range(EPOCHS)):

        scheduler.step()

        print('Epoch:', epoch,'LR:', scheduler.get_last_lr())

        

        for i in (range(0,train.shape[0],batch_size)):

            net.zero_grad()

            

            X_train = torch.from_numpy(train[i:i+batch_size].values).type(torch.float).view(-1,1,28,28)

            y_train = torch.from_numpy(test[i:i+batch_size].values)

            X_train, y_train = X_train.to(device),y_train.to(device)

            

            output = net(X_train)

            l = loss(output,y_train)

            

            l.backward()

            optimizer.step()

        

        print("train loss : " + str(l))

        loss_list.append(l)



RUN_NETWORK(data_train.drop('label',axis = 1),50,data_train['label'],85)
plt.plot(np.arange(50),list(map(torch.Tensor.item,loss_list)))

plt.show()
data_test = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')
y_pred_train = torch.argmax(net(torch.from_numpy(data_train.drop('label',axis = 1).values.reshape(-1,1,28,28)).type(torch.float).to(device)),dim = 1)

y_pred_test = torch.argmax(net(torch.from_numpy(data_test.drop('label',axis = 1).values.reshape(-1,1,28,28)).type(torch.float).to(device)),dim = 1)
from sklearn.metrics import accuracy_score

print("The obtained test accuracy is " + str(accuracy_score(data_test['label'],np.array(y_pred_test.cpu()))))