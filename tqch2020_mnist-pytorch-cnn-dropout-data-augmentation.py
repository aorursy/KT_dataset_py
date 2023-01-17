import os

import random

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.optim import Adam

from torch.utils.data import DataLoader

from torchvision import transforms

import numpy as np

import matplotlib.pyplot as plt

from torch.nn import CrossEntropyLoss

from tqdm import tqdm
root = "/kaggle/input/digit-recognizer"

# 42000 train 784 pixels + 1 label (take a while to load)

train_data = np.loadtxt(os.path.join(root,"train.csv"),delimiter=",",skiprows=1)

# 28000 test 784 pixels

test_data = np.loadtxt(os.path.join(root,"test.csv"),delimiter=",",skiprows=1)
# check gpu information

!nvidia-smi
class Dataset:

    """

    build a map-style dataset

    """

    def __init__(self,data,targets,transform=None):

        self.data = data

        self.targets = targets

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self,idx):

        if self.transform == None:

            return self.data[idx],self.targets[idx]

        else:

            return self.transform(self.data[idx]),self.targets[idx]
transform = transforms.Compose([transforms.ToPILImage(),

                                transforms.RandomAffine(degrees=15,translate=(1/7,1/7),shear=15),

                                transforms.RandomRotation(degrees=15),

                                transforms.ToTensor()])

x_train = train_data[:,1:].reshape(-1,28,28).astype(np.uint8)

y_train = torch.LongTensor(train_data[:,0])

train_dataset = Dataset(x_train,y_train,transform)
trainloader = DataLoader(train_dataset,batch_size=512,shuffle=True,pin_memory=True)

%matplotlib inline

x,y = next(iter(trainloader))

plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1).axis("off")

    plt.imshow(x[i].squeeze(0))

    plt.title(str(y[i].item()))
# define a cnn classifier

class CNN(nn.Module):

    def __init__(self):

        super(CNN,self).__init__()

        self.conv1 = nn.Conv2d(1,64,3,stride=1,padding=1)

        self.conv2 = nn.Conv2d(64,128,3,stride=1,padding=1)

        self.conv3 = nn.Conv2d(128,256,3,stride=1)

        self.conv4 = nn.Conv2d(256,512,3,stride=1,padding=1)

        self.conv5 = nn.Conv2d(512,1024,3,stride=1,padding=1)

        self.conv6 = nn.Conv2d(1024,2048,3,stride=1,padding=1)

        self.fc1 = nn.Linear(2048*3*3,2048)

        self.fc2 = nn.Linear(2048,2048)

        self.fc3 = nn.Linear(2048,512)

        self.fc4 = nn.Linear(512,10)

    def forward(self,x):

        conv1 = F.relu(self.conv1(x))

        conv2 = F.relu(self.conv2(conv1))

        maxpool1 = F.max_pool2d(conv2,2)

        conv3 = F.relu(self.conv3(maxpool1))

        conv4 = F.relu(self.conv4(conv3))

        maxpool2 = F.max_pool2d(conv4,2)

        conv5 = F.relu(self.conv5(maxpool2))

        conv6 = F.relu(self.conv6(conv5))

        maxpool3 = F.max_pool2d(conv6,2).flatten(start_dim=1)

        fc1 = F.dropout(F.relu(self.fc1(maxpool3)),0.8,training=self.training)

        fc2 = F.relu(self.fc2(fc1))

        fc3 = F.relu(self.fc3(fc2))

        out = self.fc4(fc3)

        return out
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

cnn = CNN()

cnn.to(device)

opt = Adam(cnn.parameters(),lr=1e-4)

# opt = SGD(cnn.parameters(),lr=1e-5,momentum=0.9)

loss_fn = CrossEntropyLoss()



random.seed(1234)

np.random.seed(1234)

torch.random.manual_seed(1234)

epochs = 100

batch_size = 512



trainloader = DataLoader(train_dataset,batch_size,shuffle=True,pin_memory=True)

for n in range(epochs):

    with tqdm(trainloader) as t:

        t.set_description(f"{n+1}/{epochs} epochs")

        running_loss = 0.0

        running_correct = 0

        running_total = 0

        for x,y in t:

            out = cnn(x.to(device))

            pred = out.max(dim=1)[1]

            loss = loss_fn(out,y.to(device))

            opt.zero_grad()

            loss.backward()

            opt.step()

            running_loss += loss.item()*x.size(0)

            running_correct += (pred==y.to(device)).sum().item()

            running_total += x.size(0)

            t.set_postfix({"train_loss":running_loss/running_total,"train_acc":running_correct/running_total})
# switch to eval mode

cnn.eval()

running_loss = 0.0

running_correct = 0

running_total = 0



# disable transformation

train_dataset.transform = transforms.ToTensor()

trainloader = DataLoader(train_dataset,batch_size=512,shuffle=False,pin_memory=True)



with torch.no_grad():

    for x,y in trainloader:

        out = cnn(x.to(device))

        pred = out.max(dim=1)[1]

        running_correct += (pred==y.to(device)).sum().item()

        running_total += x.size(0)

print("The training accurary is {}".format(running_correct/running_total))
# prediction

x_test = test_data.reshape(-1,1,28,28) # pytorch channel first

x_test = torch.Tensor(x_test)/255.

test_pred = []

with torch.no_grad():

    for i in range(0,len(x_test),batch_size):

        out = cnn(x_test[i:i+batch_size].to(device))

        pred = out.max(dim=1)[1]

        test_pred.append(pred.detach().cpu().numpy())

test_pred = np.concatenate(test_pred)
# store predictions

import pandas as pd

imageid = pd.Series(np.arange(len(test_pred)))+1

df = pd.DataFrame({"ImageId":imageid,"Label":test_pred})

df.set_index("ImageId")

df.to_csv("/kaggle/working/test_pred.csv",index=False)