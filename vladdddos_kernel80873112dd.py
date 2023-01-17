import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import torch

import torch.nn as nn

import torch.nn.functional as F

from sklearn.metrics import classification_report

import torch

import torchvision

import torchvision.transforms as transforms

train_dir = '../input/fruits-360_dataset/fruits-360/Training/'

test_dir = '../input/fruits-360_dataset/fruits-360/Test/'

trans_img = transforms.Compose(

    [transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train = torchvision.datasets.ImageFolder(root=train_dir,transform = trans_img)

test = torchvision.datasets.ImageFolder(root=test_dir,transform = trans_img)

test_loader = torch.utils.data.DataLoader(test, batch_size=25,shuffle=True, )

train_p = int(0.4 * len(train))

validation_set, n_train = torch.utils.data.random_split(train, [train_p,len(train)-train_p])

train_loader = torch.utils.data.DataLoader(n_train, batch_size=25,shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_set,  batch_size=25, shuffle=True )
class NN(nn.Module):

    def __init__(self):

        super(NN, self).__init__()

        self.layer1 = nn.Sequential( 

            nn.Conv2d(in_channels=3, out_channels=50, kernel_size=5, stride=1, padding=2),

            nn.ReLU(), 

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            nn.Conv2d(in_channels=50,out_channels= 100, kernel_size=5, stride=1, padding=2), 

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)) 

        self.drop_out = nn.Dropout() 

        self.fc1 = nn.Linear(25 * 25 * 100, 200) 

        self.fc2 = nn.Linear(200, 103)

    def forward(self, x): 

        x = self.layer1(x) 

        x = self.layer2(x) 

        x =  x.reshape(x.size(0), -1) 

        x = self.drop_out(x) 

        x = self.fc1(x) 

        x = self.fc2(x) 

        return x
def train(model,epoches,classes,learning_rate):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.to(device)

        total_step = len(train_loader)

        loss_list = []

        acc_list = []

        for epoch in range(epoches):

            for i, (images, labels) in enumerate(train_loader):

                images = images.to(device)

                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                loss_list.append(loss.item())

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                total = labels.size(0)

                _, predicted = torch.max(outputs.data, 1)

                correct = (predicted == labels).sum().item()

                acc_list.append(correct / total)



                if (i + 1) % 100 == 0:

                    accuracy,pr,tr=test(model,validation_loader)

                    if accuracy>95:

                        return

def test(model,loader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()

    valid_preds = []

    valid_truth = []

    with torch.no_grad():

        correct = 0

        total = 0

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            valid_preds += predicted.tolist()

            valid_truth += labels.tolist()

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        return (correct / total) * 100,valid_preds,valid_truth
model = NN()

learning_rate=0.001

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(model,5,103,learning_rate)

score,preds,labels = test(model,test_loader)

print(score)

print(classification_report(labels,preds))