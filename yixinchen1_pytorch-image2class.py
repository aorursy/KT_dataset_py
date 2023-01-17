import os

import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import TensorDataset,DataLoader,Dataset
## parameters for mmodel

# Hyper parameters

num_epochs = 8

num_classes = 2

batch_size = 128

learning_rate = 0.02



# Device configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('GPU or CPU:{}'.format(device))
# load the data

labels = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')

sub = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')

train_path = '../input/histopathologic-cancer-detection/train/'

test_path = '../input/histopathologic-cancer-detection/test/'
# splitting data into train and val

train, val = train_test_split(labels, stratify = labels.label, test_size = 0.1)

len(train),len(val)

print(train)
# create data pipe

class MyDataset(Dataset):

    def __init__(self,df_data,data_dir = './', transform=None):

        super().__init__()

        self.df = df_data.values

        self.data_dir = data_dir

        self.transform = transform

        

    def __len__(self):

        return len(self.df)

    

    #index is relative with __len__, like "for index in range(__len__)"

    def __getitem__(self,index):

        img_name,label = self.df[index]

        img_path = os.path.join(self.data_dir, img_name+'.tif')

        image = cv2.imread(img_path)

        if self.transform is not None:

            image = self.transform(image)

        return image ,label
# define transform, which is preprocess of image

trans_train = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Pad(64,padding_mode='reflect'),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(20),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.5,0.5,0.5],

        std =[0.5,0.5,0.5])    

])

trans_valid = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Pad(64,padding_mode='reflect'),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.5,0.5,0.5],

        std =[0.5,0.5,0.5])

])
dataset_train = MyDataset(df_data=train,data_dir = train_path,transform = trans_train)

dataset_valid = MyDataset(df_data=val,data_dir=train_path,transform=trans_valid)



loader_train = DataLoader(dataset = dataset_train,batch_size = batch_size,shuffle=True, num_workers=0)

loader_valid = DataLoader(dataset = dataset_valid,batch_size = batch_size//2,shuffle=False,num_workers=0)
# define CNN model

class SimpleCNN(nn.Module):

    def __init__(self):

        super(SimpleCNN,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,

                               kernel_size=3,padding=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,

                               kernel_size=3,padding=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,

                               kernel_size=3,padding=2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,

                               kernel_size=3,padding=2)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512,

                               kernel_size=3,padding=2)

        self.bn1 = nn.BatchNorm2d(32)

        self.bn2 = nn.BatchNorm2d(64)        

        self.bn3 = nn.BatchNorm2d(128)        

        self.bn4 = nn.BatchNorm2d(256)        

        self.bn5 = nn.BatchNorm2d(512)        

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.avg = nn.AvgPool2d(8)

        self.fc = nn.Linear(512*1*1,2) #!!! 2 classes, dense layer

    

    def forward(self,x):

        #convolutional layer, then batchnorm then activation and pooling

        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))

        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))

        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))

        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))

        x = self.pool(F.leaky_relu(self.bn5(self.conv5(x))))

        x = self.avg(x)

        x = x.view(-1,512*1*1)

        x = self.fc(x)

        return x



model = SimpleCNN().to(device)
# Loss and optimizer

criterion = nn.BCELoss(reduction="mean")

optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
# print(outputs[0])

# m = nn.Sigmoid()

# x = m(outputs.view(256)).view(128,2)

# print(x)



# x = nn.Sigmoid(x)

# print(x[0])



# train the model,total_step is the number of batch

total_step = len(loader_train)

m = nn.Sigmoid()

for epoch in range(num_epochs):

    for i, (images,labels) in enumerate(loader_train):

        images = images.to(device)

#         labels = labels.to(device)

        # one hot label

        class_num = 2

        batch_size = len(labels)

        labels = labels.view(len(labels),1)

        one_hot = torch.zeros(batch_size, class_num).scatter_(1, labels, 1)

        one_hot = one_hot.to(device)

        # Forward pass

        outputs = model(images)

        outputs = m(outputs.view(2*len(outputs))).view(len(outputs),2)

        loss = criterion(outputs,one_hot)

        

        # Backward and optimize

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if(i+1)%100 == 0:

            print('Epoch[{}/{}],step[{}/{},Loss:{}'

                  .format(epoch+1,num_epochs,i+1,total_step,loss.item()))

            
# Test the model

# model.eval()



with torch.no_grad():

    correct = 0

    total = 0

    for images, labels in loader_valid:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        print(torch.max(outputs.data,1))

        _, predicted = torch.max(outputs.data,1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        

    print('Test Accuracy of the model on the 22003 test images:{}%'

         .format(100*correct/total))



# save the model checkpoint

torch.save(model.state_dict(), 'model.ckpt')

        