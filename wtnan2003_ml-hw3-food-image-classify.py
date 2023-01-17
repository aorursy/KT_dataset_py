# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!python
import numpy as np
import cv2
import os
def readfile(path,label):
    img_dir = sorted(os.listdir(path))
    x = np.zeros((len(img_dir),128,128,3),dtype = np.uint8) # 图片存在这个维度
    y = np.zeros(len(img_dir),dtype = np.uint8) # 标签
    for i , file in enumerate(img_dir):
        x [i,:,:] = cv2.resize(cv2.imread(os.path.join(path,file)),(128,128))
        if label:
            y[i] = int(file.split('_')[0])
    if label:
        return x, y
    else:
        return y
workspace_dir ='/kaggle/input/ml2020spring-hw3/food-11/'
print("Reading data")
train_x , train_y = readfile(os.path.join(workspace_dir,"training"),True)
print("Size of training data = {}".format(len(train_x)))
val_x , val_y = readfile(os.path.join(workspace_dir,"validation"),True)
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir,'testing'),False)
print("Size of Testing data = {}".format(len(test_x)))
import torchvision.transforms as transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),# 
    transforms.RandomHorizontalFlip(), #水平翻转
    transforms.RandomRotation(15), # 随机旋转
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])
from torch.utils.data import DataLoader,Dataset
class ImgDataset(Dataset):
    def __init__(self,x,y=None,transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self,index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if not self.y:
            Y = self.y[index]
            return X,Y
        else:
            return X
import torch
dir(torch.nn)
batch_size = 128
train_set = ImgDataset(train_x,train_y,train_transform) # 实例化train_set
val_set = ImgDataset(val_x,val_y,test_transform)
train_loader = DataLoader(train_set,batch_size = batch_size,shuffle = True)
val_loader = DataLoader(val_set,batch_size = batch_size,shuffle=False)

import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,64,3,1,1), #[64,128,128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0), #[64,64,64]
            
            nn.Conv2d(64,128,3,1,1), # [128,64,64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),
            
            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,11)
        )
    def forward(self,x):
        out = self.cnn(x)
        out = out.view(out.size()[0],-1)
        return self.fc(out)
torch.cuda.is_available()
model  = Classifier().cuda()  # 模型也要转成cuda
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
num_epoch = 30
for epoch in range(num_epoch):
    epoch_stat_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    model.train()
    for i , data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred , cuda[1].cuda() )
        batch_loss.backward()
        optimizer.step()
        # 最大值的索引：
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(),axis = 1) == data[1].numpy())
        train_loss += batch_loss.item()
        
    model.eval()
    with torch.no_grad():
        for i,data in enuemrate(val_loader):
            val_pred = model(data[0].cuda()) # 验证集的pred
            batch_loss = loss(val_pred,data[1].cuda())
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(),axis = 1) == data[1].numpy())
            val_loss += batch_loss.item()
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
    
test_set = ImgDataset(test_x,transform = test_transform)
test_loader = DataLoader(test_set , batch_size = batch_size ,shuffle=False)
model_best.eval()
predicition = []
with torch.no_grad():
    for i,data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(),axis = 1)
        for y in test_label:
            predicition.append(y)