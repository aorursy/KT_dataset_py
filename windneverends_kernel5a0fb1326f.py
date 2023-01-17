# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

from torch.autograd import Variable

from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from PIL import Image

import pandas as pd

import matplotlib.pyplot as plt

import random
auto_save_scores_Depressing = pd.read_csv("/kaggle/input/scores/scores/auto_save_scores_Depressing.csv")

# auto_save_scores_beautiful = pd.read_csv("../input/auto_save_scores_beautiful.csv")

# auto_save_scores_boring = pd.read_csv("../input/auto_save_scores_boring.csv")

# auto_save_scores_lively = pd.read_csv("../input/auto_save_scores_lively.csv")

# auto_save_scores_safety = pd.read_csv("../input/auto_save_scores_safety.csv")

# auto_save_scores_wealthy = pd.read_csv("../input/auto_save_scores_wealthy.csv")

def default_loader(path):

    return Image.open(path).convert('RGB')

class MyDataset(Dataset):

    def __init__(self, csv, transform=None, target_transform=None, loader=default_loader):

        imgs=[]

#         self.kind=kind

        for i in range(0,len(auto_save_scores_Depressing)):

            imgname='/kaggle/input/scores/scores/depressing/'+auto_save_scores_Depressing.iloc[i,0]+'.png'

            label=auto_save_scores_Depressing.iloc[i,1]

            imgs.append((imgname,label))

        #打乱顺序

        random.shuffle(imgs)

#         train_imgs=[]

#         test_imgs=[]

       

#         for i in range(int(len(imgs)*0.8)):

#             train_imgs.append(imgs[i])

#         for i in range(int(len(imgs)*0.8)+1,len(imgs)):   

#             test_imgs.append(imgs[i])

#         print(len(train_imgs))

#         self.train_imgs = train_imgs

#         self.test_imgs=test_imgs

        self.imgs=imgs

        self.transform = transform

        self.target_transform = target_transform

        self.loader = loader



    def __getitem__(self, index):

        fn, label = self.imgs[index]

        img = self.loader(fn)

        if self.transform is not None:

            img = self.transform(img)

        return img,label



    def __len__(self):

        return len(self.imgs)

transform_train = transforms.Compose([

    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32

    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转

    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差

])



dataset=MyDataset(csv='a.csv', transform=transform_train)

dataset_size = len(dataset)

train_size = int(0.8 * dataset_size)

test_size = dataset_size - train_size

train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])



batch_size=64

#print(len(train_data))

#img,label=train_data.__getitem__(1)



#test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,drop_last=True)

test_loader = DataLoader(dataset=test_data, batch_size=batch_size)



'''ResNet-18 Image classfication for cifar-10 with PyTorch 



Author 'Sun-qian'.



'''

import torch

import torch.nn as nn

import torch.nn.functional as F



class ResidualBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1):

        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(

            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),

            nn.BatchNorm2d(outchannel),

            nn.ReLU(inplace=True),

            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),

            nn.BatchNorm2d(outchannel)

        )

        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel:

            self.shortcut = nn.Sequential(

                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(outchannel)

            )



    def forward(self, x):

        out = self.left(x)

        out += self.shortcut(x)

        out = F.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, ResidualBlock, num_classes=10):

        super(ResNet, self).__init__()

        self.inchannel = 64

        self.conv1 = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),

            nn.BatchNorm2d(64),

            nn.ReLU(),

        )

        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)

        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)

        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)

        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

        self.fc = nn.Linear(512, 1)



    def make_layer(self, block, channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]

        layers = []

        for stride in strides:

            layers.append(block(self.inchannel, channels, stride))

            self.inchannel = channels

        return nn.Sequential(*layers)



    def forward(self, x):

        out = self.conv1(x)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out





def ResNet18():



    return ResNet(ResidualBlock)



class Net(torch.nn.Module):



    def __init__(self):

        super(Net,self).__init__()

        self.conv1 = torch.nn.Conv2d(3,64,3,padding=1)

        self.conv2 = torch.nn.Conv2d(64,64,3,padding=1)

        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.bn1 = torch.nn.BatchNorm2d(64)

        self.relu1 = torch.nn.ReLU()



        self.conv3 = torch.nn.Conv2d(64,128,3,padding=1)

        self.conv4 =torch.nn.Conv2d(128, 128, 3,padding=1)

        self.pool2 = torch.nn.MaxPool2d(2, 2, padding=1)

        self.bn2 = torch.nn.BatchNorm2d(128)

        self.relu2 = torch.nn.ReLU()



        self.conv5 = torch.nn.Conv2d(128,128, 3,padding=1)

        self.conv6 = torch.nn.Conv2d(128, 128, 3,padding=1)

        self.conv7 = torch.nn.Conv2d(128, 128, 1,padding=1)

        self.pool3 = torch.nn.MaxPool2d(2, 2, padding=1)

        self.bn3 = torch.nn.BatchNorm2d(128)

        self.relu3 = torch.nn.ReLU()



        self.conv8 = torch.nn.Conv2d(128, 256, 3,padding=1)

        self.conv9 = torch.nn.Conv2d(256, 256, 3, padding=1)

        self.conv10 = torch.nn.Conv2d(256, 256, 1, padding=1)

        self.pool4 = torch.nn.MaxPool2d(2, 2, padding=1)

        self.bn4 = torch.nn.BatchNorm2d(256)

        self.relu4 = torch.nn.ReLU()



        self.conv11 = torch.nn.Conv2d(256, 512, 3, padding=1)

        self.conv12 = torch.nn.Conv2d(512, 512, 3, padding=1)

        self.conv13 = torch.nn.Conv2d(512, 512, 1, padding=1)

        self.pool5 = torch.nn.MaxPool2d(2, 2, padding=1)

        self.bn5 = torch.nn.BatchNorm2d(512)

        self.relu5 = torch.nn.ReLU()



        self.fc14 = torch.nn.Linear(512*4*4,1024)

        self.drop1 = torch.nn.Dropout2d()

        self.fc15 = torch.nn.Linear(1024,1024)

        self.drop2 = torch.nn.Dropout2d()

        self.fc16 = torch.nn.Linear(1024,1)





    def forward(self,x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.pool1(x)

        x = self.bn1(x)

        x = self.relu1(x)





        x = self.conv3(x)

        x = self.conv4(x)

        x = self.pool2(x)

        x = self.bn2(x)

        x = self.relu2(x)



        x = self.conv5(x)

        x = self.conv6(x)

        x = self.conv7(x)

        x = self.pool3(x)

        x = self.bn3(x)

        x = self.relu3(x)



        x = self.conv8(x)

        x = self.conv9(x)

        x = self.conv10(x)

        x = self.pool4(x)

        x = self.bn4(x)

        x = self.relu4(x)



        x = self.conv11(x)

        x = self.conv12(x)

        x = self.conv13(x)

        x = self.pool5(x)

        x = self.bn5(x)

        x = self.relu5(x)

        #print(" x shape ",x.size())

        F=torch.nn.functional

        x = x.view(x.size(0),-1)

        x = F.relu(self.fc14(x))

        x = self.drop1(x)

        x = F.relu(self.fc15(x))

        x = self.drop2(x)

        x = self.fc16(x)

        return x



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

#net=Net()

net=Net().cuda()

print(net)

train_loss_list=[]

test_loss_list=[]

#net=ResNet18()

#.to(device)

#print(net)

optimizer = torch.optim.Adam(net.parameters(),lr=0.01)

#loss_func = torch.nn.CrossEntropyLoss()

loss_func=torch.nn.MSELoss()

lr_list=[]

#print(len(train_data))

for epoch in range(100):

    print('epoch {}'.format(epoch + 1))

#     #kind1

#     if epoch % 33 == 0:

#         for p in optimizer.param_groups:

#             p['lr'] *= 0.1

    #kind2

    if epoch % 3 == 0:

        for p in optimizer.param_groups:

            p['lr'] *= 0.9

    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

    net.train()

    train_sum_loss=0.0

    test_sum_loss=0.0

   

    for i, data in enumerate(train_loader, 0):

        length = len(train_loader)

        #print(length)

        inputs, labels = data

        #进入GPU

        inputs, labels = inputs.cuda(), labels.cuda()

        print(inputs)

        #forward + backward

        outputs = net(inputs)

        #降维

        outputs=outputs.squeeze()

        labels=labels.float()

        #print(outputs)

        #print(labels)

        #计算损失值

        loss =loss_func(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        #print(loss.item())

        # 每训练1个batch打印一次loss和准确率

        train_sum_loss += loss.item()

    print('Train Loss: {:.6f}'.format(train_sum_loss / (len(

        train_data))))

    train_loss_list.append(train_sum_loss / (len(

        train_data)))

    net.eval()

    for i, data in enumerate(test_loader, 0):

        length = len(test_loader)

        #print(length)

        inputs, labels = data

        inputs, labels = inputs.cuda(), labels.cuda()

        #print(inputs)

        #print(labels)

        #forward + backward

        outputs = net(inputs)

        outputs=outputs.squeeze()

        labels=labels.float()

        #print(outputs)

        #print(labels)

        loss =loss_func(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        #print(loss.item())

        # 每训练1个batch打印一次loss和准确率

        test_sum_loss += loss.item()

    print('Test Loss: {:.6f}'.format(test_sum_loss / (len(

        test_data))))

    test_loss_list.append(test_sum_loss / (len(

        test_data)))

# with open("train2.txt,"+"batch_size"+str(batch_size),"w") as f:

#     for i in range(len(train_loss_list)):

#         f.write("Epoch:"+str(i+1)+",train_loss:"+str(train_loss_list[i])+"\n")  #这句话自带文件关闭功能，不需要再写f.close()

# with open("test2.txt,"+"batch_size"+str(batch_size),"w") as f:

#     for i in range(len(test_loss_list)):

#         f.write("Epoch:"+str(i+1)+",train_loss:"+str(test_loss_list[i])+"\n")  #这句话自带文件关闭功能，不需要再写f.close() 

#         print(outputs.data)

#         ten=tensor[25,3,4,2]

#         pred=torch.max(ten,0)

#         print(pred)

#       pred = torch.max(outputs.data, 1)

#         print(pred)

#        total += labels.size(0)

#         correct += predicted.eq(labels.data).cpu().sum()

#          print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '

#                % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

# print('over')

                   
# min_loss=min(train_loss_list)

# print(min_loss)

# #torch.save(net.state_dict(),'net_params.pkl')

# torch.save(net,'netfinal.pkl')

# plt.plot(range(100),lr_list,color = 'r')





# plt.plot(range(100),Loss_list,color='r')

# print(lr_list)

#print(Loss_list)

# print(net)

# #torch.save(net, '/kaggle/working/model.pkl')

# #绘制图像

# import matplotlib.pyplot as plt

# plt.plot(i, ,c='red')

# #设置纵坐标范围

# plt.ylim((8000,10000))

# #设置横坐标角度，这里设置为45度

# plt.xticks(rotation=45)

# #设置横纵坐标名称

# plt.xlabel("month")

# plt.ylabel("price")

# #设置折线图名称

# plt.title("the price of 2018")

# plt.show()
import numpy as np

x=Image.open('/kaggle/input/scores/scores/depressing/114.223568_30.551963_180_0.png').convert('RGB')

# 加载网络参数

transform_train = transforms.Compose([

    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32

    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转

    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差

])

x=transform_train(x)

x=DataLoader(x, batch_size=batch_size, shuffle=True,drop_last=True)

for i, data in enumerate(x, 0):

    inputs,label=data

inputs=inputs.cuda()



print(inputs)



net=torch.load('netfinal.pkl')



prediction=net(inputs)

print(prediction)

prediction=prediction.squeeze()

list=[]

list=prediction

print(torch.mean(list))