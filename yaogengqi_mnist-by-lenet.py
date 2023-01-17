import torch

import torch.nn as nn

import torch.optim as optim

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import MNIST

from torch.utils.data import DataLoader
class Net(nn.Module):

    

    def __init__(self):

        

        super(Net, self).__init__()

    

        self.conv1 = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding = (1,1)),

            nn.BatchNorm2d(16),

        )

        

        self.conv2 = nn.Sequential(

            nn.Conv2d(16, 32, 3, padding = (1,1)),

            nn.BatchNorm2d(32),

        )

        

        self.conv3 = nn.Sequential(

            nn.Conv2d(32, 64, 3, padding = (1,1)),

            nn.BatchNorm2d(64),

        )

        

        self.pool = nn.MaxPool2d(2, 2)

        

        self.fc =nn.Sequential(

            nn.Dropout(0.25),

            nn.Linear(64 * 3 * 3, 256),

            nn.Dropout(0.25),

            nn.Linear(256, 10),

        )



        

    def forward(self, x):

        

        x = self.conv1(x)

        x = nn.functional.relu(x)

        x = self.pool(x)

        

        x = self.conv2(x)

        x = nn.functional.relu(x)

        x = self.pool(x)

        

        x = self.conv3(x)

        x = nn.functional.relu(x)

        x = self.pool(x)

        

        x = x.view(-1,64*3*3)

        x = self.fc(x)

        

        return x

        

model = Net()

model
lr = 0.01  # learn rate



criterion = nn.CrossEntropyLoss()  # loss function



optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay=5e-4)



# weight_decay,权值衰减,一种正则化的方法，在损失函数中加入weight的计算方法，防止过拟合



if torch.cuda.is_available():

    

    model = model.cuda()

    

    criterion = criterion.cuda()
class MNIST_Dataset(Dataset):

    

    def __init__(self, data_path, transform = transforms.Compose([

                                                    transforms.ToTensor(),

                                                    # transforms.Normalize(mean=(0.5,), std=(0.5,))

                                                ])):

        

        data = pd.read_csv(data_path)

        self.transform = transform

        self.data = data

        

        if len(data.columns) == 28*28: # test

            

            self.y = None

            self.x = data.values.reshape((-1,28,28)) # .astype(np.uint8)

        

        else: # train

            

            self.y = torch.from_numpy(data.iloc[:, 0].values)

            self.x = data.iloc[:,1:].values.reshape((-1,28,28)) # .astype(np.uint8)

            

    def __len__(self):

        

        return len(self.data)

    

    def __getitem__(self, idx):

        

        if self.y == None:

            

            return self.transform(self.x[idx])

        

        else:

            

            return self.transform(self.x[idx]), self.y[idx]

        

        

train_data_path = '/kaggle/input/digit-recognizer/train.csv'

test_data_path = '/kaggle/input/digit-recognizer/test.csv'



train_data = MNIST_Dataset(train_data_path)



test_data = MNIST_Dataset(test_data_path)



train_Loader = DataLoader(train_data, batch_size=128, shuffle=True)



test_Loader = DataLoader(test_data)
model.train()



epoches = 100

train_loss = 0

correct = 0

total = 0 

current_precision = 0.0



for epoch in  range(epoches):



    for batch_idx, (inputs, targets) in enumerate(train_Loader):  # 返回批次的样本数量，输入图片和图片分类结果



        if torch.cuda.is_available():



            inputs = inputs.cuda()



            targets = targets.cuda()



        # clear the grad

        optimizer.zero_grad()



        outputs = model(inputs.float())



        loss = criterion(outputs, targets)



        loss.backward()



        optimizer.step()



        train_loss += loss.item() # 记录损失数据



        _, predicted = outputs.max(1) # 记录预测结果，选取从0-9中概率最大的结果



        total += targets.size(0) # 记录图片总数



        correct += predicted.eq(targets).sum().item() # 记录预测正确的图片数目

        

        current_precision = 100. * correct/total



#         print('批次训练进度：%d/%d'% (1+batch_idx, len(train_Loader)))



#         print('current_Loss: %.3f | average_Loss: %.3f(%f/%d)'%(loss.item(), train_loss/(batch_idx+1), train_loss, batch_idx+1))



#         print('current_ACC: %.3f%% | average_ACC:%.3f%% (%d/%d)\n'%(100.*predicted.eq(targets).sum().item()/targets.size(0) ,100. * correct/total, correct, total))

    

    print('训练进度：%d/%d,训练精确度：%.3f%%'%(epoch+1, epoches, current_precision))

    
model.eval()



test_pred = torch.LongTensor()



with torch.no_grad():

    

    for batch_idx, inputs in enumerate(test_Loader):

        

        if torch.cuda.is_available():

        

            inputs = inputs.cuda()

            

            test_pred = test_pred.cuda()

        

        outputs = model(inputs.float())

        

        _, predicted = outputs.max(1) 

            

        test_pred = torch.cat((test_pred, predicted), dim=0)

        
out_df = pd.DataFrame(np.c_[np.arange(1, len(test_data)+1)[:,None], test_pred.cpu().numpy()], 

                      columns=['ImageId', 'Label'])



out_df.to_csv('submission_version_7.csv', index=False)



out_df.head()