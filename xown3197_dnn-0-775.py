# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## load data



train=pd.DataFrame(pd.read_csv('/kaggle/input/sejongai-challenge-pretest-1/train.csv', index_col=0))

test=pd.DataFrame(pd.read_csv('/kaggle/input/sejongai-challenge-pretest-1/test_data.csv', index_col=0))

submit=pd.DataFrame(pd.read_csv('/kaggle/input/sejongai-challenge-pretest-1/submit_sample.csv'))
train_X = np.array(train.iloc[:, :-1], dtype=np.float32) # 0~7

train_Y = np.array(train.iloc[:, -1], dtype=np.float32) # 8

test = np.array(test, dtype=np.float32)



print(train.shape)

print(train_X.shape)

print(train_Y.shape)

print(test.shape)
# dataset



import torch

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset



if torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")



train_x, val_x, train_y, val_y = train_test_split(train_X, train_Y)



class CustomDataset(Dataset): 

    def __init__(self, x_data, y_data=None):

        self.x_data = x_data

        self.y_data = y_data



    # 총 데이터의 개수를 리턴

    def __len__(self): 

        return len(self.x_data)



    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴

    def __getitem__(self, idx): 

 



        x = torch.FloatTensor([self.x_data[idx]])



        if self.y_data is not None:

            y = torch.FloatTensor([self.y_data[idx]])



            return x, y

        return x







batch_size = 1



train_loader = DataLoader(CustomDataset(train_x, train_y), batch_size=batch_size)

val_loader = DataLoader(CustomDataset(val_x, val_y))

test_loader = DataLoader(CustomDataset(test))
# model



import torch

from torch import nn



class BinaryClassifer(nn.Module):

    def __init__(self, input_shape, out_shape,):

        super(BinaryClassifer, self).__init__() 

        self.layer1 = nn.Sequential(nn.Linear(input_shape, input_shape),

                                    nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(nn.Linear(input_shape, input_shape*2),

                                    nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(nn.Linear(input_shape*2, out_shape),

                                    nn.ReLU(inplace=True))

    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = self.layer3(out)

        out = torch.sigmoid(out)

        return out

    

def weights_init(m):

    if isinstance(m, nn.Linear):

        nn.init.xavier_normal_(m.weight) 

        

model = BinaryClassifer(8, 1).to(device)

model.apply(weights_init)
## train



import torch.nn.functional as F

from torchvision import datasets

import torch.optim as optim



num_epochs = 1000

lr = 0.001



criterion = nn.BCELoss().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)



costs = []

total_batch = len(train_loader)

for epoch in range(num_epochs):

    total_cost = 0

    for i, (x_data, y_data) in enumerate(train_loader):

        x_data.to(device)

        y_data.to(device)

        outputs = model(x_data)

        

        loss = criterion(outputs, y_data)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        total_cost += loss

    avg_cost = total_cost / total_batch

    if epoch % 100 == 0:

        print("Epoch:", "%03d" % (epoch+1), "Cost =", "{:.9f}".format(avg_cost))  

    costs.append(avg_cost)  
## eval



model.eval()

with torch.no_grad():

    correct = 0

    total = 0

    for i, (x_data, y_data) in enumerate(val_loader):

        outputs = model(x_data)

        outputs = outputs > 0.5

        correct += (y_data == outputs).sum().item()

    

    print('Accuracy : {:.2f}%'.format(correct / len(val_loader) * 100))   
## test and submit



model.eval()

preds = torch.zeros(len(test_loader))

for i, (x_data) in enumerate(test_loader):

    outputs = model(x_data)

    total += x_data.size(0)

    preds[i] = outputs > 0.5

    

submit["Label"] = np.array(preds.detach().cpu(), dtype=np.int)

submit.to_csv("result.csv", index=False, header=True)

    

    
submit