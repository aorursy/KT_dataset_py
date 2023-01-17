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
import numpy as np

import pandas as pd



import os
data = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

data.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



# Convert labels to their indexes

for col in data.columns:

    data[col] = le.fit_transform(data[col])

    

data.head()
cutoff = int(len(data)*0.8) + 1# add one to make it a round number. Easier for training.

train_df = data.iloc[:cutoff, :]

test_df = data.iloc[cutoff:, :]



len_train = (len(train_df))

len_test = (len(test_df))

print(float(len_train) / (float(len_test) + float(len_train)) ) # should be ~.8 for 80% train/test split or 5 fold validation
import torch



from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable

from torchvision import transforms, utils

from sklearn.preprocessing import LabelEncoder



class MushroomDataset(Dataset):

    

    def __init__(self, dataframe, transform=None):

        self.mushroom_frame = dataframe

        self.transform = transform

        

        le = LabelEncoder()

        for col in self.mushroom_frame.columns:

            self.mushroom_frame[col] = le.fit_transform(self.mushroom_frame[col])

        

    def __len__(self):

        return len(self.mushroom_frame)

    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

        inputs = torch.from_numpy(np.array(self.mushroom_frame.iloc[idx, 1:])).type(torch.float)

        label = self.mushroom_frame.iloc[idx, 0]

        label = torch.Tensor([label]).type(torch.long)

        sample = inputs, label

        

        if self.transform:

            sample = self.transform(sample)

            

        return sample

        
train_dset = MushroomDataset(train_df)

test_dset = MushroomDataset(test_df)



train_dl = torch.utils.data.DataLoader(train_dset,batch_size=50, shuffle=True,num_workers=4)

test_dl = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=4)
from torch import nn



class NeuralNet(nn.Module):

    def __init__(self):

        super(NeuralNet, self).__init__()

        self.fc1 = nn.Linear(22, 80)

        self.relu1 = nn.ReLU()

#         self.bn1 = nn.BatchNorm1d(80)

#         self.dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(80, 2)

        self.softmax = nn.Softmax()

        

    def forward(self, x, test=False):

        x = self.relu1(self.fc1(x))

#         if not test:

#             x = self.bn1(x)

#             x = self.dropout(x) 

        x = self.fc2(x)

        x = self.softmax(x)

        return x
from torch.optim import Adam



net = NeuralNet()

criterion = nn.BCELoss()

optimizer = Adam(net.parameters(), 0.001)



for epoch in range(100):

    for i, (x, y) in enumerate(train_dl):

        y_onehot = torch.FloatTensor(50, 2)

        y_onehot.zero_()

        y_onehot.scatter_(1, y, 1)

        

        y_hat = net(x)

        loss = criterion(y_hat, y_onehot)

        

        if i % 500 == 0:

            print(epoch, i, loss.item())

        

        optimizer.zero_grad()

        

        loss.backward()

        optimizer.step()
correct = 0

total = 0



with torch.no_grad():

    for i, (x, y) in enumerate(test_dl):

        y_hat = net(x, test=True)

        _, predicted = torch.max(y_hat.data, 1)

        total += y.size(0)

        correct += (predicted == y).sum().item()



print("Accuracy: ", float(correct)/total * 100)