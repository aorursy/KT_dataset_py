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
data = pd.read_csv("/kaggle/input/loto.csv.txt")

data.dropna(inplace=True)

print(len(data))

data = data.iloc[::-1]

print(data.head(4))

print(data.mean(0))

print(data.median(0))

print(data.mad(0))
tiraj_range = 1000

f = data.mean(0).values

dd = data.mad(0).values

df = pd.DataFrame({

    'tiraj':[i for i in range(tiraj_range)],

    'x1' : [np.round(np.random.normal(f[1],dd[1])) for i in range(tiraj_range)],

    'x2' : [np.round(np.random.normal(f[2],dd[2])) for i in range(tiraj_range)],

    'x3' : [np.round(np.random.normal(f[3],dd[3])) for i in range(tiraj_range)],

    'x4' : [np.round(np.random.normal(f[4],dd[4])) for i in range(tiraj_range)],

    'x5' : [np.round(np.random.normal(f[5],dd[5])) for i in range(tiraj_range)],

    'x6' : [np.round(np.random.normal(f[6],dd[6])) for i in range(tiraj_range)],

    'x7' : [np.round(np.random.normal(f[7],dd[7])) for i in range(tiraj_range)],

    'x8' : [np.round(np.random.normal(f[8],dd[8])) for i in range(tiraj_range)],

    'y' :  [np.round(np.random.normal(f[9],dd[9])) for i in range(tiraj_range)],

    

})

print(df.mean())
d = data.drop(columns=['tiraj','y']).to_numpy()

print(d)
aaa = np.zeros(20)

for i in d:

    for j in i:

        aaa[int(j)-1] +=1

i=0

while i<20:

    print(i+1,'\t', aaa[i])

    i+=1
data.dropna(inplace=True)

cover_y = data['tiraj']

cover_df = data.drop(columns=['y','tiraj'])



cover_df.drop(index=[1,2], inplace=True)

cover_y.drop(index=[1,2],inplace=True)



from sklearn.model_selection import train_test_split

df_train, df_test, y_train, y_test = train_test_split(cover_df,cover_y, test_size=0.3, random_state=42)

print(df_train)

print(y_train)
from sklearn.preprocessing import MultiLabelBinarizer

y_train_binary = np.array([])

x_train_binary = np.array([])

df_binary_train = MultiLabelBinarizer().fit_transform(df_train.values)

print(df_binary_train.shape)

for i in range(len(df_binary_train)):

    if i%2!=0:

        y_train_binary = np.append(y_train_binary,df_binary_train[i],axis=0)

    else:

        x_train_binary = np.append(x_train_binary,df_binary_train[i], axis=0)

        

print(y_train_binary.shape)

shape = 192

x_train_binary = x_train_binary.reshape(shape,20)

y_train_binary = y_train_binary.reshape(shape,20)
df_test_bin = df_test.values

if len(df_test_bin) %2!=0:

    df_test_bin = df_test_bin[1::]

y_test_binary = np.array([])

x_test_binary = np.array([])



y_val_binary = np.array([])

x_val_binary = np.array([])



df_test_train = MultiLabelBinarizer().fit_transform(df_test_bin)

print(df_test_train.shape)



lenn = len(df_test_train)

for i in range(lenn):

    if i<lenn/2:

        if i%2!=0:

            y_test_binary = np.append(y_test_binary,df_test_train[i],axis=0)

        else:

            x_test_binary = np.append(x_test_binary,df_test_train[i], axis=0)

    else:

        if i%2!=0:

            y_val_binary = np.append(y_val_binary,df_test_train[i],axis=0)

        else:

            x_val_binary = np.append(x_val_binary,df_test_train[i], axis=0)



test_shape = 41

#y_test_binary = np.append(y_test_binary,np.zeros(220*41))

#x_test_binary = np.append(x_test_binary,np.zeros(220*41))



y_test_binary = y_test_binary.reshape(test_shape,20)

x_test_binary = x_test_binary.reshape(test_shape,20)





#y_val_binary = np.append(y_val_binary,np.zeros(220*41))

#x_val_binary = np.append(x_val_binary,np.zeros(220*41))

y_val_binary = y_test_binary.reshape(test_shape,20)

x_val_binary = x_test_binary.reshape(test_shape,20)



ggg = np.array([1,2,3])

ggg2 = np.zeros(3)

print(np.append(ggg,ggg2))
import torch

from torch.utils.data import TensorDataset,DataLoader



x_tensor_train = torch.from_numpy(x_train_binary).type(torch.FloatTensor)

x_tensor_test = torch.from_numpy(x_test_binary).type(torch.FloatTensor)





y_tensor_train = torch.from_numpy(y_train_binary).type(torch.FloatTensor)

y_tensor_test = torch.from_numpy(y_test_binary).type(torch.FloatTensor)





x_val_tensor = torch.from_numpy(x_val_binary).type(torch.FloatTensor)

y_val_tensor = torch.from_numpy(y_val_binary).type(torch.FloatTensor)



train_ds = TensorDataset(x_tensor_train, y_tensor_train)

test_ds = TensorDataset(x_tensor_test, y_tensor_test)

val_ds = TensorDataset(x_val_tensor, y_val_tensor)







train_loader = DataLoader(train_ds,batch_size=16, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=4)

val_loader = DataLoader(val_ds, batch_size=4)
from sklearn import metrics

import torch.nn as nn

import torch.nn.functional as F

from torch.nn.functional import log_softmax

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.inputs = 10

        

        self.conv1 = nn.Sequential(

        nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3,padding=2),nn.Sigmoid(),nn.BatchNorm1d(100),

        nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3,padding=2),nn.Sigmoid(),nn.BatchNorm1d(100),

        )

        

        self.embed = nn.Sequential(

            #nn.Embedding(20,200),

            nn.Linear(20,self.inputs),nn.ReLU(),

            nn.Linear(self.inputs,self.inputs),nn.Sigmoid(),

            nn.Linear(self.inputs,self.inputs),nn.ReLU(),

            nn.Linear(self.inputs,self.inputs),nn.Sigmoid(),

            nn.Linear(self.inputs,self.inputs),nn.ReLU(),

            nn.Linear(self.inputs,self.inputs),nn.Sigmoid(),

            nn.Linear(self.inputs,self.inputs),nn.ReLU(),

            nn.Linear(self.inputs,20),nn.Sigmoid(),

        )



    def forward(self, x):

        x = self.embed(x)

        return x

    



        xx = x.reshape(x.size()[0],x.size()[1],1)

        xx = xx.transpose(1,2)

        xx = self.conv1(xx)

        xx = xx.view(x.size(0), -1)

        

        return  self.dense(xx) - self.dense2(x)

    

    

    def train(self,train_loader,val_loader,epoches,optimizer):

        #self.cuda()

        criterion = nn.BCELoss() 

        for i in range(epoches):

            y_true_all = []

            y_pred = []

            loss_train=0

            for xx,yy in train_loader:

                #xx,yy = xx.cuda(),yy.cuda()

                y = self.forward(xx)

                loss = criterion(y,yy)

                loss.backward()

                optimizer.step()

                optimizer.zero_grad()

            for xx,yy in val_loader:

                y = self.forward(xx)

                loss2 = criterion(y,yy)

            accuracy = metrics.accuracy_score(y_true_all, y_pred)

            print("Epoch:%d, loss:%f, val loss:%f" % (i,loss.item(),loss2.item()))

        self.cpu()
import gc

gc.collect()

model = Net()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

model.train(train_loader,val_loader,50,optimizer)
def max_8(arr,for_print=20):

    z = []

    i = 1

    for j in arr:

        z.append([i,j])

        i+=1

    z.sort(key=lambda x: x[1],reverse=True)

    for i in range(for_print):

        print(z[i])

    return z



j=10



xx = x_tensor_test[j].reshape(1,20)

p = model.forward(xx) 

d = p.detach().numpy()[0]

d = max_8(d,8)



print("\n")

from_bin(y_tensor_test[j])