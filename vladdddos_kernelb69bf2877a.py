# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import torch 

import math

# Any results you write to the current directory are saved as output.

from sklearn.datasets import fetch_openml

covertype = fetch_openml(data_id=180)

print(type(covertype), covertype)
cover_df = pd.DataFrame(data=covertype.data, columns=covertype.feature_names)

cover_df.sample(10)
print(covertype.target)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(covertype.target)

print(label_encoder.classes_) 
cover_target = label_encoder.transform(covertype.target)

print(cover_target)
print(cover_df.shape)
from sklearn.model_selection import train_test_split

df_train, df_test, y_train, y_test = train_test_split(cover_df, cover_target, test_size=0.15, stratify=cover_target)

to_normalize = [(i, col) for i, col in enumerate(cover_df.columns)

                        if not col.startswith('wilderness_area') and not col.startswith('soil_type')]

idx_to_normalize = [i for i,col in to_normalize] #номера столбцов

columns_to_normalize = [col for i, col in to_normalize] #названия



print(columns_to_normalize)

print(idx_to_normalize)


cover_df[columns_to_normalize].sample(4)
from torch.utils.data import TensorDataset,DataLoader

tensor_train = torch.from_numpy(df_train.values).type(torch.FloatTensor)

print(tensor_train[:3])
tensor_test = torch.from_numpy(df_test.values).type(torch.FloatTensor)

train_mean = torch.mean(tensor_train[:,idx_to_normalize], dim=0)

train_std = torch.std(tensor_train[:,idx_to_normalize], dim=0)

print(train_mean, train_std)
tensor_train[:,idx_to_normalize] -= train_mean

tensor_train[:,idx_to_normalize] /= train_std

tensor_test[:,idx_to_normalize] -= train_mean

tensor_test[:,idx_to_normalize] /= train_std

print(tensor_train[:3])
y_train_tensor = torch.from_numpy(y_train).type(torch.LongTensor)

y_test_tensor = torch.from_numpy(y_test).type(torch.LongTensor)

train_ds = TensorDataset(tensor_train, y_train_tensor)

test_ds = TensorDataset(tensor_test, y_test_tensor)

print(train_ds[400])
train_loader = DataLoader(train_ds,batch_size=256, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=256)

for xx, yy in train_loader:

    print(xx)

    print(yy)

    break

for xx1, yy1 in test_loader:

    print(xx1)

    print(yy1)

    break
class NN():

    def __init__(self):

        self.inputl=54

        self.outputl=7

        self.hiddenl=18

        self.bsize=0

        self.W1 = torch.randn( self.inputl, self.hiddenl) 

        self.W2 = torch.randn( self.hiddenl ,self.outputl)

        self.b1 = torch.randn(1, self.hiddenl)

        self.b2 = torch.randn(1,  self.outputl)

    def train(self, X, y,t):

        self.X=X

        self.Y=y

        self.bsize=t

        self.backp()

    def test(self, X, y,t):

        self.X=X

        self.Y=y

        self.bsize=t

        correct = 0

        o = self.forward()

        pr = o.data.max(1)[1]  

        correct += pr.eq(self.Y.data).sum()

        print(correct)

    def relu(self,s):

        s[s<0]=0

        return s

    def softmax(self, s):

        o=s;

        for j in range(self.bsize):

            ex = torch.exp(s[j,:])

            for i in range(7):

                o[j,i]=ex[i] / ex.sum()

        return o

    def forward(self):

        self.z = torch.mm(self.X, self.W1)+self.b1

        self.z2 = self.relu(self.z)

        self.z3 = torch.mm( self.z2, self.W2)+self.b2

        o = self.softmax(self.z3)

        return o

    def loss_f(self,X,Y):

        f = -torch.mean(torch.log(X[torch.arange(self.bsize),Y]))

        return f

    def backp(self):

        j=0 

        f=0

        self.W1.requires_grad_(True) 

        self.W2.requires_grad_(True)  

        self.b1.requires_grad_(True)  

        self.b2.requires_grad_(True)  

        lr = 0.05

        for iteration in range(150):

            with torch.no_grad():

                if self.W1.grad is not None:

                    self.W1.grad.zero_()

            with torch.no_grad():        

                if self.W2.grad is not None:

                    self.W2.grad.zero_() 

            with torch.no_grad():        

                if self.b1.grad is not None:

                    self.b1.grad.zero_()

            with torch.no_grad():        

                if self.b2.grad is not None:

                    self.b2.grad.zero_()    

            f=self.loss_f(self.forward(),self.Y)

            f.backward()

            with torch.no_grad():

                self.W1 -= lr *  self.W1.grad

            with torch.no_grad():    

                self.W2 -= lr *  self.W2.grad

            with torch.no_grad():   

                self.b1 -= lr *  self.b1.grad

            with torch.no_grad():    

                self.b2 -= lr *  self.b2.grad

a=NN()

a.train(xx,yy,256)

a.test(xx1,yy1,256)