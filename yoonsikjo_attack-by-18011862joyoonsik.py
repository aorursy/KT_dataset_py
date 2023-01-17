import torch

import torch.nn.functional as F

import torch.optim as optim

import torch.nn as nn

import numpy as np

import sys

import pandas as pd

import random

from sklearn.preprocessing import StandardScaler



device = 'cuda' if torch.cuda.is_available() else 'cpu'



random.seed(484)

torch.manual_seed(484)

if device == 'cuda':

  torch.cuda.manual_seed_all(484)
data = pd.read_csv('../input/city-commercialchange-analysis/train.csv')

data
scaler = StandardScaler()

#test = pd.read_csv('test_data.csv')



x_train_data = data.loc[:,'year':'Closing months_Average']

y_train_data = data.loc[:,'Commercial change']



xs_data = scaler.fit_transform(x_train_data)



x_train = torch.FloatTensor(xs_data[:551])

y_train = torch.LongTensor(y_train_data.values)

xs_data.shape
epochs = 10001



W = torch.zeros((7, 4), requires_grad=True)

b = torch.zeros(4, requires_grad=True)



optimizer = optim.SGD([W, b], lr=0.01)



for ep in range(epochs):

    hypothesis = x_train.matmul(W) + b

    cost = F.cross_entropy(hypothesis, y_train)



    optimizer.zero_grad()

    cost.backward()

    optimizer.step()



    if ep%1000 == 0:

        print('{:4}: loss: {:2.8f}'.format(ep, cost.item()))
test = pd.read_csv('../input/city-commercialchange-analysis/test.csv')

x_data = test.loc[:,'year':'Closing months_Average']

xs_data = scaler.fit_transform(x_data)
x_test = torch.FloatTensor(xs_data[:62])
x_test
with torch.no_grad():

    

    hypothesis = x_test.matmul(W) + b

    

    real_test_df = pd.DataFrame([[i, r] for i, r in enumerate(torch.argmax(hypothesis, dim=1).numpy())], columns=['ID','Label'])

    real_test_df.to_csv('result.csv', mode='w', index=False)
real_test_df