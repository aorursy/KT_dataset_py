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
import torch

import pandas as pd

import torch.optim as optim

import numpy as np

torch.manual_seed(1)

device = torch.device("cuda")

#데이터

xy_data = pd.read_csv('../input/2020-ml-w1p2/2020.AI.cancer-train.csv')

x_test = pd.read_csv('../input/2020-ml-w1p2/2020.AI.cancer-test.csv')

submit = pd.read_csv('../input/2020-ml-w1p2/2020.AI.cancer-sample-submission.csv')
xy_data = np.array(xy_data)

x_train = torch.FloatTensor(xy_data[:,1:-1]).to(device)

y_train = torch.FloatTensor(xy_data[:,0]).to(device)

x_test = np.array(x_test)

x_test = torch.FloatTensor(x_test[:,:-1]).to(device)
W = torch.zeros((30,1)).to(device).detach().requires_grad_(True)

b = torch.zeros(1).to(device).detach().requires_grad_(True)

optimizer= optim.SGD([W,b], lr=1e-4)

nb_epochs = 1000
from torch.nn import BCELoss

import torch.nn.functional as F

loss =BCELoss()

for epoch in range(nb_epochs + 1):

  hypothesis = torch.sigmoid(x_train.matmul(W)+b)

  cost = loss(hypothesis, y_train)

  optimizer.zero_grad()

  cost.backward()

  optimizer.step()

  if epoch%100==0:

    print('cost = {}'.format(cost.item()))
hypothesis = torch.sigmoid(x_test.matmul(W)+b)

predict = hypothesis>=0.5

for i in range(len(predict)):

  submit['diagnosis'][i]=predict[i]
submit=submit.astype(np.int32)

submit.to_csv('submit.csv', mode='w', header= True, index= False)