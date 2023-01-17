import pandas as pd

import numpy as np

import torch

import random

import torch.optim as optim

import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import torch.nn.functional as F
if torch.cuda.is_available() is True:

  device = torch.device('cuda')

else:

  device = torch.device('cpu')
train=pd.read_csv("../input/train-person/new_train.csv")

x_train=train.iloc[:,1:]

y_train=train.iloc[:,[0]]



x_train=np.array(x_train)

y_train=np.array(y_train)



x_train=torch.FloatTensor(x_train)

y_train=torch.FloatTensor(y_train)
torch.manual_seed(1)

torch.cuda.manual_seed_all(1)



X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size = 0.1, random_state = 100)

print(X_train.shape)

print(X_valid.shape)

print(Y_train.shape)

print(Y_valid.shape)
lr = 0.01

batch_size = 56

Epochs = 300
d = torch.utils.data.TensorDataset(X_train, Y_train)

data_loader = torch.utils.data.DataLoader(dataset = d,

                                          batch_size=batch_size,

                                          shuffle = True,

                                          drop_last=True)
torch.manual_seed(1)

torch.cuda.manual_seed_all(1)





linear1 = nn.Linear(5,20,bias=True)

linear2 = nn.Linear(20, 20, bias=True)

linear3 = nn.Linear(20, 20, bias=True)

linear4 = nn.Linear(20, 1, bias=True)

relu = nn.ReLU()



model = nn.Sequential(linear1, relu, linear2, relu, linear3,relu,linear4).to(device)





for layer in model.children():

  if isinstance(layer, nn.Linear):

    nn.init.xavier_uniform_(layer.weight)



loss = nn.BCELoss().to(device)

optimizer = optim.Adam(model.parameters(), lr = lr)
total_batch = len(data_loader);

best_acc = 0

accuracy = 0



for epoch in range(Epochs+1):

  model.train()

  avg_cost = 0;



  for X, Y in data_loader:

    X = X.view(-1,5)

    optimizer.zero_grad()

    H = torch.sigmoid(model(X))

    cost=(loss(H,Y))

    cost.backward()

    optimizer.step()



    avg_cost += cost / total_batch

    # scheduler.step(cost)

  with torch.no_grad():

    model.eval()

    # import pdb;pdb.set_trace()



    valid = torch.sigmoid(model(X_valid))

    valid[valid<0.5] = 0

    valid[valid>=0.5] = 1

    

    accuracy = accuracy_score(Y_valid.to('cpu'), valid.to('cpu').detach().numpy())*100



    if best_acc < accuracy :

      best_acc = accuracy

      print("save bestmodel, epoch: {:4d}".format(epoch))

      torch.save(model, './best_model.ptr')



  print("Epoch{:4d}/{}, cost : {:.06f}, accuracy : {:.06f}".format(epoch,

                                              Epochs,

                                              avg_cost,

                                              accuracy))

  

print('finish')
model = torch.load("./best_model.ptr")
x_test=pd.read_csv("../input/train-person/new_test.csv")

x_test = torch.FloatTensor(np.array(x_test)).to(device)
with torch.no_grad():

  

  model.eval()

  predict = torch.sigmoid(model(x_test))

#predict[predict>=0.5] = 1

#predict[predict<0.5] = 0

predict[predict>0.5350]=1

predict[predict<=0.5350]=0

predict.sum()
label = predict.to('cpu').detach().numpy()

Id = np.array([int(i) for i in range(len(predict))]).reshape(-1,1)

result = np.hstack((Id, label))
df = pd.DataFrame(result, columns=(['id','index']))

df['id'] = df['id'].astype(int)

df['index'] = df['index'].astype(int)

df.to_csv("submission.csv",index=False, header=True)