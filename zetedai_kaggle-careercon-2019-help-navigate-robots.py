# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import torch # PyTorch package

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read Kaggle datasets

X_train_old = pd.read_csv('/kaggle/input/career-con-2019/X_train.csv')

y_train_old = pd.read_csv('/kaggle/input/career-con-2019/y_train.csv')

# split X_train

samples = 20

time_series = 128

start_x = X_train_old.shape[0] - samples*time_series

X_train, X_test = X_train_old.iloc[:start_x], X_train_old.iloc[start_x:]

# split y_train

start_y = y_train_old.shape[0] - samples

y_train, y_test = y_train_old.iloc[:start_y], y_train_old.iloc[start_y:]
# Inspect the data

display(X_train, y_train, X_test, y_test)
# Merge X and y so we have the y label for each row in X

# Because there is no ommitted data (perfect), we don't need to specify how to merge

Xy_train = X_train.merge(y_train, on='series_id')

Xy_test = X_test.merge(y_test, on='series_id')

display(Xy_train, Xy_test)
# Features for predict the surface type

X_columns = ['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W', 'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z', 'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z']
# Use dictionarys to map different surface types to int for easier calculation

encode_dic = {'fine_concrete': 0, 

              'concrete': 1, 

              'soft_tiles': 2, 

              'tiled': 3, 

              'soft_pvc': 4,

              'hard_tiles_large_space': 5, 

              'carpet': 6, 

              'hard_tiles': 7, 

              'wood': 8}



decode_dic = {0: 'fine_concrete',

              1: 'concrete',

              2: 'soft_tiles',

              3: 'tiled',

              4: 'soft_pvc',

              5: 'hard_tiles_large_space',

              6: 'carpet',

              7: 'hard_tiles',

              8: 'wood'}
# Convert pandas dataframes into PyTorch tensors

X_train = torch.tensor(Xy_train[X_columns].values).float()

y_train = torch.tensor(Xy_train['surface'].map(encode_dic).values)

X_test = torch.tensor(Xy_test[X_columns].values).float()

y_test = torch.tensor(Xy_test['surface'].map(encode_dic).values)

display(X_train, y_train, X_test, y_test)
display(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
from torch import nn



model = nn.Sequential(nn.Linear(10, 63),

                      nn.ReLU(),

                      nn.Linear(63, 54),

                      nn.ReLU(),

                      nn.Linear(54, 45),

                      nn.ReLU(),

                      nn.Linear(45, 36),

                      nn.ReLU(),

                      nn.Linear(36, 27),

                      nn.ReLU(),

                      nn.Linear(27, 9)

                     )
from torch import optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.003)
# Set the computational device to GPU

device = torch.device("cuda:0")



# Move the neural network model to GPU

model.to(device)



# Move all the tensors to GPU

X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)
%%time

epochs = 2500



for e in range(epochs):

    

    # Reset the gradients

    optimizer.zero_grad()

    

    # Forward passing

    output = model(X_train)



    # Calculate the loss/cost function

    loss = criterion(output, y_train)

    

    # Back propagation

    loss.backward()

    

    # Update the weights

    optimizer.step()
with torch.no_grad():

    # Calculate the output of Neural Network

    network_output = model(X_test)

    

    # Use the softmax to calculate the probabilities of each class, "dim=1" means across the columns

    possibilities = torch.softmax(network_output, dim=1)

    

    # Use argmax to find the class has the highest probability

    predict = possibilities.argmax(dim=1)

    

    # Compare the predicted result with y_test to find out the accuracy

    acc = (1.0 * (predict==y_test).sum().item() / y_test.shape[0])

    

    print(f"Accuracy = {acc}")