import numpy as np

import torch

import torch.optim as optim

import pandas as pd

xy = pd.read_csv('../input/mlregression-cabbage-price/train_cabbage_price.csv',header=None)

xy = xy.drop(0,axis=0)



x_data = xy.loc[: , 0:4]

x_data = np.array(x_data,dtype=np.float64)





test_data = pd.read_csv('../input/mlregression-cabbage-price/test_cabbage_price.csv',header=None)

test_data = test_data.drop(0,axis=0)

test_data = np.array(test_data,dtype=np.float64)



y_data = xy[5]

y_data = np.array(y_data,dtype=np.float64)
from sklearn import neighbors

model = neighbors.KNeighborsRegressor(n_neighbors = 3 , weights = "distance")

model.fit(x_data, y_data)  #fit the model

pred = model.predict(test_data)
pred = list(pred)

index = [i for i in range(731)]

col_names = ['Id','Expected']



values = [[0]*2 for i in range(731)]

for i in range(731):

  values[i][0]=i



for i in range(731):

  values[i][1]=pred[i]



df1 = pd.DataFrame(values,columns=col_names,index=index)

df1.to_csv("result.csv",index=False,header=True)

df1