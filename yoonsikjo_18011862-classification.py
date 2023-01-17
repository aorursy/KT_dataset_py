import numpy as np

import torch

import torch.optim as optim

import pandas as pd

xy = pd.read_csv('../input/logistic-classification-diabetes-knn/train.csv',header=None)

xy = xy.drop(0,axis=0)

xy = xy.drop(0,axis=1)



x_data = xy.loc[: , 0:8]

x_data = np.array(x_data)



y_data = xy[9]

y_data = np.array(y_data)



test_data = pd.read_csv('../input/logistic-classification-diabetes-knn/test_data.csv',header=None)

test_data = test_data.drop(0,axis=0)

test_data = test_data.drop(9,axis=1)

test_data = test_data.drop(0,axis=1)

test_data = np.array(test_data)

# KNN 의 적용

from sklearn.neighbors import KNeighborsClassifier 

knn=KNeighborsClassifier(n_neighbors=5,p=2) 

knn.fit(x_data,y_data)
predict =knn.predict(test_data)  #모델을 적용한 test data의 y값 예측치
predict = list(predict)

index = [i for i in range(50)]

col_names = ['ID','Label']



values = [[0]*2 for i in range(50)]

for i in range(50):

  values[i][0]=i



for i in range(50):

  values[i][1]=predict[i]



df1 = pd.DataFrame(values,columns=col_names,index=index)

df1 = df1.astype('int')

df1.to_csv("result.csv",index=False,header=True)

df1