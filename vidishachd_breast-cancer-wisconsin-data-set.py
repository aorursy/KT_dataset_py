import numpy as np

import pandas as pd

import os
data = pd.read_csv("../input/data.csv")

print(data.describe())

print(data.head())

print(data.columns)
data.info()
#we can see diagnosis is categorical data, so convert it :

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

data.describe()

data.head()
X=data.iloc[:,2:32]

print(X.head())

print(X.columns)
y = data.iloc[:,1]

print(y.head())
#finding categorical data

#data.select_dtypes(include=['category', object]).columns
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error



train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=0)

cancer_model = DecisionTreeRegressor(random_state=0)

cancer_model.fit(train_X,train_y)

val_predictions = cancer_model.predict(val_X)

print(val_predictions)

print(val_y)

print(mean_absolute_error(val_y,val_predictions))