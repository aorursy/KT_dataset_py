import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/50-startup-companies/50_Startups.csv")
dataset.head()
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values
print(X)
print(y)
#This code will change the categorical column into binary column

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
print(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)#This is used to only get values in 2 decimals.
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)