import matplotlib.pyplot as plt

import numpy as np

import pandas as pd 

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
dataset=pd.read_csv('../input/50_Startups.csv')
dataset.head(5)
X=dataset.iloc[:,:-1]

Y=dataset.iloc[:,4]
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X['State_n']=labelencoder.fit_transform(X['State'])
X_n=X.drop('State',axis='columns')

X_n.head()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_n, Y, test_size = 0.2, random_state = 0)
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_predict = regressor.predict(X_test)
regressor.intercept_
regressor.coef_
regressor.score(X_n,Y)
regressor.predict([[162345.47,139827,236829,5]])