import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

data = pd.read_csv('../input/petrol_consumption.csv')
data.head()
data.shape
X = data.drop('Petrol_Consumption', axis=1)

y = data['Petrol_Consumption']

X.head()
y.head()
#Data Split into training and test

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=0)
X_train.head()
y_train.head()
# model selection and training

from sklearn.tree import DecisionTreeRegressor

model=DecisionTreeRegressor()

model.fit(X_train,y_train)
#prediction

prediction= model.predict(X_test)
prediction

#comparing actual vs predicted

compare=pd.DataFrame({'Actual':y_test ,'Predicted':prediction,'Difference':y_test-prediction})
compare
#Evaluating Algorithmn

#mae = mean absolute error

#mse = mean square error

#sqrt_mse = squared mean squared error

from sklearn import metrics

mae=metrics.mean_absolute_error(y_test,prediction)

mse=metrics.mean_squared_error(y_test,prediction)

sqrt_mse=np.sqrt(metrics.mean_squared_error(y_test,prediction))
mae

mse
sqrt_mse