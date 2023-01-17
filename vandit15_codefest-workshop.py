# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
# in case you get csv files, you may use pd.read_csv(). Here we are using a popular regression dataset
# already available in scikit learn.
from sklearn.datasets import load_boston
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
target = dataset.target
df.head(5)
df.info() # this is mainly to check for any null values in the dataset
# This is used to randomly split data 
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(df,target,test_size=0.2,random_state=100)
# its always good to get idea of dimensions of dataset that you are working with.
print(X_train.shape)
print(Y_train.shape)
print(X_valid.shape)
print(Y_valid.shape)
print(X_train.head())
# This besides helping in getting a glimpse of the dataset also helps to detect any outliers for
# all attributes in one go.
X_train.describe()
# This gives a relation between every pair of columns.
corr = X_train.corr()
sns.heatmap(corr)
print(corr)
for column in dataset.feature_names:
    plt.scatter(X_train[column], Y_train)
    plt.xlabel(column)
    plt.ylabel("target")
    plt.show()
# training a model using the training data.
lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)
# predicting the results using the trained model.
Y_pred = lin_reg.predict(X_valid)
error = mean_squared_error(Y_valid, Y_pred)
print(error)
# this model shows that our model produces results(in blue) that have the same pattern as shown by the dataset
plt.scatter(X_valid['LSTAT'], Y_valid,  color='black')
plt.scatter(X_valid['LSTAT'], Y_pred, color='blue')
plt.xlabel('LSTAT')
plt.ylabel('target')

plt.xticks(())
plt.yticks(())

plt.show()
# dropping unnecessary columns
print(X_train.columns)
X_train_new = X_train.drop("CHAS", axis = 1)
X_valid_new = X_valid.drop("CHAS", axis = 1)
print(X_train_new.columns)
# training the model again on edited training dataset
lin_reg.fit(X_train_new,Y_train)
# predicting validation results using trained model again.
Y_pred = lin_reg.predict(X_valid_new)
error = mean_squared_error(Y_valid, Y_pred)
print(error)
# currently using default parameters.
tree_reg = RandomForestRegressor(n_estimators = 10, random_state = 100)
tree_reg.fit(X_train, Y_train)
# similar to linear regression model, get validation results using random forest model
Y_pred = tree_reg.predict(X_valid)
error = mean_squared_error(Y_valid, Y_pred)
print(error)