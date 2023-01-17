# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# train and test data into dataframe
trainBoston = pd.read_csv("../input/boston_train.csv")
testBoston = pd.read_csv("../input/boston_test.csv")

print(trainBoston.head())
print(testBoston.head())
# print(trainBoston.info())
# print(trainBoston.describe())

trainBoston.drop('ID', axis=1, inplace=True)
trainBoston.info()

testBoston.drop('ID',axis=1, inplace=True)
testBoston.info()

# just checking correlation between no: of rooms and median value of home
trainBoston.plot.scatter('rm','medv')
# increase display size of the heatmap
plt.subplots(figsize=(12,8))
# check correlation between all the variables
sns.heatmap(trainBoston.corr(), cmap='RdGy')
# plot pairplots between medv and strongly correlated variables (+ve and -ve)
# sns.pairplot(trainBoston, vars=['lstat','ptratio','nox','age','crim','medv'])
# removing lstat outliers, i.e., where values > 30
trainBoston.drop(trainBoston[trainBoston.lstat>30].index, inplace=True)
# removing criminal outliers, i.e., where values > 70
trainBoston.drop(trainBoston[trainBoston.crim>70].index, inplace=True)
reqFtrs = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat']
X1 = trainBoston[reqFtrs]
print(X1.info())

y1 = trainBoston['medv']
print(y1.count())
# boxplot for detecting outliers
# sns.boxplot(x=[X1['age'], X1['ptratio']], orient='v')
boxFtrs = ['crim','zn','indus','chas','nox','rm','dis','rad','ptratio','lstat']
# removed age, tax, black
plt.figure(figsize=(12,8))
sns.boxplot(data=X1[boxFtrs], orient='v')
# looks like i'm gonna retain lstat completely
# print(X1.lstat.unique())
# print(X1.lstat.describe())
# zip_data_df[zip_data_df.cust_counts > 5]
# print(X1[X1.lstat>30])
# .value_counts())
# looks like zn column has only a few unique values, so i shouldn't have removed even those 6 records which i did just now; always have a copy of the dataframe that i'm working with
# print(X1.loc[:50,"zn"])
# plt.figure(figsize=(12,8))
# sns.boxplot(data=X1['lstat'], orient='v')

# dealt with outliers for lstat, zn

# train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2)

# model 1
lm = LinearRegression()
lm.fit(X1_train, y1_train)
predicted_y1 = lm.predict(X1_test)

# model 2
dt = DecisionTreeRegressor()
dt.fit(X1_train, y1_train)
predicted_y2 = dt.predict(X1_test)

# model 3
rf = RandomForestRegressor(max_depth=2)
rf.fit(X1_train, y1_train.values.ravel())
predicted_y3 = rf.predict(X1_test)

# checking randomForest pred y values with validation set
plt.scatter(y1_test, predicted_y3)
plt.xlabel('y1_test')
plt.ylabel('predicted_y1')
# calculating metrics to assess model 3, i.e. RF
from sklearn import metrics
print('Mean Absolute Error: ', metrics.mean_absolute_error(y1_test, predicted_y3))
print('Mean Squared Error: ', metrics.mean_squared_error(y1_test, predicted_y3))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y1_test, predicted_y3)))
# cross-validation
cv_lm = cross_val_score(estimator=lm, X=X1_train, y=y1_train.values.ravel(), cv=5)
print(cv_lm.mean())
cv_dt = cross_val_score(estimator=dt, X=X1_train, y=y1_train.values.ravel(), cv=5)
print(cv_dt.mean())
cv_rf = cross_val_score(estimator=rf, X=X1_train, y=y1_train.values.ravel(), cv=5)
print(cv_rf.mean())
# gridsearchCV for optimal parameters
grid_param={
  'n_estimators': [100, 300, 500, 800, 1000], 
  'criterion': ['mse','mae'],
  'bootstrap': [True, False]
}

gd_sr_rf = GridSearchCV(estimator=rf, param_grid=grid_param, scoring='r2', cv=5, n_jobs=-1)
# neg_mean_squared_error
# print(y1_train.head())
# print(y1_train.dtypes)

# print(y1_train.values.ravel())
# print(type(y1_train.values.ravel()))

gd_sr_rf.fit(X1_train, y1_train.values.ravel())
best_parameters=gd_sr_rf.best_params_
print(best_parameters)

best_results=gd_sr_rf.best_score_
print(best_results)
# contd with rf model with the optimal parameters (from GridSearchCV) for predicting values in testBoston and check RMSE
# 09 27 2018
rf2 = RandomForestRegressor(max_depth=2, n_estimators=300, criterion='mse', bootstrap=True)
rf2.fit(X1, y1.values.ravel())
pred_yrf2 = rf2.predict(testBoston)

print(pred_yrf2)
print(type(pred_yrf2))
print(pred_yrf2.size)
# testOp = pd.DataFrame(data=pred_yrf2) #, index=data[1:,0], columns=data[0,1:]) 
# print(testOp.head())
# print(testBoston.info())
# testOp.drop(columns=[0])
# del testOp
testOp = pd.DataFrame({'MedvPred': pred_yrf2})
print(testOp.info())
print(testBoston.info())
# testBostonFinal = testBoston.append(testOp)
testBostonFinal = pd.concat([testBoston, testOp], axis=1)
print(testBostonFinal.head())
# print(testBostonFinal.info())
# del testBostonFinal
# testBostonFinal.to_csv("D:\Vish_Python\Kaggle\Boston Housing\all\testFinal.csv")
testBostonFinal.to_csv("testFinal.csv",index=False)
# os.listdir('output')