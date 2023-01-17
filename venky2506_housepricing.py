# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as py

import seaborn as sns

from sklearn.svm import SVR

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.preprocessing import normalize

from sklearn.kernel_approximation import Nystroem

from sklearn.linear_model import LinearRegression,SGDRegressor

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
mo1=pd.concat([train,test],sort=False)

mo1.drop('SalePrice',axis=1,inplace=True)
len(train.columns)
py.figure(figsize=(25,5))

sns.heatmap(train.isnull())
mo1=pd.get_dummies(mo1,columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'],drop_first=True)
train1=mo1.iloc[:1460,:]

test1=mo1.iloc[1460:,:]
train1.drop(columns=['PoolQC','Alley','Fence','MiscFeature'],axis=1,inplace=True)

test1.drop(columns=['PoolQC','Alley','Fence','MiscFeature'],axis=1,inplace=True)
sim=SimpleImputer(strategy='mean',missing_values=np.nan)

X1=sim.fit_transform(train1.values)

X2=sim.fit_transform(test1.values)
X=pd.DataFrame(X1,columns=train1.columns)

X_t=pd.DataFrame(X2,columns=test1.columns)
X.shape
py.figure(figsize=(25,5))

sns.heatmap(X.isnull())
X=pd.concat([X,train['SalePrice']],axis=1)

p1=X.corr(method='pearson')['SalePrice'].sort_values(ascending=False)
p1=p1[p1<0.3]
p1
X.drop(columns=list(p1.index.values),axis=1,inplace=True)

X_t.drop(columns=list(p1.index.values),axis=1,inplace=True)
X2=normalize(X)

X=pd.DataFrame(X2,columns=X.columns)

Xt=normalize(X_t)

X_t=pd.DataFrame(Xt,columns=X_t.columns)
x_train,x_val,y_train,y_val=train_test_split(X.drop('SalePrice',axis=1),X['SalePrice'],test_size=0.3)
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_train)

tp=np.sqrt(np.sum((np.log(y_pred+1)-np.log(y_train+1))**2)/y_train.shape[0])

print('log RMSE for training set is {}'.format((tp)))

y_pred=lr.predict(x_val)

tp=np.sqrt(np.sum((np.log(y_pred+1)-np.log(y_val+1))**2)/y_val.shape[0])

print('log RMSE for validation set is {}'.format(tp))
rf=RandomForestRegressor()
rf.fit(x_train,y_train)
y_pred=rf.predict(x_train)

tp=np.sqrt(np.sum((np.log(y_pred+1)-np.log(y_train+1))**2)/y_train.shape[0])

print('log RMSE for training set is {}'.format((tp)))

y_pred=rf.predict(x_val)

tp=np.sqrt(np.sum((np.log(y_pred+1)-np.log(y_val+1))**2)/y_val.shape[0])

print('log RMSE for validation set is {}'.format(tp))
sv=SVR()
sv.fit(x_train,y_train)
y_pred=sv.predict(x_train)

tp=np.sqrt(np.sum((np.log(y_pred+1)-np.log(y_train+1))**2)/y_train.shape[0])

print('log RMSE for training set is {}'.format((tp)))

y_pred=sv.predict(x_val)

tp=np.sqrt(np.sum((np.log(y_pred+1)-np.log(y_val+1))**2)/y_val.shape[0])

print('log RMSE for validation set is {}'.format(tp))
ny=Nystroem()
x_train1=ny.fit_transform(x_train)

x_val1=ny.fit_transform(x_val)
sg=SGDRegressor()
sg.fit(x_train1,y_train)
y_pred=sg.predict(x_train1)

tp=np.sqrt(np.sum((np.log(y_pred+1)-np.log(y_train+1))**2)/y_train.shape[0])

print('log RMSE for training set is {}'.format((tp)))

y_pred=sg.predict(x_val1)

tp=np.sqrt(np.sum((np.log(y_pred+1)-np.log(y_val+1))**2)/y_val.shape[0])

print('log RMSE for validation set is {}'.format(tp))
y_test=rf.predict(X_t)
pd.DataFrame(y_test).to_csv('submission.csv')