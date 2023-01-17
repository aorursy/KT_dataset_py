# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/Train_UWu5bXk.csv')

train.columns
test = pd.read_csv('../input/Test_u94Q5KV.csv') 
# dealong with missing values 

# test columns with missing values

target=train["Item_Outlet_Sales"]

# adding a flag of train and test data 

train["Dataset_category"]= "train"

test["Dataset_category"]= "test"

data=pd.concat([train,test],axis=0)

data_var=data.drop(["Item_Outlet_Sales"],axis=1)
# let's start with the cleaning and feature eng...



data_var['Item_Identifier'] = data_var['Item_Identifier'].str[:2]

# to view the unique values in a dataset 

data_var.Item_Fat_Content.unique()

# so Low Fat,Regulat,low fat ,LF,reg

data_var['Item_Fat_Content'].replace(['Low Fat','LF'],'low fat')

#"missing data"

data_var.isna().sum()

#df['BrandName'].replace(['ABC', 'AB'], 'A')

# this converts to 3 categories
# Item weight and Outlet Size has missing values 

# first Outlet_size



# replace na valus with 0 

data_var['Item_Weight'] = data_var['Item_Weight'].fillna(0)

data_var['Outlet_Size'] = data_var['Outlet_Size'].fillna(0)



data_var.isna().sum()



# converting to dummy vaiables for applying models



data_var=pd.get_dummies(data_var)



# separate the test and train data , here we will take help of the flags applied

train_new=data_var.loc[data_var['Dataset_category_train'] == 1]

train_new.drop(['Dataset_category_train','Dataset_category_test'],axis=1,inplace=True)



test_new=data_var.loc[data_var['Dataset_category_test']==1]

test_new.drop(['Dataset_category_train','Dataset_category_test'],axis=1,inplace=True)

# let's apply algorithms and see how can the model improve the accuracy 

# Linear Regression 
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(train_new,target,test_size=0.2)
from sklearn.linear_model import LinearRegression

reg_model=LinearRegression()

reg_model.fit(x_train,y_train)

y_regpredict= reg_model.predict(x_test)

mse=np.mean((y_regpredict-y_test)**2)

rmse_reg=np.sqrt(mse)
from sklearn.linear_model import Lasso 

lasso_model= Lasso()

lasso_model.fit(x_train,y_train)

y_lass_pred=lasso_model.predict(x_test)

mse=np.mean((y_lass_pred-y_test)**2)

rmse_lasso=np.sqrt(mse)

from sklearn.linear_model import Ridge

ridge_model=Ridge()

ridge_model.fit(x_train,y_train)

ridge_pred=ridge_model.predict(x_test)

mse_ridge=np.mean((ridge_pred-y_test)**2)

rmse_ridge=np.sqrt(mse_ridge)
from sklearn.tree import DecisionTreeRegressor

tree_model=DecisionTreeRegressor()

tree_model.fit(x_train,y_train)

tree_pred=tree_model.predict(x_test)

mse_tree=np.mean((tree_pred-y_test)**2)

rmse_decision=np.sqrt(mse_tree)
from sklearn.ensemble import RandomForestRegressor

forest_model=RandomForestRegressor()

forest_model.fit(x_train,y_train)

forest_pred=forest_model.predict(x_test)

mse_forest=np.mean((forest_pred-y_test)**2)

rmse_forest=np.sqrt(mse_forest)

from sklearn.ensemble import GradientBoostingRegressor

g_boost=GradientBoostingRegressor()

g_boost.fit(x_train,y_train)

grad_predict=g_boost.predict(x_test)

mse_grad=np.mean((grad_predict-y_test)**2)

rmse_grad=np.sqrt(mse_grad)

print("RMSE for linear regression is :", rmse_reg)

print("RMSE for lasso regression is :",rmse_lasso)

print("RMSE for ridge regression is :", rmse_ridge)

print("RMSE for Decision Regressor is:", rmse_decision)

print("RMSE for Random Forest Regressor is :", rmse_forest)

print("RMSE for GradientBoostingRegressor is : ", rmse_grad)