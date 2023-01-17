# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
#data.columns
#data.info()
data[['YearBuilt','YrSold']].head()
test[['YearBuilt','YrSold']].head()
data['Year_old']=data['YrSold']-data['YearBuilt']

data['Year_old'].head()
test['Year_old']=test['YrSold']-test['YearBuilt']

test['Year_old'].head()
data.describe(include='object').columns
list_objects=[]

for i in data.describe(include='object').columns:

    if len(data[i].value_counts())==len(test[i].value_counts()):

        list_objects.append(i)
list_objects
data[list_objects].isna().sum()
test[list_objects].isna().sum()
#object_cols=['Street','LotShape','LandContour','LotConfig' ,'LandSlope','Neighborhood','Condition1','BldgType','ExterQual', 'RoofStyle',

             #'ExterCond','Foundation', 'HeatingQC',  'PavedDrive', 'SaleType', 'SaleCondition','KitchenQual','CentralAir']
data['SalePrice'].min()
data['SalePrice'].max()
plt.plot(figsize=(8,6))

sns.distplot(data['SalePrice'],hist=False)
numeric_columns=['LotArea','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea','Year_old']
data[numeric_columns].describe().T
data[numeric_columns].isna().sum()
test[numeric_columns].isna().sum()
from sklearn.tree import DecisionTreeClassifier
miss_cols=['MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageCond']
for i in miss_cols:

    X_train=data[data[i].notna()][numeric_columns].values

    y_train=data[data[i].notna()][i].values

    X_test=data[data[i].isna()][numeric_columns].values

    dtc = DecisionTreeClassifier(random_state=0)

    dtc.fit(X_train, y_train)

    y_pred = dtc.predict(X_test)

    data.loc[data[data[i].isna()][i].index,i]=y_pred
data[list_objects].isna().sum()
test['BsmtFinSF1']=test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].median())

test['BsmtUnfSF']=test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].median())

test['TotalBsmtSF']=test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].median())

test['GarageArea']=test['GarageArea'].fillna(test['GarageArea'].median())
for i in ['MSZoning','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','KitchenQual','Functional','GarageType','GarageFinish','GarageCond','SaleType']:

    X_train=test[test[i].notna()][numeric_columns].values

    y_train=test[test[i].notna()][i].values

    X_test=test[test[i].isna()][numeric_columns].values

    dtc = DecisionTreeClassifier(random_state=0)

    dtc.fit(X_train, y_train)

    y_pred = dtc.predict(X_test)

    test.loc[test[test[i].isna()][i].index,i]=y_pred

    
test[list_objects].isna().sum()
new_list_objects=['MSZoning','Street','LotShape', 'LandContour','LotConfig','LandSlope', 'Neighborhood', 'Condition1','BldgType','RoofStyle', 'MasVnrType', 'ExterQual',

                  'ExterCond','Foundation', 'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','HeatingQC', 'CentralAir','KitchenQual', 'Functional',

                  'GarageType', 'GarageFinish','GarageCond','PavedDrive','SaleType','SaleCondition']
numeric_data_train=data[numeric_columns]

numeric_data_test=test[numeric_columns]

object_data_train=data[new_list_objects]

object_data_test=test[new_list_objects]

objects_train=pd.get_dummies(object_data_train)

objects_test=pd.get_dummies(object_data_test)
X_train=pd.concat([numeric_data_train,objects_train],axis=1).values

X_test=pd.concat([numeric_data_test,objects_test],axis=1).values
print(X_train.shape)

print(X_test.shape)
y_train=data['SalePrice'].values
from sklearn.linear_model import Ridge
import xgboost as xgb
ridge = Ridge(alpha=1, normalize=True) 
train_dmatrix = xgb.DMatrix(data=X_train, label=y_train)



# Creating the parameter dictionary: params

params = {"objective":"reg:linear", "max_depth":4}



# Training the model: xg_reg

xg_reg = xgb.train(params=params, dtrain=train_dmatrix, num_boost_round=10)



# Plotting the feature importances

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(xg_reg, max_num_features=50, height=0.8, ax=ax)

plt.show()

plt.show()
cv_results = xgb.cv(dtrain=train_dmatrix, params=params, nfold=4, num_boost_round=5, metrics='rmse', as_pandas=True, seed=123)



# Print cv_results

print(cv_results)



# Extract and print final boosting round metric

print((cv_results["test-rmse-mean"]).tail(1))
xg_reg = xgb.XGBRegressor(objective='reg:linear',reg_alpha=1,max_depth=4,n_estimators=90,seed=123)
xg_reg.fit(X_train, y_train)


#ridge.fit(X_train,y_train)
y_pred=xg_reg.predict(X_test)
#y_pred=ridge.predict(X_test)
sample=pd.read_csv("../input/sample_submission.csv")
y_pred.min()
y_pred.max()
sample.index
sample['SalePrice']=y_pred
sample.to_csv('sample.csv',index=False)
print(os.listdir("../working"))