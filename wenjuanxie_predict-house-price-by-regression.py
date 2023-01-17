import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

alldata = [train,test]
train.head()
print(train.shape)

print(train.isnull().sum()[train.isnull().sum()>0])

print(test.shape)

print(test.isnull().sum()[test.isnull().sum()>0])
drop_cols = ['Alley','PoolQC','Fence','MiscFeature','FireplaceQu']

for data in alldata:

    data.drop(drop_cols,axis=1,inplace=True)
cat_cols = train.select_dtypes(['object']).columns

numeric_cols = test.select_dtypes(exclude=['object']).columns
train_cat_lack = train[cat_cols].isnull().sum()[train[cat_cols].isnull().sum()>0]
test_cat_lack=test[cat_cols].isnull().sum()[test[cat_cols].isnull().sum()>0]
commen_cat_lack = train[cat_cols].isnull().sum()[train[cat_cols].isnull().sum()>0]+test[cat_cols].isnull().sum()[test[cat_cols].isnull().sum()>0]

commen_cat_lack=commen_cat_lack[commen_cat_lack.notnull()].index

for data in alldata:

    data['MasVnrType']=data['MasVnrType'].fillna('None')
for data in alldata:

    for col in commen_cat_lack:

        data[col]=data[col].fillna('Unkown')
train['Electrical']=train['Electrical'].fillna(train['Electrical'].mode().iloc[0])
test[cat_cols].isnull().sum()[test[cat_cols].isnull().sum()>0]

for col in ['MSZoning','Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType']:

    test[col] = test[col].fillna(test[col].mode().iloc[0])
train_num_lack = train[numeric_cols].isnull().sum()[train[numeric_cols].isnull().sum()>0].index

test_num_lack = test[numeric_cols].isnull().sum()[test[numeric_cols].isnull().sum()>0].index

for data in alldata:

    for col in test_num_lack:

        data[col] = data[col].fillna(data[col].median())
from sklearn.preprocessing import LabelEncoder
Encoder = LabelEncoder()

train_Encoded = train

test_Encoded = test

for col in cat_cols:

    train_Encoded[col]=Encoder.fit_transform(train[col])

    test_Encoded[col] = Encoder.transform(test[col])
train_X = train_Encoded.drop('SalePrice',axis=1)

train_y = train_Encoded['SalePrice']

test_X = test_Encoded
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split,cross_val_score
train_final_X,val_final_X,train_final_y,val_final_y = train_test_split(train_X,train_y,random_state=0,test_size=0.2)
regressors = [LinearRegression(),

              DecisionTreeRegressor(),

              RandomForestRegressor(n_estimators=100,max_leaf_nodes=50),

              AdaBoostRegressor()]

result = {}

for reg in regressors:

    name = reg.__class__.__name__

    reg.fit(train_final_X,train_final_y)

    score = cross_val_score(reg,train_final_X,train_final_y,cv=5).mean()

    result[name]=score
result
model = RandomForestRegressor(n_estimators=100,max_leaf_nodes=50)

model.fit(train_X,train_y)

test_pred = model.predict(test_X)
submit = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

submit.head()
submission = pd.DataFrame({'Id':test_X['Id'],

                           'SalePrice':test_pred})

submission.to_csv('submission_1.csv',index=False)