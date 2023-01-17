import numpy as np

import pandas as pd

import sklearn 

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn import svm

from sklearn import preprocessing

from pandas import DataFrame

from xgboost import XGBRegressor



import os

print(os.listdir("../input"))

test_data = pd.read_csv('../input/test.csv', index_col = False)

train_data = pd.read_csv('../input/train.csv', index_col = False)

encoded_test = pd.get_dummies(test_data)

hot_encoded = pd.get_dummies(train_data)

saleprice = hot_encoded.drop(columns='SalePrice')

final_train, final_test = saleprice.align(encoded_test, join='left', axis = 1)
y = hot_encoded['SalePrice']

x = final_train

missing_val_count_by_column = (x.isnull().sum())

my_Imputer = SimpleImputer()
def lowestMae(leafs=2):

    x_train, x_val, y_train, y_val = train_test_split(x,y,random_state = 7)

    house_model = RandomForestRegressor(random_state=3)

    x_train.fillna(-999999, inplace=True)

    y_train.fillna(-999999, inplace=True)

    x_val.fillna(-999999, inplace=True)

    house_model.fit(x_train.values.reshape(-1,1),y_train)

    pred1 = house_model.predict(x_val)

    return mean_absolute_error(pred1,y_val)
x_train, x_val, y_train, y_val = train_test_split(x,y,random_state = 7)

house_model = XGBRegressor(n_estimators=1000)

x_train.fillna(-999999, inplace=True)

y_train.fillna(-999999, inplace=True)

x_val.fillna(-999999, inplace=True)

house_model.fit(x_train,y_train, early_stopping_rounds=5,eval_set=[(x_val, y_val)], verbose=False)

x.fillna(-999999,inplace=True)

y.fillna(-999999,inplace=True)

pred1 = house_model.predict(x_val)
val = mean_absolute_error(pred1,y_val)

print(val)
final_test.fillna(-9999999,inplace=True)

final_test

pred2 = house_model.predict(final_test)
df = pd.DataFrame({'Id':test_data['Id'],'SalePrice':pred2})

df.to_csv('third.csv',index=False)

print('Saved')