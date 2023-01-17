import os

import numpy as np

import pandas as pd

train=pd.read_csv('../input/train_dataset.csv')

train.head()
test=pd.read_csv('../input/test_dataset.csv')

sub=pd.read_csv('../input/submit_example.csv')

test.head()
train_id=train['用户编码']

test_id=test['用户编码']

train_label=train['信用分']

train.drop(['用户编码','信用分'],axis=1,inplace=True)



train.shape
test.drop(['用户编码'],axis=1,inplace=True)

test.shape
total_data=pd.concat([train,test],axis=0)

total_data.shape
object_cols=['用户实名制是否通过核实','是否大学生客户', '是否黑名单客户', '是否4G不健康客户','缴费用户当前是否欠费缴费', '用户话费敏感度',

            '是否经常逛商场的人','当月是否逛过福州仓山万达', '当月是否到过福州山姆会员店', '当月是否看电影', '当月是否景点游览',

       '当月是否体育场馆消费']

object_cols
for i in object_cols:

    total_data[i]=total_data[i].astype('object')
new_total_data=pd.get_dummies(total_data)
train=new_total_data.iloc[:50000,:]

test=new_total_data.iloc[50000:,:]

train.shape
test.shape
from sklearn.model_selection import KFold,cross_val_score,train_test_split

#xtrain,xval,ytrain,yval=train_test_split(train,train_label,test_size=0.2,random_state=0)

kf=KFold(n_splits=5,random_state=1,shuffle=False)

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor

    

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor









lgbm=LGBMRegressor(learning_rate=0.001,n_estimators=5000,random_state=3).fit(train,train_label)

lgbm_pred=lgbm.predict(test)

lgbm_pred
sub.head()
test_id=pd.DataFrame(test_id)

test_id.head()
score=pd.DataFrame(lgbm_pred)

score.head()
result=pd.concat([test_id,score],axis=1)



result.to_csv('result.csv',index=False)