# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("/kaggle/input/santander-value-prediction-challenge/train.csv")

test=pd.read_csv("/kaggle/input/santander-value-prediction-challenge/test.csv")
train

y=train['target']

train.drop(['target','ID'],axis=1,inplace=True)

test_ID=test['ID']

test.drop(['ID'],axis=1,inplace=True)

y=np.log1p(y)
print(test.shape)

print(train.shape)



from sklearn.model_selection import train_test_split

from sklearn import ensemble
x1,x2,y1,y2=train_test_split(train,y)
mod=ensemble.RandomForestRegressor(n_jobs=-1,n_estimators=200,min_samples_leaf=10,max_features=600,random_state=10)

mod.fit(x1,y1)

pred=mod.predict(x2)

from sklearn.metrics import mean_squared_error

print(np.sqrt(np.mean(np.power(y2 - pred, 2))))
import xgboost as xgb

import lightgbm as lgb

mod1=xgb.XGBRegressor()

mod1.fit(train,y)

pred1=mod1.predict(test)

#print(np.sqrt(np.mean(np.power(y2 - pred, 2))))
mod2=lgb.LGBMRegressor()

mod2.fit(train,y)

pred2=mod2.predict(test)

#print(np.sqrt(np.mean(np.power(y2 - pred, 2))))
ans=(np.expm1(pred1)+np.expm1(pred2))/2

sub = pd.DataFrame()

sub['ID'] = test_ID

sub['target'] = ensemble

sub.to_csv('submission.csv',index=False)