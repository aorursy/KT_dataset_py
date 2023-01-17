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
train_path='../input/train.csv'
with open(train_path) as file:
    n_rows=len(file.readlines())
print(n_rows)
train=pd.read_csv(train_path,nrows=5)
train.head()
# train_path='../input/train.csv'
# traintypes = {'fare_amount': 'float32',
#               'pickup_datetime': 'str', 
#               'pickup_longitude': 'float32',
#               'pickup_latitude': 'float32',
#               'dropoff_longitude': 'float32',
#               'dropoff_latitude': 'float32',
#               'passenger_count': 'uint8'}
# train=pd.read_csv(train_path,usecols=list(traintypes.keys()),dtype=traintypes,nrows=100000)
train=pd.read_csv('../input/train.csv',nrows=100000)
train.shape
train.head()
train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'])
train['year']=train['pickup_datetime'].dt.year
train['month']=train['pickup_datetime'].dt.month
train['weekdayname']=train['pickup_datetime'].dt.weekday_name
train['weekday']=train['pickup_datetime'].dt.weekday
train['hour']=train['pickup_datetime'].dt.hour
train.head()
import matplotlib.pyplot as plt
import seaborn as sns
fig, axarr = plt.subplots(2, 2, figsize=(15, 15))
plt.subplots_adjust(hspace=.3)
sns.barplot(x='year',y='fare_amount',data=train,ax=axarr[0][0])
sns.barplot(x='month',y='fare_amount',data=train,ax=axarr[0][1])
sns.barplot(x='weekdayname',y='fare_amount',data=train,ax=axarr[1][0])
sns.barplot(x='hour',y='fare_amount',data=train,ax=axarr[1][1])
train['abs_longi']=abs(train['pickup_longitude']-train['dropoff_longitude'])
train['abs_lat']=abs(train['pickup_latitude']-train['dropoff_latitude'])
train.head()
train['abs_lat'].plot.hist()
train['abs_longi'].plot.hist()
train=train[(train['abs_longi']<50) & (train['abs_lat']<40)]
train.shape
train['fare_amount'].plot.hist()
train=train[train['fare_amount']>0]
train.plot.scatter('passenger_count','fare_amount')
train=train[train['passenger_count']>0]
train.shape
test_path='../input/test.csv'
with open(test_path) as file:
    n_rows=len(file.readlines())
print(n_rows)
test=pd.read_csv('../input/test.csv')
test.head()
test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'])
test['year']=test['pickup_datetime'].dt.year
test['month']=test['pickup_datetime'].dt.month
test['weekdayname']=test['pickup_datetime'].dt.weekday_name
test['weekday']=test['pickup_datetime'].dt.weekday
test['hour']=test['pickup_datetime'].dt.hour
test['abs_longi']=abs(test['pickup_longitude']-test['dropoff_longitude'])
test['abs_lat']=abs(test['pickup_latitude']-test['dropoff_latitude'])
test.head()
train.dropna()
train.shape
use_cols=['passenger_count','year','month','weekday','hour','abs_longi','abs_lat']
y=train['fare_amount']
X_train=train[use_cols]
X_test=train[use_cols]
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y)
model.predict(X_test)