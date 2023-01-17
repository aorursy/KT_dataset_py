# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
csv_train=pd.read_csv('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv')

csv_test=pd.read_csv('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv')

csv_submission=pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv')

print(csv_train.head())

print(csv_test.head())

print(csv_submission.head())
from pandas import Series

dates=[]

years=[]

months=[]

for date in csv_train['year']:

  mm=date%10000

  dd=mm%100

  mm=mm//100

  months.append(mm)

  dates.append(dd)

  years.append(date//10000%100)



dates_test=[]

years_test=[]

months_test=[]

for date in csv_test['year']:

  mm=date%10000

  dd=mm%100

  mm=mm//100

  dates_test.append(dd)

  years_test.append(date//10000%100)

  months_test.append(mm)

    

csv_train['day']=Series(data=dates)

csv_train['year']=Series(data=years)

csv_train['month']=Series(data=months)



csv_test['day']=Series(data=dates_test)

csv_test['year']=Series(data=years_test)

csv_test['month']=Series(data=months_test)



print(csv_train.head())

print(csv_test.head())

print(csv_submission.head())
from sklearn.preprocessing import StandardScaler



x_train=csv_train.drop('avgPrice', axis=1)

x_train=x_train.drop('year', axis=1)

x_train=x_train.drop('day', axis=1)

y_train=csv_train['avgPrice']

x_test=csv_test

x_test=x_test.drop('year', axis=1)

x_test=x_test.drop('day', axis=1)

sc=StandardScaler()

sc.fit(x_train)



x_train_std = sc.transform(x_train)

x_test_std = sc.transform(x_test)
from sklearn.neighbors import KNeighborsRegressor



knn = KNeighborsRegressor()

knn.fit(x_train_std, y_train)



y_predict_train = knn.predict(x_train)

y_predict_test = knn.predict(x_test)
from sklearn.metrics import mean_squared_error

import math



mean_squared_error(y_train, y_predict_train)

math.sqrt(mean_squared_error(y_train, y_predict_train))
for idx, y in enumerate(y_predict_test.astype(int)):

  csv_submission['Expected'][idx]=y



print(csv_submission.head())

csv_submission.dtypes
csv_submission.to_csv('submission.csv',index=False)