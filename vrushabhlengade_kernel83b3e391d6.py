# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')



df_test = pd.read_csv(r'/kaggle/input/covid19-global-forecasting-week-4/test.csv')
df.head()
df.tail()
df_test.head()
df.shape
df_test.shape
df = df.drop(['Id'],axis = 1)

df.head()
df_test1 = df_test.drop(['ForecastId'],axis = 1)

df_test1.head()
df.describe()
df.dtypes
df_test1.dtypes
tem = ['Province_State','Country_Region','Date']



for i in tem:

    print('-----',i,'--------')

    print(df[i].value_counts())

    print('')

    
df['Province_State'].value_counts()
df['Country_Region'].value_counts()
df.isnull().sum()
df = pd.get_dummies(df.drop(['Date'],axis=1))

df.head()
df['ConfirmedCases']
df_test1 = pd.get_dummies(df_test1.drop(['Date'],axis=1))

df_test1.head()
train_x = df.drop(['ConfirmedCases','Fatalities'],axis=1)

train_y = df[['ConfirmedCases','Fatalities']]



train_x.head()
train_y.head()
test_x = df_test1
test_x
from sklearn.linear_model import LinearRegression as LR

from sklearn.metrics import mean_absolute_error as mae
lr = LR()



lr.fit(train_x,train_y)
train_predict=lr.predict(train_x)



test_predict=lr.predict(test_x)



k = mae(train_predict, train_y)

print("Train Error    ",k)



test_predict
test_predict[:,0]
test_predict[:,1]
test_predict = np.array(test_predict, dtype=np.int64)   ## convert it to int64



output = pd.DataFrame({'ForecastId': df_test.ForecastId, 'ConfirmedCases': test_predict[:,0], 'Fatalities' : test_predict[:,1]})

output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")