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
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')
train.head()
train.info()
test.head()
test.info()
train['date']=pd.to_datetime(train['date'])
test['date']=pd.to_datetime(test['date'])
def date_format(df):
    df['year']=df['date'].dt.year
    df['month']=df['date'].dt.month
    df['day']=df['date'].dt.day
    return df 
date_format(train)
date_format(test)
train['store'].unique()
train['item'].unique()
train['store_item']=train['store'].astype(str)+'_'+train['item'].astype(str)
test['store_item']=test['store'].astype(str)+'_'+test['item'].astype(str)
train.head()
test.head()
train_x=train[['store','item','year','month','day','store_item']]
train_y=train['sales']
test_x=test[['store','item','year','month','day','store_item']]
reg=LinearRegression()
reg.fit(train_x,train_y)
test_y=reg.predict(test_x)
sample['sales'] = test_y
sample.to_csv('simple_starter.csv', index=False)
