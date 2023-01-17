# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
TARGET = 'trip_duration'
df = pd.read_csv("../input/train/train.csv",index_col="id")



#Filtre : on prend uniquement les trajets qui dure moins d'une heure, et les trajets avec des passagers 

df = df[(df['trip_duration']<3600)]

df = df[(df['passenger_count']>0)]

df.shape
def create_dataset(df, features=None, target=TARGET):

    X = df[features]

    y = df[target]

    return X, y
X , y = create_dataset(df,features=df.columns)

X= X.drop(['trip_duration','store_and_fwd_flag','vendor_id','dropoff_datetime'],axis=1)
df_test = pd.read_csv('../input/test/test.csv',index_col='id')

df_test=df_test.drop(['store_and_fwd_flag','vendor_id'],axis=1)
def split_date(df,name_column):

        date=pd.to_datetime(df[name_column])

        df[name_column+'_second']=date.dt.second

        df[name_column+'_minute']=date.dt.minute

        df[name_column+'_hour']=date.dt.hour

        df[name_column+'_year']=date.dt.year

        df[name_column+'_day']=date.dt.day

        df= df.drop([name_column],axis=1)

        return df

        
X = split_date(X,'pickup_datetime')

df_test = split_date(df_test,'pickup_datetime')
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

from sklearn.model_selection import (

    cross_val_score, ShuffleSplit)

from sklearn.metrics import accuracy_score
rs = ShuffleSplit(n_splits=3, test_size=.25, train_size=.12)

cv_scores = -cross_val_score(rf, X, y, cv=rs)

cv_scores.mean()
rf.fit(X,y)
df_test.head()
y_pred= rf.predict(df_test)

y_pred.mean()
submission = pd.read_csv('../input/sample_submission/sample_submission.csv')

submission['trip_duration']=y_pred

submission.to_csv('submission.csv',index=False)