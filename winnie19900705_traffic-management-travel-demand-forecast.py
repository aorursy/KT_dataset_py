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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



df_train = pd.read_csv('../input/training.csv')

df_train.head()
df_train.shape
len(df_train.geohash6.unique())
df_train['hours'] = df_train['timestamp'].map(lambda x: int(x.split(':')[0]))

df_train['mins'] = df_train['timestamp'].map(lambda x: int(x.split(':')[1]))

df_train.head()
df_train['time'] = 24*60*(df_train['day']-1) + 60*df_train['hours'] + df_train['mins']

df_train.head()
import Geohash

df_train['Latitude'] = df_train.geohash6.map(lambda x: float(Geohash.decode_exactly(x)[0]))

df_train['Longitude'] = df_train.geohash6.map(lambda x: float(Geohash.decode_exactly(x)[1]))

df_train = df_train.sort_values(by=['time','Latitude','Longitude'], ascending=True)

df_train = df_train.reset_index().drop('index',axis=1)

df_train.head()
df_train[['geohash6','demand']].groupby('geohash6').count().head(10)
max_day = df_train.day.max()

max_time = df_train.time.max()

train_start = df_train[df_train.day==61-13].index[0]

test_start = df_train[df_train.time==max_time-15*4].index[0]



Xtrain = df_train[['time', 'Latitude','Longitude']].iloc[train_start:test_start,:]

Xtest = df_train[['time', 'Latitude','Longitude']].iloc[test_start:,:]



ytrain = df_train.demand.iloc[train_start:test_start]

ytest = df_train.demand.iloc[test_start:]
Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error



model = RandomForestRegressor(n_estimators=30, max_depth=40)

model.fit(Xtrain, ytrain)

ytest_pred = model.predict(Xtest)

rmse = np.sqrt(mean_squared_error(ytest, ytest_pred))

print('RMSE:',rmse)
from xgboost import XGBRegressor



model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=35)

model.fit(Xtrain, ytrain)

ytest_pred = model.predict(Xtest)

rmse = np.sqrt(mean_squared_error(ytest, ytest_pred))

print('RMSE:',rmse)
def convert_time(time):

    day = int(time/(24*60)) + 1

    hour = int((time-(day-1)*24*60)/60)

    minute = time-(day-1)*24*60-hour*60

    timestamp = ':'.join((str(hour),str(minute)))

    return (day, hour, minute, timestamp)
def predict5ts(link, n_estimators=500, learning_rate=0.05, max_depth=35):

    df = pd.read_csv(link)

    df['hours'] = df['timestamp'].map(lambda x: int(x.split(':')[0]))

    df['mins'] = df['timestamp'].map(lambda x: int(x.split(':')[1]))

    df['time'] = 24*60*(df['day']-1) + 60*df['hours'] + df['mins']

    

    import Geohash

    df['Latitude'] = df.geohash6.map(lambda x: float(Geohash.decode_exactly(x)[0]))

    df['Longitude'] = df.geohash6.map(lambda x: float(Geohash.decode_exactly(x)[1]))



    df = df.sort_values(by=['time','Latitude','Longitude'], ascending=True)

    df = df.reset_index().drop('index',axis=1)

    

    X = df[['time', 'Latitude','Longitude']]

    y = df.demand

    

    from xgboost import XGBRegressor

    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

    model.fit(X, y)

    

    T = df.time.max()

    T1 = T+15

    T2 = T+15*2

    T3 = T+15*3

    T4 = T+15*4

    T5 = T+15*5

    

    geohashes = df_train.geohash6.unique()

    geohashes2 = []

    latitudes = []

    longitudes = []

    times = []

    days = []

    timestamps = []



    for t in (T1,T2,T3,T4,T5):

        for gh in geohashes:

            geohashes2.append(gh)

            latitudes.append(float(Geohash.decode_exactly(gh)[0]))

            longitudes.append(float(Geohash.decode_exactly(gh)[1]))

            times.append(t)

            days.append(convert_time(t)[0])

            timestamps.append(convert_time(t)[-1])



    df_pred = pd.DataFrame({'geohash6': geohashes2, 'day': days, 'timestamp': timestamps,

                        'time': times, 'Latitude': latitudes, 'Longitude': longitudes})

    Xtest = df_pred[['time', 'Latitude','Longitude']]

    ypred = model.predict(Xtest)



    df_pred['demand'] = ypred

    output = df_pred[['geohash6', 'day', 'timestamp', 'demand']]

    output.to_csv('output.csv', index=False)
df_trial = df_train[['geohash6','day','timestamp','demand']].iloc[-20000:,:]

df_trial.to_csv('df_trial.csv', index=False)



trial_link = 'df_trial.csv'

predict5ts(link=trial_link)



output = pd.read_csv('output.csv')

print(output.shape)

output.head()
os.remove("df_trial.csv")

os.remove("output.csv")
#test_link = '...'

#predict5ts(link=test_link)