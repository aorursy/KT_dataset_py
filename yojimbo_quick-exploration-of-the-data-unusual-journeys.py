# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

train.shape
days = ['Mon', 'Tue', 'Wed', 'Thr', 'Fri', 'Sat', 'Sun']
train['day'] = pd.to_datetime(train['pickup_datetime']).dt.weekday

train['date'] = pd.to_datetime(train['pickup_datetime']).dt.date

train['hour'] = pd.to_datetime(train['pickup_datetime']).dt.hour

train['rushhour'] = np.where(train['hour'].isin([7,8,16,17]) & train['day'].isin([0,1,2,3,4,]), 'yes', 'no')

train['days'] = train['day'].apply(lambda x: days[x])
##is there a dfference between rushhour and non rush hour journey mean



pd.pivot_table(train[['rushhour','day','trip_duration']], 

               index='day', columns='rushhour', aggfunc='mean').plot.bar()
## number of journeys per day

pd.pivot_table(train[['id','days']], columns='days', aggfunc="count")
## number of journeys per day/hour

pd.pivot_table(train[['hour','days','id']], index='hour', columns='days', aggfunc="count").plot.line(figsize=(10,8))
#mean journey time by hour and day

pd.pivot_table(train[['hour','day','trip_duration']], index='hour', columns='day', aggfunc="mean").plot.line(figsize=(10,8))
#lets just look at trips on a saturday



dayview = train.loc[train['days'] == 'Sat']

pd.pivot_table(dayview[['date','hour','trip_duration']],

               index='hour', columns='date', aggfunc='mean').plot.line(figsize=(10,8), legend=False)
pd.pivot_table(dayview[['date','hour','trip_duration']].loc[dayview['hour']==22],

               index='hour', columns='date', aggfunc='mean').sum()
dayview.loc[(dayview['date'] == dt.date(2016, 2, 13)) & (dayview['trip_duration'] >12000)].describe()
3526282/3600/24
train.loc[train['trip_duration'] >= 18000].describe()
#mean journey time by hour and day , journeys less than 5 hours

pd.pivot_table(train[['hour','days','trip_duration']].loc[train['trip_duration'] <= 18000],

                                                         index='hour',

                                                         columns='days',

                                                         aggfunc="mean").plot.line(figsize=(10,8))
mondayview = train.loc[(train['days'] == 'Mon')]# & (train['trip_duration'] < 18000)]

pd.pivot_table(mondayview[['date','hour','trip_duration']],

               index='hour', columns='date', aggfunc='mean').plot.line(figsize=(10,8), legend=False)
mondayview.loc[(mondayview['trip_duration'] < mondayview.loc[mondayview['hour'] ==

                                                    15]['trip_duration'].mean()) &

            (mondayview['hour'] == 15)].sort_values('trip_duration')
train.loc[train['trip_duration'] <= 15].describe()
trained = train.loc[(train['trip_duration'] > 15) & (train['trip_duration'] < 18000) ].copy()
#mean journey time by hour and day

pd.pivot_table(trained[['hour','days','trip_duration']],

                index='hour',

                columns='days',

                aggfunc="mean").plot.line(figsize=(10,8))
##do the rush hours look any fi=different with these stripped out



pd.pivot_table(train[['rushhour','day','trip_duration']], 

               index='day', columns='rushhour', aggfunc='mean').plot.bar()