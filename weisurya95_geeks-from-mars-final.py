!pip install tsfresh

!pip install requests tabulate future

!pip install "colorama>=0.3.8"

!pip install h2o
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tsfresh

import h2o

from h2o.automl import H2OAutoML

import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
sub1 = pd.read_csv('../input/ensemble/submission_autoreg.csv')

sub2 = pd.read_csv('../input/ensemble/test_sub_1.csv')

sub3 = pd.read_csv('../input/ensemble/test_sub_9.csv')
sub = sub1.merge(sub2, on='Id').merge(sub3, on='Id')
sub['Incidents'] = (sub['Incidents'] + sub['Incidents_x'] + sub['Incidents_y'])/3
sub['Incidents'] = np.where(sub.Incidents < 0, 0, sub.Incidents)
sub[['Id', 'Incidents']].to_csv('ensemble.csv', index=False)
ERROR
# train = pd.read_csv('../input/train.csv')

# test = pd.read_csv('../input/test.csv')

# hexagon = pd.read_csv("../input/hexagon_centers.csv")

# landmarks = pd.read_csv("../input/landmarks.csv")

# public_holidays = pd.read_csv("../input/public_holidays.csv")

# weather = pd.read_csv("../input/weather.csv")

# temp = pd.read_csv("../input/temperatures.csv")
train_melted = train.iloc[:,1:].melt(var_name='hexagon_id', value_name='crimes')

# train_melted = train_melted.merge(df, left_on='hexagon_id', right_on='hexagon_id', how='left')



# train_melted['date'] = pd.to_datetime(train['Date'], format='%Y/%m/%d %H:%M:%S').tolist() * 319

# train_melted['date_only'] = pd.DatetimeIndex(train['Date']).date.tolist() * 319

train_melted['year'] = pd.to_datetime(train['Date']).dt.year.tolist() * 319

train_melted['year'].replace([2015, 2016, 2017, 2018], [1, 2, 3, 4], inplace = True)

train_melted['year'] = pd.Categorical(train_melted['year'])



train_melted['month'] = pd.to_datetime(train['Date']).dt.month.tolist() * 319

train_melted['month'] = pd.Categorical(train_melted['month'])



train_melted['day'] = pd.to_datetime(train['Date']).dt.day.tolist() * 319

train_melted['day'] = pd.Categorical(train_melted['day'])



train_melted['dayofweek'] = pd.to_datetime(train['Date']).dt.dayofweek.tolist() * 319

train_melted['dayofweek'] = pd.Categorical(train_melted['dayofweek'])



train_melted['hour'] = pd.to_datetime(train['Date']).dt.hour.tolist() * 319

train_melted['hour'].replace([6, 14, 22], [1, 2, 3], inplace = True)

train_melted['hour'] = pd.Categorical(train_melted['hour'])



train_melted['hexagon_id'] = pd.Categorical(train_melted['hexagon_id'])

train_melted['hexagon_id'] = train_melted['hexagon_id'].cat.codes
test_melted = test.iloc[:,1:].melt(var_name='hexagon_id', value_name='crimes')

# test_melted = test_melted.merge(df, left_on='hexagon_id', right_on='hexagon_id', how='left')



# test_melted['date'] = pd.to_datetime(test['Date'], format='%Y/%m/%d %H:%M:%S').tolist() * 319

# test_melted['date_only'] = pd.DatetimeIndex(test['Date']).date.tolist() * 319

test_melted['year'] = pd.to_datetime(test['Date']).dt.year.tolist() * 319

test_melted['year'].replace([2015, 2016, 2017, 2018], [1, 2, 3, 4], inplace = True)

test_melted['year'] = pd.Categorical(test_melted['year'])



test_melted['month'] = pd.to_datetime(test['Date']).dt.month.tolist() * 319

test_melted['month'] = pd.Categorical(test_melted['month'])



test_melted['day'] = pd.to_datetime(test['Date']).dt.day.tolist() * 319

test_melted['day'] = pd.Categorical(test_melted['day'])



test_melted['dayofweek'] = pd.to_datetime(test['Date']).dt.dayofweek.tolist() * 319

test_melted['dayofweek'] = pd.Categorical(test_melted['dayofweek'])



test_melted['hour'] = pd.to_datetime(test['Date']).dt.hour.tolist() * 319

test_melted['hour'].replace([6, 14, 22], [1, 2, 3], inplace = True)

test_melted['hour'] = pd.Categorical(test_melted['hour'])



test_melted['hexagon_id'] = pd.Categorical(test_melted['hexagon_id'])

test_melted['hexagon_id'] = test_melted['hexagon_id'].cat.codes
# group_dayofweek = train_melted.groupby('dayofweek').crimes.mean().reset_index().rename(columns={ 'crimes': 'mean_dayofweek' })

# group_day = train_melted.groupby('day').crimes.mean().reset_index().rename(columns={ 'crimes': 'mean_day' })

# group_hexagonid = train_melted.groupby('hexagon_id').crimes.mean().reset_index().rename(columns={ 'crimes': 'mean_hexagon_id' })

# group_hour = train_melted.groupby('hour').crimes.mean().reset_index().rename(columns={ 'crimes': 'mean_hour' })

# group_month = train_melted.groupby('month').crimes.mean().reset_index().rename(columns={ 'crimes': 'mean_month' })

# group_year = train_melted.groupby('year').crimes.mean().reset_index().rename(columns={ 'crimes': 'mean_year' })
# train_melted = train_melted.merge(group_dayofweek, on='dayofweek')

# train_melted = train_melted.merge(group_day, on='day')

# train_melted = train_melted.merge(group_hexagonid, on='hexagon_id')

# train_melted = train_melted.merge(group_hour, on='hour')

# train_melted = train_melted.merge(group_month, on='month')

# train_melted = train_melted.merge(group_year, on='year')
train_melted.dtypes
# test_melted = test_melted.merge(group_dayofweek, on='dayofweek')

# test_melted = test_melted.merge(group_day, on='day')

# test_melted = test_melted.merge(group_hexagonid, on='hexagon_id')

# test_melted = test_melted.merge(group_hour, on='hour')

# test_melted = test_melted.merge(group_month, on='month')

# test_melted = test_melted.merge(group_year, on='year')
test_melted.dtypes
train_set = train_melted[:-28710]

validation_set = train_melted[-28710:]
x_train = train_set

y_train = train_set[['crimes']]



x_validation = validation_set

y_validation = validation_set[['crimes']]
h2o.init()
x_train_h2o = h2o.H2OFrame(train_set)

x_validation_h2o = h2o.H2OFrame(validation_set)
aml = H2OAutoML(max_models = 10, seed = 1, max_runtime_secs=600, stopping_metric='RMSE', sort_metric='RMSE',

               nfolds=0)

aml.train(y = 'crimes', x=x_train_h2o.drop('crimes',1).columns, training_frame = x_train_h2o,

         leaderboard_frame=x_validation_h2o)
aml.leaderboard
preds = aml.leader.predict(x_validation_h2o)

preds = preds.as_data_frame()
test_h2o = h2o.H2OFrame(test_melted.drop('crimes', 1))
preds = aml.leader.predict(test_h2o)

preds = preds.as_data_frame()
ss = pd.read_csv('../input/sampleSubmission.csv')
ss.dtypes
preds['predict'].astype
ss['Incidents'] = preds['predict']
ss.head()
ss.to_csv('test_sub.csv', index=False)