# !pip install tsfresh

# !pip install requests tabulate future

# !pip install "colorama>=0.3.8"

# !pip install h2o
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import tsfresh

# import h2o

# from h2o.automl import H2OAutoML

import datetime

import lightgbm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

hexagon = pd.read_csv("../input/hexagon_centers.csv")

landmarks = pd.read_csv("../input/landmarks.csv")

public_holidays = pd.read_csv("../input/public_holidays.csv")

weather = pd.read_csv("../input/weather.csv")

temp = pd.read_csv("../input/temperatures.csv")
train_melted = train.iloc[:,1:].melt(var_name='hexagon_id', value_name='crimes')

# train_melted = train_melted.merge(df, left_on='hexagon_id', right_on='hexagon_id', how='left')



train_melted['date'] = pd.to_datetime(train['Date'], format='%Y/%m/%d %H:%M:%S').tolist() * 319

train_melted['date_only'] = pd.DatetimeIndex(train['Date']).date.tolist() * 319

train_melted['year'] = pd.to_datetime(train['Date']).dt.year.tolist() * 319

train_melted['year'].replace([2015, 2016, 2017, 2018], [1, 2, 3, 4], inplace = True)

train_melted['month'] = pd.to_datetime(train['Date']).dt.month.tolist() * 319

train_melted['day'] = pd.to_datetime(train['Date']).dt.day.tolist() * 319

train_melted['dayofweek'] = pd.to_datetime(train['Date']).dt.dayofweek.tolist() * 319

train_melted['hour'] = pd.to_datetime(train['Date']).dt.hour.tolist() * 319

train_melted['hour'].replace([6, 14, 22], [1, 2, 3], inplace = True)



train_melted['hexagon_id'] = pd.Categorical(train_melted['hexagon_id'])

train_melted['hexagon_id'] = train_melted['hexagon_id'].cat.codes
test_melted = test.iloc[:,1:].melt(var_name='hexagon_id', value_name='crimes')

# test_melted = test_melted.merge(df, left_on='hexagon_id', right_on='hexagon_id', how='left')



test_melted['date'] = pd.to_datetime(test['Date'], format='%Y/%m/%d %H:%M:%S').tolist() * 319

test_melted['date_only'] = pd.DatetimeIndex(test['Date']).date.tolist() * 319

test_melted['year'] = pd.to_datetime(test['Date']).dt.year.tolist() * 319

test_melted['year'].replace([2015, 2016, 2017, 2018], [1, 2, 3, 4], inplace = True)

test_melted['month'] = pd.to_datetime(test['Date']).dt.month.tolist() * 319

test_melted['day'] = pd.to_datetime(test['Date']).dt.day.tolist() * 319

test_melted['dayofweek'] = pd.to_datetime(test['Date']).dt.dayofweek.tolist() * 319

test_melted['hour'] = pd.to_datetime(test['Date']).dt.hour.tolist() * 319

test_melted['hour'].replace([6, 14, 22], [1, 2, 3], inplace = True)



test_melted['hexagon_id'] = pd.Categorical(test_melted['hexagon_id'])

test_melted['hexagon_id'] = test_melted['hexagon_id'].cat.codes
# train_melted.drop(columns=['date', 'date_only'], inplace=True)

# test_melted.drop(columns=['date', 'date_only'], inplace=True)
group_dayofweek = train_melted.groupby('dayofweek').crimes.mean().reset_index().rename(columns={ 'crimes': 'mean_dayofweek' })

group_day = train_melted.groupby('day').crimes.mean().reset_index().rename(columns={ 'crimes': 'mean_day' })

group_hexagonid = train_melted.groupby('hexagon_id').crimes.mean().reset_index().rename(columns={ 'crimes': 'mean_hexagon_id' })

group_hour = train_melted.groupby('hour').crimes.mean().reset_index().rename(columns={ 'crimes': 'mean_hour' })

group_month = train_melted.groupby('month').crimes.mean().reset_index().rename(columns={ 'crimes': 'mean_month' })

group_year = train_melted.groupby('year').crimes.mean().reset_index().rename(columns={ 'crimes': 'mean_year' })
train_melted = train_melted.merge(group_dayofweek, on='dayofweek')

train_melted = train_melted.merge(group_day, on='day')

train_melted = train_melted.merge(group_hexagonid, on='hexagon_id')

train_melted = train_melted.merge(group_hour, on='hour')

train_melted = train_melted.merge(group_month, on='month')

train_melted = train_melted.merge(group_year, on='year')
train_melted = train_melted.sort_values(['hexagon_id', 'date'])
test_melted = test_melted.merge(group_dayofweek, on='dayofweek')

print(test_melted.shape)

test_melted = test_melted.merge(group_day, on='day')

print(test_melted.shape)

test_melted = test_melted.merge(group_hexagonid, on='hexagon_id')

print(test_melted.shape)

test_melted = test_melted.merge(group_hour, on='hour')

print(test_melted.shape)

test_melted = test_melted.merge(group_month, on='month')

print(test_melted.shape)

test_melted = test_melted.merge(group_year, on='year', how='left')

print(test_melted.shape)
test_melted['mean_year'] = test_melted.mean_year.fillna(group_year[group_year.year==4].mean_year.values[0])

test_melted = test_melted.sort_values(['hexagon_id', 'date'])

print(test_melted.shape)
#Hitesh Code



for i in range(landmarks.shape[0]):

    if len(landmarks.loc[i,'hexagon_id']) < 7:

        landmarks.loc[i,'hexagon_id'] = f"hex_0{(landmarks.loc[i,'hexagon_id'].split('_')[1])}"

        

df = landmarks.merge(right=hexagon, how = 'outer', left_on='hexagon_id', right_on='hex_id')



weather.time = pd.to_datetime(weather.time)

temp.time = pd.to_datetime(temp.time)

weather.time = weather.time.dt.tz_localize(tz=None)

temp.time = temp.time.dt.tz_localize(tz=None)

        

df_climate = weather.merge(right=temp, how = 'outer', right_on='time', left_on='time')



df_climate['date']=df_climate['time'].dt.date

    

public_holidays = public_holidays.rename(columns={'date':'public_holiday_date'})

public_holidays['public_holiday_date'] = pd.to_datetime(public_holidays['public_holiday_date'])

public_holidays['public_holiday_date'] = public_holidays['public_holiday_date'].dt.date



df_climate = df_climate.merge(right=public_holidays, how='left', right_on='public_holiday_date', left_on='date')



df_climate.loc[df_climate['public_holiday_date'].notna(),'public_holiday_date'] = 1

df_climate.loc[df_climate['public_holiday_date'].isna(),'public_holiday_date'] = 0



# So you have two dataframes 'df' and 'df_climate'. You can merge 'df' with train and test on 'hexagon_id' and 'df_climate' with train and test on 

# 'Date'
df['hexagon_id'] = pd.Categorical(df['hexagon_id'])

df['hexagon_id'] = df['hexagon_id'].cat.codes



train_melted = train_melted.merge(df, how='left', left_on='hexagon_id', right_on='hexagon_id')

test_melted = test_melted.merge(df, how='left', left_on='hexagon_id', right_on='hexagon_id')



train_melted['date']=pd.to_datetime(train_melted['date'])

test_melted['date']=pd.to_datetime(test_melted['date'])



train_melted = train_melted.merge(df_climate, how = 'left', left_on='date', right_on='time')

test_melted = test_melted.merge(df_climate, how = 'left', left_on='date', right_on='time')
train_melted.drop(columns=['hex_id', 'gust', 'wdir', 'date_y'], inplace=True)

train_melted.rename(columns={'date_x': 'date'}, inplace=True)

test_melted.drop(columns=['hex_id', 'gust', 'wdir', 'date_y'], inplace=True)

test_melted.rename(columns={'date_x': 'date'}, inplace=True)
print(test_melted.shape)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# le.fit_transform(train_melted['clds'])

# le.fit_transform(train_melted['day_ind'])

# le.fit_transform(train_melted['uv_desc'])

# le.fit_transform(train_melted['wdir_cardinal'])

# le.fit_transform(train_melted['wx_phrase'])
# cat_cols = ['month','day', 'dayofweek', 'hour']

# for col in cat_cols:

#     le.fit_transform(train_melted[col])

#     le.transform(test_melted[col])
train_melted = train_melted.drop(columns=['date', 'date_only', 'time', 'clds', 'day_ind', 'uv_desc', 'wdir_cardinal', 'wx_phrase']

, axis=1)
test_melted = test_melted.drop(columns=['date', 'date_only', 'time', 'clds', 'day_ind', 'uv_desc', 'wdir_cardinal', 'wx_phrase']

, axis=1)
train_melted.head()
train_melted = train_melted.drop(columns=['dewPt', 'heat_index', 'icon_extd', 'pressure', 'rh', 'uv_index', 'wc', 'wx_icon', 'temperature', 'wspd']

, axis=1)

test_melted = test_melted.drop(columns=['dewPt', 'heat_index', 'icon_extd', 'pressure', 'rh', 'uv_index', 'wc', 'wx_icon', 'temperature', 'wspd']

, axis=1)
train_set = train_melted[:-28710]

validation_set = train_melted[-28710:]
x_train = train_set

y_train = train_set[['crimes']]



x_validation = validation_set

y_validation = validation_set[['crimes']]
cat_cols = ['month','day', 'dayofweek', 'hour']

lgbmr=lightgbm.LGBMRegressor()

lgbmr.fit(x_train,y_train, categorical_feature=cat_cols)
val_results = lgbmr.predict(x_validation)
test_results = lgbmr.predict(test_melted)
len(test_results.tolist())
ss = pd.read_csv('../input/sampleSubmission.csv')
ss['Incidents'] = test_results.tolist()
ss.to_csv('test_sub.csv', index=False)