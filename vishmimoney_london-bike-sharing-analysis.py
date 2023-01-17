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
df = pd.read_csv('/kaggle/input/london-bike-sharing-dataset/london_merged.csv', parse_dates=['timestamp'])

df.tail()
df.shape
df['season'] = df['season'].map({0:'spring', 1:'summer', 2:'fall', 3:'winter'})

df['weather'] = df['weather_code'].map({1:'clear', 2:'scattered clouds', 3:'Broken clouds', 4:'Cloudy', 7:'Light rain', 10:'rain with thunderstorm',26:'snowfall',94:'Freezing Fog'})



df = df.drop(['weather_code'], axis=1)

df.tail()
df.dtypes
cat_cols = ['season', 'weather', 'is_holiday', 'is_weekend']



for col in cat_cols:

    df[col] = df[col].astype('category')

df.dtypes
df.isnull().sum()

df.describe()
plt.figure(figsize=(16,8))



cols = ['is_holiday', 'is_weekend']

for i in range(len(cols)):

    plt.subplot(1,2,i+1)

    df.groupby(cols[i])['cnt'].hist(bins=50,grid=False)

    plt.xlabel(cols[i])

    plt.legend(df[cols[i]].unique())
plt.style.use('ggplot')

fig, axs = plt.subplots(1, 2, figsize=(10,6))



cols = ['season', 'weather']

for i in range(len(cols)):

    sns.barplot(x=cols[i], y='cnt', data=df, ax=axs[i])

    axs[i].xaxis.set_tick_params(rotation=90)
df.plot(kind='scatter', x='t1', y='t2')
df.plot(kind='scatter', x='t2', y='hum')
df.plot(kind='scatter', x='hum', y='wind_speed')
df.plot(kind='scatter', x='wind_speed', y='t2')
num_cols = ['t2', 'hum', 'wind_speed']

fig, axs = plt.subplots(1, 3, figsize=(10,6))



i = 0

for col in num_cols:

    sns.lineplot(x=col, y='cnt', data=df, ax=axs[i])

    i+=1