# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



import os

print(os.listdir('../input/earthquake'))
# basic information about the data

data = pd.read_csv('../input/earthquake/earthquake.csv', header=0)

print(data.shape)

data.head()
# list dtype, non-null values of each columns

data.info()
data.describe()
# display data by latitude and longitude

plt.scatter(data.lat, data.long, alpha=0.1, color='silver')

plt.xlim(data.lat.min(), data.lat.max())

plt.ylim(data.long.min(), data.long.max())

plt.xlabel('latitude')

plt.ylabel('longitude')
# display correlations between numerical features

corr = data.corr().values

plt.figure(figsize=(8,8))

plt.imshow(corr, cmap=plt.get_cmap('YlOrBr'))

plt.xticks(range(len(corr)), data.columns, rotation=30)

plt.yticks(range(len(corr)), data.columns)

plt.colorbar()

plt.title('the correlations of features')
# frequency of earthquake distribute over year, month, day

data['date'] = pd.to_datetime(data.date)

data['time'] = pd.to_datetime(data.time, format='%H:%M:%S %p').dt.time

data['year'] = data.date.dt.year

data['month'] = data.date.dt.month

data['day'] = data.date.dt.day

plt.subplot(3,1,1)

byyear = data.groupby(['year'])['id'].count().sort_index()

plt.plot(byyear)

plt.title('distribute over year')

plt.subplot(3,1,2)

bymonth = data.groupby(['month'])['id'].count().sort_index()

plt.plot(bymonth)

plt.title('distribute over month')

plt.subplot(3,1,3)

byday = data.groupby(['day'])['id'].count().sort_index()

plt.plot(byday)

plt.title('distribute over day')

plt.tight_layout()
# relationship between frequency and city

plt.figure(figsize=(30, 6))

bycity = data.groupby(['city'])['id'].count().sort_values(ascending=False)

plt.plot(bycity)

plt.xticks(range(len(bycity)), bycity.index, rotation=30)
# relationship between frequency and direction

bydirection = data.groupby(['direction'])['id'].count().sort_values(ascending=False)

plt.plot(bydirection)

plt.xticks(rotation=30)
# distribution of depth, mx, ect. over year

plt.figure(figsize=(15, 6))

plt.subplot(2,1,1)

byyeardepth = data.groupby(['year'])['depth'].mean().sort_index()

plt.plot(byyeardepth)

plt.title('mean of depth over year')

plt.subplot(2,1,2)

for item,c in zip(['xm','md','mw','ms','mb'], ['b','r','g','b','m']):

    byitem = data.groupby(['year'])[item].mean().sort_index()

    plt.plot(byitem, color=c)

plt.legend()

plt.tight_layout()

plt.show()