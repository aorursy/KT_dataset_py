# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/barcelona-data-sets/accidents_2017.csv')

df.dropna(how='any', inplace=True)

df.head()

df.describe()
df.info()
category=df['Street'].value_counts()

plt.figure(figsize=[20,5])

x=category[:15].index

y=category[:15].values

sns.barplot(x, y)

plt.xlabel('Year')

plt.ylabel('Number of heros that exist')

plt.title('Number of heros exist in years mostly')
category=df['Day'].value_counts()

plt.figure(figsize=[31,10])

x=category.index

y=category.values

sns.barplot(x, y)

plt.xlabel('Day of the month')

plt.ylabel('Number of Accidents')

plt.title('Day of the month distrubiton')
category=df['District Name'].value_counts()

plt.figure(figsize=[17,5])

x=category.index

y=category.values

sns.barplot(x, y)

plt.xlabel('Districts')

plt.ylabel('Number of accident on the districts')

#plt.title('Number of heros exist in years mostly')
category=df['Hour'].value_counts()

plt.figure(figsize=[24,7])

x=category[:24].index

y=category[:24].values

sns.barplot(x, y)

plt.xlabel('Hour')

plt.ylabel('Number of accident')

plt.title('Number of accidents per hour')
category=df['Month'].value_counts()

plt.figure(figsize=[12,5])

x=category[:15].index

y=category[:15].values

sns.barplot(x, y)

plt.xlabel('months')

plt.ylabel('number of accident')

plt.title('accident numbers by month')
max_long, max_lat, min_long, min_lat=max(df['Longitude'].values), max(df['Latitude'].values), min(df['Longitude'].values), min(df['Latitude'].values)

print(max_long, min_long, max_lat, min_lat)
map_of_barcelona=np.zeros((542, 431), dtype=int)#kaggle now doesn't support PNG or jpg format 



area=(min_long, max_lat, max_long, min_lat)
def Street_map(df, area, map_of_barcelona, s=10):#using longtitude and latitude on map

    fig, axs = plt.subplots(figsize=(30, 25))

    datafilter=df[df['Hour']>12]

    datafilter2=df[df['Hour']<=12]

    axs.scatter(datafilter.Longitude, datafilter.Latitude, alpha=1, color='red', label="Afternoon")

    axs.scatter(datafilter2.Longitude, datafilter2.Latitude, alpha=1,  color='blue', label="Before noon")

    axs.set_xlim(area[0], area[2])

    axs.set_ylim((area[3], area[1]))

    axs.set_title('accident points on map')

    axs.imshow(map_of_barcelona, zorder=1, extent=area)

Street_map(df, area, map_of_barcelona)  