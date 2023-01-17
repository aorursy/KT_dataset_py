# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.




# "timestamp" - timestamp field for grouping the data

# "cnt" - the count of a new bike shares

# "t1" - real temperature in C

# "t2" - temperature in C "feels like"

# "hum" - humidity in percentage

# "wind_speed" - wind speed in km/h

# "weather_code" - category of the weather

# "is_holiday" - boolean field - 1 holiday / 0 non holiday

# "is_weekend" - boolean field - 1 if the day is weekend

# "season" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.



# "weathe_code" category description:

# 1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity

# 2 = scattered clouds / few clouds

# 3 = Broken clouds

# 4 = Cloudy

# 7 = Rain/ light Rain shower/ Light rain

# 10 = rain with thunderstorm

# 26 = snowfall

# 94 = Freezing Fog
df=pd.read_csv('..//input/london-bike-sharing-dataset/london_merged.csv',parse_dates=True)
df[df['is_holiday']==1].count()
df.describe()

df.info()

df.isnull().any()

sns.barplot(x=df['is_weekend'],y=df['cnt'])
sns.barplot(x=df['is_holiday'],y=df['cnt'])






ax=df['cnt'].groupby([df['weather_code']]).sum()

ax.index

d=ax.reset_index()

s=d.sort_values(by=['cnt'])

sns.barplot(x=s['weather_code'],y=s['cnt'],order=s['weather_code'])



sns.barplot(x=df['season'],y=df['cnt'],)


df['timestamp']=pd.to_datetime(df['timestamp'],format='%Y-%m-%d %H:%M:%S')

df.plot(x='timestamp',y='t1', rot=90,c="red",title='Time vs real temperature in C')

#plt.title('time vs real temperature in C')

plt.ylabel('t1')



plt.show()




df.plot(x='timestamp',y='t2', rot=90,c="red",title='Time vs feels like temperature in c')

#plt.title('time vs real temperature in C')

plt.ylabel('t2')



plt.show()

df.plot(x='timestamp',y='hum', rot=90,c="red",title='Time vs humidity change')

#plt.title('time vs real temperature in C')

plt.ylabel('t2')



plt.show()

df.plot(x='timestamp',y='wind_speed', rot=90,c="red",title='Time vs windspeed')

#plt.title('time vs real temperature in C')

plt.ylabel('t2')



plt.show()

df.plot(x='timestamp',y='season', rot=90,c="green",title='Time vs season')

#plt.title('time vs real temperature in C')

plt.ylabel('t2')



plt.show()