# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/uber-raw-data-sep14.csv')

df.head()
df.info()
import matplotlib.pyplot as plt

%matplotlib inline

df['Date/Time'] = pd.to_datetime(df['Date/Time'], format="%m/%d/%Y %H:%M:%S")

df['DayofweekNum'] = df['Date/Time'].dt.dayofweek
df['Dayofweek'] = df['Date/Time'].dt.weekday_name

df['MonthdayNum'] = df['Date/Time'].dt.day

df['Hourofday'] = df['Date/Time'].dt.hour
df.head(15)
df_weekdays = df.pivot_table(index=['DayofweekNum','Dayofweek'], values='Base',aggfunc='count')

df_weekdays.plot(kind='bar',figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys By WeekDay')
df_monthdays = df.pivot_table(index=['MonthdayNum'], values='Base', aggfunc='count')

df_monthdays.plot(kind='bar',figsize=(8,6))

plt.title('Journeys by Month Day')
df_hours = df.pivot_table(index=['Hourofday'], values='Base', aggfunc='count')

df_hours.plot(kind='bar',figsize=(8,6))

plt.title('Journeys by Hours')
from mpl_toolkits.basemap import Basemap

from matplotlib import cm
west, south, east, north = -74.26, 40.50, -73.70, 40.92



fig = plt.figure(figsize=(14,10))

ax = fig.add_subplot(111)

m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')

x, y = m(df['Lon'].values, df['Lat'].values)

m.hexbin(x, y, gridsize=1000,

         bins='log', cmap=cm.YlOrRd_r);