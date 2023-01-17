# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import statsmodels.api as sm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
source = pd.read_csv('../input/GlobalLandTemperaturesByMajorCity.csv')
%matplotlib inline
df = source.copy()  # I only do this to avoid repeated readings of CSV file.
df.loc[:, 'dt'] = pd.to_datetime(df.loc[:, 'dt'])  # Req'd for decomposition.
df = source.set_index(df.loc[:, 'dt'])  # note: makes multiple rows with same index.

cities = list(source['City'].unique())
df.drop(['Latitude', 'Longitude'], axis=1, inplace=True)
dfd= {}
df_trend = pd.DataFrame()
basedt = '1882-01-01'  # somewhat arbitrary starting point.

for city in cities:
    dfd[city] = df[df['City']==city]
    dfd[city].interpolate(inplace=True)  # decompose requires data to be nan-free. Linearly interpolating.
    df_trend[city] = sm.tsa.seasonal_decompose(dfd[city]['AverageTemperature']).trend
df_trend.dropna(inplace=True)
rolling = df_trend.rolling(window=36, center=False).std()  # pd.rolling_std() is deprecated.
rolling = rolling.dropna()
n = 2
largest = rolling.max().nlargest(n)
smallest = rolling.max().nsmallest(n)
pt = pd.pivot_table(df,
                    values='AverageTemperature',
                    index='dt',
                    columns='City',
                    aggfunc=sum,
                    dropna=False)
pt = pt.set_index(pd.to_datetime(pt.index))
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(20, 14))

pt.loc[basedt:].plot(ax=ax1,
                     legend=False,
                     title='AverageTemperature (difficult to learn much)')
df_trend.plot(ax=ax2,
              legend=False,
              title='Trend component of AverageTemperature (still difficult to learn much)')
rolling[largest.index | smallest.index].plot(ax=ax3,
              title='Rolling Deviations of Most and Least Volatile (we can learn something)')
df_trend.loc[basedt:, largest.index | smallest.index].plot(ax=ax4,
              title='AverageTemperature of Most and Least Volatile (we can learn something)')
plt.show()