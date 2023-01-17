# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import statsmodels.api as sm

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_json('/kaggle/input/imgwdatarzbiala10/data10.json')
df.wartosc.skew()

data = df['wartosc']
x = data.index.values

z = np.polyfit(data.index.values, list(data), 1)

p = np.poly1d(z)

trendcolname = 'trend - {}'.format(z[0])

df[trendcolname] = p(x)
ax = plt.gca()

df.plot(kind='line',x='czas',y='wartosc',ax=ax, figsize=(32,18))

df.plot(kind='line',x='czas',y=trendcolname, color='red', ax=ax)

plt.title('rz. Biała - Tarnów')

plt.ylabel('Wysokość lustra wody [cm]')

plt.show()
ax = plt.gca()

df.plot(kind='line',x='czas',y='wartosc',ax=ax, figsize=(32,18), logy=True)

df.plot(kind='line',x='czas',y=trendcolname, color='red', ax=ax, logy=True)

plt.title('rz. Biała - Tarnów')

plt.ylabel('Wysokość lustra wody (logscale) [cm]')

plt.show()
p(0) - p(len(data.index.values)-1)
df['parsed_time'] = df['czas'].apply(pd.to_datetime)
df['year'] = df['parsed_time'].dt.year 

df['season'] = df['parsed_time'].dt.dayofyear.map(season) 
spring = range(80, 172)

summer = range(172, 264)

fall = range(264, 355)



def season(x):

    if x in spring:

       return 'Spring'

    if x in summer:

       return 'Summer'

    if x in fall:

       return 'Fall'

    else:

       return 'Winter'

df[df['wartosc'] < df['p_ostrzegawczy']].hist(column='wartosc', bins=100, by='year', figsize=(32,18))
df[df['wartosc'] < 100].hist(column='wartosc', bins=100, by='year', figsize=(32,18))
df_non_alarm = df[df['wartosc'] < df['p_ostrzegawczy']]
df_non_alarm['wartosc'].describe()
df_non_alarm.groupby(['year']).wartosc.describe()
df_non_alarm.boxplot(column='wartosc', by='year', figsize=(32,18))
df_non_alarm.hist(column='wartosc', bins=100, by='season', figsize=(32,18))