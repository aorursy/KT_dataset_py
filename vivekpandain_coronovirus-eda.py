# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#import chart_studio.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import re

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

dfc=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

dfr=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

dfd=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
df.head()
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])

df['Last Update'] = pd.to_datetime(df['Last Update'])
df.info()
df.describe()
#missing data

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(15)
col_to_use = list(df.select_dtypes(include=['float64','int64']).columns)

for column in col_to_use:

    print(f"{column}: Number of unique values {df[column].nunique()}")

    print(f"Maximum: {df[column].max()}")

    print(f"Minumum: {df[column].min()}")

    print(f"Number of Null values: {df[column].isnull().sum()}")

    print("----------------------------------------------")
df.columns
grp_date = df.groupby(["ObservationDate"])["Deaths","Confirmed","Recovered"].sum()
grp_date.reset_index(level=0, inplace=True)

grp_date.head()
#Interative plot to see death, confirmed, and recovered on  a given day

fig = go.Figure()

fig.add_trace(go.Scatter(x=grp_date.ObservationDate, y=grp_date.Deaths,

                    mode='markers',

                    name='Deaths'))

fig.add_trace(go.Scatter(x=grp_date.ObservationDate, y=grp_date.Confirmed,

                    mode='markers',

                    name='Confirmed'))

fig.add_trace(go.Scatter(x=grp_date.ObservationDate, y=grp_date.Recovered,

                    mode='markers',

                    name='Recovered'))



fig.show()
#Adding column to mark weekdays (0) and weekends(1) for time series evaluation

grp_date['WEEKDAY'] = (((grp_date['ObservationDate']).dt.dayofweek)// 5 == 1).astype(float)

grp_date['WEEKDAY'].value_counts()

grp_date = grp_date.set_index('ObservationDate')

grp_date['month'] = grp_date.index.month

grp_date['weekday'] = grp_date.index.weekday

grp_date['week'] = grp_date.index.week
def daily(x,df=grp_date):

    return df.groupby('weekday')[x].mean()

daily('Confirmed').plot(kind = 'bar', figsize=(10,8))

ticks = list(range(0, 7, 1)) 

labels = "Mon Tues Weds Thurs Fri Sat Sun".split()

plt.xlabel('Day')

plt.ylabel('Number of People')

plt.title('Mean Confirmed count per Day of Week')

plt.xticks(ticks, labels);
daily('Recovered').plot(kind = 'bar', figsize=(10,8))

ticks = list(range(0, 7, 1)) 

labels = "Mon Tues Weds Thurs Fri Sat Sun".split()

plt.xlabel('Day')

plt.ylabel('Number of People')

plt.title('Mean Recovered count per Day of Week')

plt.xticks(ticks, labels);
daily('Deaths').plot(kind = 'bar', figsize=(10,8))

ticks = list(range(0, 7, 1)) 

labels = "Mon Tues Weds Thurs Fri Sat Sun".split()

plt.xlabel('Day')

plt.ylabel('Number of People')

plt.title('Mean Death count per Day of Week')

plt.xticks(ticks, labels);

grp_country=df.groupby(["Country/Region"])["Deaths","Confirmed","Recovered"].sum()
grp_country.reset_index(level=0, inplace=True)

grp_country.head()
import plotly.express as px





data=grp_country.sort_values('Confirmed',ascending = False).head(5)



fig = px.bar(data, x='Country/Region', y='Confirmed',

             hover_data=['Confirmed'], color='Confirmed',

             labels={'pop':'population of Canada'}, height=400)

fig.show()


data=grp_country.sort_values('Deaths',ascending = False).head(5)



fig = px.bar(data, x='Country/Region', y='Deaths',

             hover_data=['Deaths'], color='Deaths',

              height=400)

fig.show()
data=grp_country.sort_values('Recovered',ascending = False).head(5)



fig = px.bar(data, x='Country/Region', y='Recovered',

             hover_data=['Recovered'], color='Recovered',

              height=400)

fig.show()