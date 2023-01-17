# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
PATH = '/kaggle/input/2008.csv.bz2'

df = pd.read_csv(PATH)

pd.set_option('display.max_columns', 30)

df.head(1)
# coordinates = ['UniqueCarrier', 'FlightNum']



topCarrier = df['UniqueCarrier'].value_counts().sort_values(ascending = False).head(10).index.values

sns.boxplot(x="UniqueCarrier", y="FlightNum", data=df[df['UniqueCarrier'].isin(topCarrier)])
df.groupby('CancellationCode').size().plot(kind='bar');
df.groupby(['Origin','Dest'])['FlightNum'].count().sort_values(ascending = False).iloc[:5].plot(kind = 'bar', rot = 0);
Delay_df = df[df['DepDelay']>0]

topDelay = Delay_df.groupby(['Origin','Dest'])['DepDelay'].count().sort_values(ascending = False).iloc[:5]

topDelay.plot(kind = 'bar', rot=0)
t1 = df[(df['Origin']=='LAX')&(df['Dest']=='SFO')&(df['WeatherDelay']>0)]

t2 = df[(df['Origin']=='DAL')&(df['Dest']=='HOU')&(df['WeatherDelay']>0)]

t3 = df[(df['Origin']=='SFO')&(df['Dest']=='LAX')&(df['WeatherDelay']>0)]

t4 = df[(df['Origin']=='ORD')&(df['Dest']=='LGA')&(df['WeatherDelay']>0)]

t5 = df[(df['Origin']=='HOU')&(df['Dest']=='DAL')&(df['WeatherDelay']>0)]

len(t1) + len(t2) + len(t3) + len(t4) + len(t5)
sns.distplot(df['DepTime'].dropna())
df['Date'] = pd.to_datetime(df.rename(columns={'DayofMonth': 'Day'})[['Year', 'Month', 'Day']])

df.groupby('Date').size().plot()

#dayNumber 

df.hist('DayOfWeek', bins=20);

#weekNumber

df.hist('DayofMonth', bins=80);

#MonthNumber

df.hist('Month', bins=60);
Delay_df = df.groupby('Month')[['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']].sum()

Delay_df.columns = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']



pos1 = go.Bar(

    x=Delay_df.index,

    y=Delay_df.CarrierDelay,

    name='Carrier Delay'

)

pos2 = go.Bar(

    x=Delay_df.index,

    y=Delay_df.WeatherDelay,

    name='Weather Delay'

)

pos3 = go.Bar(

    x=Delay_df.index,

    y=Delay_df.NASDelay,

    name='NAS Delay'

)

pos4 = go.Bar(

    x=Delay_df.index,

    y=Delay_df.SecurityDelay,

    name='Security Delay'

)

pos5 = go.Bar(

    x=Delay_df.index,

    y=Delay_df.LateAircraftDelay,

    name='Late Aircraft Delay'

)



graf = go.Figure(data=[pos1, pos2, pos3, pos4, pos5], layout={'xaxis': {'title': 'Month'}})

iplot(graf)
sns.jointplot('Month', 'CarrierDelay', data=df);
dFrame = df[df['Month']==4]

graf = dFrame.groupby('UniqueCarrier').size().head(10).sort_values(ascending = False)

graf.plot(rot=0);
Arr = df[df['ArrDelay']>0]

sns.boxplot(x="UniqueCarrier", y="ArrDelay", data=Arr, showfliers=False, whis = 0)
Dep = df[df['DepDelay']>0]

sns.boxplot(x="UniqueCarrier", y="DepDelay", data=Dep, showfliers=False, whis = 0)