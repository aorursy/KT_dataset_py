import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf)



selected_country='South Korea'

df=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

#print(np.unique(df['Country/Region'].values))

df = df[df['Country/Region']==selected_country]

df=df.groupby('ObservationDate').sum()

print(df)
df['daily_confirmed'] = df['Confirmed'].diff()

df['daily_deaths'] = df['Deaths'].diff()

df['daily_recovery'] = df['Recovered'].diff()

df['daily_confirmed'].plot()

df['daily_recovery'].plot()

df['daily_deaths'].plot()
print(df)
from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')

daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')

daily_recovery_object=go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily recoveries')



layout_object= go.Layout(title='South Korea daily cases 20M51990',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)

iplot(fig)

fig.write_html('South Korea_daily_cases_20M51990.html')


df1 = df#[['daily_confirmed']]

df1 = df1.fillna(0.)

styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')

display(styled_object)

f = open('table_20M51990.html','w')

f.write(styled_object.render())

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df1 = df.groupby(['ObservationDate','Country/Region']).sum()

df2 = df[df['ObservationDate']=='06/10/2020'].sort_values(by=['Confirmed'],ascending=False).reset_index()

print(df2[df2['Country/Region']=='South Korea'])
import numpy as np 

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.head()
data.index=data['ObservationDate']

data = data.drop(['SNo','ObservationDate'],axis=1)

data.head()
data_SK = data[data['Country/Region']=='South Korea']

data_SK.tail()
latest = data[data.index=='06/08/2020']

latest = latest.groupby('Country/Region').sum()

latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 

print('Ranking of South Korea: ', latest[latest['Country/Region']=='South Korea'].index.values[0]+1)