import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from plotly.offline import iplot

import plotly.graph_objs as go

np.set_printoptions(threshold=np.inf)



selected_country = 'Brazil'

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

#print(np.unique(df['Country/Region'].values))

df = df[df['Country/Region']==selected_country]



df = df.groupby('ObservationDate').sum()

df['daily_confirmed'] = df['Confirmed'].diff()

df['daily_deaths'] = df['Deaths'].diff()

df['daily_recoveries'] = df['Recovered'].diff()



daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')

daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')

daily_recoveries_object = go.Scatter(x=df.index,y=df['daily_recoveries'].values,name='Daily recoveries')

layout_object = go.Layout(title='Brazil daily cases 20M51790',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recoveries_object],layout=layout_object)

iplot(fig)

fig.write_html('Brazil_daily_cases_20M51790.html')
df1 = df#[['daily_confirmed']]

df1 = df1.fillna(0.)

styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')

display(styled_object)

f = open('table_20M51790.html','w')

f.write(styled_object.render())
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df = df.groupby(['ObservationDate','Country/Region'],as_index=False).sum()

df = df[df['ObservationDate']=='06/10/2020'].sort_values(by=['Confirmed'],ascending=False).reset_index()

print('Grobal Rankign of '+ selected_country +':',df[df['Country/Region']==selected_country].index.values[0]+1)