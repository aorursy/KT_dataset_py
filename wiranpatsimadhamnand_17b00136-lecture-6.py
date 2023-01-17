import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf) # so that we can see the whole dataframe
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)

#print(df.columns)

#print(np.unique(df['Country/Region'].values)) # list names of country
selected_country = 'Peru'

df1 = df[df['Country/Region']==selected_country]

#print(df1)
df = df1.groupby('ObservationDate').sum()

#print(df) # sum the same date in all the states
df['ActiveCase'] = df['Confirmed']-df['Recovered']-df['Deaths']

df['Confirmed'].plot(legend=True, color='#5ec691')

df['Recovered'].plot(legend=True, color='#fe7241')

df['Deaths'].plot(legend=True, color='#95389e')

df['ActiveCase'].plot(legend=True, color='#fe346e')
df['DailyConfirmed'] = df['Confirmed'].diff()

df['DailyDeaths'] = df['Deaths'].diff()

df['DailyRecovery'] = df['Recovered'].diff()

df['DailyConfirmed'].plot(legend=True, color='#40bad5')

df['DailyRecovery'].plot(legend=True, color='#fcbf1e')

df['DailyDeaths'].plot(legend=True, color='#f35588')
from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_object = go.Scatter(x=df.index,y=df['DailyConfirmed'].values,name='Daily confirmed')

daily_deaths_object = go.Scatter(x=df.index,y=df['DailyDeaths'].values,name='Daily deaths')



layout_object = go.Layout(title='Peru daily case',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object],layout=layout_object)

iplot(fig)

fig.write_html('Peru_daily_cases_17B00136.html')
df1 = df#[['daily_confirmed']]

df1 = df1.fillna(0.)

styled_object = df1.style.background_gradient(cmap='YlGnBu').highlight_max('DailyConfirmed').set_caption('Daily Summaries')

display(styled_object)

f = open('table_17B00136.html','w')

#f.write(styled_object.render())
import pandas as pd

import numpy as np



df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

#df.index[df['ObservationDate']=='06/12/2020'].tolist()

df1 = df[40076:]  # data of the date 06/12/2020

df1 = df.groupby(['Country/Region']).sum()

df2 = df1.sort_values(by=['Confirmed'], ascending=False)

df2['Rank'] = np.arange(1,224)  # adding a column of worldwide rank

df2.head(20)