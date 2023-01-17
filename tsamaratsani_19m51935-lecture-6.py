import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

selected_country='Iran'
data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df = data[data['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)
df['daily_confirmed']=df['Confirmed'].diff()
df['daily_deaths']=df['Deaths'].diff()
df['daily_recovery']=df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_deaths'].plot()
df['daily_recovery'].plot()
plt.legend()
plt.show()
from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily Confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily Deaths')
daily_recovery_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily Recovery')

layout_object = go.Layout(title='Iran daily cases 19M51935',xaxis=dict(title='Date'),yaxis=dict(title='Number of People'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)
iplot(fig)
fig.write_html('Iran_daily_cases.html')
print(df.describe())
df1 = df
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='jet').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)

latest = data[data['ObservationDate']=='06/12/2020']
latest = latest.groupby('Country/Region').sum()
latest = latest.sort_values(by='Confirmed',ascending=False).reset_index()
print('Ranking of Iran Confirmed Cases: ', latest[latest['Country/Region']=='Iran'].index.values[0]+1)

#Fatality Rate is defined as the ration of death per confirmed cases in certained period of time in the country
latest['Fatality_rate']=latest['Deaths']/latest['Confirmed']
print('Iran Fatality Rate:', latest[latest['Country/Region']=='Iran']['Fatality_rate'].values)

#Fatality Rate Ranking
latest_fat = latest.sort_values(by='Fatality_rate',ascending=False).reset_index()
print('Ranking of Iran Fatality Rate: ', latest_fat[latest_fat['Country/Region']=='Iran'].index.values[0]+1)