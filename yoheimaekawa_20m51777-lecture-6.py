import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.Inf)

selected_country='Germany'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)
df['daily_confirmed'] = df['Confirmed'].diff()
df['daily_deaths'] = df['Deaths'].diff()
df['daily_recoveries'] = df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_deaths'].plot()
df['daily_recoveries'].plot()

plt.show()
from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')
daily_recoveries_object = go.Scatter(x=df.index,y=df['daily_recoveries'].values,name='Daily recovery')

layout_object = go.Layout(title='Germany daily cases 20M51777',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recoveries_object],layout=layout_object)
iplot(fig)
fig.write_html('Germany_cases_20M51777.html')
df1 = df#[['daily_confirmed']]
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
#f = open('table_20M51777.html','w')
#f.write(styled_object)
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df1 = df[df.index=='06/08/2020']
df1 = df.groupby('Country/Region').sum()
df1 = df.sort_values(by='Confirmed',ascending=False).reset_index()

print('Ranking of Germany: ', df1[df1['Country/Region']=='Germany'].index.values[0]+1)