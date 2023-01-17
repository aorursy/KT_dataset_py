import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country = 'South Korea'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)
df = df[df['Country/Region']==selected_country]

df = df.groupby('ObservationDate').sum()
print(df)








df['daily_confirmed']= df['Confirmed'].diff()
df['daily_deaths']= df['Deaths'].diff()
df['daily_recovery']= df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_deaths'].plot()
df['daily_recovery'].plot()
plt.show()

from plotly.offline import iplot
import plotly.graph_objs as go

daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily_deaths')
daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily_confirmed')
layout_object = go.Layout(title='South Korea daily cases 19M58400',xaxis=dict(title='Date'),yaxis=dict(title='Number of prople'))
fig = go.Figure(data=[daily_deaths_object,daily_confirmed_object],layout=layout_object)
iplot(fig)
fig.write_html('South Korea_daily cases_19M58400')



df1 = df#[['daily_confirmed']]
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
f = open('table_19M58400.html','w')
f.write(styled_object.render())



df_rk = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)
df_rk1 = df_rk[df_rk['ObservationDate']=='06/07/2020'].sort_values(by=['Confirmed'],ascending=False).reset_index()

df_rk2 = df_rk1.groupby(['Country/Region']).sum().sort_values(by=['Confirmed'],ascending=False)
print(df_rk2)



