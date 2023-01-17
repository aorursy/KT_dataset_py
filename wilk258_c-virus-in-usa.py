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
!pip install calmap
import matplotlib.pyplot as plt 
import seaborn as sns 
import calmap
import plotly.express as px
import plotly.graph_objects as go
from fbprophet import Prophet
import plotly.offline as py
from fbprophet.plot import plot_plotly
df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
df.head(10)
df.isnull().sum()
print(df.state.value_counts())
print("case sum:",df.cases.sum())
print("case mean:",df.cases.mean())
print("deaths sum :",df.deaths.sum())
print("deaths mean:",df.deaths.mean())
state_sum = df.groupby('state')['cases','deaths'].sum().reset_index()
state_sum.sort_values(by='state', ascending=False)
cm = sns.light_palette("green", as_cmap=True)
state_sum.style.background_gradient(cmap=cm)
fig = px.scatter(state_sum, x="cases", y="deaths",
               color="state",
                 hover_name="state", log_x=True, size_max=60)
fig.show()
date_sum = df.groupby('date')['cases','deaths'].sum().reset_index()
date_sum.sort_values(by='date', ascending=False)
date_sum1 = country_sum[country_sum.deaths > 0] # country_sum1 no included zero death 
date_sum1.style.background_gradient(cmap='viridis')
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=country_sum['date'], 
                         y=country_sum['cases'],
                         mode='lines+markers',
                         name='Confirmed',
                         line=dict(color='Yellow', width=2)))
fig.add_trace(go.Scatter(x=country_sum['date'], 
                         y=country_sum['deaths'],
                         mode='lines+markers',
                         name='Deaths',
                         line=dict(color='Red', width=2)))

fig.show()
country_sum = df.groupby('county')['cases','deaths'].sum().reset_index()
country_sum.sort_values(by='county', ascending=False)
country_sum.style.background_gradient(cmap='viridis')
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=country_sum['county'], 
                         y=country_sum['cases'],
                         mode='lines+markers',
                         name='Confirmed',
                         line=dict(color='Yellow', width=2)))
fig.add_trace(go.Scatter(x=country_sum['county'], 
                         y=country_sum['deaths'],
                         mode='lines+markers',
                         name='Deaths',
                         line=dict(color='Red', width=2)))

fig.show()
case_df = pd.DataFrame(df.groupby('date')['cases'].sum().reset_index()).rename(columns={'date': 'ds', 'cases': 'y'})
death_df = pd.DataFrame(df.groupby('date')['deaths'].sum().reset_index()).rename(columns={'date': 'ds', 'deaths': 'y'})
m = Prophet()
m.fit(case_df)
m1 = Prophet()
m1.fit(death_df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast1 = m1.predict(future)
fig = plot_plotly(m, forecast)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Predictions for Total cases in US State',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig.update_layout(annotations=annotations)
fig
fig = plot_plotly(m, forecast1)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Predictions for Total Death in US State',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig.update_layout(annotations=annotations)
fig