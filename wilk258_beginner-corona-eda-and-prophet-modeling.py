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
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
from fbprophet.plot import plot_plotly
import math
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
train.head(3)
train.dtypes
train.shape
train.isnull().sum()
train=train.rename(columns={'Province_State': 'Province','Country_Region': 'Country',
                           'ConfirmedCases': 'Confirmed'})
#train data confirmed case mean and sum in world 
print("confirmed mean:", train.Confirmed.mean())
print("confirmed sum:", train.Confirmed.sum())
print("Fatalities mean:", train.Fatalities.mean())
print("Fatalities sum:", train.Fatalities.sum())
# date sum
datastata = train.groupby('Date')['Date', 'Confirmed', 'Fatalities'].sum().reset_index()
cm = sns.light_palette("green", as_cmap=True)
datastata.style.background_gradient(cmap=cm)
#Date confirm and fatalities sum visualization 
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=datastata['Date'], 
                         y=datastata['Confirmed'],
                         mode='lines+markers',
                         name='Confirmed',
                         line=dict(color='Yellow', width=2)))
fig.add_trace(go.Scatter(x=datastata['Date'], 
                         y=datastata['Fatalities'],
                         mode='lines+markers',
                         name='Fatalities',
                         line=dict(color='Red', width=2)))
fig.show()
# Date mean 
datastata_ = train.groupby('Date')['Date', 'Confirmed', 'Fatalities'].mean().reset_index()
cm = sns.light_palette("red", as_cmap=True)
datastata_.style.background_gradient(cmap=cm)
#Date confirm and fatalities mean visualization 
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=datastata_['Date'], 
                         y=datastata_['Confirmed'],
                         mode='lines+markers',
                         name='Confirmed mean',
                         line=dict(color='Yellow', width=2)))
fig.add_trace(go.Scatter(x=datastata_['Date'], 
                         y=datastata_['Fatalities'],
                         mode='lines+markers',
                         name='Fatalities mean',
                         line=dict(color='Red', width=2)))
fig.show()
countrystat = train.groupby('Country')['Country', 'Confirmed', 'Fatalities'].sum().reset_index()
countrystat.style.set_properties(**{'background-color': 'black',
                           'color': 'lawngreen',
                           'border-color': 'white'})
countrystat = train.groupby('Country')['Country', 'Confirmed', 'Fatalities'].mean().reset_index()
cm = sns.light_palette("blue", as_cmap=True)
countrystat.style.background_gradient(cmap=cm)
fig = px.bar(countrystat.sort_values('Confirmed', ascending=False), 
             y="Confirmed", x="Country", color='Country', 
             log_y=True, template='ggplot2', title='country confirm sum')
fig.show()
fig = px.bar(countrystat.sort_values('Fatalities', ascending=False), 
             y="Fatalities", x="Country", color='Country', 
             log_y=True, template='ggplot2', title='country fatality sum')
fig.show()
china=train.loc[train.Country == 'China']
china.head(3)
chinasum_date = china.groupby('Date')['Date', 'Confirmed', 'Fatalities'].sum().reset_index()
china_province = china.groupby('Province')['Province', 'Confirmed', 'Fatalities'].sum().reset_index()
print("China Confirm mean:", china.Confirmed.mean())
print("China Fatalities mean:", china.Fatalities.mean())
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=chinasum_date['Date'], 
                         y=chinasum_date['Confirmed'],
                         mode='lines+markers',
                         name='China Confirmed',
                         line=dict(color='Yellow', width=2)))
fig.add_trace(go.Scatter(x=chinasum_date['Date'], 
                         y=chinasum_date['Fatalities'],
                         mode='lines+markers',
                         name='China Fatalities',
                         line=dict(color='Red', width=2)))
fig.show()
fig = px.bar(china_province.sort_values('Fatalities', ascending=False), 
             y="Fatalities", x="Province", color='Province', 
             log_y=True, template='ggplot2', title='China Every Province Fatalities')
fig.show()
fig = px.bar(china_province.sort_values('Confirmed', ascending=False), 
             y="Confirmed", x="Province", color='Province', 
             log_y=True, template='ggplot2', title='China Every Province Confirm')
fig.show()
fig = px.pie(china_province, values='Confirmed', names='Province',
             title='China province Confirmed')
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()
fig = px.pie(china_province, values='Fatalities', names='Province',
             title='China province fatalities')
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()
#Spain viualiztion
spain=train.loc[train.Country == 'Spain']
spainsum_date = spain.groupby('Date')['Date', 'Confirmed', 'Fatalities'].sum().reset_index()
print("Spain Confirm mean:", spain.Confirmed.mean())
print("Spain Fatalities mean:", spain.Fatalities.mean())
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=spainsum_date['Date'], 
                         y=spainsum_date['Confirmed'],
                         mode='lines+markers',
                         name='Spain Confirmed',
                         line=dict(color='Yellow', width=2)))
fig.add_trace(go.Scatter(x=spainsum_date['Date'], 
                         y=spainsum_date['Fatalities'],
                         mode='lines+markers',
                         name='Spain Fatalities',
                         line=dict(color='Red', width=2)))
fig.show()
#Italy visualization 
italy=train.loc[train.Country == 'Italy']
italy_date = italy.groupby('Date')['Date', 'Confirmed', 'Fatalities'].sum().reset_index()
print("Italy Confirm mean:", italy.Confirmed.mean())
print("Italy Fatalities mean:", italy.Fatalities.mean())
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=italy_date['Date'], 
                         y=italy_date['Confirmed'],
                         mode='lines+markers',
                         name='Italy Confirmed',
                         line=dict(color='Yellow', width=2)))
fig.add_trace(go.Scatter(x=italy_date['Date'], 
                         y=italy_date['Fatalities'],
                         mode='lines+markers',
                         name='Italy Fatalities',
                         line=dict(color='Red', width=2)))
fig.show()
#iran visualization 
iran=train.loc[train.Country == 'Iran']
iran_date = iran.groupby('Date')['Date', 'Confirmed', 'Fatalities'].sum().reset_index()
print("Iran Confirm mean:", iran.Confirmed.mean())
print("Iran Fatalities mean:", iran.Fatalities.mean())
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=iran_date['Date'], 
                         y=iran_date['Confirmed'],
                         mode='lines+markers',
                         name='Iran Confirmed',
                         line=dict(color='Yellow', width=2)))
fig.add_trace(go.Scatter(x=iran_date['Date'], 
                         y=iran_date['Fatalities'],
                         mode='lines+markers',
                         name='Iran Fatalities',
                         line=dict(color='Red', width=2)))
fig.show()
#South Korea visualization 
korea=train.loc[train.Country == 'Korea, South']
korea_date = italy.groupby('Date')['Date', 'Confirmed', 'Fatalities'].sum().reset_index()
print("Korea Confirm mean:", korea.Confirmed.mean())
print("Korea Fatalities mean:", korea.Fatalities.mean())
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=korea_date['Date'], 
                         y=korea_date['Confirmed'],
                         mode='lines+markers',
                         name='Korea Confirmed',
                         line=dict(color='Yellow', width=2)))
fig.add_trace(go.Scatter(x=korea_date['Date'], 
                         y=korea_date['Fatalities'],
                         mode='lines+markers',
                         name='Korea Fatalities',
                         line=dict(color='Red', width=2)))
fig.show()
# usa visualization 
# now there is world number one 
us=train.loc[train.Country == 'US']
us.head(3)
us_date = us.groupby('Date')['Date', 'Confirmed', 'Fatalities'].sum().reset_index()
us_province = us.groupby('Province')['Province', 'Confirmed', 'Fatalities'].sum().reset_index()
print("China Confirm mean:", china.Confirmed.mean())
print("China Fatalities mean:", china.Fatalities.mean())
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=us_date['Date'], 
                         y=us_date['Confirmed'],
                         mode='lines+markers',
                         name='US Confirmed',
                         line=dict(color='Yellow', width=2)))
fig.add_trace(go.Scatter(x=us_date['Date'], 
                         y=us_date['Fatalities'],
                         mode='lines+markers',
                         name='US Fatalities',
                         line=dict(color='Red', width=2)))
fig.show()
fig = px.bar(us_province.sort_values('Confirmed', ascending=False), 
             y="Confirmed", x="Province", color='Province', 
             log_y=True, template='ggplot2', title='USA Every Province Confirmed')
fig.show()
fig = px.bar(us_province.sort_values('Fatalities', ascending=False), 
             y="Fatalities", x="Province", color='Province', 
             log_y=True, template='ggplot2', title='USA Every Province Fatalities')
fig.show()
fig = px.pie(us_province, values='Confirmed', names='Province',
             title='USA province Confirmed')
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()
fig = px.pie(us_province, values='Fatalities', names='Province',
             title='USA province Fatalities')
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()
train['Province'] = train['Province'].fillna('')
temp = train[[col for col in train.columns if col != 'Province']]

world = train[train['Date'] == max(train['Date'])].reset_index()
world = world.groupby('Country')['Confirmed', 'Fatalities'].sum().reset_index()
fig = px.choropleth(world, locations="Country", 
                    locationmode='country names', color="Confirmed", 
                    hover_name="Confirmed", range_color=[1,5000], 
                    color_continuous_scale="peach", 
                    title=' Confirmed Cases in the world')
fig.show()

fig = px.choropleth(world, locations="Country", 
                    locationmode='country names', color="Fatalities", 
                    hover_name="Fatalities", range_color=[1,5000], 
                    color_continuous_scale="peach", 
                    title=' Fatalities Cases in the world')
fig.show()
fig = px.choropleth(world, locations="Country", 
                    locationmode='country names', color="Confirmed", 
                    hover_name="Country", range_color=[1,8000], 
                    color_continuous_scale='balance', 
                    title='Confirmed Cases in europe', scope='europe', height=800)
fig.show()

fig = px.choropleth(world, locations="Country", 
                    locationmode='country names', color="Fatalities", 
                    hover_name="Country", range_color=[1,8000], 
                    color_continuous_scale='balance', 
                    title='Fatalities Cases in europe', scope='europe', height=800)
fig.show()
fig = px.choropleth(world, locations="Country", 
                    locationmode='country names', color="Confirmed", 
                    hover_name="Country", range_color=[1,3000], 
                    color_continuous_scale='balance', 
                    title='Confirmed Cases in europe', scope='europe', height=800)
fig.show()

fig = px.choropleth(world, locations="Country", 
                    locationmode='country names', color="Fatalities", 
                    hover_name="Country", range_color=[1,3000], 
                    color_continuous_scale='balance', 
                    title='Fatalities Cases in europe', scope='europe', height=800)
fig.show()
fig = px.choropleth(world, locations="Country", 
                    locationmode='country names', color="Confirmed", 
                    hover_name="Country", range_color=[1,3000], 
                    color_continuous_scale='balance', 
                    title='Confirmed Cases in Asia', scope='asia', height=800)
fig.show()

fig = px.choropleth(world, locations="Country", 
                    locationmode='country names', color="Fatalities", 
                    hover_name="Country", range_color=[1,3000], 
                    color_continuous_scale='balance', 
                    title='Fatalities Cases in Asia', scope='asia', height=800)
fig.show()
fig = px.choropleth(world, locations="Country", 
                    locationmode='country names', color="Confirmed", 
                    hover_name="Country", range_color=[1,3000], 
                    color_continuous_scale='balance', 
                    title='Confirmed Cases in North America', scope='north america', height=800)
fig.show()

fig = px.choropleth(world, locations="Country", 
                    locationmode='country names', color="Fatalities", 
                    hover_name="Country", range_color=[1,3000], 
                    color_continuous_scale='balance', 
                    title='Fatalities Cases in North America', scope='north america', height=800)
fig.show()
fig = px.choropleth(world, locations="Country", 
                    locationmode='country names', color="Confirmed", 
                    hover_name="Country", range_color=[1,3000], 
                    color_continuous_scale='balance', 
                    title='Confirmed Cases in South America', scope='south america', height=800)
fig.show()

fig = px.choropleth(world, locations="Country", 
                    locationmode='country names', color="Fatalities", 
                    hover_name="Country", range_color=[1,3000], 
                    color_continuous_scale='balance', 
                    title='Fatalities Cases in South America', scope='south america', height=800)
fig.show()
fig = px.choropleth(world, locations="Country", 
                    locationmode='country names', color="Confirmed", 
                    hover_name="Country", range_color=[1,3000], 
                    color_continuous_scale='balance', 
                    title='Confirmed Cases in Africa', scope='africa', height=800)
fig.show()

fig = px.choropleth(world, locations="Country", 
                    locationmode='country names', color="Fatalities", 
                    hover_name="Country", range_color=[1,3000], 
                    color_continuous_scale='balance', 
                    title='Fatalities Cases in Africa', scope='africa', height=800)
fig.show()
predict_confirm=datastata.rename(columns={'Date': 'ds', 'Confirmed': 'y'})
predict_fatalities=datastata.rename(columns={'Date': 'ds', 'Fatalities': 'y'})
from fbprophet import Prophet
m = Prophet()
m1 = Prophet()
m.fit(predict_confirm)
m1.fit(predict_fatalities)
future = m.make_future_dataframe(periods=365)
future1 = m1.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast1 = m1.predict(future1)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = plot_plotly(m, forecast) 
py.iplot(fig)
fig = m.plot(forecast,xlabel='Date',ylabel='Preidcted Confirmed')

fig = plot_plotly(m1, forecast1) 
py.iplot(fig)
fig = m.plot(forecast1,xlabel='Date',ylabel='Preidcted Fatality')
