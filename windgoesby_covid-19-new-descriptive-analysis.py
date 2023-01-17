# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

pd.set_option('display.max_rows', None)

from plotly.subplots import make_subplots

import seaborn as sns

import datetime

import missingno as msno



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read the data of the spread of covid-19

df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df.head()
# View Missing values

msno.matrix(df, labels=True)
print("Before:", df.dtypes.Confirmed)

# Transform type of some columns from float to int (You can chooes one of the following method)

df[['Confirmed', 'Deaths', 'Recovered']] = df[['Confirmed', 'Deaths', 'Recovered']].astype('int')

# df = df.astype({'Confirmed': 'int', 'Deaths': 'int', 'Recovered': 'int'})

print("After:", df.dtypes.Confirmed)
# Standardized country name

df['Country/Region'] = df['Country/Region'].replace('Mainland China', 'China')
# Create a new series called Actived

df['Actived'] = df['Confirmed'] - df['Deaths'] - df['Recovered']

df.head()
# Get newest cases data

data = df[df['ObservationDate'] == max(df['ObservationDate'])].reset_index()

data.head()
# Group by country

world_total_cases = data.groupby('ObservationDate')[['Confirmed', 'Deaths', 'Recovered', 'Actived']].sum().reset_index()

world_total_cases.head()
# Draw table

labels = ["LastUpdate","Confirmed","Actived","Recovered","Deaths"]

values = world_total_cases.loc[0,["ObservationDate","Confirmed","Actived","Recovered","Deaths"]]

fig = go.Figure(data=[go.Table(header=dict(values=labels), cells=dict(values=values))])

fig.update_layout(title='The latest number of cases worldwide : ',)

fig.show()
# Donut chart

labels = ['Actived', 'Recovered', 'Deaths']

values = world_total_cases.loc[0,["Actived","Recovered","Deaths"]]

fig = px.pie(data_frame=world_total_cases, values=values, names=labels, color_discrete_sequence=['rgb(77,146,33)','rgb(69,144,185)','rgb(77,77,77)'], hole=0.7)

fig.update_layout(title="World total cases is " + str(world_total_cases["Confirmed"][0]))

fig.show()
# Spread thread

data_time_series = df.groupby("ObservationDate")[["Confirmed", "Actived", "Recovered", "Deaths"]].sum()

data_time_series.head()
# Confirmed

fig = go.Figure()

fig.add_trace(go.Scatter(x=data_time_series.index, y=data_time_series.Confirmed, mode='lines', name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=data_time_series.index, y=data_time_series.Actived, mode='lines', name='Actived Cases', marker_color='red', line=dict( dash='dot')))

fig.add_trace(go.Scatter(x=data_time_series.index, y=data_time_series.Recovered, mode='lines', name='Recovered Cases', marker_color='green', line=dict( dash='dot')))

fig.add_trace(go.Scatter(x=data_time_series.index, y=data_time_series.Deaths, mode='lines', name='Deaths Cases', marker_color='grey', line=dict( dash='dot')))

fig.update_layout(title='Evolution of Confirmed cases over time in the word', template='plotly_white', yaxis_title="Cases", xaxis_title="Days")

fig.show()
# Actived

fig = go.Figure()

fig.add_trace(go.Scatter(x=data_time_series.index, y=data_time_series.Actived, mode='lines', name='Actived Cases', marker_color='red', line=dict( dash='dot')))

fig.update_layout(title='Evolution of Actived cases over time in the word', template='plotly_dark', yaxis_title="Actived Cases", xaxis_title="Days")

fig.show()
# Recovered

fig = go.Figure()

fig.add_trace(go.Scatter(x=data_time_series.index, y=data_time_series.Recovered, mode='lines', name='Recovered Cases', marker_color='green', line=dict( dash='dot')))

fig.update_layout(title='Evolution of Recovered cases over time in the word', template='plotly_dark', yaxis_title="Recovered Cases", xaxis_title="Days")

fig.show()
# Deaths

fig = go.Figure()

fig.add_trace(go.Scatter(x=data_time_series.index, y=data_time_series.Deaths, mode='lines', name='Deaths Cases', marker_color='grey', line=dict( dash='dot')))

fig.update_layout(title='Evolution of Deaths cases over time in the word', template='plotly_dark', yaxis_title="Deaths Cases", xaxis_title="Days")

fig.show()
# per country

data_per_country = df.groupby(["Country/Region"])["Confirmed","Actived","Recovered","Deaths"].sum().reset_index().sort_values("Confirmed",ascending=False).reset_index(drop=True)

data_per_country.head()
fig = px.choropleth(data_per_country, locations=data_per_country['Country/Region'], color=data_per_country['Confirmed'],locationmode='country names', hover_name=data_per_country['Country/Region'], color_continuous_scale=px.colors.sequential.Tealgrn,template='plotly_dark', )

fig.update_layout(title='Confirmed Cases In Each Country')

fig.show()
# Times Series thread

data_per_country_series = df.groupby(["Country/Region","ObservationDate"])[["Confirmed","Actived","Recovered","Deaths"]].sum().reset_index().sort_values("ObservationDate",ascending=True).reset_index(drop=True)
fig = px.choropleth(

    data_per_country_series, 

    locations=data_per_country_series['Country/Region'], 

    color=data_per_country_series['Confirmed'], 

    locationmode='country names', 

    hover_name=data_per_country_series['Country/Region'], 

    color_continuous_scale=px.colors.sequential.deep, 

    animation_frame='ObservationDate'

)

fig.update_layout(title='Evolution of confirmed cases In Each Country')

fig.show()