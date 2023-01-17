import pandas as pd

import plotly.express as px

import json

from geojson import dump

import datetime
data = pd.read_csv('/kaggle/input/co2-ghg-emissionsdata/co2_emission.csv')
data = data.rename(columns={'Entity':'Country', 'Annual CO₂ emissions (tonnes )':'value'})

data.head()
world_data = data.query("Country == 'World'")



fig = px.line(world_data,x='Year', y='value')

fig.update_layout(title='<b>Evolution of CO2 emissions per year in the world</b>',xaxis_title='', yaxis_title='')

fig.show()
europe_data = data.query("Country == ['France','United Kingdom','Spain','Italy','Germany', 'Russia']")



fig = px.line(europe_data,x='Year', y='value', color='Country')

fig.update_layout(title='<b>Evolution of CO2 emissions per year in Europe</b>',xaxis_title='', yaxis_title='')

fig.show()
asia_data = data.query("Country == ['China','Japan','India','Iran','Mongolia']")



fig = px.line(asia_data,x='Year', y='value', color='Country')

fig.update_layout(title='<b>Evolution of CO2 emissions per year in Asia</b>',xaxis_title='', yaxis_title='')

fig.show()
america_data = data.query("Country == ['United States','Canada','Brazil','Mexico','Argentina']")



fig = px.line(america_data,x='Year', y='value', color='Country')

fig.update_layout(title='<b>Evolution of CO2 emissions per year in America</b>',xaxis_title='', yaxis_title='')

fig.show()
other_data = data.query("Country == ['United States','France','China','United Kingdom']")



fig = px.line(other_data,x='Year', y='value', color='Country')

fig.update_layout(title='<b>Evolution of CO2 emissions per year</b>',xaxis_title='', yaxis_title='')

fig.show()
data_2017 = data.query("Country != 'World' and Year=='2017'")

data_2017 = data_2017[~pd.isna(data_2017['Code'])]



fig = px.choropleth(data_2017, locations="Code",

                    color="value",

                    hover_name="Country",

                    color_continuous_scale='Reds',

                   title='<b>CO2 Emission map by country in 2017</b>')

fig.show()
fig = px.pie(data_2017, values='value', names='Country', title='<b>Repartition of CO2 emission by country in 2017</b>')

fig.show()