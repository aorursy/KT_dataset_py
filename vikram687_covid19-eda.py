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
train = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
train.head()
train.shape
train.info()
train['Date'] = pd.to_datetime(train['Date'])

train['Date'] = train['Date'].dt.strftime('%m/%d/%Y')
#Checking misisng values

train.isnull().sum()
#replace missing value with 0

train.fillna(0, inplace=True)
train.describe()
#load data for country code

codes = pd.read_csv('/kaggle/input/country-codes/country_stats.csv')
#Join code with train data

train = pd.merge(train, codes[['country_name','country_code3']],left_on=['Country/Region'], right_on=['country_name'], how='left')
train.drop(columns=['country_name'], inplace=True)
train.head(2)
#Top 10 Country for Confirmed

train.groupby(['Country/Region'])['Confirmed'].sum().reset_index().sort_values(by='Confirmed',ascending=False).head(10)
temp=train.groupby(['Country/Region'])['Confirmed'].sum().reset_index().sort_values(by='Confirmed',ascending=False).head(10)

import plotly.express as px

fig = px.bar(temp, x='Country/Region', y='Confirmed')

fig.show()
#Top 10 Country for Confirmed

train.groupby(['Country/Region'])['Deaths'].sum().reset_index().sort_values(by='Deaths',ascending=False).head(10)
temp=train.groupby(['Country/Region'])['Deaths'].sum().reset_index().sort_values(by='Deaths',ascending=False).head(10)

import plotly.express as px

fig = px.bar(temp, x='Country/Region', y='Deaths')

fig.show()
#Top 10 Country for Confirmed

train.groupby(['Country/Region'])['Recovered'].sum().reset_index().sort_values(by='Recovered',ascending=False).head(10)
temp=train.groupby(['Country/Region'])['Recovered'].sum().reset_index().sort_values(by='Recovered',ascending=False).head(10)

import plotly.express as px

fig = px.bar(temp, x='Country/Region', y='Recovered')

fig.show()
train['Active'] = train['Confirmed'] - train['Deaths'] - train['Recovered']
temp=train.groupby(['Date'])['Confirmed','Deaths','Recovered','Active'].sum().reset_index()
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=temp.Date, y=temp['Confirmed'], name="Confirmed", line_color='deepskyblue'))

fig.add_trace(go.Scatter(x=temp.Date, y=temp['Deaths'], name="Deaths",line_color='dimgray'))

fig.add_trace(go.Scatter(x=temp.Date, y=temp['Recovered'], name="Recovered", line_color='red'))

fig.add_trace(go.Scatter(x=temp.Date, y=temp['Active'], name="Active",line_color='yellow'))

fig.update_layout(title_text='Time Series with Rangeslider',xaxis_rangeslider_visible=True)

fig.show()
import plotly.graph_objects as go

from plotly.subplots import make_subplots

temp=train[train['Deaths']>0]

freq = temp['Country/Region'].value_counts().reset_index().rename(columns={"index": "x"})



# Initialize figure with subplots

fig = make_subplots(

    rows=2, cols=2,

    column_widths=[0.6, 0.4],

    row_heights=[0.4, 0.6],

    specs=[[{"type": "scattergeo", "rowspan": 2}, {"type": "bar"}],

           [            None                    , {"type": "surface"}]])



# Add scattergeo globe map of volcano locations

fig.add_trace(

    go.Scattergeo(lat=train["Lat"],

                  lon=train["Long"],

                  mode="markers",

                  hoverinfo="text",

                  showlegend=False,

                  marker=dict(color="crimson", size=4, opacity=0.8)),

    row=1, col=1

)



# Add locations bar chart

fig.add_trace(

    go.Bar(x=freq["x"][0:10],y=freq["Country/Region"][0:10], marker=dict(color="crimson"), showlegend=False),

    row=1, col=2

)



# Update geo subplot properties

fig.update_geos(

    projection_type="orthographic",

    landcolor="white",

    oceancolor="MidnightBlue",

    showocean=True,

    lakecolor="LightBlue"

)



# Rotate x-axis labels

fig.update_xaxes(tickangle=45)



# Set theme, margin, and annotation in layout

fig.update_layout(

    template="plotly_dark",

    margin=dict(r=10, t=25, b=40, l=60),

    annotations=[

        dict(

            

            showarrow=False,

            xref="paper",

            yref="paper",

            x=0,

            y=0)

    ]

)



fig.show()
#Distribution of Fatalities accross the world 

import plotly.express as px

fig = px.density_mapbox(train, lat='Lat', lon='Long', z='Deaths', radius=10,

                        center=dict(lat=0, lon=180), zoom=0,mapbox_style="stamen-terrain")

fig.show()
import folium 

world_map_recovered = folium.Map(location=[30, 0], zoom_start=1.5,tiles='Stamen Toner')

world_data_totaly_recovered=train.copy()

for lat, lon, value, name in zip(world_data_totaly_recovered['Lat'], world_data_totaly_recovered['Long'], world_data_totaly_recovered['Deaths'], 

                                 world_data_totaly_recovered['Country/Region']):

    folium.CircleMarker([lat, lon], radius=3,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>''<strong>Recovered</strong>: ' + str(value) + '<br>'),

                        color='green',fill_color='green',fill_opacity=0.7 ).add_to(world_map_recovered)
world_map_recovered
temp = train.groupby(['Country/Region','country_code3'])['Deaths','Confirmed'].sum().reset_index()

temp['size'] = temp['Deaths'].pow(0.3)

temp.head(2)
import plotly.express as px

fig = px.choropleth(temp, locations="country_code3",

                    color="size", # lifeExp is a column of gapminder

                    hover_name="Country/Region", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.show()
train.groupby(['Country/Region'])['Deaths'].sum().reset_index().sort_values(by='Deaths', ascending=False).head(10).style.background_gradient(cmap='Reds')
train.groupby(['Province/State'])['Confirmed','Deaths','Recovered','Active'].sum().reset_index().sort_values(by='Confirmed', ascending=False).head(20).style.background_gradient(cmap='Reds')
temp3 = train.groupby(['Date','Country/Region'])['Confirmed','Deaths','Recovered','Active'].max().reset_index()

temp3['size'] = temp3['Deaths'].pow(0.4)

temp3.head(10)
import plotly.express as px

df = px.data.gapminder()

fig = px.scatter_geo(temp3, locations="Country/Region", color="Deaths",locationmode='country names',

                     hover_name="Country/Region", size="size",range_color= [0, 1000],

                     animation_frame="Date",color_continuous_scale="portland",

                     projection="natural earth")

fig.show()
import plotly.express as px

df = px.data.gapminder()

fig = px.scatter_geo(temp3, locations="Country/Region", color="Confirmed",locationmode='country names',

                     hover_name="Country/Region", size="size",range_color= [0, 2000],

                     animation_frame="Date",color_continuous_scale="portland",

                     projection="natural earth")

fig.show()
import plotly.express as px

df = px.data.gapminder()

fig = px.scatter_geo(temp3, locations="Country/Region", color="Active",locationmode='country names',

                     hover_name="Country/Region", size="size",range_color= [0, 1000],

                     animation_frame="Date",color_continuous_scale="portland",

                     projection="natural earth")

fig.show()
import plotly.express as px

df = px.data.gapminder()

fig = px.scatter_geo(temp3, locations="Country/Region", color="Recovered",locationmode='country names',

                     hover_name="Country/Region", size="size",range_color= [0, 1000],

                     animation_frame="Date",color_continuous_scale="portland",

                     projection="natural earth")

fig.show()
import plotly.express as px

temp4 = train.groupby(['Date','Country/Region'])['Deaths','Active','Recovered','Confirmed'].mean().reset_index()

fig = px.line(temp4, x="Date", y="Deaths", title='cases of Deaths')

fig.show()
import plotly.graph_objects as go

# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=temp4['Date'], y=temp4['Active'],mode='lines',name='Active Cases'))

fig.add_trace(go.Scatter(x=temp4['Date'], y=temp4['Deaths'],mode='lines+markers',name='Deaths'))

fig.show()
import plotly.graph_objects as go

# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=temp4['Date'], y=temp4['Active'],mode='lines',name='Active Cases'))

fig.add_trace(go.Scatter(x=temp4['Date'], y=temp4['Recovered'],mode='lines+markers',name='Recovered'))

fig.show()
import plotly.graph_objects as go

# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=temp4['Date'], y=temp4['Active'],mode='lines',name='Active Cases'))

fig.add_trace(go.Scatter(x=temp4['Date'], y=temp4['Confirmed'],mode='lines+markers',name='Confirmed'))

fig.show()
# scatter between Confirmed vs Active



import plotly.express as px

fig = px.scatter(x=train['Confirmed'], y=train['Active'])

fig.show()
# scatter between Confirmed vs Active



import plotly.express as px

fig = px.scatter(x=train['Deaths'], y=train['Active'])

fig.show()
temp4 = train.groupby(['Date','Country/Region'])['Deaths','Active'].mean().reset_index()

temp4.head(2)
import plotly.express as px

fig = px.line(temp4, x="Date", y="Deaths", color='Country/Region')

fig.show()
import plotly.express as px

fig = px.line(temp4, x="Date", y="Active", color='Country/Region')

fig.show()
import plotly.express as px

fig = px.line(train, x="Date", y="Confirmed", color='Country/Region')

fig.show()
import plotly.express as px

fig = px.line(train, x="Date", y="Recovered", color='Country/Region')

fig.show()
import plotly.express as px

fig = px.scatter_mapbox(train, lat="Lat", lon="Long", hover_name="Country/Region", hover_data=["Country/Region"],

                        color_discrete_sequence=["fuchsia"], zoom=0.01, height=300)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
train.groupby(['Country/Region'])['Deaths'].sum().reset_index().sort_values(by='Deaths', ascending=False)[0:10]