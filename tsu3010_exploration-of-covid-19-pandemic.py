##  Library Imports
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from collections import Counter, defaultdict
from plotly import tools
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
import folium
import warnings
warnings.filterwarnings('ignore')
py.init_notebook_mode(connected=True)
import plotly.io as pio
pio.templates.default = "plotly_dark"
# color pallette
cnf = '#FDFBFB' # confirmed - white
dth = '#ff2e63' # death - red
rec = '#21bf73' # recovered - cyan
act = '#fe9801' # active case - yellow
df_master = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
df_master.rename(columns={'Country/Region':'Country'}, inplace=True)
df_master.rename(columns={'Province/State':'State'}, inplace=True)
df_master.rename(columns={'ObservationDate':'Date'}, inplace=True)
# Rename country labels
df_master['Country'].replace(['Mainland China'], 'China',inplace=True)
df_master['Country'].replace(['US'], 'United States',inplace=True)
df_master['Country'].replace(['UK'], 'United Kingdom',inplace=True)
## Load Lat Long 
coordinates = pd.read_csv('/kaggle/input/latitude-and-longitude-for-every-country-and-state/world_country_and_usa_states_latitude_and_longitude_values.csv')
country_coordinates = coordinates[['country_code','latitude','longitude','country']]
# Load Population
population = pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv',converters={'Country': str.strip})
population = population[['Country','Region','Population']]
df_master.head(3)
# Create derived column called Active Cases 
df_master['Active'] = df_master['Confirmed'] - df_master['Recovered'] - df_master['Deaths']
# Convert date column to appropriate data type
df_master['Date'] = pd.to_datetime(df_master['Date'])
temp = df_master.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)
temp.style.background_gradient(cmap='Set1')
cur_date = str(temp['Date'][0].date())
print(f"As of {cur_date}, we have {round((temp['Recovered'][0]/temp['Confirmed'][0])*100,3)}% of the infected who have recovered from the virus globally")
tm = temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'])
fig = px.treemap(tm, path=["variable"], values="value", height=400, width=600,
                 color_discrete_sequence=[rec, act, dth])
title = f"CoVid19 case split as of {cur_date}"
fig.update_layout(title=title)
fig.show()
print(f"We can see from the tree map visualization above that {round((temp['Active'][0]/temp['Confirmed'][0])*100,2)}% infected cases worldwide are currently active.")
temp = df_master.copy()
temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)
temp = temp.groupby(['Country'])['Confirmed','Recovered','Deaths','Active'].sum().sort_values(by='Confirmed',ascending=False).reset_index()
# Merge with country coordinates
temp = temp.merge(country_coordinates, left_on="Country", right_on="country")
population = population[['Country','Region','Population']]
# Merge With Population
temp = temp.merge(population, left_on="Country", right_on="Country")
# Derive Columns 
temp['ConfirmedPer10kPeople'] = ( temp['Confirmed'] / temp['Population'] ) * 10000
temp['DeathsPer10kPeople'] = ( temp['Deaths'] / temp['Population'] ) * 10000
fig = px.choropleth(temp, 
                    locations="Country", 
                    color="Confirmed",
                    locationmode = 'country names',
                    hover_name="Country",
                    range_color=[0,round(temp['Confirmed'].max())], 
                    color_continuous_scale="peach",
                    title = "CoVid-19 : Confirmed Cases")

fig.show()
fig = px.choropleth(temp, 
                    locations="Country", 
                    color="Deaths",
                    locationmode = 'country names',
                    hover_name="Country",
                    range_color=[0,round(temp['Deaths'].max())], 
                    color_continuous_scale='portland',
                    title = "CoVid-19 : Death Cases")

fig.show()
# Filter countries with atleast 5 million people to avoid skewing the scales
is_1mn = temp['Population'] >= 1000000
temp = temp[is_1mn]
fig = px.choropleth(temp, 
                    locations="Country", 
                    color="ConfirmedPer10kPeople",
                    locationmode = 'country names',
                    hover_name="Country",
                    range_color=[0,round(temp['ConfirmedPer10kPeople'].max())], 
                    color_continuous_scale="peach",
                    title = "Confirmed Cases Per 10000 People")

fig.show()
fig = px.choropleth(temp, 
                    locations="Country", 
                    color="DeathsPer10kPeople",
                    locationmode = 'country names',
                    hover_name="Country",
                    range_color=[0,round(temp["DeathsPer10kPeople"].max())], 
                    color_continuous_scale='portland',
                    title = "Deaths Per 10000 People")

fig.show()
def visualize_trends(fig, region, data, done=True):
    trace1 = go.Scatter(x=data['Date'],
                        y=data['Confirmed'],
                        mode='lines+markers',
                        name='Confirmed',
                        marker=dict(color=cnf,))

    trace2 = go.Scatter(x=data['Date'],
                        y=data['Recovered'],
                        mode='lines+markers',
                        name='Recovered',
                        marker=dict(color=rec,))

    trace3 = go.Scatter(x=data['Date'],
                        y=data['Deaths'],
                        mode='lines+markers',
                        name='Deaths',
                        marker=dict(color=dth,))

    trace4 = go.Scatter(x=data['Date'],
                        y=data['Active'],
                        mode='lines+markers',
                        name='Active',
                        marker=dict(color=act,))

    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)
    fig.add_trace(trace4)
    title = f"{region} Trend for CoVid-19 Cases"
    fig.update_layout(xaxis_title="Date", yaxis_title="Count", title=title)
    if done:
        fig.show()
    
temp = df_master.groupby('Date')['Confirmed','Recovered','Deaths','Active'].sum().reset_index()
fig = go.Figure()
visualize_trends(fig=fig,
                 region="Global",
                 data=temp)
is_china = df_master['Country'] == 'China'
temp = df_master[is_china].groupby('Date')['Confirmed','Recovered','Deaths','Active'].sum().reset_index()
fig = go.Figure()
visualize_trends(fig=fig,
                 region="Mainland China",
                 data=temp,
                 done=False)
fig.add_shape(
        dict(
            type="line",
            x0='2020-01-23',
            y0=-10000,
            x1='2020-01-23',
            y1=10000,
            line=dict(
                color="Blue",
                width=3
            )))
fig.add_shape(
        dict(
            type="line",
            x0='2020-02-13',
            y0=50000,
            x1='2020-02-13',
            y1=70000,
            line=dict(
                color="Purple",
                width=3
            )
))
fig.add_shape(
        dict(
            type="line",
            x0='2020-02-20',
            y0=65000,
            x1='2020-02-20',
            y1=85000,
            line=dict(
                color="Green",
                width=3
            )
))
fig.show()
is_row = df_master['Country'] != 'Mainland China'
temp = df_master[is_row].groupby('Date')['Confirmed','Recovered','Deaths','Active'].sum().reset_index()
fig = go.Figure()
visualize_trends(fig=fig,
                 region="Rest of World",
                 data=temp)
is_italy = df_master['Country'] == 'Italy'
temp = df_master[is_italy].groupby('Date')['Confirmed','Recovered','Deaths','Active'].sum().reset_index()
fig = go.Figure()
visualize_trends(fig=fig,
                 region="Italy",
                 data=temp, 
                 done=False)
fig.add_shape(
        dict(
            type="line",
            x0='2020-03-09',
            y0=5000,
            x1='2020-03-09',
            y1=15000,
            line=dict(
                color="deeppink",
                width=3
            )))
fig.show()
is_spain = df_master['Country'] == 'Spain'
temp = df_master[is_spain].groupby('Date')['Confirmed','Recovered','Deaths','Active'].sum().reset_index()
fig = go.Figure()
visualize_trends(fig=fig,
                 region="Spain",
                 data=temp, 
                 done=False)
fig.add_shape(
        dict(
            type="line",
            x0='2020-03-15',
            y0=6000,
            x1='2020-03-15',
            y1=10000,
            line=dict(
                color="deeppink",
                width=3
            )))
fig.show()
is_france = df_master['Country'] == 'France'
temp = df_master[is_france].groupby('Date')['Confirmed','Recovered','Deaths','Active'].sum().reset_index()
fig = go.Figure()
visualize_trends(fig=fig,
                 region="France",
                 data=temp, 
                 done=False)
fig.add_shape(
        dict(
            type="line",
            x0='2020-03-15',
            y0=3000,
            x1='2020-03-15',
            y1=6000,
            line=dict(
                color="deeppink",
                width=3
            )))
fig.show()
is_us = df_master['Country'] == 'United States'
temp = df_master[is_us].groupby('Date')['Confirmed','Recovered','Deaths','Active'].sum().reset_index()
fig = go.Figure()
visualize_trends(fig=fig,
                 region="US",
                 data=temp)
is_uk = df_master['Country'] == 'United Kingdom'
temp = df_master[is_uk].groupby('Date')['Confirmed','Recovered','Deaths','Active'].sum().reset_index()
fig = go.Figure()
visualize_trends(fig=fig,
                 region="UK",
                 data=temp, 
                 done=False)

fig.add_shape(
        dict(
            type="line",
            x0='2020-03-22',
            y0=5000,
            x1='2020-03-22',
            y1=6500,
            line=dict(
                color="deeppink",
                width=3
            )))
fig.show()
is_singapore = df_master['Country'] == 'Singapore'
temp = df_master[is_singapore].groupby('Date')['Confirmed','Recovered','Deaths','Active'].sum().reset_index()
fig = go.Figure()
visualize_trends(fig=fig,
                 region="Singapore",
                 data=temp)
is_sk = df_master['Country'] == 'South Korea'
temp = df_master[is_sk].groupby('Date')['Confirmed','Recovered','Deaths','Active'].sum().reset_index()
fig = go.Figure()
visualize_trends(fig=fig,
                 region="South Korea",
                 data=temp, 
                 done=False)
fig.add_shape(
        dict(
            type="line",
            x0='2020-02-29',
            y0=2000,
            x1='2020-02-29',
            y1=4000,
            line=dict(
                color="Blue",
                width=3
            )))
fig.show()
is_india = df_master['Country'] == 'India'
temp = df_master[is_india].groupby('Date')['Confirmed','Recovered','Deaths','Active'].sum().reset_index()
fig = go.Figure()
visualize_trends(fig=fig,
                 region="India",
                 data=temp, 
                 done=False)

fig.add_shape(
        dict(
            type="line",
            x0='2020-03-22',
            y0=300,
            x1='2020-03-22',
            y1=500,
            line=dict(
                color="deeppink",
                width=3
            )))
fig.show()
## Extract top countries with most cases
temp = df_master.copy()
temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)
top10_df = temp.groupby(['Country'])['Confirmed','Recovered','Deaths','Active'].sum().sort_values(by='Confirmed',ascending=False).head(8)
top10_countries = top10_df.index.values

## Filter for the top 10 countries in base data and plot for confirmed per 10k people and deaths per 10k people
temp = df_master.copy()
## Merge With Population
temp = temp.merge(population, left_on="Country", right_on="Country")
# Derive Columns 
temp['ConfirmedPer10kPeople'] = ( temp['Confirmed'] / temp['Population'] ) * 10000
temp['DeathsPer10kPeople'] = ( temp['Deaths'] / temp['Population'] ) * 10000

# Aggregate at country, date level and filter data for top 10 countries only
temp = temp.groupby(['Country','Date'])['ConfirmedPer10kPeople','DeathsPer10kPeople'].mean().reset_index()
is_top10 = temp['Country'].isin(top10_countries)
temp = temp[is_top10]

# Filter out cases from Feb 25 when multiple countries started showing cases
temp = temp[temp['Date'] >= datetime.date(2020,2,25)]

# Convert date to correct format 
temp["Date"] = pd.to_datetime(temp["Date"] , format="%m/%d/%Y").dt.date
temp.sort_values(by=["Country","Date"]).reset_index(drop=True)
temp["Date"] = temp["Date"].astype(str)
fig = go.Figure()
fig = px.bar(temp,
                x="Country",
                y="ConfirmedPer10kPeople",
                color="Country",
                animation_frame="Date",
                animation_group="Country",
                title="Time Lapse - Confirmed Cases Per 10000 People")
fig.show()
fig = go.Figure()
fig = px.bar(temp,
                x="Country",
                y="DeathsPer10kPeople",
                color="Country",
                animation_frame="Date",
                animation_group="Country",
                title="Time Lapse - Deaths Per 10000 People")
fig.show()
temp = df_master[['State','Country','Date','Confirmed']].copy()

is_hubei = (temp['State'].isin(['Hubei'])) & (temp['Country'].isin(['China']))
temp_hubei = temp[is_hubei]
is_ro_china = (temp['Country'].isin(['China'])) & (temp['State'] != 'Hubei')
temp_ro_china = temp[is_ro_china]
temp_ro_china = temp_ro_china.groupby(['Country','Date'])['Confirmed'].sum().reset_index()

def growth_rate_compute(temp):
    """
    Function to derive the growth rate column for given region
    """
    temp = temp.sort_values(by='Date')
    temp['Confirmed_t-1'] = temp['Confirmed'].shift(1, axis=0)
    temp = temp[temp['Confirmed_t-1'].notna()]
    temp.reset_index(inplace=True, drop=True)
    temp['New_Cases_t'] = temp['Confirmed'] - temp['Confirmed_t-1']
    # Filter the data points on days with New_Cases_t is zero (reporting anomalies)
    temp = temp[temp['New_Cases_t'] != 0.0 ]
    temp['New_Cases_t-1'] = temp['New_Cases_t'].shift(1, axis=0)
    temp['Growth_Rate'] = temp['New_Cases_t'] / temp['New_Cases_t-1']
    # Remove anomalies in growth rate (excess of > 10) 
    temp = temp[temp['Growth_Rate'] <= 20.0 ]
    temp = temp[temp['Growth_Rate'].notna()]
    return temp

temp_hubei = growth_rate_compute(temp=temp_hubei)
temp_ro_china = growth_rate_compute(temp=temp_ro_china)
fig = go.Figure()
#trace1 = go.Bar(x=temp_hubei['Date'], 
 #               y=temp_hubei['Growth_Rate'],
 #               hovertext = "Hubei,China",
 #               marker=dict(color="red",))

trace1 = go.Scatter(x=temp_hubei['Date'],
                    y=temp_hubei['Growth_Rate'],
                    mode='lines+markers',
                    hovertext='Hubei,China',
                    marker=dict(color="red",))

fig.add_trace(trace1)
fig.update_layout(title=' Growth Rate Trend in Hubei, China')
fig.show()
fig = go.Figure()
trace1 = go.Scatter(x=temp_ro_china['Date'],
                    y=temp_ro_china['Growth_Rate'],
                    mode='lines+markers',
                    hovertext='Rest of China',
                    marker=dict(color="orange",))

fig.add_trace(trace1)
fig.update_layout(title=' Growth Rate Trend in China (Outside Hubei)')
fig.show() 
is_italy = temp['Country'] == 'Italy'
temp_italy = temp[is_italy]
temp_italy = temp_italy.groupby(['Country','Date'])['Confirmed'].sum().reset_index()
temp_italy = growth_rate_compute(temp=temp_italy)
fig = go.Figure()
trace1 = go.Scatter(x=temp_italy['Date'],
                    y=temp_italy['Growth_Rate'],
                    mode='lines+markers',
                    hovertext='Italy',
                    marker=dict(color="red",))

fig.add_trace(trace1)
fig.update_layout(title=' Growth Rate Trend in Italy')
fig.show() 
is_spain = temp['Country'] == 'Spain'
temp_spain = temp[is_spain]
temp_spain = temp_spain.groupby(['Country','Date'])['Confirmed'].sum().reset_index()
temp_spain = growth_rate_compute(temp=temp_spain)
fig = go.Figure()
trace1 = go.Scatter(x=temp_spain['Date'],
                    y=temp_spain['Growth_Rate'],
                    mode='lines+markers',
                    hovertext='Spain',
                    marker=dict(color="red",))

fig.add_trace(trace1)
fig.update_layout(title=' Growth Rate Trend in Spain')
fig.show() 
is_us= temp['Country'] == 'United States'
temp_us = temp[is_us]
temp_us = temp_us.groupby(['Country','Date'])['Confirmed'].sum().reset_index()
temp_us = growth_rate_compute(temp=temp_us)
fig = go.Figure()
trace1 = go.Scatter(x=temp_us['Date'],
                    y=temp_us['Growth_Rate'],
                    mode='lines+markers',
                    hovertext='US',
                    marker=dict(color="red",))

fig.add_trace(trace1)
fig.update_layout(title=' Growth Rate Trend in US')
fig.show() 
is_uk = temp['Country'] == 'United Kingdom'
temp_uk = temp[is_uk]
temp_uk = temp_uk.groupby(['Country','Date'])['Confirmed'].sum().reset_index()
temp_uk = growth_rate_compute(temp=temp_uk)
fig = go.Figure()
trace1 = go.Scatter(x=temp_uk['Date'],
                    y=temp_uk['Growth_Rate'],
                    mode='lines+markers',
                    hovertext='United Kingdom',
                    marker=dict(color="red",))

fig.add_trace(trace1)
fig.update_layout(title=' Growth Rate Trend in United Kingdom')
fig.show() 
is_india = temp['Country'] == 'India'
temp_india = temp[is_india]
temp_india = temp_india.groupby(['Country','Date'])['Confirmed'].sum().reset_index()
temp_india = growth_rate_compute(temp=temp_india)
fig = go.Figure()
trace1 = go.Scatter(x=temp_uk['Date'],
                    y=temp_uk['Growth_Rate'],
                    mode='lines+markers',
                    hovertext='India',
                    marker=dict(color="red",))

fig.add_trace(trace1)
fig.update_layout(title=' Growth Rate Trend in India')
fig.show() 
is_sk = temp['Country'] == 'South Korea'
temp_sk = temp[is_sk]
temp_sk = temp_sk.groupby(['Country','Date'])['Confirmed'].sum().reset_index()
temp_sk = growth_rate_compute(temp=temp_sk)
fig = go.Figure()
trace1 = go.Scatter(x=temp_sk['Date'],
                    y=temp_sk['Growth_Rate'],
                    mode='lines+markers',
                    hovertext='South Korea',
                    marker=dict(color="green",))

fig.add_trace(trace1)
fig.update_layout(title=' Growth Rate Trend in South Korea')
fig.show() 