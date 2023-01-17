# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import plotly

import plotly.offline as offline

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import datetime

import plotly.express as px



offline.init_notebook_mode(connected=True)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



confirmed_raw = 0

deaths_raw = 0

recovered_raw = 0



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if 'confirmed' in filename and 'US' not in filename:

            confirmed_raw = pd.read_csv(str(os.path.join(dirname, filename)))

        elif 'deaths' in filename and 'US' not in filename:

            deaths_raw = pd.read_csv(str(os.path.join(dirname, filename)))

        elif 'recovered' in filename and 'US' not in filename:

            recovered_raw = pd.read_csv(str(os.path.join(dirname, filename)))

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
confirmed_raw.head()
def covid_rename_columns(input_data):

    output_data = input_data.rename(columns={'Province/State':'subregion', 'Country/Region':'country','Lat':'lat', 'Long':'long'})

    return output_data
def covid_melt_data(input_data, value_var_name):

    output_data = input_data.melt(id_vars=['subregion','country','lat','long'], var_name='date_raw', value_name=value_var_name)

    return output_data
def covid_convert_dates(input_data):

    output_data = input_data.assign(date = pd.to_datetime(input_data.date_raw, format='%m/%d/%y'))

    output_data.drop(columns=['date_raw'], inplace = True)

    return output_data
def covid_rearrange_data(input_data, value_var_name):

    output_data = (input_data.filter(['country','subregion','date','lat','long', value_var_name])

                   .sort_values(['country','subregion','date','lat','long'])

                   .reset_index(drop=True))             

    return output_data
def covid_get_data(covid_data, value_var_name):

    covid_data = covid_rename_columns(covid_data)

    covid_data = covid_melt_data(covid_data, value_var_name)

    covid_data = covid_convert_dates(covid_data)

    covid_data = covid_rearrange_data(covid_data, value_var_name)

    return covid_data
covid_confirmed = covid_get_data(confirmed_raw, 'confirmed')

covid_deaths  = covid_get_data(deaths_raw, 'deaths')

covid_recovered = covid_get_data(recovered_raw, 'recovered')
covid_data_merged = (covid_confirmed.merge(covid_deaths, on=['country','subregion','lat','long','date'], how='left')

 .merge(covid_recovered, on=['country','subregion','date', 'lat','long'], how='left'))
covid_data_sum = covid_data_merged.filter(['country','confirmed','deaths','recovered']).groupby('country').agg('max')
def top_ten_sum(df,col):

    return df.filter([col]).sort_values(by=col, ascending=False)[:10]
top_10_confirmed = top_ten_sum(covid_data_sum,'confirmed')

top_10_recovered = top_ten_sum(covid_data_sum,'recovered')

top_10_deaths = top_ten_sum(covid_data_sum,'deaths')
fig = px.bar(top_10_confirmed, x=top_10_confirmed.index, y='confirmed', text='confirmed', color='confirmed')

fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode = 'hide')

fig.show()
fig = px.bar(top_10_recovered, x=top_10_recovered.index, y='recovered', text='recovered', color='recovered')

fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode = 'hide')

fig.show()
fig = px.bar(top_10_deaths, x=top_10_deaths.index, y='deaths', text='deaths', color='deaths')

fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode = 'hide')

fig.show()
total_per_day = (covid_data_merged.filter(['country','date','confirmed','deaths','recovered'])

                 .groupby(['date'])

                 .agg('sum'))
import pandas as pd



fig = go.Figure()



fig.add_trace(go.Scatter(

                x=total_per_day.index,

                y=total_per_day.confirmed,

                name="confirmed",

                mode='lines+markers',

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=total_per_day.index,

                y=total_per_day.recovered,

                name="recovered",

                mode='lines+markers',

                line_color='green',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=total_per_day.index,

                y=total_per_day.deaths,

                name="deaths",

                mode='lines+markers',

                line_color='red',

                opacity=0.8))



fig.update_layout(

    title='Total cases per day from January, 2020 to March, 2020',

    yaxis = dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0,0.85],

    ),



    xaxis = dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0,0.85],

    ),

    

    legend=dict(x=0.029,y=1.038, font_size=10),

    margin=dict(l=100, r=20, t=70, b=70),

    paper_bgcolor='rgb(248, 248, 255)',

    plot_bgcolor='rgb(248, 248, 255)',



)



fig.show()
def country_per_day(df,country):

    return (covid_data_merged.filter(['country','date','confirmed','deaths','recovered'])

            .query("country == '{}'".format(country)).groupby(['date']).agg('sum'))
China = country_per_day(covid_data_merged, 'China')

US = country_per_day(covid_data_merged, 'US')

Italy = country_per_day(covid_data_merged, 'Italy')

Nigeria = country_per_day(covid_data_merged, 'Nigeria')
fig = go.Figure()



fig.add_trace(go.Scatter(

                x=US.index,

                y=US.confirmed,

                name="confirmed (United States)",

                mode='lines+markers',

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=China.index,

                y=China.confirmed,

                name="confirmed (China)",

                mode='lines+markers',

                line_color='green',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=Italy.index,

                y=Italy.confirmed,

                name="confirmed (Italy)",

                mode='lines+markers',

                line_color='purple',

                opacity=0.8))



fig.update_layout(

    title='Total confirmed cases per day for the US, China, and Italy',

    yaxis = dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0,0.85],

    ),



    xaxis = dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0,0.85],

    ),

    

    legend=dict(x=0.029,y=1.038, font_size=10),

    margin=dict(l=100, r=20, t=70, b=70),

    paper_bgcolor='rgb(248, 248, 255)',

    plot_bgcolor='rgb(248, 248, 255)',



)



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

                x=US.index,

                y=US.deaths,

                name="deaths (US)",

                mode='lines+markers',

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=China.index,

                y=China.deaths,

                name="deaths (China)",

                mode='lines+markers',

                line_color='green',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=Italy.index,

                y=Italy.deaths,

                name="deaths (Italy)",

                mode='lines+markers',

                line_color='purple',

                opacity=0.8))





fig.update_layout(

    title='Total death cases per day for the US, China, and Italy',

    yaxis = dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0,0.85],

    ),



    xaxis = dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0,0.85],

    ),

    

    legend=dict(x=0.029,y=1.038, font_size=10),

    margin=dict(l=100, r=20, t=70, b=70),

    paper_bgcolor='rgb(248, 248, 255)',

    plot_bgcolor='rgb(248, 248, 255)',



)



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

                x=US.index,

                y=US.recovered,

                name="recovered (US)",

                mode='lines+markers',

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=China.index,

                y=China.recovered,

                name="recovered (China)",

                mode='lines+markers',

                line_color='green',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=Italy.index,

                y=Italy.recovered,

                name="recovered (Italy)",

                mode='lines+markers',

                line_color='purple',

                opacity=0.8))





fig.update_layout(

    title='Total recovered cases per day for the US, China, and Italy',

    yaxis = dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0,0.85],

    ),



    xaxis = dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0,0.85],

    ),

    

    legend=dict(x=0.029,y=1.038, font_size=10),

    margin=dict(l=100, r=20, t=70, b=70),

    paper_bgcolor='rgb(248, 248, 255)',

    plot_bgcolor='rgb(248, 248, 255)',



)



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

                x=Nigeria.index,

                y=Nigeria.confirmed,

                name="confirmed (Nigeria)",

                mode='lines+markers',

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=China.index,

                y=China.confirmed,

                name="confirmed (China)",

                mode='lines+markers',

                line_color='green',

                opacity=0.8))



fig.update_layout(

    title='Total confirmed cases per day for China, and Nigeria',

    yaxis = dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0,0.85],

    ),



    xaxis = dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0,0.85],

    ),

    

    legend=dict(x=0.029,y=1.038, font_size=10),

    margin=dict(l=100, r=20, t=70, b=70),

    paper_bgcolor='rgb(248, 248, 255)',

    plot_bgcolor='rgb(248, 248, 255)',



)



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

                x=Nigeria.index,

                y=Nigeria.deaths,

                name="deaths (Nigeria)",

                mode='lines+markers',

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=China.index,

                y=China.deaths,

                name="deaths (China)",

                mode='lines+markers',

                line_color='green',

                opacity=0.8))



fig.update_layout(

    title='Total death cases per day for China, and Nigeria',

    yaxis = dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0,0.85],

    ),



    xaxis = dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0,0.85],

    ),

    

    legend=dict(x=0.029,y=1.038, font_size=10),

    margin=dict(l=100, r=20, t=70, b=70),

    paper_bgcolor='rgb(248, 248, 255)',

    plot_bgcolor='rgb(248, 248, 255)',



)



fig.show()