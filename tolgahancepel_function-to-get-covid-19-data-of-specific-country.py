import numpy as np

import pandas as pd

from datetime import date

from datetime import timedelta
def specify_country(f_date, country):

    df_tr = pd.DataFrame()

    l_date = date.today()

    temp_date = f_date

    delta = l_date - f_date

    dlink = f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{temp_date.strftime("%m-%d-%Y")}.csv'

    for x in range(delta.days):

        df = pd.read_csv(dlink)

        

        # these columns for avoiding inconstant column names, hope gonna they're gonna be constant

        country_col = [col for col in df.columns if 'Country' in col]

        country_col = "".join(country_col)

        province_col = [col for col in df.columns if 'Province' in col]

        province_col = "".join(province_col)

        

        df = df[(df[country_col] == country) & ((df[province_col].isnull()) | (df[province_col] == country))]

        df['Date'] = temp_date

        df_tr = df_tr.append(df)

        dlink = list(dlink)

        temp_date = temp_date + timedelta(days = 1)

        dlink[112:122] = temp_date.strftime("%m-%d-%Y")

        dlink = "".join(dlink)

    df_tr.reset_index(inplace = True)

    return df_tr
import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot
#You can edit: starting date and country name

f_date = date(2020, 3, 13)

country = 'Turkey'



df_turkey = specify_country(f_date, country)

df_turkey = df_turkey[['Confirmed','Deaths', 'Recovered', 'Date']]

df_turkey
fig = go.Figure()



trace0 = go.Scatter(

    x = df_turkey['Date'],

    y = df_turkey['Confirmed'],

    name = "Cases",

    fill='tozeroy',

    mode = 'lines+markers',

    hovertemplate = 'Cases: %{y}<br>%{x}',

    marker=dict(

        size=8,

        color = "#1f77b4"

    )

)



trace1 = go.Scatter(

    x = df_turkey['Date'],

    y = df_turkey['Deaths'],

    name = "Deaths",

    fill='tozeroy',

    mode = 'lines+markers',

    hovertemplate = 'Deaths: %{y}<br>%{x}',

    marker=dict(

        size=8,

        color = "#d62728"

    )

)



trace2 = go.Scatter(

    x = df_turkey['Date'],

    y = df_turkey['Recovered'],

    name = "Recovered",

    fill='tozeroy',

    mode = 'lines+markers',

    hovertemplate = 'Recovered: %{y}<br>%{x}',

    marker=dict(

        size=8,

        color = "#00b200"

    )

)





data = [trace0, trace1, trace2]

layout = go.Layout(title = 'Cumulative COVID-19 cases in Turkey',

              xaxis = dict(title = 'Date'),

              yaxis = dict(title = 'Confirmed Cases'),)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
f_date = date(2020, 2, 26)

country = 'Brazil'



df_brazil = specify_country(f_date, country)

df_brazil = df_brazil[['Confirmed','Deaths', 'Recovered', 'Date']]

df_brazil
fig = go.Figure()



trace0 = go.Scatter(

    x = df_brazil['Date'],

    y = df_brazil['Confirmed'],

    name = "Cases",

    fill='tozeroy',

    mode = 'lines+markers',

    hovertemplate = 'Cases: %{y}<br>%{x}',

    marker=dict(

        size=8,

        color = "#1f77b4"

    )

)



trace1 = go.Scatter(

    x = df_brazil['Date'],

    y = df_brazil['Deaths'],

    name = "Deaths",

    fill='tozeroy',

    mode = 'lines+markers',

    hovertemplate = 'Deaths: %{y}<br>%{x}',

    marker=dict(

        size=8,

        color = "#d62728"

    )

)



trace2 = go.Scatter(

    x = df_brazil['Date'],

    y = df_brazil['Recovered'],

    name = "Recovered",

    fill='tozeroy',

    mode = 'lines+markers',

    hovertemplate = 'Recovered: %{y}<br>%{x}',

    marker=dict(

        size=8,

        color = "#00b200"

    )

)





data = [trace0, trace1, trace2]

layout = go.Layout(title = 'Cumulative COVID-19 cases in Brazil',

              xaxis = dict(title = 'Date'),

              yaxis = dict(title = 'Confirmed Cases'),)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
f_date = date(2020, 2, 27)

country = 'Netherlands'



df_netherlands = specify_country(f_date, country)

df_netherlands = df_netherlands[['Confirmed','Deaths', 'Recovered', 'Date']]

df_netherlands
fig = go.Figure()



trace0 = go.Scatter(

    x = df_netherlands['Date'],

    y = df_netherlands['Confirmed'],

    name = "Cases",

    fill='tozeroy',

    mode = 'lines+markers',

    hovertemplate = 'Cases: %{y}<br>%{x}',

    marker=dict(

        size=8,

        color = "#1f77b4"

    )

)



trace1 = go.Scatter(

    x = df_netherlands['Date'],

    y = df_netherlands['Deaths'],

    name = "Deaths",

    fill='tozeroy',

    mode = 'lines+markers',

    hovertemplate = 'Deaths: %{y}<br>%{x}',

    marker=dict(

        size=8,

        color = "#d62728"

    )

)



trace2 = go.Scatter(

    x = df_netherlands['Date'],

    y = df_netherlands['Recovered'],

    name = "Recovered",

    fill='tozeroy',

    mode = 'lines+markers',

    hovertemplate = 'Recovered: %{y}<br>%{x}',

    marker=dict(

        size=8,

        color = "#00b200"

    )

)





data = [trace0, trace1, trace2]

layout = go.Layout(title = 'Cumulative COVID-19 cases in Netherlands',

              xaxis = dict(title = 'Date'),

              yaxis = dict(title = 'Confirmed Cases'),)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
f_date = date(2020, 1, 22)

country = 'Germany'



df_germany = specify_country(f_date, country)

df_germany = df_germany[['Confirmed','Deaths', 'Recovered', 'Date']]

df_germany
fig = go.Figure()



trace0 = go.Scatter(

    x = df_germany['Date'],

    y = df_germany['Confirmed'],

    name = "Cases",

    fill='tozeroy',

    mode = 'lines+markers',

    hovertemplate = 'Cases: %{y}<br>%{x}',

    marker=dict(

        size=8,

        color = "#1f77b4"

    )

)



trace1 = go.Scatter(

    x = df_germany['Date'],

    y = df_germany['Deaths'],

    name = "Deaths",

    fill='tozeroy',

    mode = 'lines+markers',

    hovertemplate = 'Deaths: %{y}<br>%{x}',

    marker=dict(

        size=8,

        color = "#d62728"

    )

)



trace2 = go.Scatter(

    x = df_germany['Date'],

    y = df_germany['Recovered'],

    name = "Recovered",

    fill='tozeroy',

    mode = 'lines+markers',

    hovertemplate = 'Recovered: %{y}<br>%{x}',

    marker=dict(

        size=8,

        color = "#00b200"

    )

)





data = [trace0, trace1, trace2]

layout = go.Layout(title = 'Cumulative COVID-19 cases in Germany',

              xaxis = dict(title = 'Date'),

              yaxis = dict(title = 'Confirmed Cases'),)

fig = go.Figure(data=data, layout=layout)

iplot(fig)