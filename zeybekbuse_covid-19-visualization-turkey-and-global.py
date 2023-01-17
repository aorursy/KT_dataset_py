# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import datetime
import plotly.express as px
import plotly.graph_objects as go
data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data_line_list = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
data_open_list = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')
data_confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
data_deaths = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
data_recovered = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
data_confirmed.head(2)
#data_line_list.head(2)
data.info()
data_turkey = data[data['Country/Region'] == 'Turkey']
data_turkey

df = data_turkey.copy()
df["Date"] = pd.to_datetime(df["ObservationDate"]).dt.strftime('%Y-%m-%d')
df
df_total = df.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum()
df_total.reset_index(inplace=True)
df_total['Active'] = df_total['Confirmed'] - df_total['Deaths'] - df_total['Recovered']
df_total
fig = px.line(df_total,x='Date',y=['Confirmed','Deaths','Recovered','Active'],title='Total Cases in Turkey')
fig.data[0].update(mode='markers+lines')
fig.data[1].update(mode='markers+lines')
fig.data[2].update(mode='markers+lines')
fig.data[3].update(mode='markers+lines')
fig.update_layout(template='ggplot2')
fig.show()
values = [df_total['Deaths'].iloc[-1], df_total['Recovered'].iloc[-1]
         , df_total['Active'].iloc[-1]]

fig = px.pie(df_total, values=values, names=['Deaths','Recovered','Active']
             , title='Distribution of Cases in Turkey - {}'.format(df_total['Date'].iloc[-1])
             , color=['Deaths','Recovered','Active'],color_discrete_map={'Deaths':'rgb(238,77,90)',
                                                                         'Recovered': 'rgb(129,180,227)',
                                                                         'Active':'rgb(209,175,232)'})

#fig = go.Figure(data=[go.Pie(labels=['Deaths','Recovered','Active'], values=values, pull=[0, 0.2, 0])])
fig.show()
df_total['New Cases'] = df_total['Confirmed'] - df_total['Confirmed'].shift(periods=1,fill_value=0)
df_total['New Deaths'] = df_total['Deaths'] - df_total['Deaths'].shift(periods=1,fill_value=0)
df_total['Recovered_New'] = df_total['Recovered'] - df_total['Recovered'].shift(periods=1,fill_value=0)
df_total['Mortality Rate'] = (df_total['Deaths']/df_total['Confirmed'])*100
df_total['Recovery Rate'] = (df_total['Recovered']/df_total['Confirmed'])*100
df_total
print(px.colors.sequential.Viridis) #Color Scale link: https://plotly.com/python/builtin-colorscales/
fig = px.bar(df_total, x='Date', y='New Cases', title='Number of Cases Per Day in Turkey'
             , color_discrete_sequence=['rgb(131, 75, 160)'])
fig.update_layout(template='ggplot2')
fig.show()
fig = px.bar(df_total, x='Date', y='New Deaths', title='Number of Deaths Per Day in Turkey'
             , color_discrete_sequence=['rgb(235, 74, 64)'])
fig.update_layout(template='ggplot2')
fig.show()
fig = px.bar(df_total, x='Date', y='Recovered_New', title='Number of Recovered Per Day in Turkey'
             , color_discrete_sequence=['rgb(0,152,255)'])
fig.update_layout(template='ggplot2')
fig.show()
fig = px.line(df_total,x='Date',y=['New Cases','Recovered_New'],title='Daily New Confirmed Cases vs Recovered in Turkey')
fig.data[0].update(mode='markers+lines',line=dict(color='firebrick'))
fig.data[1].update(mode='markers+lines',line=dict(color='rgb(115, 154, 228)'))
fig.update_layout(template='ggplot2')
fig.show()
fig = px.line(df_total,x='Date',y=['Mortality Rate'],title='Mortality Rate in Turkey')
fig.data[0].update(mode='markers+lines',line=dict(color='firebrick'))
fig.update_layout(template='ggplot2')
fig.show()
fig = px.line(df_total,x='Date',y=['Recovery Rate'],title='Recovery Rate in Turkey')
fig.data[0].update(mode='markers+lines',line=dict(color='rgb(115, 154, 228)'))
fig.update_layout(template='ggplot2')
fig.show()
df_country = pd.read_csv('../input/corona-virus-report/country_wise_latest.csv')
df_confirmed20 = df_country.sort_values('Confirmed',ascending = False).head(20)
df_Deaths20 = df_country.sort_values('Deaths',ascending = False).head(20)
df_recovered20 = df_country.sort_values('Recovered',ascending = False).head(20)
df_active20 = df_country.sort_values('Active',ascending = False).head(20)
df_newcases20 = df_country.sort_values('New cases',ascending = False).head(20)
df_newcases20
fig = px.pie(df_confirmed20, values='Confirmed', names='Country/Region'
             , title='Top 20 Countries (Confirmed Cases)', color_discrete_sequence=px.colors.sequential.Viridis)
fig.show()
fig = px.bar(df_Deaths20, x='Country/Region', y='Deaths'
             , title='Top 20 Countries (Deaths Cases)')
fig.update_traces(marker_color='rgb(56,178,163)')
fig.update_layout(template='ggplot2')
fig.show()
fig = px.bar(df_recovered20, x='Country/Region', y='Recovered'
             , title='Top 20 Countries (Recovered Cases)')
fig.update_traces(marker_color='rgb(252,146,114)')
fig.update_layout(template='ggplot2')
fig.show()
fig = px.bar(df_active20, x='Country/Region', y='Active'
             , title='Top 20 Countries (Active Cases)')
fig.update_traces(marker_color='rgb(185,152,221)')
fig.update_layout(template='ggplot2')
fig.show()
fig = px.bar(df_newcases20, x='Country/Region', y='New cases'
             , title='Top 20 Countries (New Cases)')
fig.update_traces(marker_color='rgb(78,179,211)')
fig.update_layout(template='ggplot2')
fig.show()
df_comp_ = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

df_comp = df_comp_.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()
df_comp.reset_index(inplace=True)

df_comp
fig = px.line(df_comp,x='Date',y=['Confirmed'],title='Total Confirmed Cases')
fig.data[0].update(mode='markers+lines',line=dict(color='rgb(242, 211, 56)'))
fig.update_layout(template='plotly_white')
fig.show()
fig = px.line(df_comp,x='Date',y=['Recovered'],title='Total Recovered Cases')
fig.data[0].update(mode='markers+lines',line=dict(color='rgb(123, 204, 196)'))
fig.update_layout(template='plotly_white')
fig.show()
fig = px.line(df_comp,x='Date',y=['Deaths'],title='Total Deaths Cases')
fig.data[0].update(mode='markers+lines',line=dict(color='rgb(242, 143, 56)'))
fig.update_layout(template='plotly_white')
fig.show()
fig = px.line(df_comp,x='Date',y=['Active'],title='Total Active Cases')
fig.data[0].update(mode='markers+lines',line=dict(color='rgb(201, 148, 199)'))
fig.update_layout(template='plotly_white')
fig.show()
fig = px.scatter_geo(df_country, locations="Country/Region", locationmode='country names'
                     , color="WHO Region", hover_name="Country/Region", size="Confirmed",
                     projection="natural earth", title="Confirmed Cases")
fig.show()
fig = px.scatter_geo(df_country, locations="Country/Region", locationmode='country names'
                     , color="WHO Region", hover_name="Country/Region", size="Recovered",
                     projection="natural earth", title="Recovered Cases")
fig.show()
fig = px.scatter_geo(df_country, locations="Country/Region", locationmode='country names'
                     , color="WHO Region", hover_name="Country/Region", size="Deaths",
                     projection="natural earth", title="Deaths Cases")
fig.show()
fig = px.scatter_geo(df_country, locations="Country/Region", locationmode='country names'
                     , color="WHO Region", hover_name="Country/Region", size="Active",
                     projection="natural earth", title="Active Cases")
fig.show()
fig = px.choropleth(df_comp_, locations="Country/Region", locationmode='country names',
                    color=np.log(df_comp_["Confirmed"]), 
                    #color="Confirmed",
                    hover_name="Country/Region", 
                    color_continuous_scale=px.colors.sequential.OrRd,
                    animation_frame="Date", title="Confirmed Cases Over Time")
fig.update(layout_coloraxis_showscale=False)
fig.update_layout(
    geo=dict(
        showframe=False,
    ))
fig.show()
fig = go.Figure(data=go.Choropleth(
    locations = df_country['Country/Region'], locationmode='country names',
    z = df_country['Confirmed'],
    text = df_country['Country/Region'],
    #colorscale = 'Blues',
    colorscale = 'Teal',
    autocolorscale=False,
    #reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'Confirmed'
))

fig.update_layout(
    title_text='Confirmed Cases',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ))
fig.show()
fig = go.Figure(data=go.Choropleth(
    locations = df_country['Country/Region'], locationmode='country names',
    z = df_country['Confirmed'],
    text = df_country['Country/Region'],
    #colorscale = 'YlGnBu',
    colorscale = 'Teal',
    autocolorscale=False,
    #reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'Confirmed'
))

fig.update_layout(
    title_text='Confirmed Cases (Different projection type)',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='orthographic' #different projection type
    ))
fig.show()