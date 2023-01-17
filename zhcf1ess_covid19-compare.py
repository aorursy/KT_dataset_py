import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns

from plotnine import *

import plotly.express as px

import folium



from IPython.display import Javascript

from IPython.core.display import display, HTML



cdr = ['#393e46', '#ff2e63', '#30e3ca'] # grey - red - blue

idr = ['#f8b400', '#ff2e63', '#30e3ca'] # yellow - red - blue



s = '#f0134d'

h = '#12cad6'

e = '#4a47a3'

m = '#42e6a4'

c = '#333333'



shemc = [s, h, e, m, c]

sec = [s, e, c]
covid_19 = pd.read_csv('../input/covid19dataset/COVID-19-Compared/corona-virus-report/covid_19_clean_complete.csv', 

                       parse_dates=['Date'])



# 选取较重要的数据

covid_19 = covid_19[['Date', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered']]



# 换个名字(中国大陆念起来怪怪的)

covid_19['Country/Region'] = covid_19['Country/Region'].replace('Mainland China', 'China')



# 重命名

covid_19.columns = ['Date', 'Country', 'Cases', 'Deaths', 'Recovered']



# 按日期和国家分组

covid_19 = covid_19.groupby(['Date', 'Country'])['Cases', 'Deaths', 'Recovered']

covid_19 = covid_19.sum().reset_index()



# 最新数据

c_lat = covid_19[covid_19['Date'] == max(covid_19['Date'])].reset_index()



# 最新数据通过国家分组

c_lat_grp = c_lat.groupby('Country')['Cases', 'Deaths', 'Recovered'].sum().reset_index()



# nth day

covid_19['nth_day'] = (covid_19['Date'] - min(covid_19['Date'])).dt.days



# day by day

c_dbd = covid_19.groupby('Date')['Cases', 'Deaths', 'Recovered'].sum().reset_index()



# nth day

c_dbd['nth_day'] = covid_19.groupby('Date')['nth_day'].max().values



# 病例排名(国家)

temp = covid_19[covid_19['Cases']>0]

c_dbd['n_countries'] = temp.groupby('Date')['Country'].apply(len).values



c_dbd['new_cases'] = c_dbd['Cases'].diff()

c_dbd['new_deaths'] = c_dbd['Deaths'].diff()

c_dbd['epidemic'] = 'COVID-19'



covid_19.head()
ebola_14 = pd.read_csv("../input/covid19dataset/COVID-19-Compared/ebola-outbreak-20142016-complete-dataset/ebola_2014_2016_clean.csv", 

                       parse_dates=['Date'])





ebola_14 = ebola_14[['Date', 'Country', 'No. of confirmed, probable and suspected cases',

                     'No. of confirmed, probable and suspected deaths']]



ebola_14.columns = ['Date', 'Country', 'Cases', 'Deaths']

ebola_14.head()



ebola_14 = ebola_14.groupby(['Date', 'Country'])['Cases', 'Deaths']

ebola_14 = ebola_14.sum().reset_index()



# 用0填充空项

ebola_14['Cases'] = ebola_14['Cases'].fillna(0)

ebola_14['Deaths'] = ebola_14['Deaths'].fillna(0)



# 转换数据类型

ebola_14['Cases'] = ebola_14['Cases'].astype('int')

ebola_14['Deaths'] = ebola_14['Deaths'].astype('int')



# latest

e_lat = ebola_14[ebola_14['Date'] == max(ebola_14['Date'])].reset_index()



# latest grouped by country

e_lat_grp = e_lat.groupby('Country')['Cases', 'Deaths'].sum().reset_index()



# nth day

ebola_14['nth_day'] = (ebola_14['Date'] - min(ebola_14['Date'])).dt.days



# day by day

e_dbd = ebola_14.groupby('Date')['Cases', 'Deaths'].sum().reset_index()



# nth day

e_dbd['nth_day'] = ebola_14.groupby('Date')['nth_day'].max().values



# no. of countries

temp = ebola_14[ebola_14['Cases']>0]

e_dbd['n_countries'] = temp.groupby('Date')['Country'].apply(len).values



e_dbd['new_cases'] = e_dbd['Cases'].diff()

e_dbd['new_deaths'] = e_dbd['Deaths'].diff()

e_dbd['epidemic'] = 'EBOLA'



ebola_14.head()
sars_03 = pd.read_csv("../input/covid19dataset/COVID-19-Compared/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv", 

                       parse_dates=['Date'])



sars_03 = sars_03[['Date', 'Country', 'Cumulative number of case(s)', 

                   'Number of deaths', 'Number recovered']]



sars_03.columns = ['Date', 'Country', 'Cases', 'Deaths', 'Recovered']



sars_03 = sars_03.groupby(['Date', 'Country'])['Cases', 'Deaths', 'Recovered']

sars_03 = sars_03.sum().reset_index()



s_lat = sars_03[sars_03['Date'] == max(sars_03['Date'])].reset_index()



s_lat_grp = s_lat.groupby('Country')['Cases', 'Deaths', 'Recovered'].sum().reset_index()



# nth day

sars_03['nth_day'] = (sars_03['Date'] - min(sars_03['Date'])).dt.days



# day by day

s_dbd = sars_03.groupby('Date')['Cases', 'Deaths', 'Recovered'].sum().reset_index()



# nth day

s_dbd['nth_day'] = sars_03.groupby('Date')['nth_day'].max().values



# no. of countries

temp = sars_03[sars_03['Cases']>0]

s_dbd['n_countries'] = temp.groupby('Date')['Country'].apply(len).values





s_dbd['new_cases'] = s_dbd['Cases'].diff()

s_dbd['new_deaths'] = s_dbd['Deaths'].diff()

s_dbd['epidemic'] = 'SARS'



s_dbd.head()
mers_cntry = pd.read_csv("../input/covid19dataset/COVID-19-Compared/mers-outbreak-dataset-20122019/country_count_latest.csv")

mers_weekly = pd.read_csv("../input/covid19dataset/COVID-19-Compared/mers-outbreak-dataset-20122019/weekly_clean.csv")



mers_weekly['Year-Week'] = mers_weekly['Year'].astype(str) + ' - ' + mers_weekly['Week'].astype(str)

mers_weekly['Date'] = pd.to_datetime(mers_weekly['Week'].astype(str) + 

                                     mers_weekly['Year'].astype(str).add('-1'),format='%V%G-%u')



mers_weekly.head()
mers_cntry.head()
mers_weekly.head()
# COVID19

fig = px.choropleth(c_lat_grp, locations="Country", locationmode='country names',

                    color="Cases", hover_name="Country", 

                    color_continuous_scale="Emrld", title='COVID-19')

fig.update(layout_coloraxis_showscale=False)

fig.show()



# EBOLA

fig = px.choropleth(e_lat_grp, locations="Country", locationmode='country names',

                    color="Cases", hover_name="Country", 

                    color_continuous_scale="Emrld", title='EBOLA 2014')

fig.update(layout_coloraxis_showscale=False)

fig.show()



# SRAS

fig = px.choropleth(s_lat_grp, locations="Country", locationmode='country names',

                    color="Cases", hover_name="Country", 

                    color_continuous_scale="Emrld", title='SARS 2003')

fig.update(layout_coloraxis_showscale=False)

fig.show()



# MERS

fig = px.choropleth(mers_cntry, locations="Country", locationmode='country names',

                    color="Confirmed", hover_name="Country", 

                    color_continuous_scale='Emrld', title='MERS')

fig.update(layout_coloraxis_showscale=False)

fig.show()
# COVID19

fig = px.choropleth(c_lat_grp[c_lat_grp['Deaths']>0], locations="Country", locationmode='country names',

                    color="Deaths", hover_name="Country", 

                    color_continuous_scale="Sunsetdark", title='COVID-19')

fig.update(layout_coloraxis_showscale=False)

fig.show()



# EBOLA

fig = px.choropleth(e_lat_grp[e_lat_grp['Deaths']>0], locations="Country", locationmode='country names',

                    color="Deaths", hover_name="Country", 

                    color_continuous_scale="Sunsetdark", title='EBOLA 2014')

fig.update(layout_coloraxis_showscale=False)

fig.show()



# SRAS

fig = px.choropleth(s_lat_grp[s_lat_grp['Deaths']>0], locations="Country", locationmode='country names',

                    color="Deaths", hover_name="Country", 

                    color_continuous_scale="Sunsetdark", title='SARS 2003')

fig.update(layout_coloraxis_showscale=False)

fig.show()
# COVID19

fig = px.treemap(c_lat_grp.sort_values(by='Cases', ascending=False).reset_index(drop=True), 

                 path=["Country"], values="Cases", title='COVID-19',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.show()



# EBOLA

fig = px.treemap(e_lat_grp.sort_values(by='Cases', ascending=False).reset_index(drop=True), 

                 path=["Country"], values="Cases", title='EBOLA',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.show()



# SRAS

fig = px.treemap(c_lat_grp.sort_values(by='Cases', ascending=False).reset_index(drop=True), 

                 path=["Country"], values="Cases", title='SARS',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.show()



# MERS

fig = px.treemap(mers_cntry, 

                 path=["Country"], values="Confirmed", title='MERS',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.show()
# 总病例数

c_cases = sum(c_lat_grp['Cases'])

c_deaths = sum(c_lat_grp['Deaths'])

c_no_countries = len(c_lat_grp['Country'].value_counts())



s_cases = sum(s_lat_grp['Cases'])

s_deaths = sum(s_lat_grp['Deaths'])

s_no_countries = len(s_lat_grp['Country'].value_counts())



e_cases = sum(e_lat_grp['Cases'])

e_deaths = sum(e_lat_grp['Deaths'])

e_no_countries = len(e_lat_grp['Country'].value_counts())
epidemics = pd.DataFrame({

    'epidemic' : ['COVID-19', 'SARS', 'EBOLA', 'MERS', 'H1N1'],

    'start_year' : [2019, 2003, 2014, 2012, 2009],

    'end_year' : [2020, 2004, 2016, 2017, 2010],

    'confirmed' : [c_cases, s_cases, e_cases, 2494, 6724149],

    'deaths' : [c_deaths, s_deaths, e_deaths, 858, 19654],

    'no_of_countries' : [c_no_countries, s_no_countries, e_no_countries, 27, 178]

})



epidemics['mortality'] = round((epidemics['deaths']/epidemics['confirmed'])*100, 2)

epidemics = epidemics.sort_values('end_year').reset_index(drop=True)

epidemics.head()
fig = px.bar(epidemics.sort_values('confirmed',ascending=False), 

             x="confirmed", y="epidemic", color='epidemic', 

             text='confirmed', orientation='h', title='No. of Cases', 

             range_x=[0,7500000],

             color_discrete_sequence = [h, c, e, s, m])

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
fig = px.bar(epidemics.sort_values('deaths',ascending=False), 

             x="deaths", y="epidemic", color='epidemic', 

             text='deaths', orientation='h', title='No. of Deaths',

             range_x=[0,25000],

             color_discrete_sequence = [h, e, c, m, s])

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
fig = px.bar(epidemics.sort_values('mortality',ascending=False),

             x="mortality", y="epidemic", color='epidemic', 

             text='mortality', orientation='h', title='Moratlity rate', 

             range_x=[0,100],

             color_discrete_sequence = [e, m, s, c, h])

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
fig = px.bar(epidemics.sort_values('no_of_countries', ascending=False),

             x="no_of_countries", y="epidemic", color='epidemic', 

             text='no_of_countries', orientation='h', title='No. of Countries', 

             range_x=[0,200],

             color_discrete_sequence = [h, c, s, m, e])

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
temp = pd.concat([s_dbd, e_dbd, c_dbd], axis=0, sort=True)

                

fig = px.line(temp, x="Date", y="Cases", color='epidemic', 

             title='No. of new cases',

             color_discrete_sequence = sec)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()



fig = px.line(temp, x="Date", y="Deaths", color='epidemic', 

             title='No. of new deaths',

             color_discrete_sequence = sec)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
fig = px.line(temp, x="nth_day", y="Cases", color='epidemic', 

             title='Cases', color_discrete_sequence = sec)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()



fig = px.line(temp, x="nth_day", y="Deaths", color='epidemic', 

             title='Deaths', color_discrete_sequence = sec)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()



fig = px.line(temp, x="nth_day", y="n_countries", color='epidemic', 

             title='No. of Countries', color_discrete_sequence = sec)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
fig = px.scatter(epidemics, x='start_year', y = [1 for i in range(len(epidemics))], 

                 size=epidemics['confirmed']**0.3, color='epidemic', title='Confirmed Cases',

                 color_discrete_sequence = shemc, hover_name='epidemic', height=400,

                 text=epidemics['epidemic']+'<br> Cases : '+epidemics['confirmed'].apply(str))

fig.update_traces(textposition='bottom center')

fig.update_yaxes(showticklabels=False)

fig.update_layout(showlegend=False)

fig.show()



fig = px.scatter(epidemics, x='start_year', y = [1 for i in range(len(epidemics))], 

                 size=epidemics['deaths']**0.5, color='epidemic', title='Deaths',

                 color_discrete_sequence = shemc, hover_name='epidemic', height=400,

                 text=epidemics['epidemic']+'<br> Deaths : '+epidemics['deaths'].apply(str))

fig.update_traces(textposition='bottom center')

fig.update_yaxes(showticklabels=False)

fig.update_layout(showlegend=False)

fig.show()