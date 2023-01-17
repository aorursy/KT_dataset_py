# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Disable warnings 
import warnings
warnings.filterwarnings('ignore')
# import data set
df = pd.read_csv("../input/distribution-of-covid19-cases-worldwide/covid_19.csv",encoding='ISO-8859-1')
# Look at dimension of data set and types of each attribute
df.info()
# Summarize attribute distributions of the data frame
df.describe(include='all')
# Take a peek at the first rows of the data
df.head(10)
# Drop columns not used
df_new = df.drop(['day','month','year'], axis=1)
# Rename some features for a practical use
df_new = df_new.rename(columns={"dateRep":"date","countriesAndTerritories":"country","popData2018":"pop",
                                "continentExp":"continent","countryterritoryCode":"ccode"}) 
# format date
df_new['date'] = pd.to_datetime(df_new['date'], format='%d/%m/%Y')
# peek a sample
df_new.sample(10)
# groupby country
df_country = df_new.groupby(['country','pop','continent','ccode'], as_index=False)['cases', 'deaths'].sum()
# dropping not matching countries
df_country = df_country.dropna()
# new columns: population in million, cases x population and deaths x population
df_= df_country.copy()
df_['pop(ml)'] = round((df_['pop']/10**6),2)
df_['cases x pop(ml)'] = round((df_['cases']/df_['pop(ml)']),2)
df_['deaths x pop(ml)'] = round((df_['deaths']/df_['pop(ml)']),2)
df_.sample(5)
# select countries with population > 1 million
df_ = df_[(df_['pop(ml)'] > 1)]
# ranking countries with cases x population
df_c = df_.sort_values(['cases x pop(ml)'], ascending = False).reset_index(drop=True)
print('Top 15 countries with the most cases per population (ml)')
df_c.drop(columns = ['deaths', 'deaths x pop(ml)','pop','continent','ccode']).head(15).style.background_gradient(cmap='cool')

# ranking countries with deaths per population
df_d = df_.sort_values(['deaths x pop(ml)'], ascending = False).reset_index(drop=True)
print('Top 15 countries with the most deaths per population (ml)')
df_d.drop(columns = ['cases', 'cases x pop(ml)','pop','continent','ccode']).head(15).style.background_gradient(cmap='Reds')
# groupby continent
df_continent = df_new.groupby(['continent'], as_index=False)['cases', 'deaths','pop'].sum()
# dropping NA rows
df_continent = df_continent.dropna()
# new columns: population in million, cases x population and deaths x population
df_cont= df_continent.copy()
df_cont['pop(ml)'] = round((df_cont['pop']/10**6),2)
df_cont['cases x pop(ml)'] = round((df_cont['cases']/df_cont['pop(ml)']),2)
df_cont['deaths x pop(ml)'] = round((df_cont['deaths']/df_cont['pop(ml)']),2)
# select countries with population > 1 million
df_cont = df_cont[(df_cont['pop(ml)'] > 1)]
# ranking continent with cases x population
df_ca = df_cont.sort_values(['cases x pop(ml)'], ascending = False).reset_index(drop=True)
print('Continent ranking with the most cases per population (ml)')
df_ca.drop(columns = ['deaths', 'deaths x pop(ml)','pop']).style.background_gradient(cmap='summer')
# ranking continent with deaths per population
df_de = df_cont.sort_values(['deaths x pop(ml)'], ascending = False).reset_index(drop=True)
print('Continent ranking with the most deaths per population (ml)')
df_de.drop(columns = ['cases', 'cases x pop(ml)','pop']).style.background_gradient(cmap='winter')
fig = px.choropleth(df_c, locations="ccode",
                    color="cases x pop(ml)",
                    hover_name="country",
                    color_continuous_scale=px.colors.sequential.Plotly3)

layout = go.Layout(
    title=go.layout.Title(
        text="Covid-19 cases per population (million)",
        x=0.5
    ),
    font=dict(size=14),
    width = 750,
    height = 350,
    margin=dict(l=0,r=0,b=0,t=30)
)

fig.update_layout(layout)

fig.show()
fig = px.choropleth(df_d, locations="ccode",
                    color="deaths x pop(ml)",
                    hover_name="country",
                    color_continuous_scale=px.colors.sequential.Agsunset)

layout = go.Layout(
    title=go.layout.Title(
        text="Covid-19 deaths per population (million)",
        x=0.5
    ),
    font=dict(size=14),
    width = 750,
    height = 350,
    margin=dict(l=0,r=0,b=0,t=30)
)

fig.update_layout(layout)

fig.show()
# groupby country and date
ts_country = df_new.groupby(['country','date'], as_index=False)['cases','deaths'].sum()
# dropping NA rows
ts_country = ts_country.dropna()
ts_country.sample(5)
# create pivot table for cases
covid_c = ts_country.pivot(index='date', columns='country', values='cases')
# select countries to visualize time series
covid_cases = covid_c[['Italy','Spain','United_Kingdom','Germany','France','United_States_of_America','Belgium',
                       'Switzerland','Netherlands','Sweden']]
# cumulative time series 
covid_cases.sort_index().cumsum().iplot(title = 'Time series of cumulative cases per country')
# time series per day
covid_cases.iplot(title = 'Time series of cases per day and per country')
# create pivot table for deaths
covid_d = ts_country.pivot(index='date', columns='country', values='deaths')
# select countries to visualize time series
covid_deaths = covid_d[['Italy','Spain','United_Kingdom','Germany','France','United_States_of_America','Belgium',
                        'Switzerland', 'Netherlands','Sweden']]
# cumulative time series
covid_deaths.sort_index().cumsum().iplot(title = 'Time series of cumulative deaths per country')
# time series per day
covid_deaths.iplot(title = 'Time series of deaths per day and per country')
# groupby continent and date
ts_continent = df_new.groupby(['continent','date'], as_index=False)['cases','deaths'].sum()
# dropping NA rows
ts_continent = ts_continent.dropna()
ts_continent.sample(10)
# create pivot table for cases
covid_C = ts_continent.pivot(index='date', columns='continent', values='cases')
# select countries to visualize time series
covid_C_cases = covid_C[['Africa','America','Asia','Europe','Oceania']]
# cumulative time series 
covid_C_cases.sort_index().cumsum().iplot(title = 'Time series of cumulative cases per continent')
# time series per day
covid_C_cases.iplot(title = 'Time series of cases per day and per continent')
# create pivot table for deaths
covid_C_deaths = ts_continent.pivot(index='date', columns='continent', values='deaths')
# select countries to visualize time series
covid_D = covid_C_deaths[['Africa','America','Asia','Europe','Oceania']]
# cumulative time series
covid_D.sort_index().cumsum().iplot(title = 'Time series of cumulative deaths per continent')
# time series per day
covid_D.iplot(title = 'Time series of deaths per day and per continent')