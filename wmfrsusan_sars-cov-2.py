# ----- Libraries -----



# essential

import json

import random

from urllib.request import urlopen



# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.figure_factory as ff

import folium



# color pallette

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow



# converter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   



# hide warnings

import warnings

warnings.filterwarnings('ignore')



# html embedding

from IPython.display import Javascript

from IPython.core.display import display

from IPython.core.display import HTML
# importing datasets

full_table = pd.read_csv('../input/covid_19_clean_complete.csv', 

                         parse_dates=['Date'])

# full_table.head()

# full_table.isna().sum()
# cases 

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']



# replacing Mainland china with just China

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

full_table[['Province/State']] = full_table[['Province/State']].fillna('')

full_table[cases] = full_table[cases].fillna(0)
# cases in the ships

ship = full_table[full_table['Province/State'].str.contains('Grand Princess')|full_table['Country/Region'].str.contains('Cruise Ship')]



# china and the row

china = full_table[full_table['Country/Region']=='China']

row = full_table[full_table['Country/Region']!='China']



# latest

full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()

china_latest = full_latest[full_latest['Country/Region']=='China']

row_latest = full_latest[full_latest['Country/Region']!='China']



# latest condensed

flg = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)

temp.style.background_gradient(cmap='Pastel1')
us_temp = full_table[full_table['Country/Region'] == 'US']

us_temp = us_temp.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

us_temp = us_temp[us_temp['Date']==max(us_temp['Date'])].reset_index(drop=True)

us_temp.style.background_gradient(cmap='Pastel1')
# active + recovered + deaths = confirmed

# death rate = deaths/confirmed



us_temp = full_table[full_table['Country/Region'] == 'US']

us_temp = us_temp.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

us_temp = us_temp[us_temp['Date']==max(us_temp['Date'])].reset_index(drop=True)

us_temp.style.background_gradient(cmap='Pastel1')





cn_temp = full_table[full_table['Country/Region'] == 'China']

cn_temp = cn_temp.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

cn_temp = cn_temp[cn_temp['Date']==max(cn_temp['Date'])].reset_index(drop=True)

cn_temp.style.background_gradient(cmap='Pastel1')



ita_temp = full_table[full_table['Country/Region'] == 'Italy']

ita_temp = ita_temp.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

ita_temp = ita_temp[ita_temp['Date']==max(ita_temp['Date'])].reset_index(drop=True)

ita_temp.style.background_gradient(cmap='Pastel1')



ira_temp = full_table[full_table['Country/Region'] == 'Iran']

ira_temp = ira_temp.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

ira_temp = ira_temp[ira_temp['Date']==max(ira_temp['Date'])].reset_index(drop=True)

ira_temp.style.background_gradient(cmap='Pastel1')



kr_temp = full_table[full_table['Country/Region'] == 'Korea, South']

kr_temp = kr_temp.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

kr_temp = kr_temp[kr_temp['Date']==max(kr_temp['Date'])].reset_index(drop=True)

kr_temp.style.background_gradient(cmap='Pastel1')



sp_temp = full_table[full_table['Country/Region'] == 'Spain']

sp_temp = sp_temp.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

sp_temp = sp_temp[sp_temp['Date']==max(sp_temp['Date'])].reset_index(drop=True)

sp_temp.style.background_gradient(cmap='Pastel1')







ratios = {'death-to-case ratio': [(temp.Deaths/temp.Confirmed)[0], 

                                  (us_temp.Deaths/us_temp.Confirmed)[0],

                                  (cn_temp.Deaths/cn_temp.Confirmed)[0],

                                  (ita_temp.Deaths/ita_temp.Confirmed)[0],

                                  (ira_temp.Deaths/ira_temp.Confirmed)[0],

                                  (kr_temp.Deaths/kr_temp.Confirmed)[0],

                                  (sp_temp.Deaths/sp_temp.Confirmed)[0]

                                 ], 

          'death-to-recovery ratio': [(temp.Deaths/temp.Recovered)[0], 

                                  (us_temp.Deaths/us_temp.Recovered)[0],

                                  (cn_temp.Deaths/cn_temp.Recovered)[0],

                                  (ita_temp.Deaths/ita_temp.Recovered)[0],

                                  (ira_temp.Deaths/ira_temp.Recovered)[0],

                                  (kr_temp.Deaths/kr_temp.Recovered)[0],

                                  (sp_temp.Deaths/sp_temp.Recovered)[0]

                                 ],

          'recovery-to-case ratio': [(temp.Recovered/temp.Confirmed)[0], 

                                  (us_temp.Recovered/us_temp.Confirmed)[0],

                                  (cn_temp.Recovered/cn_temp.Confirmed)[0],

                                  (ita_temp.Recovered/ita_temp.Confirmed)[0],

                                  (ira_temp.Recovered/ira_temp.Confirmed)[0],

                                  (kr_temp.Recovered/kr_temp.Confirmed)[0],

                                  (sp_temp.Recovered/sp_temp.Confirmed)[0]

                                 ]}

ratio_df = pd.DataFrame(data=ratios, index=['WW','US','China','Italy','Iran','S. Korea','Spain'])

ratio_df.style.background_gradient(cmap='Reds')
temp_f = flg.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f.head(10).style.background_gradient(cmap='Reds')
temp = full_table.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case',

             title='Worldwide Cases over time', color_discrete_sequence = [rec, dth, act])

fig.show()
temp = full_table[full_table['Country/Region'] == 'China'].groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case',

             title='China Cases over time', color_discrete_sequence = [rec, dth, act])

fig.show()

temp = full_table[full_table['Country/Region'] == 'Italy'].groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case',

             title='Italy Cases over time', color_discrete_sequence = [rec, dth, act])

fig.show()



temp = full_table[full_table['Country/Region'] == 'Iran'].groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case',

             title='Iran Cases over time', color_discrete_sequence = [rec, dth, act])

fig.show()



temp = full_table[full_table['Country/Region'] == 'Spain'].groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case',

             title='Spain Cases over time', color_discrete_sequence = [rec, dth, act])

fig.show()



temp = full_table[full_table['Country/Region'] == 'Korea, South'].groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case',

             title='S. Korea Cases over time', color_discrete_sequence = [rec, dth, act])

fig.show()



temp = full_table[full_table['Country/Region'] == 'US'].groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case',

             title='US Cases over time', color_discrete_sequence = [rec, dth, act])

fig.show()
fig = px.bar(flg.sort_values('Confirmed', ascending=False).head(10).sort_values('Confirmed', ascending=True), 

             x="Confirmed", y="Country/Region", title='Confirmed Cases', text='Confirmed', orientation='h', 

             width=1000, height=500, range_x = [0, max(flg['Confirmed'])+10000])

fig.update_traces(marker_color='#46cdcf', opacity=0.8, textposition='outside')

fig.show()
fig = px.bar(flg.sort_values('Recovered', ascending=False).head(10).sort_values('Recovered', ascending=True), 

             x="Recovered", y="Country/Region", title='Recovered', text='Recovered', orientation='h', 

             width=1000, height=500, range_x = [0, max(flg['Recovered'])+10000])

fig.update_traces(marker_color=rec, opacity=0.6, textposition='outside')

fig.show()
fig = px.bar(flg.sort_values('Deaths', ascending=False).head(10).sort_values('Deaths', ascending=True), 

             x="Deaths", y="Country/Region", title='Deaths', text='Deaths', orientation='h', 

             width=1000, height=500, range_x = [0, max(flg['Deaths'])+500])

fig.update_traces(marker_color=dth, opacity=0.6, textposition='outside')

fig.show()
fig = px.bar(flg.sort_values('Active', ascending=False).head(10).sort_values('Active', ascending=True), 

             x="Active", y="Country/Region", title='Active', text='Active', orientation='h', 

             width=1000, height=500, range_x = [0, max(flg['Active'])+3000])

fig.update_traces(marker_color='#f0134d', opacity=0.6, textposition='outside')

fig.show()
temp = full_table.groupby(['Date', 'Country/Region'])['Confirmed'].sum().reset_index().sort_values('Date', ascending=False)

key_countries = ['US', 'China', 'Italy','Korea, South','Iran','Spain']

temp_key = temp[temp['Country/Region'].isin(key_countries)]

px.line(temp_key, x="Date", y="Confirmed", color='Country/Region', title='Cases Spread', height=600)
temp = full_table.groupby(['Date', 'Country/Region'])['Confirmed'].sum().reset_index().sort_values('Date', ascending=False)

key_countries = ['US', 'Italy','Korea, South','Iran','Spain']

temp_key = temp[temp['Country/Region'].isin(key_countries)]

px.line(temp_key, x="Date", y="Confirmed", color='Country/Region', title='Cases Spread (excluding China)', height=600)
temp = full_table.groupby(['Date', 'Country/Region'])['Confirmed'].sum().reset_index().sort_values('Date', ascending=False)

temp_key = temp[temp['Country/Region'] == 'US']

px.line(temp_key, x="Date", y="Confirmed", color='Country/Region', title='Cases Spread (only US)', height=600)
# In China

temp = china.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

temp = temp.reset_index()

temp = temp.melt(id_vars="Date", 

                 value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(temp, x="Date", y="value", color='variable', 

             title='In China',

             color_discrete_sequence=[cnf, dth, rec])

fig.update_layout(barmode='group')

fig.show()



#-----------------------------------------------------------------------------



# ROW

temp = row.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

temp = temp.reset_index()

temp = temp.melt(id_vars="Date", 

                 value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(temp, x="Date", y="value", color='variable', 

             title='Outside China',

             color_discrete_sequence=[cnf, dth, rec])

fig.update_layout(barmode='group')

fig.show()
def from_china_or_not(row):

    if row['Country/Region']=='China':

        return 'From China'

    else:

        return 'Outside China'

    

temp = full_table.copy()

temp['Region'] = temp.apply(from_china_or_not, axis=1)

temp = temp.groupby(['Region', 'Date'])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()

mask = temp['Region'] != temp['Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



fig = px.bar(temp, x='Date', y='Confirmed', color='Region', barmode='group', 

             text='Confirmed', title='Confirmed', color_discrete_sequence= [cnf, dth, rec])

fig.update_traces(textposition='outside')

fig.show()



fig = px.bar(temp, x='Date', y='Deaths', color='Region', barmode='group', 

             text='Confirmed', title='Deaths', color_discrete_sequence= [cnf, dth, rec])

fig.update_traces(textposition='outside')

fig.update_traces(textangle=-90)

fig.show()
gdf = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()

gdf = gdf.reset_index()



temp = gdf[gdf['Country/Region']=='China'].reset_index()

temp = temp.melt(id_vars='Date', value_vars=['Confirmed', 'Deaths', 'Recovered'],

                var_name='Case', value_name='Count')

fig = px.bar(temp, x="Date", y="Count", color='Case', facet_col="Case",

            title='China', color_discrete_sequence=[cnf, dth, rec])

fig.show()



temp = gdf[gdf['Country/Region']!='China'].groupby('Date').sum().reset_index()

temp = temp.melt(id_vars='Date', value_vars=['Confirmed', 'Deaths', 'Recovered'],

                var_name='Case', value_name='Count')

fig = px.bar(temp, x="Date", y="Count", color='Case', facet_col="Case",

             title='Outside China', color_discrete_sequence=[cnf, dth, rec])

fig.show()
### Key Countries
temp = gdf[gdf['Country/Region']=='US'].groupby('Date').sum().reset_index()

temp = temp.melt(id_vars='Date', value_vars=['Confirmed', 'Deaths', 'Recovered'],

                var_name='Case', value_name='Count')

fig = px.bar(temp, x="Date", y="Count", color='Case', facet_col="Case",

             title='US', color_discrete_sequence=[cnf, dth, rec])

fig.show()



temp = gdf[gdf['Country/Region']!='Italy'].groupby('Date').sum().reset_index()

temp = temp.melt(id_vars='Date', value_vars=['Confirmed', 'Deaths', 'Recovered'],

                var_name='Case', value_name='Count')

fig = px.bar(temp, x="Date", y="Count", color='Case', facet_col="Case",

             title='Italy', color_discrete_sequence=[cnf, dth, rec])

fig.show()



temp = gdf[gdf['Country/Region']!='Iran'].groupby('Date').sum().reset_index()

temp = temp.melt(id_vars='Date', value_vars=['Confirmed', 'Deaths', 'Recovered'],

                var_name='Case', value_name='Count')

fig = px.bar(temp, x="Date", y="Count", color='Case', facet_col="Case",

             title='Iran', color_discrete_sequence=[cnf, dth, rec])

fig.show()



temp = gdf[gdf['Country/Region']!='Spain'].groupby('Date').sum().reset_index()

temp = temp.melt(id_vars='Date', value_vars=['Confirmed', 'Deaths', 'Recovered'],

                var_name='Case', value_name='Count')

fig = px.bar(temp, x="Date", y="Count", color='Case', facet_col="Case",

             title='Spain', color_discrete_sequence=[cnf, dth, rec])

fig.show()



key_countries = ['US', 'Italy','Korea, South','Iran','Spain']

temp = gdf[gdf['Country/Region'].isin(key_countries)]



temp_confirmed = temp.melt(id_vars=['Date', 'Country/Region'], value_vars=['Confirmed'], var_name='Case', value_name='Count')

temp_deaths = temp.melt(id_vars=['Date', 'Country/Region'], value_vars=['Deaths'], var_name='Case', value_name='Count')

temp_recovered = temp.melt(id_vars=['Date', 'Country/Region'], value_vars=['Recovered'], var_name='Case', value_name='Count')





fig = px.line(temp_confirmed, x="Date", y="Count", color='Country/Region', title='Confirmed')

fig.show()





fig = px.line(temp_deaths, x="Date", y="Count", color='Country/Region', title='Deaths')

fig.show()





fig = px.line(temp_recovered, x="Date", y="Count", color='Country/Region', title='Recovered')

fig.show()



us_temp = gdf[gdf['Country/Region'] == 'US']

us_temp['Date'] = pd.to_datetime(us_temp['Date'])

us_temp['lag_date'] = us_temp['Date'] - pd.DateOffset(days=14)

us_temp['Date'] = us_temp['lag_date']

us_temp.drop(columns=['lag_date'], inplace = True)

italy_temp = gdf[gdf['Country/Region'] == 'Italy']

lagged =  pd.concat([us_temp, italy_temp])



lagged_filter = lagged["Date"] <= "2020-3-5"

filtered = lagged.loc[lagged_filter]

filtered



lagged_confirmed = filtered.melt(id_vars=['Date', 'Country/Region'], value_vars=['Confirmed'], var_name='Case', value_name='Count')

lagged_deaths = filtered.melt(id_vars=['Date', 'Country/Region'], value_vars=['Deaths'], var_name='Case', value_name='Count')



fig = px.bar(lagged_confirmed, x="Date", y="Count", color='Country/Region', title='Confirmed', barmode='group')

fig.show()





fig = px.bar(lagged_deaths, x="Date", y="Count", color='Country/Region', title='Deaths', barmode='group')

fig.show()







co_positives = pd.read_csv('../input/colorado_testing_results.csv')

co_positives.columns = ['date','positives']

fig = px.line(co_positives, x="date", y="positives", title='Colorado COVID-19 Test Results')

fig.show()
# format time

co_positives['date']=pd.to_datetime(co_positives['date'])

co_positives
# Import time series library

p = np.poly1d(np.polyfit(co_positives.index, co_positives.positives, 3))

today = pd.Timestamp('today')

time_passed = pd.Timestamp('today') - co_positives.head(1).date

days_passed = time_passed.dt.days

days_passed
projected = []



for day in range(int(days_passed), int(days_passed+10), 1):

    projected.append(p(day))

days = [today.date() + pd.DateOffset(days=x) for x in range(0,10,1)]



projection = zip(days, projected)



projection = pd.DataFrame(list(projection))

projection.columns = ['date', 'positives']

projection['type'] = 'projected'

projection
co_positives['type'] = 'measured'

co_positives
co_projection = pd.concat([co_positives, projection])

co_projection.reset_index(drop=True)
fig = px.line(co_projection, x="date", y="positives", color='type', title='CO Covid-19 Test Results - projected with CO data')

fig.show()