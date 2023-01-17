# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install calmap

!pip install folium

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import calmap

import folium



cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 



# converter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   



# hide warnings

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)
covid_19 = pd.read_csv('../input/covid_19.csv', parse_dates=['Date'])

covid_19_india=covid_19.rename(columns={"Cured":"Recovered"})

covid_19_india.sample(6)
#covid_19_india.info

#covid_19_india.describe
covid_19_india.isna().sum()
# Active Case = confirmed - deaths - Recovered

covid_19_india['Active'] = covid_19_india['Confirmed'] - covid_19_india['Deaths'] - covid_19_india['Recovered']



#len(covid_19_india['State/UnionTerritory'].unique())

covid_19_india['State/UnionTerritory'].unique()

covid_19_india.sample(6)
#state_wise breakdown

grouped_state=covid_19_india.groupby(['Date', 'State/UnionTerritory'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

grouped_state
temp = grouped_state.groupby(['State/UnionTerritory', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()

mask = temp['State/UnionTerritory'] != temp['State/UnionTerritory'].shift(1)

#mask

temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan
temp.columns = ['State/UnionTerritory','Date', 'New cases', 'New deaths', 'New recovered']

grouped_state = pd.merge(grouped_state, temp, on=['State/UnionTerritory', 'Date'])
#filling NaN with 0

grouped_state=grouped_state.fillna(0)
cols=['New cases','New deaths','New recovered']

grouped_state[cols]=grouped_state[cols].astype('int')



grouped_state['New cases']=grouped_state['New cases'].apply(lambda x: 0 if x<0 else x)

grouped_state.head()
#daywise table

day_wise = grouped_state.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases'].sum().reset_index()



# number cases per 100 cases

day_wise['Deaths/100 Cases'] = round((day_wise['Deaths']/day_wise['Confirmed'])*100, 2)

day_wise['Recovered/100 Cases'] = round((day_wise['Recovered']/day_wise['Confirmed'])*100, 2)

day_wise['Deaths/100 Recovered'] = round((day_wise['Deaths']/day_wise['Recovered'])*100, 2)



# no. of states

day_wise['No. of States'] = grouped_state[grouped_state['Confirmed']!=0].groupby('Date')['State/UnionTerritory'].unique().apply(len).values



# fillna by 0

cols = ['Deaths/100 Cases', 'Recovered/100 Cases', 'Deaths/100 Recovered']

day_wise[cols] = day_wise[cols].fillna(0)



day_wise.head()
# State wise

# getting latest values

state_wise = grouped_state[grouped_state['Date']==max(grouped_state['Date'])].reset_index(drop=True).drop('Date', axis=1)



# group by state

state_wise = state_wise.groupby('State/UnionTerritory')['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases'].sum().reset_index()



# # per 100 cases

state_wise['Deaths/100 Cases'] = round((state_wise['Deaths']/state_wise['Confirmed'])*100, 2)

state_wise['Recovered/100 Cases'] = round((state_wise['Recovered']/state_wise['Confirmed'])*100, 2)

state_wise['Deaths/100 Recovered'] = round((state_wise['Deaths']/state_wise['Recovered'])*100, 2)



cols = ['Deaths/100 Cases', 'Recovered/100 Cases', 'Deaths/100 Recovered']

state_wise[cols] = state_wise[cols].fillna(0)

state_wise.head()
population = pd.read_csv('../input/population_india_census2011.csv')

# select only population

population = population.iloc[:, 1:3]



# rename column names

population.columns = ['State/UnionTerritory', 'Population']

#population

state_wise['State/UnionTerritory']

# merged data

state_wise = pd.merge(state_wise, population, on='State/UnionTerritory', how='left')

state_wise.sample(5)
from datetime import timedelta

today = grouped_state[grouped_state['Date']==max(grouped_state['Date'])].reset_index(drop=True).drop('Date', axis=1)[['State/UnionTerritory', 'Confirmed']]

last_week = grouped_state[grouped_state['Date']==max(grouped_state['Date'])-timedelta(days=7)].reset_index(drop=True).drop('Date', axis=1)[['State/UnionTerritory', 'Confirmed']]



temp = pd.merge(today, last_week, on='State/UnionTerritory', suffixes=(' today', ' last week'))



# temp = temp[['State/UnionTerritory', 'Confirmed last week']]

temp['1 week change'] = temp['Confirmed today'] - temp['Confirmed last week']



temp = temp[['State/UnionTerritory', 'Confirmed last week', '1 week change']]



state_wise = pd.merge(state_wise, temp, on='State/UnionTerritory')

state_wise.sample(2)
cases_over = covid_19_india.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

cases_over = cases_over.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

#temp.sample(4)

fig = px.area(cases_over, x="Date", y="Count", color='Case', height=800,

             title='Cases over time', color_discrete_sequence = [rec, dth, act])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
fig_c = px.bar(day_wise, x="Date", y="Confirmed", color_discrete_sequence = [act])

fig_d = px.bar(day_wise, x="Date", y="Deaths", color_discrete_sequence = [dth])



fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                    subplot_titles=('Confirmed cases', 'Deaths reported'))



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)



fig.update_layout(height=480)
fig_1 = px.line(day_wise, x="Date", y="Deaths/100 Cases", color_discrete_sequence = [dth])

fig_2 = px.line(day_wise, x="Date", y="Recovered/100 Cases", color_discrete_sequence = [rec])

fig_3 = px.line(day_wise, x="Date", y="Deaths/100 Recovered", color_discrete_sequence = ['#333333'])



fig = make_subplots(rows=1, cols=3, shared_xaxes=False, 

                    subplot_titles=('Deaths/100 Cases', 'Recovered/100 Cases', 'Deaths/100 Recovered'))



fig.add_trace(fig_1['data'][0], row=1, col=1)

fig.add_trace(fig_2['data'][0], row=1, col=2)

fig.add_trace(fig_3['data'][0], row=1, col=3)



fig.update_layout(height=480)
fig_c = px.bar(day_wise, x="Date", y="New cases", color_discrete_sequence = [act])

fig_d = px.bar(day_wise, x="Date", y="No. of States", color_discrete_sequence = [dth])



fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                     subplot_titles=('No. of new cases everyday', 'No. of States'))



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)



fig.update_layout(height=480)
# confirmed - deaths

fig_c = px.bar(state_wise.sort_values('Confirmed').tail(15), x="Confirmed", y="State/UnionTerritory", 

               text='Confirmed', orientation='h', color_discrete_sequence = [act])

fig_d = px.bar(state_wise.sort_values('Deaths').tail(15), x="Deaths", y="State/UnionTerritory", 

               text='Deaths', orientation='h', color_discrete_sequence = [dth])



# recovered - active

fig_r = px.bar(state_wise.sort_values('Recovered').tail(15), x="Recovered", y="State/UnionTerritory", 

               text='Recovered', orientation='h', color_discrete_sequence = [rec])

fig_a = px.bar(state_wise.sort_values('Active').tail(15), x="Active", y="State/UnionTerritory", 

               text='Active', orientation='h', color_discrete_sequence = ['#333333'])



# death - recoverd / 100 cases

fig_dc = px.bar(state_wise.sort_values('Deaths/100 Cases').tail(15), x="Deaths/100 Cases", y="State/UnionTerritory", 

               text='Deaths/100 Cases', orientation='h', color_discrete_sequence = ['#f38181'])

fig_rc = px.bar(state_wise.sort_values('Recovered/100 Cases').tail(15), x="Recovered/100 Cases", y="State/UnionTerritory", 

               text='Recovered/100 Cases', orientation='h', color_discrete_sequence = ['#a3de83'])



# new - week change

fig_nc = px.bar(state_wise.sort_values('New cases').tail(15), x="New cases", y="State/UnionTerritory", 

               text='New cases', orientation='h', color_discrete_sequence = ['#574b90'])

fig_wc = px.bar(state_wise.sort_values('1 week change').tail(15), x="1 week change", y="State/UnionTerritory", 

               text='1 week change', orientation='h', color_discrete_sequence = ['#9e579d'])



fig = make_subplots(rows=5, cols=2, shared_xaxes=False, horizontal_spacing=0.13, vertical_spacing=0.08,

                    subplot_titles=('Confirmed cases', 'Deaths reported', 'Recovered', 'Active cases', 

                                    'Deaths/100 cases', 'Recovered/100 cases', 'New cases', 

                                    '1 week change'))





fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)

fig.add_trace(fig_r['data'][0], row=2, col=1)

fig.add_trace(fig_a['data'][0], row=2, col=2)



fig.add_trace(fig_dc['data'][0], row=3, col=1)

fig.add_trace(fig_rc['data'][0], row=3, col=2)

fig.add_trace(fig_nc['data'][0], row=4, col=1)

fig.add_trace(fig_wc['data'][0], row=4, col=2)



fig.update_layout(height=1500,width=1100)
temp = grouped_state[grouped_state['New cases']>0].sort_values('State/UnionTerritory', ascending=False)

fig = px.scatter(temp, x='Date', y='State/UnionTerritory', size='New cases', color='New cases', height=4000, 

           color_continuous_scale=px.colors.sequential.Viridis)

fig.update_layout(yaxis = dict(dtick = 1),height=900)

fig.show()
fig = px.bar(grouped_state, x="Date", y="Confirmed", color='State/UnionTerritory', height=600,

             title='Confirmed', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()



# =========================================



fig = px.bar(grouped_state, x="Date", y="Deaths", color='State/UnionTerritory', height=600,

              title='Deaths', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()



# # =========================================



fig = px.bar(grouped_state, x="Date", y="New cases", color='State/UnionTerritory', height=600,

             title='New cases', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
fig = px.line(grouped_state, x="Date", y="Confirmed", color='State/UnionTerritory', height=600,

             title='Confirmed', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()



# =========================================



fig = px.line(grouped_state, x="Date", y="Deaths", color='State/UnionTerritory',height=600,

             title='Deaths', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()



# =========================================



fig = px.line(grouped_state, x="Date", y="New cases", color='State/UnionTerritory', height=600,

             title='New cases', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
gt_100 = grouped_state[grouped_state['Confirmed']>100]['State/UnionTerritory'].unique()

temp = covid_19_india[grouped_state['State/UnionTerritory'].isin(gt_100)]

temp = temp.groupby(['State/UnionTerritory', 'Date'])['Confirmed'].sum().reset_index()

temp = temp[temp['Confirmed']>100]

#print(temp.head())



min_date = temp.groupby('State/UnionTerritory')['Date'].min().reset_index()

min_date.columns = ['State/UnionTerritory', 'Min Date']

#print(min_date.head())



from_100th_case = pd.merge(temp, min_date, on='State/UnionTerritory')

from_100th_case['N days'] = (from_100th_case['Date'] - from_100th_case['Min Date']).dt.days

#print(from_100th_case.head())

fig = px.line(from_100th_case, x='N days', y='Confirmed', color='State/UnionTerritory', title='N days from 100 case', height=800)

fig.show()
latest_data = covid_19_india[covid_19_india['Date'] == max(covid_19_india['Date'])]

                         

fig = px.treemap(latest_data.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 

                 path=["State/UnionTerritory"], values="Confirmed", height=700,

                 title='Number of Confirmed Cases',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'

fig.show()



fig = px.treemap(latest_data.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 

                 path=["State/UnionTerritory"], values="Deaths", height=700,

                 title='Number of Deaths reported',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'

fig.show()
temp = covid_19_india.groupby(['Date', 'State/UnionTerritory'])['Confirmed'].sum()

temp = temp.reset_index().sort_values(by=['Date', 'State/UnionTerritory'])

temp = temp[temp['State/UnionTerritory'].isin(gt_100)]



plt.style.use('seaborn')

g = sns.FacetGrid(temp, col="State/UnionTerritory", hue="State/UnionTerritory", sharey=False, col_wrap=5)

g = g.map(plt.plot, "Date", "Confirmed")

g.set_xticklabels(rotation=90)

plt.show()
temp = covid_19_india.groupby('Date')['Confirmed'].sum()

temp = temp.diff()



plt.figure(figsize=(20, 5))

ax = calmap.yearplot(temp, fillcolor='white', cmap='Reds', linewidth=0.5)
spread = covid_19_india[covid_19_india['Confirmed']!=0].groupby('Date')

spread = spread['State/UnionTerritory'].unique().apply(len).diff()



plt.figure(figsize=(20, 5))

ax = calmap.yearplot(spread, fillcolor='white', cmap='Greens', linewidth=0.5)