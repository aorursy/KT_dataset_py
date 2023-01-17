import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
dl=pd.read_csv('../input/uncover/UNCOVER/New_York_Times/covid-19-county-level-data.csv')
df.head()
df.info()
df.describe()
sns.heatmap(df.isnull())
fig=plt.gcf()
fig.set_size_inches(13,5)
msno.matrix(df)
df[df['fips'].isnull()].count()
df['fips'].count()
(3555/346810) * 100
df['fips']
df[df['fips'].isnull()].groupby(df['county']).count()
df[df['fips'].fillna(0).isnull()].count()
df.dropna(inplace = True)
msno.matrix(df)
df['date'] = pd.to_datetime(df['date'])
state_wise = df.groupby('state').sum()['cases'].sort_values(ascending=False)[0:10]
state_wise = pd.DataFrame(state_wise).reset_index()
plt.figure(figsize=(14,10))
s = sns.barplot(x='state', y='cases', data=state_wise, palette = 'plasma_r', alpha=0.6)
s.set_xlabel('Top States')
s.set_ylabel('Cases (in Millions)')
s.set_title ('Total cases in top 10 States')
sns.despine(left=True)

county_wise = df.groupby('county').sum()['cases'].sort_values(ascending=False)[0:10]
county_wise = pd.DataFrame(county_wise).reset_index()
plt.figure(figsize=(14,10))
c = sns.barplot(x='county', y='cases', data=county_wise, palette = 'magma')
c.set_xlabel('Top Counties')
c.set_ylabel('Cases (in 10,000)')
c.set_title ('Total cases in top 10 Counties')
sns.despine(left=True)
state = df[['cases', 'deaths']].groupby(df['state']).sum().sort_values('cases', ascending=False)[0:10]
state = state.reset_index()
plt.figure(figsize=(14,8))
d =sns.stripplot(x='deaths', y='cases', data=state, color= 'maroon', hue='state', s=30, palette='magma')
d.set_xlabel('Total Deaths')
d.set_ylabel('Total Cases')
d.set_title ('Deaths and Cases Matrix for top States')
sns.despine(left=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
county = df[['deaths', 'cases']].groupby(df['county']).sum().sort_values('deaths', ascending=False)
county = pd.DataFrame(county)
fig, ax =plt.subplots(nrows= 2, ncols = 2, figsize= (14,12))
w = sns.distplot(county['cases'][0:30], ax =ax[0][0],color='red')
g = sns.distplot(county[31:60], ax =ax[0][1])
h = sns.distplot(county[61:90], ax =ax[1][0])
j =sns.distplot(county[91:120], ax =ax[1][1], color='green')
sns.despine(left=True)
w.set_title('Cases Vs Deaths for Counties Top 30')
w.set_ylabel('Number of deaths(in Thousands)')
w.set_xlabel ('Cases (in 10,000)')

g.set_title('Cases Vs Deaths for Counties 31-60')
g.set_ylabel('Number of deaths(in Thousands)')
g.set_xlabel ('Cases (in 10,000)')

h.set_title('Cases Vs Deaths for Counties 61-91')
h.set_ylabel('Number of deaths(in Thousands)')
h.set_xlabel ('Cases (in 10,000)')

j.set_title('Cases Vs Deaths for Counties 91-120')
j.set_ylabel('Number of deaths(in Thousands)')
j.set_xlabel ('Cases (in 10,000)')

plt.figure(figsize=(12,8))
sns.lineplot(x='date', y='deaths', data=df, color='darkblue', label='Deaths')
de= sns.lineplot(x='date', y='cases', data=df, color='maroon', label='Cases')
de.set_xlabel('Date ')
de.set_ylabel('Total Cases and Deaths (in Thousands)')
de.set_title ('Timeline for Virus Spread')
sns.despine(left=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
import plotly.graph_objs as go

state_wise_1 = df.groupby('state').sum()
state_wise_1 = pd.DataFrame(state_wise_1)
state_wise_1 = state_wise_1.reset_index()
state_codes = {
    'District of Columbia' : 'dc','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME', 'Northern Mariana Islands' : 'MP', 'Puerto Rico': 'PR', 'Virgin Islands' : 'VI'}

state_wise_1['state_code'] = state_wise_1['state'].apply(lambda x : state_codes[x])
state_wise_1.head()
data = dict(type='choropleth',colorscale='RdBu_r', locations = state_wise_1['state_code'], locationmode = 'USA-states', z= state_wise_1['cases'], text = state_wise_1['state'], colorbar={'title':'Scale'},  marker = dict(line=dict(width=0))) 
layout = dict(title = 'COVID 19 Cases till Now!', geo = dict(scope='usa', showlakes=True, lakecolor = 'darkblue'))
Choromaps2 = go.Figure(data=[data], layout=layout)
iplot(Choromaps2)

data = dict(type='choropleth',colorscale='magma_r', locations = state_wise_1['state_code'], locationmode = 'USA-states', z= state_wise_1['deaths'], text = state_wise_1['state'], colorbar={'title':'Scale'},  marker = dict(line=dict(width=0))) 
layout = dict(title = 'COVID 19 Deaths till Now!', geo = dict(scope='usa', showlakes=True, lakecolor = 'grey'))
Choromaps2 = go.Figure(data=[data], layout=layout)
iplot(Choromaps2)
