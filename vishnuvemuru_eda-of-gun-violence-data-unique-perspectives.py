# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt, matplotlib
import matplotlib.cm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import seaborn as sns

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
import string, random
init_notebook_mode(connected=True)

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rcParams.update({'font.size': 20})
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
gun = pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv', parse_dates=True)

percent_NA = pd.Series(["{0:.2f}%".format(val * 100) for val in gun.isnull().sum()/gun.shape[0]], index=list(gun.keys()))
print(percent_NA)
gun_NA_gs = gun[['gun_stolen', 'gun_type', 'n_guns_involved']].isnull().any(axis=1)
NA_gs_bystate = gun[gun_NA_gs].groupby('state')['n_killed'].value_counts().unstack('state')
kills_bystate = gun.groupby('state')['n_killed'].value_counts().unstack('state')

plt.figure(1, figsize=(16, 6))
# (100* NA_gs_bystate.sum(axis=0)/kills_bystate.sum(axis=0)).sort_values().plot.bar()
val = (100* NA_gs_bystate.sum(axis=0)/kills_bystate.sum(axis=0)).sort_values()
g = sns.barplot(x=val.index, y=val.values)
g.set_xticklabels(labels=val.index, rotation=90)
plt.title('All reported cases considered')
plt.ylabel('% of missing Gun Data per Incident')
city_fatalities = gun.groupby('city_or_county')['n_killed'].sum().sort_values(ascending=False)[:20]
city_latandlong = gun.groupby('city_or_county')[['latitude', 'longitude']].mean().loc[city_fatalities.index]
state_injured = gun.groupby('state')['n_injured'].sum().sort_values(ascending=False)
state_killed = gun.groupby('state')['n_killed'].sum().sort_values(ascending=False)
state_incidents = gun.groupby('state')['n_killed'].count()
# city_fatalities = gun.groupby('city_or_county')['n_killed'].sum().sort_values(ascending=False)[:20]
# latandlong = gun.groupby('city_or_county')[['latitude', 'longitude']].mean().loc[city_fatalities.index]

def plot_statewide_trend(state_trend, plot_title = 'State wise trends'):
    state_to_code = {'District of Columbia' : 'dc', 'Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI', 'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV', 'Maine': 'ME'}
    df = pd.DataFrame()
    df['state'] = state_trend.index
    df['counts'] = state_trend.values
    df['state_code'] = df['state'].apply(lambda x : state_to_code[x])
    data = [ dict(type='choropleth', colorscale = 'Reds', 
            autocolorscale = False, locations = df['state_code'],
            z = df['counts'], locationmode = 'USA-states',
            text = df['state'], marker = dict(line = dict(color = 'rgb(0, 0, 0)', width = 1)),
            colorbar = dict())]

    layout = dict(title = plot_title, geo=dict(scope='usa', projection=dict(type='albers usa'), 
                                               showlakes=True, lakecolor='rgb(255, 255, 255)'))
    fig = dict(data=data,layout=layout)
    iplot(fig, filename='map')
plot_statewide_trend(state_incidents, 'State wise number of Gun Violence Incidents')
plot_statewide_trend(state_killed, 'State wise number of lives lost to Gun Violence')
def plot_city_trend(city_trend, latandlong, plot_title = 'City wise trends'):
    data = [ dict(type='scattergeo', locationmode = 'USA-states',# colorscale = 'Reds', autocolorscale = False, 
            lon=latandlong['longitude'], lat=latandlong['latitude'], text=city_trend.index,
            mode='markers',
            marker = dict(size = city_trend.values/20, opacity=0.7, cmin=0))]

    layout = dict(title = plot_title, colorbar=True, geo=dict(scope='usa', projection=dict(type='albers usa'), 
                                                              subunitcolor='rgb(0, 0, 0)', subunitwidth=0.5))
    fig = dict(data=data,layout=layout)
    iplot(fig, validate=False)
plot_city_trend(city_fatalities, city_latandlong, 'Top 20 Deadliest cities in US')
print(city_fatalities)

gun_i = gun[gun['state'] == 'Illinois']
gun_c = gun[gun['state'] == 'California']
gun_t = gun[gun['state'] == 'Texas']
gun_f = gun[gun['state'] == 'Florida']
gun_m = gun[gun['state'] == 'Michigan']

gun_is = gun_i.groupby(['n_killed', 'city_or_county'])['n_injured'].value_counts().unstack('city_or_county')
gun_cs = gun_c.groupby(['n_killed', 'city_or_county'])['n_injured'].value_counts().unstack('city_or_county')
gun_ts = gun_t.groupby(['n_killed', 'city_or_county'])['n_injured'].value_counts().unstack('city_or_county')
gun_fs = gun_f.groupby(['n_killed', 'city_or_county'])['n_injured'].value_counts().unstack('city_or_county')
gun_ms = gun_m.groupby(['n_killed', 'city_or_county'])['n_injured'].value_counts().unstack('city_or_county')

gun_is.fillna(0, inplace=True)
gun_cs.fillna(0, inplace=True)
gun_ts.fillna(0, inplace=True)
gun_fs.fillna(0, inplace=True)

# Gun Database grouped by state and city
gun_groupedbystateandcity = gun.groupby(['state', 'city_or_county'])
# Gun Database longitude and latitude mean locations 
gun_bystateandcity_loc = gun_groupedbystateandcity[['latitude', 'longitude']].mean()
# Gun Database - statistics of number killed
gun_bystateandcity_kills = gun_groupedbystateandcity['n_killed'].value_counts()
# Gun Database - statistics of number injured
gun_bystateandcity_injured = gun_groupedbystateandcity['n_injured'].value_counts()

def get_state_stats(state, statistics_scale=20):
    # Incident statistic indices where no fatalities occur
    killzero = gun_bystateandcity_kills.loc[state].index.get_level_values(1) == 0
    counties = gun_bystateandcity_kills.loc[state][killzero].sort_values(ascending=False)[:20].index.get_level_values(0)
    killzerostats = gun_bystateandcity_kills.loc[state][killzero].loc[list(counties)].sort_values(ascending=False).values
    # Scaling the statistics using some measure to show 
    killzerostats = (killzerostats)/statistics_scale
    nkillednumbers = gun_groupedbystateandcity['n_killed'].sum().loc[state].sort_values(ascending=False)[:20]
    latandlong = gun_bystateandcity_loc.loc[state].loc[nkillednumbers.index]
    # Scaling the statistics using some measure to show 
    nkillednumbers = nkillednumbers/statistics_scale
    return killzerostats, nkillednumbers, latandlong

def plot_stats_stackedbar(gun_state):
    highcrimeareas = gun_state.sum().sort_values()[-20:].index

    data =[go.Bar(
        x=[str(indval) for indval in gun_state[highcrimeareas].loc[0:2].index],
        y=gun_state[highcrimeareas].loc[0:2].values[:,i],
        name=gun_state[highcrimeareas].loc[0:2].columns[i]
    ) for i in range(20)]

    layout = go.Layout(barmode='stack', width=800, height=600, 
                       xaxis=dict(title='Number of deaths, Number of injuries'),
                      yaxis=dict(title='Incident Statistics'))
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
   

def plot_state_focused(state_focused, latandlong, lonaxis, lataxis, state_scope, plot_title = 'State trends'):
    data = [ dict(type='scattergeo', locationmode = 'usa', # colorscale = 'Reds', autocolorscale = False, 
            lon=latandlong['longitude'], lat=latandlong['latitude'], text=nkillednumbers.index + ' - ' + nkillednumbers.astype('str'),#state_focused,
            mode='markers', marker = dict(size = state_focused/10, opacity=0.7, cmin=0))]

    layout = dict(title = plot_title, colorbar=True, 
                  geo=dict(resolution = 50, width=1000, height=1000, scope = state_scope, #['CA', 'AZ', 'Nevada', 'Oregon', ' Idaho'], 
                           showframe = True, showland = True, landcolor = "rgb(229, 229, 229)", showrivers = True,
                           showlakes = True,
                           showsubunits=True, subunitcolor = "#111",subunitwidth = 2,
#                           countrycolor = "rgb(0, 0, 0)", 
                           coastlinecolor = "rgb(0, 0, 0)", 
                           projection = dict(type = "Mercator"), # Mercator
                          county_outline={'color': 'rgb(0,0,0)', 'width': 2.5}, 
                           lonaxis = dict(range=lonaxis), lataxis = dict(range=lataxis), domain = dict(x = [0, 1], y = [0, 1])))
    fig = dict(data=data,layout=layout)
    iplot(fig, validate=False)

killzerostats, nkillednumbers, latandlong = get_state_stats('California', 1)

plot_stats_stackedbar(gun_cs)
plot_state_focused(nkillednumbers, latandlong, [-125, -114], [30.0, 40], 'CA', 'California State Numer of lives lost to gun violence')
killzerostats, nkillednumbers, latandlong = get_state_stats('Texas', 1)

plot_stats_stackedbar(gun_ts)
plot_state_focused(nkillednumbers, latandlong, [-105, -90], [25.0, 35], 'TX', 'Texas Gun Violence trends')
killzerostats, nkillednumbers, latandlong = get_state_stats('Florida', 1)

plot_stats_stackedbar(gun_fs)
plot_state_focused(nkillednumbers, latandlong, [-90, -78], [23.0, 35], 'FL', 'Florida Gun Violence trends')
killzerostats, nkillednumbers, latandlong = get_state_stats('Illinois', 1)

plot_stats_stackedbar(gun_is)
plot_state_focused(nkillednumbers, latandlong, [-95, -85], [35.0, 45], 'IL', 'Illinois Gun Violence trends')
gun_bystate = gun.groupby(['state'])
xval = gun_bystate['n_killed'].sum()
yval = gun_bystate['n_injured'].sum()
zval = xval + yval

data = [{'x': gun_bystate['n_killed'].sum()[0:6],'y': gun_bystate['n_injured'].sum()[0:6],
        'mode': 'markers',
        'marker': {'size': gun_bystate['n_killed'].sum(), 'showscale': True}}]
data = [{'x': xval, 'y': yval,
        'text': gun_bystate['n_killed'].sum().index,
        'mode': 'markers',
        'marker': { 'color': zval/100, 'size': zval/100, 'showscale': True}}]
layout = go.Layout(autosize=False,
    width=800, height=700,
    title='State wise Gun Fatality Statistics',
    xaxis=dict(title='Number of people killed'),
    yaxis=dict(title='Number of people injured'),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)
fig = dict(data=data, layout=layout)
iplot(fig)
from collections import Counter

total_incidents = []
for i, each_inc in enumerate(gun['incident_characteristics'].fillna('Not Available')):
    split_vals = [x for x in re.split('\|', each_inc) if len(x)>0]
    total_incidents.append(split_vals)
    if i == 0:
        unique_incidents = Counter(split_vals)
    else:
        for x in split_vals:
            unique_incidents[x] +=1

unique_incidents = pd.DataFrame.from_dict(unique_incidents, orient='index')
colvals = unique_incidents[0].sort_values(ascending=False).index.values
find_val = lambda searchList, elem: [[i for i, x in enumerate(searchList) if (x == e)][0] for e in elem]

a = np.zeros((gun.shape[0], len(colvals)))
for i, incident in enumerate(total_incidents):
    aval = find_val(colvals, incident)
    a[i, np.array(aval)] = 1
incident = pd.DataFrame(a, index=gun.index, columns=colvals)
prominent_incidents = incident.sum()[[4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                                      21, 23]]
fig = {
    'data': [
        {
            'labels': prominent_incidents.index,
            'values': prominent_incidents,
            'type': 'pie',
            'hoverinfo':'label+percent+name',
            "domain": {"x": [0, .45]},
        }
    ],
    'layout': {'title': 'Prominent Incidents of Gun Violence',
               'showlegend': False}
}
iplot(fig)

print('Number of people affected by Felons with guns')
print(gun[incident.iloc[:, 9]>0][['n_killed', 'n_injured']].sum())
make_dict_from_entry = lambda value: dict([re.split(':+', item) for item in [s for s in re.split(r'(\|)', value) if len(s)>1]])
d = {'Gender' : pd.DataFrame(gun['participant_gender'].fillna('0::Not Available').map(make_dict_from_entry).tolist()), 
     'Type': pd.DataFrame(gun['participant_type'].fillna('0::Not Available').map(make_dict_from_entry).tolist()),
     'Status': pd.DataFrame(gun['participant_status'].fillna('0::Not Available').map(make_dict_from_entry).tolist())
    }
p = pd.concat(d.values(), axis=1, keys=d.keys())

find_stats = lambda x: x.unstack().transpose().groupby(['Gender', 
                                        'Type'])['Status'].value_counts().unstack(['Type', 
                                                                                   'Gender']).sum()
Participant = p.apply(find_stats, axis=1)
Participant.loc[49384]['Victim', 'Male'] = 1
Participant1 = Participant.drop(['Male, female', 'Not Available'], axis=1, level=1)
Participant1 = Participant1.fillna(0)
# Filtering only male suspect incidents
Male_suspect = (Participant1['Subject-Suspect', 'Male']>0) & (Participant1['Subject-Suspect', 'Female']==0)
# Filtering only female suspect incidents
Female_suspect  = (Participant1['Subject-Suspect', 'Female']>0) & (Participant1['Subject-Suspect', 'Male']==0)
# Filtering incidents where suspects are male and female
Mix_suspect = (Participant1['Subject-Suspect', 'Male']>0) & (Participant1['Subject-Suspect', 'Female']>0)
# Filtering Domestic Violence incidents
Domestic_indices = (incident['Domestic Violence']>0)

gun[Male_suspect]['n_killed'].sum(), gun[Female_suspect]['n_killed'].sum(), gun[Mix_suspect]['n_killed'].sum()
fig = {
    'data': [
        {
            'labels': ['Male suspect only', 'Female suspect only', 'Both Male as well as Female suspects', 'Suspect information unavailable'],
            'values': [gun[Male_suspect]['n_killed'].sum(), gun[Female_suspect]['n_killed'].sum(), gun[Mix_suspect]['n_killed'].sum(), gun[~(Male_suspect | Female_suspect | Mix_suspect)]['n_killed'].sum()],
            'type': 'pie',
            'hoverinfo':'label+percent+name',
            "domain": {"x": [0, .45]},
        }
    ],
    'layout': {'title': 'Gender wise decomposition of the fatalities caused by suspects',
               'showlegend': True}
}
iplot(fig)
male_incident = incident[Male_suspect].sum().sort_values(ascending=False)[[3, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17, 18,21, 23, 24, 25, 27, 28, 29, 30, 31, 32, 34, 36, 37, 38, 40, 41, 42, 43, 44, 45, 47, 48, 50, 53]]/incident[Male_suspect].sum().sum()
fig = {
    'data': [
        {
            'labels': male_incident.index,
            'values': male_incident,
            'type': 'pie',
            'hoverinfo':'label+percent+name',
            "domain": {"x": [0, .45]},
        }
    ],
    'layout': {'title': 'When suspect is male',
               'showlegend': False}
}
iplot(fig)


female_incident = incident[Female_suspect].sum().sort_values(ascending=False)[[3, 5, 6, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20]]

fig = {
    'data': [
        {
            'labels': female_incident.index,
            'values': female_incident,
            'type': 'pie',
            'hoverinfo':'label+percent+name',
            "domain": {"x": [0, .35]},
        }
    ],
    'layout': {'title': 'When suspect is female',
               'showlegend': False}
}
iplot(fig)


trace1 = go.Bar(x=['Female Suspect','Male Suspect','Female Victim','Male Victim'], 
                y=Participant1[Male_suspect & Domestic_indices].sum().values,
               marker=dict(color=['rgba(28,45,204,1)', 'rgba(222,45,38,1)',
               'rgba(204,28,104,1)', 'rgba(24,204,20,1)']))
trace2 = go.Bar(x=['Female Suspect','Male Suspect','Female Victim','Male Victim'], 
                y=Participant1[Female_suspect & Domestic_indices].sum().values,
               marker=dict(color=['rgba(28,45,204,1)', 'rgba(222,45,38,1)',
               'rgba(204,28,104,1)', 'rgba(24,204,20,1)']), yaxis='y2')

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Male suspect only', 'Female suspect only'), shared_yaxes=True, print_grid=False)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)

layout = go.Layout(
    xaxis=dict(domain=[0, 0.45], title='Considering only Male Suspects'), xaxis2=dict(domain=[0.5, 1], title='Considering only Female Suspects'),
    yaxis=dict(title='Number of Suspects and Victims')
)
fig = go.Figure(data=[trace1, trace2], layout=layout)
iplot(fig)


