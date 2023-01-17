# datetime operations
from datetime import timedelta

# for numerical analyis
import numpy as np

# to store and process data in dataframe
import pandas as pd

# to interface with operating system
import os

# basic visualization package
import matplotlib.pyplot as plt

# advanced ploting
import seaborn as sns

# interactive visualization
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# for offline ploting
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)

# for trendlines
import statsmodels

# hide warnings
import warnings
warnings.filterwarnings('ignore')
# list files
!ls ../input/corona-virus-report
# Create an empty list
files = []

# Fill the list with the file names of the CSV files in the Kaggle folder
for dirname, _, filenames in os.walk('../input/corona-virus-report/'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))

# Sort the file names
files = sorted(files)

# Output the list of sorted file names
files
series = [pd.read_csv(f, na_values=['.']) for f in files]

# Define series name, which becomes the dictionary key
series_name = ['1','2','3','4','5','6']

# series name = dictionary key, series = dictionary value
series_dict = dict(zip(series_name, series))
country_wise = series_dict['1']
temp1 = country_wise['Country/Region'].str.contains('Singapore')
SG_latest = country_wise[temp1]
SG_latest = SG_latest[['Deaths','Recovered','Active']]
SG_latest = SG_latest.melt(value_vars=['Active', 'Deaths', 'Recovered'])
fig = px.treemap(SG_latest, path=["variable"], values="value", height=225)
fig.data[0].textinfo = 'label+text+value'
fig.show()
full_grouped = series_dict['4']
temp2 = full_grouped['Country/Region'].str.contains('Singapore')
SG_trend = full_grouped[temp2]
SG_trend['Date'] = pd.to_datetime(SG_trend['Date'])
SG_trend_accumulate = SG_trend[['Date','Active','Deaths','Recovered']]
SG_trend_daily = SG_trend[['Date','New cases','New deaths','New recovered']]
acc = SG_trend_accumulate.melt(id_vars="Date", value_vars=['Deaths', 'Active', 'Recovered'],var_name='Case', value_name='Count')
fig = px.area(acc, x="Date", y="Count", color='Case', height=600, width=700,
             title='Accumulated Cases over time')
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
daily = SG_trend_daily.melt(id_vars="Date", value_vars=['New deaths','New cases','New recovered'],var_name='Case', value_name='Count')
fig = px.area(daily, x="Date", y="Count", color='Case', height=600, width=1200,
             title='New Cases over time')
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
worldmeter = series_dict['6']
worldmeter['InfectionRate'] = worldmeter['TotalCases']/worldmeter['Population']
worldmeter['DeathRate'] = worldmeter['TotalDeaths']/worldmeter['TotalCases']
worldmeter['SeriousRate'] = worldmeter['Serious,Critical']/worldmeter['TotalCases']
worldmeter['TestRate'] = worldmeter['TotalTests']/worldmeter['Population']
worldmeter.head()
world1 = worldmeter[['Country/Region','WHO Region','TotalCases','InfectionRate']].dropna().sort_values('InfectionRate',ascending=False)
world2 = worldmeter[['Country/Region','WHO Region','InfectionRate','DeathRate']].dropna().sort_values('DeathRate',ascending=False)
world3 = worldmeter[['Country/Region','WHO Region','SeriousRate']].dropna().sort_values('SeriousRate',ascending=False)
world4 = worldmeter[['Country/Region','WHO Region','TestRate']].dropna().sort_values('TestRate',ascending=False)
world1.reset_index(inplace=True)
world1.drop(['index'], axis=1,inplace=True)
world1[world1['Country/Region']=='Singapore']
fig = px.scatter(world1,x='Country/Region', y='InfectionRate',size='TotalCases',color='WHO Region',color_discrete_sequence = px.colors.qualitative.Dark2)
fig.update_layout(title='Infection Rate', xaxis_title="", yaxis_title="InfectionRate",xaxis_categoryorder = 'total ascending',
                  uniformtext_minsize=8, uniformtext_mode='hide',xaxis_rangeslider_visible=True)
fig.show()
world2.reset_index(inplace=True)
world2.drop(['index'], axis=1,inplace=True)
world2[world2['Country/Region']=='Singapore']
fig = px.scatter(world2,x='Country/Region', y='DeathRate',color='WHO Region',size='InfectionRate',color_discrete_sequence = px.colors.qualitative.Dark2)
fig.update_layout(title='Death Rate', xaxis_title="InfectionRate", yaxis_title="DeathRate",xaxis_categoryorder = 'total ascending',
                  uniformtext_minsize=8, uniformtext_mode='hide',xaxis_rangeslider_visible=True)
fig.show()
world3.reset_index(inplace=True)
world3.drop(['index'], axis=1,inplace=True)
world3[world3['Country/Region']=='Singapore']
world4.reset_index(inplace=True)
world4.drop(['index'], axis=1,inplace=True)
world4[world4['Country/Region']=='Singapore']
fig = px.bar(world4,x='Country/Region', y='TestRate',color='WHO Region',color_discrete_sequence = px.colors.qualitative.Dark2)
fig.update_layout(title='Death Rate', xaxis_title="InfectionRate", yaxis_title="DeathRate",xaxis_categoryorder = 'total ascending',
                  uniformtext_minsize=8, uniformtext_mode='hide',xaxis_rangeslider_visible=True)
fig.show()