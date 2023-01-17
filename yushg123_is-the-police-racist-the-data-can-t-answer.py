import numpy as np

import pandas as pd



import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

import plotly.express as px
data = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')
data
a = data['race'].value_counts().reset_index()

fig = px.pie(a, values='race', names='index', title='Race of Victim')

fig.show()
counts = []

counts.append(data[data['threat_level'] == 'attack'].groupby('race').count()['id'].values)

counts.append(data[data['threat_level'] == 'undetermined'].groupby('race').count()['id'].values)

counts.append(data[data['threat_level'] == 'other'].groupby('race').count()['id'].values)

race = ['A', 'B', 'H', 'N', 'O', 'W']



fig = go.Figure(data=[

    go.Bar(name='Attack', x=race, y=counts[0]),

    go.Bar(name='Undetermined', x=race, y=counts[1]),

    go.Bar(name='Other', x=race, y=counts[2])

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
data['armed'].unique()[0:6]
graphs = []

race = ['A', 'B', 'H', 'N', 'O', 'W']





for weapon in data['armed'].unique()[0:6]:

    graphs.append(go.Bar(name=weapon, x=race, y=data[data['armed'] == weapon].groupby('race').count()['id'].values))





fig = go.Figure(data=graphs)

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
graphs = []

race = ['A', 'B', 'H', 'N', 'O', 'W']





for weapon in data['flee'].unique():

    graphs.append(go.Bar(name=weapon, x=race, y=data[data['flee'] == weapon].groupby('race').count()['id'].values))





fig = go.Figure(data=graphs)

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
graphs = []

race = ['A', 'B', 'H', 'N', 'O', 'W']



#Plotly doesn't accept boolean values

data['signs_of_mental_illness'] = data['signs_of_mental_illness'].astype('object')



for weapon in data['signs_of_mental_illness'].unique():

    graphs.append(go.Bar(name=weapon, x=race, y=data[data['signs_of_mental_illness'] == weapon].groupby('race').count()['id'].values))





fig = go.Figure(data=graphs)

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
trace1 = go.Violin(



        x = data['age'],

    text = 'Age'

)

layout = dict(title = 'Distribution of Age')



iplot(go.Figure(data=trace1, layout=layout))
a = data['state'].value_counts().reset_index()

fig = px.pie(a, values='state', names='index', title='State Wise Shooting Count')

fig.show()
data
data[data['name'] == 'TK TK']
data[data['age'].isnull()]