from datetime import datetime as dt



import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv('../input/Competitions.csv')
df['DeadlineDate'] = pd.to_datetime(df['DeadlineDate'], format='%m/%d/%Y %I:%M:%S %p')
df = df[df['DeadlineDate'] > pd.to_datetime('2018-10-01 00:00:00')][df['TotalSubmissions'] > 1000]
df.head()
# import plotly

import plotly.plotly as py

import plotly.graph_objs as go



# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis

data = [

    go.Bar(x=df.Title, y=df.TotalTeams, name='TotalTeams'),

    go.Bar(x=df.Title, y=df.TotalCompetitors, name='TotalCompetitors'),

    go.Bar(x=df.Title, y=df.TotalSubmissions, name='TotalSubmissions', yaxis='y2'),

]



# specify the layout of our figure

layout = dict(title = "Kaggle Competitions",

              xaxis = dict(title = 'Competition Title'),

              yaxis = dict(title = 'TotalTeams/TotalCompetitors'),

              yaxis2 = dict(title = 'TotalSubmissions', overlaying='y', side ='right'),

              barmode = 'group',

             )



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)