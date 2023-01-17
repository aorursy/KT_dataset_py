import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML

import plotly.express as px

import plotly.graph_objects as go

import seaborn as sns

from plotly.subplots import make_subplots

%matplotlib inline





import plotly.tools as tls

import cufflinks as cf

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True)



print(__version__) # requires version >= 1.9.0

cf.go_offline()
df1 = pd.read_csv('../input/uncover/New_York_Times/covid-19-state-level-data.csv')

df1.head()
df1_reg=df1.groupby(['state']).agg({'cases':'sum','deaths':'sum'}).sort_values(["cases"],ascending=False).reset_index()

df1_reg.head(10)
fig = go.Figure(data=[go.Table(

    columnwidth = [50],

    header=dict(values=('state', 'cases', 'deaths'),

                fill_color='#104E8B',

                align='center',

                font_size=14,

                font_color='white',

                height=40),

    cells=dict(values=[df1_reg['state'].head(10), df1_reg['cases'].head(10), df1_reg['deaths'].head(10)],

               fill=dict(color=['#509EEA', '#A4CEF8',]),

               align='right',

               font_size=12,

               height=30))

])



fig.show()
df1_reg.iplot(kind='box')
fig = px.pie(df1_reg.head(10),

             values="cases",

             names="state",

             title="cases",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')

fig.show()
fig = px.pie(df1_reg.head(10),

             values="deaths",

             names="state",

             title="deaths",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')

fig.show()
df1_state=df1.groupby(['date','state']).agg({'cases':'sum','deaths':'sum'}).sort_values(["cases"],ascending=False)

df1_state.head(10)
dfd = df1_state.groupby('date').sum()

dfd.head()
dfd[['cases','deaths']].iplot(title = 'US States Situation Over Time')
dfd['Active'] = dfd['cases']-dfd['deaths']

dfd['Active'] 
df1_state.head()
df_st_ct = pd.value_counts(df1['state'])

df_st_ct.head()