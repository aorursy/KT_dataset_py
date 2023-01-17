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
df = pd.read_csv('../input/covid19-us-county-jhu-data-demographics/covid_us_county.csv')

df.head()
df.drop(['county'], axis=1)

df.drop(['fips'], axis=1)

df.drop(['lat'], axis=1)

df.drop(['long'], axis=1)
df_reg=df.groupby(['county']).agg({'cases':'sum','deaths':'sum'}).sort_values(["cases"],ascending=False).reset_index()

df_reg.head(10)
fig = go.Figure(data=[go.Table(

    columnwidth = [50],

    header=dict(values=('county', 'cases', 'deaths'),

                fill_color='#104E8B',

                align='center',

                font_size=14,

                font_color='white',

                height=40),

    cells=dict(values=[df_reg['county'].head(10), df_reg['cases'].head(10), df_reg['deaths'].head(10)],

               fill=dict(color=['#509EEA', '#A4CEF8',]),

               align='right',

               font_size=12,

               height=30))

])



fig.show()
df_reg.iplot(kind='box')
fig = px.pie(df_reg.head(10),

             values="cases",

             names="county",

             title="cases",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')

fig.show()
fig = px.pie(df_reg.head(10),

             values="deaths",

             names="county",

             title="deaths",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')

fig.show()
df_county=df.groupby(['date','county']).agg({'cases':'sum','deaths':'sum'}).sort_values(["cases"],ascending=False)

df_county.head(10)
dfd = df_county.groupby('date').sum()

dfd.head()
dfd[['cases','deaths']].iplot(title = 'US Counties Situation Over Time')
dfd['Active'] = dfd['cases']-dfd['deaths']

dfd['Active'] 
df_county.head()