import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import matplotlib.ticker as ticker





%matplotlib inline
df = pd.read_csv("../input/us-minimum-wage-by-state-from-1968-to-2017/Minimum Wage Data.csv", encoding="latin")

df.head()
df.describe().T
df.info()
df['State'].unique()
df_reg=df.groupby(['State']).agg({'High.Value':'sum','Low.Value':'sum'}).sort_values(["State"],ascending=False).reset_index()

df_reg.head(10)
fig = go.Figure(data=[go.Table(

    columnwidth = [50],

    header=dict(values=('State', 'High Value', 'Low Value'),

                fill_color='#104E8B',

                align='center',

                font_size=14,

                font_color='white',

                height=40),

    cells=dict(values=[df_reg['State'].head(10), df_reg['High.Value'].head(10), df_reg['Low.Value'].head(10)],

               fill=dict(color=['#509EEA', '#A4CEF8',]),

               align='right',

               font_size=12,

               height=30))

])



fig.show()
import plotly.express as px

fig = px.pie(df_reg.head(10),

             values="High.Value",

             names="State",

             title="High Value",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')

fig.show()
import plotly.express as px

fig = px.pie(df_reg.head(10),

             values="Low.Value",

             names="State",

             title="Low Value",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')

fig.show()